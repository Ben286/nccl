/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "sym_kernels.h"
#include "kernel.cuh"
#include "primitives.cuh"
#include "gin_scratch__types.h"

// ============================================================================
// ncclSymkRun_AllGather_RailRing_LsaSTMC
// ============================================================================
// 这是 Symmetric Kernel 模式下 AllGather 的核心实现。
// 名称拆解：
//   AllGather   - 执行 AllGather collective（每个 rank 贡献一块数据，最终所有 rank 收到全部数据）
//   RailRing    - 算法结构：在 Rail 维度（同一 NIC rail 内的 rank 们）使用 Ring 算法
//   Lsa         - LSA（Local Symmetric Aggregate）：节点内 GPU 之间的数据共享层
//   STMC        - STore with MultiCast：使用 NVLink MultiCast 写入节点内多 GPU
//
// ===== 整体架构概述 =====
// 这个 kernel 同时处理两类数据传输：
//   1. 跨节点数据传输（via GIN）：GIN warp（threadIdx.x < 32）负责，用 gin.put() 沿 ring 转发
//   2. 节点内数据广播（via MultiCast）：Worker warps 负责，用 bcastMultimem() 把数据写到所有本地 GPU
//
// ===== 为什么需要 RailRing 而不是普通 Ring =====
// 在多节点场景中，NCCL 使用"Rail"概念：
//   - Rail 是跨节点的 rank 分组，例如 node0_gpu0, node1_gpu0, node2_gpu0 同属 Rail 0
//   - 节点内所有 GPU 共享同一段内存（symmetric memory），所以节点内传输用 MultiCast 而非 ring
//   - Ring 算法只在 Rail 维度运行（跨节点），节点内用 MultiCast 一次性广播
//
// ===== GIN（GPU Initiated Network）的作用 =====
//   gin.put(rail, peer, dst, src, size, signal, local_action, warps)：
//     GPU kernel 直接向远端写数据（RDMA write），不需要 CPU 介入
//     写完后向目标 rank 发送一个信号（Signal Increment），通知对方"我的 step N 数据已写好"
//   gin.waitSignal(warps, signalPtr, expectedValue, timeout)：
//     等待某个 signal 达到期望值（类似 spin wait on atomic counter）
//     等到意味着上一个节点已经把数据写到了 output buffer
//
// ===== Ring 流水线步骤解析 =====
// Ring AllGather 需要 (nRanks-1) 步：
//   Step 0：每个 rank 把自己的数据发给 ring 中的 nextPeer
//   Step 1：等待 prevPeer 的信号，再把收到的数据转发给 nextPeer
//   ...（每步转发一个不同 rank 的数据块）
// 最终每个 rank 都拥有所有 nRanks 的数据。
//
// ===== Signal 机制解析 =====
//   railSignals[prevPeer]：本 rank 从 prevPeer 那里监听信号
//   localSignalValue：本地跟踪的信号期望值（初始化为 *localSignalPtr，滚动递增）
//   每次 step 后 localSignalValue++，所以 waitSignal 等待的是单调递增的序列号
//   这是典型的 credit-based flow control：prevPeer 每发完一 step 就 signal +1，
//   本 rank 收到信号后才允许转发该 step 的数据
//
// ===== 两个并行路径的分工 =====
//   GIN warp（threadIdx.x < ringThreads=32）：
//     负责跨节点的 ring 传输，通过 gin.put() + gin.waitSignal() 实现环形流水线
//   Worker warps（threadIdx.x >= 32）：
//     负责节点内的 MultiCast 广播，将已收到的数据用 bcastMultimem() 写到本节点所有 GPU
//     Worker warps 跟 GIN warp 用相同的 waitSignal() 逻辑做同步，确保数据已到达再广播
//
// ===== ncclCoopCta / ncclCoopWarpSpan 概念 =====
//   ncclCoopCta：代表整个 CTA（CUDA Thread Block）的协作组
//   ncclCoopWarpSpan{warp0, nWarps, id}：代表从 warp0 开始的 nWarps 个 warp 的子协作组
//   这是 NCCL 的 Cooperative Thread Array 抽象，用于让一部分 warp 独立完成任务
//
// ===== bar.sync 的含义 =====
//   bar.sync(cta, memory_order, fence_level) 是 GIN Barrier：
//   确保所有参与者完成了 GIN 相关操作（包括网络 fence），以及 MultiCast 的内存一致性
//   在 kernel 开始时用 Relaxed 做一次全局同步，结束时用 release 保证对外可见性
// ============================================================================
__device__ __forceinline__ void ncclSymkRun_AllGather_RailRing_LsaSTMC(struct ncclSymkDevWorkArgs const* args) {
  ncclCoopCta cta;  // 整个 CTA 协作组（所有线程）
  ncclSymkArgsHandler handler(args);  // 解析 ncclSymkDevWorkArgs：提取 comm、devWork、ginSyncHandle 等
  ncclTeam rail = ncclTeamRail(handler.comm);  // Rail 团队：跨节点同一 NIC 对应的 rank 集合
  ncclGin gin(handler.comm, (int)(blockIdx.x % handler.comm.ginContextCount));
                                          // GIN 上下文：每个 block 使用一个 GIN context（round-robin）
                                          // ginContextCount 是预分配的总 context 数（连接建立时确定）
                                          // context 包含：到每个 peer 的 QP/channel、signal/counter 空间
  constexpr int chunkSize = ncclSymkAllGather_RailRing_ChunkSize;  // 编译期常量：每次 gin.put() 传输的最大字节数
                                          // 由编译器内联展开，要求 chunkSize 足够小使得流水线有意义
  ncclGinSignal_t railSignals = handler.ginSyncHandle.railSignals + blockIdx.x * rail.nRanks;
                                          // railSignals[blockIdx.x * rail.nRanks + r] 是本 block 对第 r 个 rail rank 的信号 slot
                                          // 各 block 独立：blockIdx.x * rail.nRanks 保证各 block 不共享信号 slot
                                          // railSignals 是远端可写的内存映射地址，即 GIN 内部的 signal 数组
  ncclBarrierSession<ncclCoopCta> bar(cta, ncclTeamTagWorld(), gin, blockIdx.x, /*multimem=*/true);
                                          // GIN Barrier session：用于整个 world 的同步
                                          // ncclTeamTagWorld()：标识这是一个 world-scope 同步
                                          // multimem=true：还要同步 NVLink MultiCast 内存操作
  int nextPeer = (rail.rank + 1) % rail.nRanks;  // Ring 中"下游"节点：本 rank 要把数据发给的下一个 rank
  int prevPeer = (rail.rank + rail.nRanks - 1) % rail.nRanks;  // Ring 中"上游"节点：数据从哪里来
  uint64_t* localSignalPtr = gin.getSignalShadowPtr(railSignals + prevPeer);
                                          // getSignalShadowPtr：获取 prevPeer 信号的本地 shadow 指针
                                          // shadow 是信号的本地副本，GPU 可直接读取，避免从网络侧读
                                          // 内部实现：经过内存映射的共享内存地址，CPU proxy 更新，GPU 读取
  uint64_t localSignalValue = *localSignalPtr;  // 当前已观测到的最新信号值（初始化为上一轮 kernel 结束时的值）
                                          // 每个 AllGather step 完成后递增，实现滚动信号追踪
                                          // 关键：跨 kernel 调用保持信号连续性，防止误触发
  const int ringThreads = WARP_SIZE;  // GIN warp 的线程数 = 1个 warp = 32 threads

  // ===== 全局初始 barrier =====
  // 等待所有参与本 collective 的 rank 都完成 setup，确保：
  //   1. symmetric memory（output buffer）已被所有节点注册
  //   2. GIN 连接已就绪（signal shadow 页面已映射）
  // 第二次 bar.sync（结束时）使用 memory_order_release，把结果写对外可见
  bar.sync(cta, cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  // ===== 主循环：遍历每个 work item（支持多个 AllGather 任务融合执行）=====
  // forEachWorkNoFusion<T>：迭代 devWork 列表中的每一个集合通信任务
  //   nElts：本 rank 贡献的元素数量
  //   nAllElts：AllGather 结果总元素数（= nElts * rail.nRanks）
  //   input：本 rank 的输入 buffer（symmetric 内存地址）
  //   output：AllGather 输出 buffer（接收所有 rank 的数据），大小 = nElts * rail.nRanks
  handler.template forEachWorkNoFusion<uint8_t>(
    [&]__device__(size_t nElts, size_t nAllElts, ncclSymPtr<uint8_t> input, ncclSymPtr<uint8_t> output) {

      // ===== 路径 A：GIN warp（threadIdx.x 0..31）- 跨节点 Ring 传输 =====
      // 1个 warp（32线程）负责所有跨节点的 GIN put 操作
      // 为什么只需要 1 warp：GIN put 是 DMA 操作，CPU 只需 post 一次描述符，不需要大量线程
      if (threadIdx.x < ringThreads) {
        ncclCoopWarpSpan warps(0, 1, 0);  // warp0 开始，共 1 个 warp，id=0
        // Ring AllGather 共 (nRanks-1) 步：
        // step=0: 发送本 rank 自己的数据（from input）→ nextPeer
        // step=1: 从 prevPeer 收到 step=0 的数据后，转发给 nextPeer
        // ...以此类推，最后每个 rank 都有了全部 nRanks 份数据
        for (int step = 0; step < rail.nRanks - 1; step++) {
          // dataPeer：本 step 正在流通的数据属于哪个 rail rank
          // 例如 rank=2, nRanks=4:
          //   step=0: dataPeer=2（自己的数据）
          //   step=1: dataPeer=1（来自 prevPeer 的数据）
          //   step=2: dataPeer=0
          int dataPeer = (rail.rank - step + rail.nRanks) % rail.nRanks;
          int dgrank = ncclTeamRankToWorld(handler.comm, rail, dataPeer);
          // dgrank：dataPeer 对应的 world rank（用于计算 output 中的偏移地址）
          // output buffer 布局：[rank0_data | rank1_data | ... | rankN-1_data]
          // 每段大小均为 nAllElts bytes
          size_t remainingElts = nElts;
          size_t offset = 0;
          if (dataPeer == rail.rank) {
            // === 后续步骤（step>=1）：转发来自 prevPeer 的数据 ===
            // 必须先等待 prevPeer 写好数据（即 prevPeer 的 signal 到达 localSignalValue+1）
            // === 第一步（step=0）：发送自己的数据 ===
            // 数据源：input（本 rank 自己的 send buffer）
            // 数据目标：nextPeer 的 output + dgrank * nAllElts（即 nextPeer 的 output 中属于本 rank 的那一段）
            // 信号：发送完成后向 nextPeer 递增 railSignals[rail.rank]，通知其"你的 step 0 数据已到"
            while (remainingElts) {
              size_t chunkElts = min(remainingElts, size_t(chunkSize));
              // Send data chunk to next peer in ring
              // 数据已到 output[dgrank*nAllElts + offset]，直接转发给 nextPeer
              // src 和 dst 都是 output 中同一位置（就地转发，达到后直接发出而不复制到中间缓冲区）
              // gin.put 参数详解：
              //   rail         - Rail 团队，确定目标 rank 的 GIN connection
              //   nextPeer     - Rail 内目标 rank 索引（不是 world rank）
              //   output + dgrank * nAllElts + offset - 远端目标地址（symmetric 内存，连续物理地址）
              //   input + offset  - 本地源地址
              //   chunkElts       - 传输字节数
              //   ncclGin_SignalInc{ railSignals + rail.rank }：
              //     完成信号：在目标端将 railSignals[rail.rank] +1
              //     在 GIN_PROXY 下：CPU proxy 收到 IB CQ 完成事件后写入 signal 内存
              //     在 GIN_GDAKI 下：GPU 直接写入目标 GPU 的 signal 内存
              //   ncclGin_None{}  - 不需要本地完成回调
              //   warps           - 执行此操作的 warp 组
              gin.put(rail, nextPeer, output + dgrank * nAllElts + offset,
                input + offset, chunkElts,
                ncclGin_SignalInc{ railSignals + rail.rank }, ncclGin_None{}, warps);
              offset += chunkElts;
              remainingElts -= chunkElts;
              // 需要每次 chunk 都递增：因为 prevPeer 每发送一个 chunk 就 signal +1
              // 如果 nElts 比 chunkSize 大，就需要多次 wait/put/signal
            }

          } else {
            // === 本 rank 自己的数据：直接 MultiCast 写到节点内所有 GPU ===
            // 不需要等待信号，因为 input 是本 rank 的数据，已经准备好了
            while (remainingElts) {
              size_t chunkElts = min(remainingElts, size_t(chunkSize));
              // Wait for ready signal from next peer before sending
              // 等待 prevPeer 完成对本 rank 的写入
              // railSignals + prevPeer：监听来自 prevPeer 的信号 shadow 地址
              // localSignalValue + 1：期望信号达到此値（单调递增，每 step +1）
              // 32：超时等待轮数（spin count）
              // 实质是 spin wait on shadow：GPU 不断读取内存映射页面中的 signal 値，
              // CPU proxy 当 IB RDMA write 完成时更新该値
              gin.waitSignal(warps, railSignals + prevPeer, localSignalValue + 1, 32);
              // Send data chunk to next peer in ring
              gin.put(rail, nextPeer, output + dgrank * nAllElts + offset,
                output + dgrank * nAllElts + offset, chunkElts,
                ncclGin_SignalInc{ railSignals + rail.rank }, ncclGin_None{}, warps);
              offset += chunkElts;
              remainingElts -= chunkElts;
              localSignalValue++;  // 记录已消费了一个信号，下次等待 +1
            }
          }
        }
        // flush：确保所有 GIN 描述符都被提交到 NIC 发送队列
        // GIN_PROXY 下：将尚未写入 GFD 队列的最后少量描述符的 pi 更新，让 CPU proxy 能看到
        // GIN_GDAKI 下：GPU 直接刷新 QP 发送队列（doorbell flush）
        gin.flush(warps);
      } else {
        // ===== 路径 B：Worker warps（threadIdx.x >= 32）- 节点内 MultiCast 广播 =====
        // 多个 worker warp 负责把 GIN 收到的数据用 NVLink MultiCast 写到本节点所有 GPU
        // bcastMultimem() = MultiCast Store：一次写，所有本节点 GPU 都收到（免去逐 GPU 循环）
        ncclCoopWarpSpan warps(1, blockDim.x / WARP_SIZE - 1, 1);
        // Loop through rail ranks starting from itself
        // Worker warps 需要处理 nRanks 步（包括自己的数据），比 GIN warp 多一步
        // 选择 nRanks 而非 nRanks-1：自己的 MultiCast 不需要等信号，直接发送
        for (int step = 0; step < rail.nRanks; step++) {
          int dataPeer = (rail.rank - step + rail.nRanks) % rail.nRanks;
          int dgrank = ncclTeamRankToWorld(handler.comm, rail, dataPeer);
          size_t remainingElts = nElts;
          size_t offset = 0;
          if (dataPeer == rail.rank) {
            // === 其他 rank 的数据：等 GIN warp 收到后，再 MultiCast 广播 ===
            // 这里的 waitSignal 与 GIN warp 共用同一个信号数组（railSignals + prevPeer）
            // 同步语义：等 prevPeer 的信号递增到 localSignalValue+1，说明数据已写好
            // 注意：GIN warp 和 Worker warp 使用同一个 localSignalValue，
            //   因此两者的步调是对齐的（都等同一个信号，处理同一 step 的数据）
            while (remainingElts) {
              size_t chunkElts = min(remainingElts, size_t(chunkSize));
              // Put self rank's data
              // bcastMultimem 参数详解：
              //   handler             - 提供 comm 信息和 MultiCast 红柄
              //   warps.num_threads() - worker warp 的总线程数
              //   warps.thread_rank() - 当前线程在 worker warp 中的索引
              //   input + offset      - 本地数据源
              //   output + dgrank * nAllElts + offset - 远端 MultiCast 目标地址
              //                         本节点所有 GPU 将收到写入该地址的数据
              //   chunkElts           - 传输字节数
              // NVLink MultiCast 硬件实现：一次内存写操作，通过 NVLink 气组一对多广播
              // 比逐 GPU 循环写入快得多，随节点内 GPU 数多益切
              bcastMultimem(handler, warps.num_threads(), warps.thread_rank(), input + offset, output + dgrank * nAllElts + offset, chunkElts);
              offset += chunkElts;
              remainingElts -= chunkElts;
            }
          } else {
            while (remainingElts) {
              size_t chunkElts = min(remainingElts, size_t(chunkSize));
              // Wait for signal from other peers before putting their data
              gin.waitSignal(warps, railSignals + prevPeer, localSignalValue + 1, 32);
              // 数据已到本 rank 的 output buffer，用 MultiCast 扩散到本节点所有 GPU
              // 同一个地址同时是读源和写目标（GIN RDMA write 已写入，现在再 MultiCast 广播）
              bcastMultimem(handler, warps.num_threads(), warps.thread_rank(), output + dgrank * nAllElts + offset, output + dgrank * nAllElts + offset, chunkElts);
              offset += chunkElts;
              remainingElts -= chunkElts;
              localSignalValue++;
            }
          }
        }
      }
    }
  );

  // update the shadow signal value
  // ===== 更新信号 shadow 値 =====
  // threadIdx.x == ringThreads（即第一个 worker warp 的第一个线程）负责把
  // 本轮结束时的 localSignalValue 写回 shadow 指针。
  // 下一次执行此 kernel 时，从 *localSignalPtr 恢复，实现跨 kernel 调用的信号连续性。
  // 为什么要 shadow：GIN signal 的完整値存在 GIN device handle 侧，
  //   GPU 读取需要通过 shadow ptr（mapped memory / P2P 内存）而非网络
  // 为什么由 threadIdx.x == ringThreads 写入：
  //   这是 worker warp 0 的第 0 线程，与 GIN warp 互不干扰。
  //   只需一个线程写回即可（GIN warp 和 worker warp 的 localSignalValue 最终相同）
  if (threadIdx.x == ringThreads) {
    *localSignalPtr = localSignalValue;
  }
  // 结尾 barrier：memory_order_release 确保所有写操作（output buffer + MultiCast）对外可见
  // 下一个 kernel 或 CPU 侧操作看到这个 barrier 完成就意味着 AllGather 结果已就绪
  bar.sync(cta, cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
}
