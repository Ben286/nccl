#include "sym_kernels.h"
#include "kernel.cuh"
#include "primitives.cuh"
#include "data_ops.cuh"

// ============================================================================
// rsAlgoHier、ncclSymkRun_ReduceScatter_RailA2A_LsaLD、…LsaLDMC
// ============================================================================
// 这是 Symmetric Kernel 模式下 ReduceScatter 的核心实现。
// 名称拆解：
//   ReduceScatter  - ReduceScatter collective：每个 rank 贡献数据并展开 reduce，最终每个 rank 拥有全部的一部分 reduce 结果
//   RailA2A        - Rail 维度用 All-to-All 模式：每个 rank 将自己的资料展开后分别发给所有 rail 同伴
//   Lsa            - LSA（Local Symmetric Aggregate）：节点内的 GPU 之间共享内存/reduce 层
//   LD / LDMC      - Load（普通内存读） / Load with MultiCast（NVLink MultiCast 读）
//
// ===== 整体算法部层结构 =====
//
//  层 1：Rail A2A（跨节点，通过 GIN）
//    所有 rail rank 将各自所负责并加的 chunk 发给其他节点同一位置的 GPU
//    具体：rank r 将 input[dstWorld*nAllElts : dstWorld*nAllElts+nElts] 发给 rail 中属于 dstWorld 的 rank
//    这里用 All-to-All 而非 Ring，因为居于 Rail 内所有 rank 都只需要一个 chunk，不全流转
//
//  层 2：LSA Reduce（节点内，通过 shared memory / NVLink MultiCast）
//    每个 GPU 将本节点内所有 GPU 的相同位置 chunk reduce 在一起
//    得到 world-rank 絔维的 reduce 结果
//
// ===== 两阶流水线设计 =====
//  这个 kernel 将 CTA 内的 warp 分成两组，每组负责一个 stage，两个 stage 并行执行：
//  Stage 0：发送阶段（post GIN sends）
//    - GIN warp：把本 rank 的数据通过 gin.put() 发给各 rail 伙伴
//    - Worker warps（如果 lsa.nRanks != 1）：预先在 outbox 中汇聚 LSA reduce
//  Stage 1：接收阶段（wait GIN recvs + reduce）
//    - GIN warp：报告 inbox buffer 已完成接收（finishRecvs）
//    - Worker warps：小 inbox buffer 汇聚 reduce，写入 output
//  两个 stage 并行的好处：Stage 0 发送的同时 Stage 1 在 reduce，隐藏网络延迟
//
// ===== 关键概念解释 =====
//  ncclGinInboxA2ASession："inbox" 表示本 rank 将从各 rail 伙伴接收的 chunk 缓冲区
//    它是一块專属内存，通过 GIN 报告"哪块 buffer 已接收完成"
//  ncclGinOutboxSession："outbox" 表示本 rank 将要发送的 chunk 缓冲区
//    Stage 0 worker 先将 LSA reduce 结果写入 outbox，再由 GIN warp 通过 GIN put 发完
//  ncclLsaBarrierSession：LSA 同步屏障，确保本节点所有 GPU 在进入主循环前同步好
//  reduceLsaBatch：引用 lsa 共享内存或 MultiCast 对多个 rank 的数据做 reduce
//  ncclGinScratchMaxBufs：GIN inbox/outbox 的帧数上限，限制 chunk 大小的下界
//
// ===== multimem 参数 =====
//  multimem=false：LsaLD   — LSA reduce 通过普通内存读取（逐 GPU）
//  multimem=true： LsaLDMC — LSA reduce 通过 NVLink MultiCast 读取（一次读全部）
//
// ===== 最外层入口 =====
//  ncclSymkRun_ReduceScatter_RailA2A_LsaLD<Red,T>   — 无 MultiCast
//  ncclSymkRun_ReduceScatter_RailA2A_LsaLDMC<Red,T> — 有 MultiCast
// ============================================================================

template<template<typename> typename Red, typename T, bool multimem>
static __device__ void rsAlgoHier(ncclSymkDevWorkArgs const* args, BoolTag<multimem> multimemTag) {
  ncclCoopCta cta;
  ncclSymkArgsHandler handler{args};
  ncclTeam world = ncclTeamWorld(handler.comm);  // World 团队：所有 rank
  ncclTeam rail = ncclTeamRail(handler.comm);  // Rail 团队：跨节点同 NIC 的 rank
  ncclTeam lsa = ncclTeamLsa(handler.comm);  // LSA 团队：节点内所有 GPU
  ncclGin gin{handler.comm, int(blockIdx.x % handler.comm.ginContextCount)};

  using AccT = typename ncclSymkGinAccumType<Red, T>::Type;  // 累加类型（例如 T=fp16 时 AccT=fp32）
  // 为什么需要 AccT：fp16 缩减时会有羮差累积，用 fp32 做中间计算可提高精度
  Red<AccT> red(handler.devWork->redOpArg);  // 普通 reduce 算子（容化 AccT 类型）
  Red<T> mmRed(handler.devWork->redOpArg);  // MultiCast reduce 算子（容化 T 类型）

  // ===== Warp 分配 =====
  // 总 warp 数 = blockDim.x/32
  // 其中 2 个 warp 为 GIN warp（stage0 1个，stage1 1个）
  // 剩下的 nWorkWarps 平均分配给 stage0 和 stage1 的 worker
  int nWorkWarps = blockDim.x/32 - 2;
  int stage0_nWorkWarps;  // stage0 分到的 worker warp 数
  if (lsa.nRanks == 1) {
    // 纯 Rail 模式（节点内只有一个 GPU）：Stage 0 只需发送，不需要 worker
    stage0_nWorkWarps = 0; // Stage 0 just posts sends so no workers.
  } else {  // !!! Pipeline Stage 1: 等待接收 + reduce !!!
    // Only count reads because they dominate writes.
    // 混合模式（节点内多 GPU）：根据负载比例分配 worker warp
    // stage0_work = LSA reduce 的工作量：读取 (multimem ? 1 : lsa.nRanks) 个源 * (rail.nRanks-1) 个 chunk
    // stage1_work = 接收后 reduce 的工作量：(multimem ? 1 : lsa.nRanks) + (rail.nRanks-1) 个 recv
    // 为什么只计 reads：reads 占主导，writes 的开销相对较小
    int stage0_work = lsa.nRanks == 1 ? 0 : (multimem ? 1 : lsa.nRanks)*(rail.nRanks-1);
    int stage1_work = (multimem ? 1 : lsa.nRanks) + rail.nRanks-1;
    // 按工作量比例分配 warp，用浮点数近似防止整数除法偏差
    stage0_nWorkWarps = __float2int_rn(__fdividef(nWorkWarps*stage0_work, stage0_work + stage1_work));
    stage0_nWorkWarps = min(stage0_nWorkWarps, nWorkWarps-1); // Stage 1 requires at least 1 worker.
    // 强制 Stage 1 至少有 1 个 worker，防止 Stage 1 无线程可用
  }

  // 2 stage pipeline, one coop per stage.
  // stage 0：warp 0...(1+stage0_nWorkWarps-1)
  // stage 1：warp (1+stage0_nWorkWarps)...(blockDim.x/32-1)
  // 第 0 个 warp 由 stage 0 独占（stage0 GIN warp），第 1+stage0_nWorkWarps 个由 stage 1 GIN warp 占用
  int stage = threadIdx.x/32 < 1 + stage0_nWorkWarps ? 0 : 1;
  ncclCoopWarpSpan coopStage{
    /*warp0=*/stage == 0 ? 0 : 1 + stage0_nWorkWarps,
    /*nWarps=*/1 + (stage == 0 ? stage0_nWorkWarps : nWorkWarps-stage0_nWorkWarps),
    /*id=*/stage
    // id=2+stage 与 coopStage.id 区分，用于 role 级别的屏障/共享操作
  };
  // Within each stage we have 2 roles: GIN warp, worker warps.
  // roleIsWorker=false：该 stage 的第一个 warp，就是 GIN warp
  // roleIsWorker=true：其予 warp，处理 compute 工作
  bool roleIsWorker = 32 <= coopStage.thread_rank();
  ncclCoopWarpSpan coopRole{
    /*warp0=*/(stage == 0 ? 0 : 1 + stage0_nWorkWarps) + (roleIsWorker ? 1 : 0),
    /*nWarps=*/!roleIsWorker ? 1 : (stage == 0 ? stage0_nWorkWarps : nWorkWarps - stage0_nWorkWarps),
    /*id=*/2 + stage
  };

  // Construct outbox only for stage=0 when lsa peers exist.
  // outbox：Stage 0 的发送缓冲区（worker reduce 结果 → GIN put 源）
  // 只有 stage==0 且 lsa.nRanks != 1 时才需要；否则直接从 input 发送
  // 使用 placement new + reinterpret_cast 让 outbox_storage 存放对象而不使用堆内存
  // 这样可以避免内核中的 malloc（强制禁止，只能用 smem 或栈内存）
  alignas(ncclGinOutboxSession<ncclCoopWarpSpan>) char outbox_storage[sizeof(ncclGinOutboxSession<ncclCoopWarpSpan>)];
  ncclGinOutboxSession<ncclCoopWarpSpan>& outbox =
    stage == 0 && lsa.nRanks != 1
      ? *::new(&outbox_storage) ncclGinOutboxSession<ncclCoopWarpSpan>
        {coopStage, gin, handler.ginOutbox, blockIdx.x}
      : reinterpret_cast<ncclGinOutboxSession<ncclCoopWarpSpan>&>(outbox_storage);

  __shared__ int totalSends;  // stage0 GIN warp 用于记录发送总次数（纯 Rail 时用 counter 实现流控）
  if (stage == 0 && !roleIsWorker && lsa.nRanks == 1) {
    if (coopRole.thread_rank() == 0) {
      totalSends = 0;
      // When pure rail we use a counter to track sends. We could leave them untracked
      // and end with a flush but by using a counter we can reuse same postSends code
      // for the rail-only and hybrid (lsa!=1) cases.
      // ginCounterPerBlock[blockIdx.x]：每个 block 独立的 GIN 计数器，记录已提交的 put 数量
      // 关铭意义：当 totalSends 个 put 都完成时，计数器达到 totalSends，才能进入结尾 barrier
      // resetCounter：将该 counter slot 重置为 0，取消上一轮剩余值
      gin.resetCounter(handler.ginCounterPerBlock + blockIdx.x);
    }
    coopRole.sync();  // 确保所有 worker warp 线程都看到 inbox 数据已就绪
  }

  // inbox：本 rank 将接收到的来自各 rail 伙伴的 chunk 缓冲区
  // A2ASession：All-to-All 语义，即每个 rank 向本 rank 各发一个 chunk
  // ginInboxRail：描述 inbox 内存区域的句柄（大小、起始地址）
  // blockIdx.x：每个 block 有独立的 inbox 区域不重叠
  ncclGinInboxA2ASession<ncclCoopCta> inbox
    {cta, gin, rail, handler.ginInboxRail, blockIdx.x};

  // LSA 初始屏障：确保本节点内所有 GPU 同步好时才开始
  // 为什么需要 LSA 屏障而非 GIN 屏障：
  //   ReduceScatter 需要节点内所有 GPU 小协调（LSA reduce）
  //   只有 GIN 屏障可以确保跨节点马舚，但不够确保节点内 GPU 同步
  ncclLsaBarrierSession<ncclCoopCta> lsaBar
    {cta, handler.comm, ncclTeamTagLsa(), blockIdx.x, multimem};
  lsaBar.sync(cta, cuda::memory_order_relaxed);  // Relaxed 尼备：此处只需要同步函数，不需要内存序

  int maxChunkElts = args->maxDynamicSmem/sizeof(AccT);

  // maxDynamicSmem：每个 block 可用的动态 shared memory（由 ncclInitKernelsForDevice 设置）
  // maxChunkElts：单个 chunk 最多能容纳的元素数（用 smem 做中间 accumulator）
  handler.template forEachWorkNoFusion<T>(
    [&]__device__(size_t nElts, size_t nAllElts, ncclSymPtr<T> input, ncclSymPtr<T> output) {
      AccT* accum;  // Stage 1 将 reduce 中间结果存在 shared memory
      ncclSymkSmemPartition(&accum, maxChunkElts);  // 从动态 smem 中分配一个片段为 accumulator

      // ===== Chunk 大小计算 =====
      // chunk 大小的设计要点：
      //   1. 不能超过 smem 容量（maxChunkElts）
      //   2. 不能超过 inbox/outbox 大小的一半（流水线重叠需要）
      //   3. 不能使 chunk 数量超过 inbox credit 数（否则颗面匹配失败）
      int chunkBytes_log2 = log2Up(nElts) + log2Up(sizeof(T));

      // Chunk size should not be larger than what host dictates and 1/2 of
      // total capacity so we enjoy some pipeline overlap. This is a soft constraint
      // because chunks which are too big can always be partially used.
      // maxChunkBytes_log2 的两个上限来源：
      //   log2Up(maxChunkElts)+log2Up(sizeof(T))： smem 容量限制
      //   handler.ginInboxRail.size_log2-1： inbox 大小的一半（保证流水线可重叠）
      int maxChunkBytes_log2 = min(
        log2Up(maxChunkElts) + log2Up(sizeof(T)),
        handler.ginInboxRail.size_log2-1);  // inbox 大小的一半
      chunkBytes_log2 = min(chunkBytes_log2, maxChunkBytes_log2);

      // Chunk size must not be so small that the chunk count exceeds either the
      // per peer number or total number. This is a hard constraint imposed
      // by inbox credit logic so we enforce after the max chunk size.
      // 小 chunk 会导致 chunk 数过多，超过 inbox credit 数量，必须强制增大
      // ncclGinScratchMaxBufsPerPeer_log2：单个 peer 的 inbox 帧数 log2
      // ncclGinScratchMaxBufs_log2：总 inbox 帧数 log2
      int minChunkBytes_log2 = handler.ginInboxRail.size_log2 - min(
        log2Down(rail.nRanks-1) + ncclGinScratchMaxBufsPerPeer_log2,
        ncclGinScratchMaxBufs_log2);
      chunkBytes_log2 = max(chunkBytes_log2, minChunkBytes_log2);

      // nBufs_log2：inbox 中除了当前 chunk 外能共配置的帧数（log2）
      // 高值意味着流水线深度大（数个 chunk 同时在飞）
      int nBufs_log2 = handler.ginInboxRail.size_log2 - chunkBytes_log2;

      maxChunkElts = min(maxChunkElts, (1u<<chunkBytes_log2)/(unsigned)sizeof(T));

      auto nop = []__device__(auto...) {};  // 空操作占位符
      // ===== skeleton lambda =====
      // 主循环骨架：逐 chunk 执行 initFn/stepFn/finishFn
      // 每次处理 nChunkElts 个元素，多次迭代直到 loopElts 耗尽
      // 内层的 step 循环是对 rail 内 (rail.nRanks-1) 个 peer 的处理
      // nSteps 控制每次循环内流水线的 step 数，受 inbox 帧数的一半限制以确保重叠
      auto skeleton = [&]__device__(auto initFn, auto stepFn, auto finishFn) {
        size_t loopElts = nElts;
        ncclSymPtr<T> loopInput = input;
        ncclSymPtr<T> loopOutput = output;
        #pragma unroll 1
        while (loopElts != 0) {
          int nChunkElts = min(loopElts, (size_t)maxChunkElts);
          // nSteps 是一次迭代处理的 step 数，不超过 inbox 可用帧数的一半（流水线要求）
          int nSteps = min(rail.nRanks-1, 1<<(nBufs_log2-1));
          int step = 0;
          initFn(nChunkElts, loopInput);
          do {
            nSteps = min(nSteps, rail.nRanks-1 - step);  // 不超过剩余 step
            stepFn(step, nSteps, nChunkElts, loopInput);
            step += nSteps;
          } while (step != rail.nRanks-1);
          finishFn(nChunkElts, loopOutput);
          inbox.endRound(cta);  // 通知 inbox：本轮的所有帧已消费，可以重用
          loopInput += nChunkElts;
          loopOutput += nChunkElts;
          loopElts -= nChunkElts;
        }
      };

      if (stage == 0) { // !!! Pipeline Stage 0 !!!
        // ===== Stage 0 初始化 =====
        // 把 inbox 帧分配给 coopStage（Stage 0 的全部 warp）
        // subcoopIsNonTrivial=false 表示 inbox 的分配不需要 subcoop（直对出就不用分配）
        inbox.apportion(cta, /*subcoop=*/coopStage, /*subcoopIsNonTrivial=*/false, nBufs_log2);
        if (lsa.nRanks != 1) {
          // 有 LSA：outbox 需要分配给 worker warp，deferSync=true 延迟同步（避免阻塞整个 stage）
          outbox.apportion(coopStage, /*subcoop=*/coopRole, /*subcoopIsNonTrivial=*/roleIsWorker, nBufs_log2, /*deferSync=*/true);
        }

        // Generalize worker and non-worker logic with compile time BoolTag to differentiate.
        // 用 BoolTag 静态化 roleIsWorker，允许编译器就分支内联展开不同分支
        auto stage0_impl = [&](/*BoolTag<roleIsWorker>*/auto roleIsWorker_tag) {
          // Shadow runtime value with compile time value.
          constexpr bool roleIsWorker = roleIsWorker_tag.value;
          skeleton(
            /*initFn*/[&]__device__(int nChunkElts, ncclSymPtr<T> inPtr)->void {
              if (!roleIsWorker) {
                // === Stage 0 GIN warp initFn：累加 totalSends ===
                // 每个 chunk 需要对 (rail.nRanks-1) 个目标发送
                // 用于纯 Rail 模式下的 counter 流控
                if (coopRole.thread_rank() == 0) {
                  // totalSends += rail.nRanks-1;
                  // 使用 PTX inline assembly 进行 shared memory 的原子加法
                  // red.relaxed.shared.add.s32：对 shared memory 地址进行 relaxed 原子加
                  // 为什么用 PTX 而不用 C++ atomicAdd：避免不必要的内存屏障，
                  //   relaxed 确保不需要 acquire/release 语义
                  // __cvta_generic_to_shared：将通用地址转换为 shared memory 地址
                  #if __CUDA_ARCH__ >= 700
                  asm volatile("red.relaxed.shared.add.s32 [%0],%1;" :: "r"((uint32_t)__cvta_generic_to_shared(&totalSends)), "r"(rail.nRanks-1) : "memory");
                  #else
                  __trap();  // SM70 以下不支持此指令
                  #endif
                }
              } else {
                // outbox.apportion() was told to defer sync so that we don't sync
                // the whole stage. We sync just this warp.
                // === Stage 0 Worker warp initFn：等待 outbox apportion 同步 ===
                coopRole.sync();
              }
            },
            /*stepFn=*/[&]__device__(int step0, int nSteps, int nChunkElts, ncclSymPtr<T> inPtr)->void {
              // getInputOffset：根据发送 step 计算目标 rank 在 input 中的偏移
              // 定义：peer 是 inbox 第 step 个发送对象（rail rank）
              //         dstWorld = 该 rail rank 对应的 world rank
              //         返回 dstWorld * nAllElts（input 中属于该 rank 的片段偏移）
              auto getInputOffset = [&]__device__(int step)->size_t {
                int peer = inbox.getSendPeer(step, /*step_lt_nPeers=*/true);
                int dstWorld = ncclTeamRankToTeam(world, rail, peer);
                return dstWorld*nAllElts;
              };

              if (lsa.nRanks != 1) { // No need to process data when we can send from input buf.
                if (roleIsWorker) {
                  // Wait for outbox bufs to free up. Since outbox advances with each call of this
                  // function we always index starting at 0.
                  // waitBufs：等待 outbox 中从 idx=0 开始的 nSteps 个 buf 空闲
                  //   空闲的定义：GIN put 已完成通知（outbox counter 递增）
                  // === Stage 0 Worker：等 outbox buf 空闲，先 reduce LSA ===
                  // outbox buf 空闲后，把本节点内所有 GPU 的该 chunk 做 reduce，结果存入 outbox
                  outbox.waitBufs(coopRole, 0, nSteps);
                  coopRole.sync();
                  // Make `outbox.getBufPtr()` as cheap as possible within reduction loop.
                  // make_getBufPtr 返回一个优化过的指针获取 lambda，将下层地址计算内联化
                  auto outbox_getBufPtr = outbox.make_getBufPtr(0);
                  // reduceLsaBatch：将 lsa.nRanks 个 GPU 的同一片段 reduce，写入 outbox buf
                  // 语义：outbox[i] = reduce(input[dstWorld_i*nAllElts : +nChunkElts] across LSA)
                  // multimemTag：控制是否用 NVLink MultiCast 读取（LsaLDMC 模式）
                  // nBatch=nSteps：一次对多个目标进行批量 reduce（隐藏多次 LSA 读取的延迟）
                  reduceLsaBatch(coopRole, /*nBatch=*/nSteps, nChunkElts,
                    /*dstMem=*/GMemTag(), /*dstAlignMin=*/16,
                    /*getDst=*/[&]__device__(int i)->T* {
                      return (T*)outbox_getBufPtr(i);
                    },
                    /*srcRedUc=*/red, /*srcRedMc=*/mmRed, /*srcBase=*/inPtr,
                    /*getSrcOffset=*/[&]__device__(int i)->size_t {
                      return getInputOffset(step0 + i);
                    },
                    handler.comm, multimemTag
                  );
                }
                coopStage.sync();  // GIN warp 等 worker reduce 完成
              }

              if (!roleIsWorker) {
                // === Stage 0 GIN warp：提交 GIN put 发送数据 ===
                // 如果 lsa.nRanks==1：直接发送 input[dstOffset]（不经过 outbox）
                // 如果 lsa.nRanks!=1：发送 outbox[i]（已 LSA reduce 的结果）
                // inbox.postSends：对 [step0, step0+nSteps) 个目标进行 GIN put
                //   getPtr：返回本次发送的数据源地址
                //   getEltCount：返回传输元素数
                //   getCompletion：返回完成信号描述
                //     lsa.nRanks==1：完成后将 ginCounterPerBlock[blockIdx.x] +1
                //     lsa.nRanks!=1：完成后将 outbox counter +1（全知 outbox buf 已空闲）
                inbox.postSends(coopRole, step0, nSteps,
                  /*getPtr*/[&]__device__(int i, int peer) {
                    return lsa.nRanks == 1 ? inPtr + getInputOffset(step0 + i)
                                           : (ncclSymPtr<T>)outbox.getBuf(i);
                  },
                  /*getEltCount*/[&]__device__(int i, int peer) {
                    return nChunkElts;
                  },
                  /*getCompletion*/[&]__device__(int i, int peer) {
                    return ncclGin_CounterInc{
                      lsa.nRanks == 1 ? handler.ginCounterPerBlock + blockIdx.x
                                      : outbox.getCounter(i)
                    };
                  }
                );
              }

              if (lsa.nRanks != 1) {
                // We advance outbox with every iteration because it is only guaranteed
                // to have `nSteps` buffers. If we advanced it after the step loop it
                // would need rail.nRanks-1 buffers.
                // advance：释放已使用的 outbox buf，准备给下一轮复用
                outbox.advance(coopStage, nSteps);  // 释放已使用的 outbox buf
              }
            },
            /*finishFn=*/nop
          );
        };

        // Instantiate stage0_impl specialized to each case of roleIsWorker.
        if (!roleIsWorker) stage0_impl(BoolTag</*roleIsWorker=*/false>());
        else stage0_impl(BoolTag</*roleIsWorker=*/true>());

      } else { // !!! Pipeline Stage 1 !!!

        // Generalize worker and non-worker logic with compile time BoolTag to differentiate.
        // ===== Stage 1 整体说明 =====
        // 本 rank 接收来自 rail 中所有其他 rank 通过 GIN 发来的 chunk，
        // 与本地 LSA reduce 结果累加，写入最终 output。
        //
        // 角色分工（coopRole.id = 2+stage = 3）：
        //   GIN warp（roleIsWorker=false，coopRole 内首个 warp）：
        //     负责向 GIN 后端通报"我已消费该 inbox buf"（finishRecvs），
        //     释放 inbox credit，允许发送端的下一轮 put 进入该 slot。
        //     注意：GIN warp 不参与 reduce 计算，只做信号/信用管理。
        //   Worker warps（roleIsWorker=true，coopRole 内其余 warp）：
        //     等待 inbox 中数据就绪（waitRecvs），
        //     累加 reduce：accum = accum op inbox[s] for each step s，
        //     最后将 accum（shared memory）写回全局 output。
        //
        // 与 Stage 0 的并行关系：
        //   Stage 0 正在对下一轮 chunk 进行 LSA reduce + postSends（发送），
        //   Stage 1 同时在等待上一轮 chunk 到达并做 reduce，
        //   两者通过 inbox credit 协调（Stage 1 调用 finishRecvs 归还 credit，
        //   Stage 0 的 postSends 等待有空闲 credit 才发送）。
        auto stage1_impl = [&]__device__(/*BoolTag<roleIsWorker>*/auto roleIsWorker_tag)->void {
          constexpr bool roleIsWorker = roleIsWorker_tag.value;
          // inbox.apportion：将 inbox 的帧（slots）分配给当前 role 使用
          //   subcoop=coopRole：按当前 role（GIN warp 或 worker warp）分配
          //   subcoopIsNonTrivial=!roleIsWorker：
          //     GIN warp（!roleIsWorker=true）需要内部同步（apportion 会在 GIN warp 内部广播 slot 偏移）
          //     Worker warp（!roleIsWorker=false）不需要额外同步
          //   nBufs_log2：inbox 的帧数 log2（流水线深度上界）
          // 作用：确保 GIN warp 和 worker warp 看到同一套 inbox 帧分配结果
          inbox.apportion(cta, /*subcoop=*/coopRole, /*subcoopIsNonTrivial=*/!roleIsWorker, nBufs_log2);
              // 同步：等待 worker 完成 reduce 后，再让 GIN warp 执行 finishRecvs
              // 原因：finishRecvs 会将 inbox slot 的 credit 归还给发送方，
              //   如果在 reduce 尚未完成前就归还，Stage 0 可能立刻复用该 slot 写入新数据，
              //   导致 worker 还在读取的同一片内存被覆盖（数据竞争）
          // apportion 之后必须同步：GIN warp 和 worker warp 都完成 apportion 后才能协调
          // 若不同步，worker warp 可能在 GIN warp 完成分配前就开始访问 inbox slot
          coopStage.sync();

          skeleton(
            /*initFn*/[&]__device__(int nChunkElts, ncclSymPtr<T> inPtr)->void {
              if (roleIsWorker) {
                // === Stage 1 Worker initFn：初始化 accum = 本 rank 自有分片的 LSA reduce 结果 ===
                //
                // 背景：ReduceScatter 中，每个 world rank 最终拥有 input[rank * nAllElts] 那一片的
                //   全局 reduce 结果。本 rank 持有该片中属于本节点所有 GPU 的部分（LSA），
                //   需要先把节点内所有 GPU 的对应片段 reduce 成初始值，存入 smem accum。
                //
                // reduceLsa 参数详解：
                //   coopRole     — 参与计算的 warp 组（worker warps）
                //   nChunkElts   — 本 chunk 的元素数
                //   dstMem=SMemTag() — 写入目标为 shared memory（accum 在 smem 中）
                //                      使用 smem 的原因：避免中间结果写回全局内存产生 DRAM 带宽浪费
                //   dstAlignMin=16   — 目标对齐要求 16 字节（对应 128-bit 向量化访问）
                //   dstPtr=accum     — shared memory 中的 accumulator 数组
                //   srcRedUc=red     — 非 MultiCast 路径使用的 reduce 算子（AccT 类型，高精度）
                //   srcRedMc=mmRed   — MultiCast 路径使用的 reduce 算子（T 类型，匹配广播数据类型）
                //   srcPtr=inPtr + world.rank*nAllElts
                //                  — 源地址：本 rank（world.rank）所负责的那段 input 的起始位置
                //                    注意：inPtr 是 symmetric 内存，所有节点内 GPU 共享同一物理地址空间
                //                    加上 world.rank*nAllElts 偏移后指向本 rank 在该 chunk 中的基地址
                //   handler.comm     — comm 描述符（含 LSA 节点内 GPU 列表、MultiCast handle 等）
                //   multimemTag      — 编译期 bool：控制是否用 NVLink MultiCast 广播读取
                //                     true（LsaLDMC）：一次 multimem.ld 广播读所有节点内 GPU 的数据
                //                     false（LsaLD） ：逐 GPU 逐一读取并累加（带宽受 NVLink 限制）
                reduceLsa(coopRole, nChunkElts,
                  /*dstMem=*/SMemTag(), /*dstAlignMin=*/16, /*dstPtr=*/accum,
                  /*srcRedUc=*/red, /*srcRedMc=*/mmRed,
                  /*srcPtr=*/inPtr + world.rank*nAllElts,
                  handler.comm, multimemTag
                );
                // reduce 完成后，accum = (节点内 LSA reduce 结果) op (各远端 rank 发来的 chunk 的累加)
                // 即：accum 已包含本 chunk 在 [当前节点内所有 GPU] + [step0..step0+nSteps-1 对应的远端 rank] 的 reduce
                // reduceLsa 完成后，accum[0..nChunkElts-1] 中存有节点内所有 GPU 对该片段的 reduce 结果
                // 后续 stepFn 中再累加来自其他节点（通过 GIN inbox 到达）的 chunk
              }
            },
            /*stepFn*/[&]__device__(int step0, int nSteps, int nChunkElts, ncclSymPtr<T> inPtr)->void {
              if (roleIsWorker) {
                // === Stage 1 Worker stepFn：等待 GIN 接收完成，将 inbox chunk 累加进 accum ===
                //
                // waitRecvs 参数详解：
                //   coopRole       — 参与等待的 warp 组（worker warps）
                //   step0          — 本轮 step 的起始索引（inbox slot 编号的基）
                //   nSteps         — 本轮等待的 inbox slot 数量
                //
                // 内部实现：waitRecvs 对每个 inbox slot[step0..step0+nSteps-1] 轮询其 credit 计数
                //   发送方（Stage 0 GIN warp）在 put 完成时调用 finishSends（通过 IB CQ completion），
                //   将对应 inbox slot 的 credit +1（通过 ncclGin_CounterInc 完成信号）
                //   waitRecvs spin-poll 该 credit，直到值达到预期才返回
                //   本质是：等待 GIN 后端通知"远端 put 已写入本地 inbox buf 对应 slot"
                inbox.waitRecvs(coopRole, step0, nSteps);
                // coopRole.sync() 的作用：防止 initFn 过早覆写 accum
                //   skeleton 是 while 循环，每次迭代 finishFn 完成后立即进入下一轮 initFn
                //   initFn 会调用 reduceLsa 写入 accum，与本次 copy 读取 accum 形成竞争
                //   sync 确保 copy 写入全局内存已完成（所有 warp 都完成了写回）后，
                //   才允许下一轮 initFn 开始覆写 accum
                coopRole.sync();
                // Make `inbox.getBufPtr()` as cheap as possible within reduction loop.
                // make_getBufPtr：内联化 inbox buf 指针获取
                //   inbox.getBufPtr(s) 在循环中会重复计算地址，make_getBufPtr(step0)
                //   返回一个预先固定 base 地址的 lambda，使编译器能更好地内联和优化
                //   返回的 inbox_getBufPtr(s) 等价于 inbox.getBuf(step0 + s)
                auto inbox_getBufPtr = inbox.make_getBufPtr(step0);
                // reduce：将 nSteps 个 inbox buf 依次累加进 accum（in-place reduce）
                //
                // reduce 参数详解：
                //   coopRole           — 参与计算的 warp 组
                //   red                — reduce 算子（AccT 类型，如 fp16→fp32 精度累加）
                //   inPlace=true       — 目标 dst（accum）同时作为第一个源（accum = accum op src[s]）
                //                        若 inPlace=false 则 dst 被第一个 src 覆盖，不累加已有值
                //   nChunkElts         — 本 chunk 的元素数
                //   dstMem=SMemTag()   — 目标在 shared memory（accum）
                //   dstAlignMin=16     — 对齐要求 16 字节
                //   dst=accum          — 目标：smem accumulator
                //   nSrcs=nSteps       — 源的个数（每个来自一个 inbox slot，对应一个远端 rank）
                //   srcPtrCommonMask=16-1 — 所有 src 指针的公共对齐掩码（低 4 位相同）
                //                          用于让编译器产生对齐优化的向量化访存指令
                //   srcPtrMasked=0     — 公共低位值（= 0，表示所有 buf 起始地址低 4 位为 0）
                //   getSrc(s)          — 返回第 s 个 inbox buf 的指针（已接收的远端 chunk 数据）
                reduce(coopRole, red, /*inPlace=*/true, nChunkElts,
                  /*dstMem=*/SMemTag(), /*dstAlignMin=*/16, /*dst=*/accum,
                  /*nSrcs=*/nSteps, /*srcPtrCommonMask=*/16-1, /*srcPtrMasked=*/0,
                  /*getSrc=*/[&]__device__(int s)->T* {
                    return (T*)inbox_getBufPtr(s);
                  }
                );
              }
              coopStage.sync();
              if (!roleIsWorker) {
                // === Stage 1 GIN warp：归还 inbox credit（finishRecvs）===
                //
                // finishRecvs 参数详解：
                //   coopRole       — GIN warp（只有 GIN warp 调用此路径）
                //   step0          — 本轮归还的起始 slot 索引
                //   nSteps         — 本轮归还的 slot 数量
                //
                // 内部语义：对 inbox slot[step0..step0+nSteps-1] 发出"消费完毕"信号
                //   在 GIN_PROXY 模式下：CPU proxy 更新对应的 inbox credit（通过 shared memory 写入）
                //   在 GIN_GDAKI 模式下：GPU 直接写入 GIN 硬件的 credit 寄存器
                //   发送方的 postSends 等待 credit 可用才会提交下一轮 put，
                //   因此 finishRecvs 是流水线"背压"（back-pressure）控制的关键路径
                inbox.finishRecvs(coopRole, step0, nSteps);
              }
            },
            /*finishFn*/[&]__device__(int nChunkElts, ncclSymPtr<T> outPtr)->void {
              if (roleIsWorker) {
                // === Stage 1 Worker finishFn：将最终 reduce 结果从 smem 写回全局 output ===
                //
                // 此时 accum 中已包含该 chunk 的完整全局 reduce 结果：
                //   accum = input[world.rank*nAllElts .. +nChunkElts] 在所有 world rank 上的 reduce 结果
                //
                // copy 参数详解：
                //   coopRole           — 参与写回的 warp 组（worker warps）
                //   nChunkElts         — 本 chunk 的元素数
                //   dstMem=GMemTag()   — 目标在全局内存（output buffer）
                //   dst=outPtr.localPtr()
                //                      — 只写入本地 GPU 的 output 指针（非 MultiCast 指针）
                //                        原因：ReduceScatter 结果只属于本 rank，不需要广播给其他 GPU
                //                        若用 MultiCast 指针写入，会触发 NVLink 广播，写入节点内所有 GPU
                //                        的对应位置，导致其他 rank 的 output 被覆盖（语义错误）
                //   srcMem=SMemTag()   — 源在 shared memory（accum）
                //   src=accum          — smem accumulator，含最终 reduce 结果
                copy(coopRole, nChunkElts,
                  /*dstMem=*/GMemTag(), /*dst=*/outPtr.localPtr(),
                  /*srcMem=*/SMemTag(), /*src=*/accum,
                  [&]__device__(auto x) { return applyPostOp(red, x); }
                );
                coopRole.sync(); // prevent initFn from trampling accum
              }
            }
          );
        };

        // Instantiate stage1_impl specialized to each case of roleIsWorker.
        if (!roleIsWorker) stage1_impl(BoolTag</*roleIsWorker=*/false>());
        else stage1_impl(BoolTag</*roleIsWorker=*/true>());
      }
    }
  );

  if (stage == 0) {  // !!! Pipeline Stage 0: 发送 chunk 给 rail 伙伴 !!!
    if (lsa.nRanks == 1) {
      // === 纯 Rail 模式（lsa.nRanks==1）：等待所有 GIN put 完成 ===
      //
      // 背景：纯 Rail 时 Stage 0 没有 Worker warp，所有 put 由 GIN warp 直接提交，
      //   完成信号写入 ginCounterPerBlock[blockIdx.x] 计数器（每完成一次 put +1）。
      //   totalSends 在 initFn 中通过 PTX atomic add 累计（每轮 += rail.nRanks-1）。
      //
      // waitCounter 参数详解：
      //   ncclCoopThread()       — 单线程 coop（只有 coopRole.thread_rank()==0 的线程调用）
      //   handler.ginCounterPerBlock + blockIdx.x
      //                          — 指向本 block 专用的 GIN put 完成计数器
      //                            每完成一次 gin.put，GIN 后端（CPU proxy 或 GPU 直接）将该计数器 +1
      //   totalSends             — 期望的完成次数（= 所有 chunk 的 (rail.nRanks-1) 之和）
      //   32                     — 轮询间隔（每 32 次 load 检查一次，节省 memory bus 带宽）
      //
      // 等待完成后才进入结尾 lsaBar.sync，确保所有发送已落地（远端 rank 已接收全部数据）
      if (!roleIsWorker && coopRole.thread_rank() == 0) {
        gin.waitCounter(ncclCoopThread(), handler.ginCounterPerBlock + blockIdx.x, totalSends, 32);
      }
    } else {
      // === 混合模式（lsa.nRanks!=1）：析构 outbox session ===
      //
      // ncclGinOutboxSession 的析构函数语义：
      //   等待所有已提交 outbox buf 的 put 完成（通过 outbox 内部 counter 轮询），
      //   然后释放 outbox 持有的 GIN context 资源（解除与 ginOutbox 的绑定）。
      //
      // 为什么需要显式析构：
      //   outbox_storage 是栈上的 char 数组（placement new 构造），编译器不会自动调用析构函数；
      //   必须手动调用 ~ncclGinOutboxSession 确保内部 counter 同步和资源释放。
      //   若不等待，后续 lsaBar.sync 可能在 Stage 0 的 put 仍未完成时越过，
      //   导致接收方（Stage 1）在 output 未完整写入前就读取数据（数据竞争）。
      //
      // 注意：只有 stage==0 才进入此分支，stage==1 的 outbox 是未构造的存储，不能调用析构
      outbox.template ~ncclGinOutboxSession<ncclCoopWarpSpan>();
    }
  }

  // === 结尾 LSA barrier：全节点同步，确保所有 rank 完成 Stage 0 + Stage 1 ===
  //
  // 为什么需要结尾 lsaBar.sync：
  //   ReduceScatter 要求所有 rank 的 output 都已写入后，调用方才能读取 output；
  //   Stage 1 在写入 output 后并不知道其他节点内 GPU 是否也完成，
  //   结尾 lsaBar 确保本节点内所有 GPU（LSA 组）都越过了 Stage 1 的 finishFn，
  //   使 output buffer 对本节点内所有 GPU 可见。
  //
  // memory_order_relaxed 原因：
  //   此处只需要 execution barrier（所有 GPU 都到达该点），不需要 memory fence，
  //   因为 Stage 1 finishFn 中的 coopRole.sync() 已经保证了 accum→output 的顺序性，
  //   lsaBar.sync 只需协调"谁先到谁等"，relaxed 语义足够且性能更优。
  lsaBar.sync(cta, cuda::memory_order_relaxed);
}

template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_RailA2A_LsaLD(ncclSymkDevWorkArgs const* args) {
  // LsaLD 版本：LSA reduce 通过普通内存读取实现（逐个 GPU 读）
  rsAlgoHier<Red, T>(args, /*multimem=*/BoolTag<false>{});
}
template<template<typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_RailA2A_LsaLDMC(ncclSymkDevWorkArgs const* args) {
  // LsaLDMC 版本：LSA reduce 通过 NVLink MultiCast 读取实现（一次读广播到节点内所有 GPU）
  rsAlgoHier<Red, T>(args, /*multimem=*/BoolTag<true>{});
}
