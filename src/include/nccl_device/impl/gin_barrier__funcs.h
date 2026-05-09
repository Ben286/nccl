/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// ===== gin_barrier__funcs.h：GIN Barrier 函数实现 =====
// 本文件实现 ncclGinBarrierSession<Coop> 的所有成员函数。
//
// 关键实现技术：
//
// 1. 旋转发送模式（Rotating Pattern）：
//    每个 rank 按不同起始顺序遍历 team 中的 peer，避免所有 rank 同时发送给 rank 0
//    导致 rank 0 的 GIN 接收端成为热点（hot spot）。
//    旋转后，rank r 的发送顺序为：
//      rank r+1, r+2, ..., nRanks-1, 0, 1, ..., r-1
//    即 peer = (1 + team.rank + i) % team.nRanks，i = 0..nRanks-2
//
// 2. Shadow 计数器机制（Shadow Pointer）：
//    GIN signal 值存储在 GIN 网络硬件内（device 不可直接读），
//    shadow 副本（uint32_t*）是在 device 可访问内存中的镜像。
//    每次调用 waitSignal 前，先对 shadow 副本 ++（预期值），
//    作为 waitSignal 的目标值 waitVal，避免二次读取 GIN 硬件。
//    公式：waitVal = ++(*shadowPtr)
//      即第 k 次 barrier 后该 shadow 值 = k（从 0 开始计数）
//
// 3. 发送与等待交替（Latency Hiding）：
//    先发送 GIN signal（net.signal），然后立即去更新 shadow 计数器（内存操作），
//    这段内存操作覆盖了 GIN 信号传输的网络延迟，等 shadow 更新完成后再 waitSignal，
//    从而隐藏一部分网络延迟。
//
// 4. GIN fence 语义（cuda::memory_order + ncclGinFenceLevel）：
//    - 若 ord 包含 release 语义（acquireOrderOf(ord) != relaxed），
//      则在发送 GIN signal 时附带 thread_scope_thread 的 fence，
//      确保此前的写操作对接收方可见（fence 级别由 GIN 驱动层处理）。
//    - 若 ord 为 relaxed，则使用 thread_scope_system（仅触发 DMA，不加 fence）。

#ifndef _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
#define _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
#include "gin_barrier__types.h"

// -------------------------------------------------------------------------
// sync()：执行一次完整的 GIN barrier（arrive + wait 合并）
// -------------------------------------------------------------------------
// 语义：当 sync() 返回时，所有 team 中的 rank 都已到达此 barrier，
//       并根据 ord 提供相应的内存可见性保证。
//
// 实现细节（旋转发送 + shadow 计数）：
//
//   Phase 1: coop.sync()
//     协作组内所有线程先做组内同步，确保 barrier 之前的工作已完成。
//
//   Phase 2: 按旋转顺序 send + wait（并行化到 coop 中的线程）
//     #pragma unroll 1：禁止循环展开，因为循环体内有 waitSignal（可能 spin-wait），
//       展开会导致寄存器压力过高且无法正确处理 spin-wait 语义。
//
//     for i in [coop.thread_rank(), nRanks-1, step=coop.size()]：
//       每个 coop 线程负责一部分 peer，多线程并行 overlap send/wait
//
//     peer = (1 + team.rank + i) % team.nRanks
//       旋转起点：rank r 的第 i 个 peer 是 (r+1+i) mod nRanks
//       例：nRanks=4, rank=2 → peer 序列为 3,0,1（跳过自己 2）
//
//     Step A: net.signal(team, peer, ncclGin_SignalInc{signal + team.rank}, ...)
//       向 peer 发送 GIN signal，通知"我（team.rank）已到达"
//       signal + team.rank：本 rank 的"发信箱" slot
//       ncclGin_SignalInc：signal value 递增操作（+1）
//       最后参数（thread_scope）：
//         若 ord 有 release 语义（releaseOrderOf(ord) != relaxed）→ thread_scope_thread（附 fence）
//         否则 → thread_scope_system（仅触发 DMA，不加 fence）
//
//     Step B: shadow 计数更新（延迟隐藏关键步骤）
//       shadowPtr = net.getSignalShadowPtr(signal + peer)
//         获取 peer 的"收信箱" shadow 副本地址（device 可访问）
//       waitVal = ++(*shadowPtr)
//         预递增：期望 peer 的 signal 计数已增加到 waitVal
//         ++*shadowPtr 在 device 端执行（普通内存写，无原子），
//         因为同一个 shadow slot 只被一个 rank 单向递增（无竞争）
//
//     Step C: net.waitSignal(ncclCoopThread(), signal + peer, waitVal, 32, acquireOrderOf(ord))
//       等待 peer 的"收信箱"（signal + peer）达到 waitVal
//       ncclCoopThread()：以单线程 coop 方式 wait（因为每个 peer 的 slot 由单线程负责）
//       waitVal：上一步预计算的目标值
//       32：超时阈值（每 32 次轮询检查一次 abortFlag）
//       acquireOrderOf(ord)：获取对应的 acquire 内存序，确保读到对方写的数据
//
//   Phase 3: coop.sync()
//     所有线程完成各自负责的 peer 的 send+wait 后，再做一次组内同步，
//     确保整个 coop 完成了所有 peer 的等待。
// -------------------------------------------------------------------------
// 析构函数
// -------------------------------------------------------------------------
// GIN barrier 的状态完全存储在 GIN 硬件 signal slot 和 shadow 计数器中，
// 无需在析构时持久化（shadow 计数器在每次 waitSignal 后已是最新值）。
// 因此析构函数为空，仅满足 RAII 接口要求。
// -------------------------------------------------------------------------
// 构造函数（rail tag 版本）
// -------------------------------------------------------------------------
// 便捷构造：使用 ncclTeamTagRail 标签时，自动推断：
//   team   = ncclTeamRail(net.comm)       —— rail team（跨节点的同编号 GPU）
//   handle = net.comm.railGinBarrier      —— rail 专用的 GIN barrier 句柄
// 其余参数不变，委托给通用版构造函数。
// -------------------------------------------------------------------------
// 构造函数（通用版本）
// -------------------------------------------------------------------------
// 参数：
//   coop        — 协作组（负责在多线程间分配 send/wait 任务）
//   net         — ncclGin 对象（含 comm 引用和 GIN 操作方法）
//   team        — 逻辑 team（rank + nRanks + stride）
//   handle      — GIN barrier 句柄（含 signal0）
//   barrierIndex— 本次使用的 barrier 实例编号（0..nBarriers-1）
//
// 构造流程：
//   1. 调用基类 ncclGinBarrierSession_internal<Coop> 构造，存储 coop/net/team/handle/index
//   2. 计算 this->signal = handle.signal0 + barrierIndex * team.nRanks
//      即本次 barrier 实例对应的 signal slot 基地址
#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(
    Coop coop, ncclGin net, ncclTeam team, ncclGinBarrierHandle handle, uint32_t barrierIndex
  ):
  ncclGinBarrierSession_internal<Coop>{coop, net, team, handle, (int)barrierIndex} {
  // signal = signal0 + barrierIndex * nRanks
  // 该值是整个 barrier 的信号基准，各 rank 的 slot 编号为 signal + r（r = 0..nRanks-1）
  this->signal = handle.signal0 + barrierIndex * team.nRanks;
  // 无需清理：GIN 信号 slot 由 host 端资源管理
  // 通过 tag dispatch 选择正确的 team 和 handle，避免调用方硬编码
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(
    Coop coop, ncclGin net, ncclTeamTagRail, uint32_t barrierIndex
  ):
  ncclGinBarrierSession(coop, net, ncclTeamRail(net.comm), net.comm.railGinBarrier, barrierIndex) {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(
    Coop coop, ncclGin net, ncclTeamTagWorld, uint32_t barrierIndex
  ):
  ncclGinBarrierSession(coop, net, ncclTeamWorld(net.comm), net.comm.worldGinBarrier, barrierIndex) {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::~ncclGinBarrierSession() {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
template<bool EnableTimeout>
NCCL_DEVICE_INLINE ncclResult_t ncclGinBarrierSession_internal<Coop>::syncInternal(Coop, cuda::memory_order ord,
                                                                      ncclGinFenceLevel fence, uint64_t timeoutCycles) {
  uint64_t startCycle;
  ncclResult_t ret = ncclSuccess;
  // Phase 1: 组内同步，确保所有线程的前置工作已完成
  this->coop.sync();
  if NCCL_IF_CONSTEXPR (EnableTimeout) {
    startCycle = clock64();
  }

  // Phase 2: 旋转 send + wait（线程并行）
  #pragma unroll 1
  for (int i=this->coop.thread_rank(); i < this->team.nRanks-1; i += this->coop.size()) {
    // Use a rotating pattern to avoid hot spots
    // 旋转发送顺序：peer = (team.rank + 1 + i) mod nRanks
    // 从 team.rank+1 开始，绕一圈排除自己，均匀分散发送压力
    int peer = 1 + this->team.rank + i;
    if (this->team.nRanks <= peer) peer -= this->team.nRanks;

    // Initiate signal
    // Step A: 向 peer 发送 GIN signal（通知"我已到达"）
    // signal + team.rank：本 rank 的"发信箱" slot 编号
    // ncclGin_SignalInc：原子递增该 slot 的计数器值
    // 内存序：有 release → 附 thread 级 fence；无 → 仅 DMA
    this->net.signal(
      this->team, peer, ncclGin_SignalInc{this->signal + this->team.rank}, ncclCoopThread(), ncclGin_None(),
      nccl::utility::releaseOrderOf(ord) != cuda::memory_order_relaxed
        ? cuda::thread_scope_thread  // 有 release 语义：附加内存 fence，确保 peer 能看到我的写操作
        : cuda::thread_scope_system  // 仅触发 DMA，不加额外 fence
    );

    // Load and update barrier state in memory. The load/store should be covered by the GIN signal latency.
    // Step B: 更新 shadow 计数器（利用 GIN 信号传输延迟隐藏内存操作开销）
    // GIN 信号在网络上传输需要一定时间，此时 CPU 执行 shadow 内存更新
    // 这样当 waitSignal 被调用时，信号可能已经到达，减少 spin-wait 时间
    uint32_t* shadowPtr = (uint32_t*)this->net.getSignalShadowPtr(this->signal + peer);
    // waitVal = 当前期望的累计信号次数（++ 后 = 第 waitVal 次 barrier）
    // 无需原子操作：每个 shadowPtr slot 仅由一个线程写（无竞争）
    int waitVal = ++*shadowPtr;

    if NCCL_IF_CONSTEXPR (EnableTimeout) {
      while (true) {
        uint64_t got = this->net.readSignal(this->signal + peer, 32, nccl::utility::acquireOrderOf(ord));
        if (nccl::utility::rollingLessEq(static_cast<uint64_t>(waitVal), got, 32)) break;
        if (clock64() - startCycle >= timeoutCycles) {
          ret = ncclTimeout;
          goto exit;
        }
      }
    } else {
    // Step C: 等待 peer 向"我的 signal slot"发送的信号到达
    // signal + peer：peer 的"发信箱"（等待 peer 发送它自己的 slot 上的通知）
    // waitVal：期望计数值
    // 32：每隔 32 次轮询检查一次 abortFlag（故障容忍）
    // acquireOrderOf(ord)：获取 acquire 内存序，保证 peer 的写对我可见
      this->net.waitSignal(ncclCoopThread(), this->signal + peer, waitVal, 32, nccl::utility::acquireOrderOf(ord));
    }
  }
  goto exit; // Silence a compiler warning.
exit:

  // Phase 3: 所有线程完成各自 peer 的 send+wait，组内同步收尾
  this->coop.sync();
  return ret;
}
#endif


#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrierSession<Coop>::sync(Coop coop, cuda::memory_order ord, ncclGinFenceLevel fence) {
  (void)(this->template syncInternal</*EnableTimeout=*/false>(coop, ord, fence, 0ULL));
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclResult_t ncclGinBarrierSession<Coop>::sync(
    Coop coop, cuda::memory_order ord, ncclGinFenceLevel fence, uint64_t timeoutCycles) {
  return this->template syncInternal</*EnableTimeout=*/true>(coop, ord, fence, timeoutCycles);
}
#endif

#endif // _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
