/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// ===== barrier__funcs.h：组合 Barrier 函数实现 =====
// 本文件实现 ncclBarrierSession<Coop> 的所有成员函数。
//
// 关键设计：nccl::utility::present(...) / Absent()
//   present(args...) — 创建 Optional<T>，present=true，并用 args... 构造 T
//   Absent()         — 创建 Optional<T>，present=false，T 不被构造（节省资源）
//
//   这允许在同一个 ncclBarrierSession 对象中，根据需要选择性地激活
//   innerLsaBar 和 outerGinBar，而不引入运行时分支开销（编译器可优化）。
//
// sync() 的内存序拆分原理：
//   当两层 barrier 都存在时（World barrier），必须保证：
//     1. inner sync 完成后的所有写对外节点可见（release 语义）
//     2. outer sync 完成后能读到所有外节点的写（acquire 语义）
//   因此内存序在两层之间拆分：
//     innerLsaBar.sync(release)  → 发布本节点的写操作
//     outerGinBar.sync(acquire)  → 获取其他节点的写操作
//   合起来等价于 seq_cst 语义的全局同步。
//
//   如果只有一层（LSA-only 或 Rail-only），则直接使用原始的 ord。

#ifndef _NCCL_DEVICE_BARRIER__FUNCS_H_
#define _NCCL_DEVICE_BARRIER__FUNCS_H_
#include "barrier__types.h"
#include "lsa_barrier__funcs.h"
#if defined(NCCL_OS_LINUX)
#include "gin_barrier__funcs.h"
#endif
#include "../utility.h"

// -------------------------------------------------------------------------
// lsaBarrier()：获取内层 LSA barrier 的可变引用
// -------------------------------------------------------------------------
// 注意：仅在 innerLsaBar.present == true 时调用，否则行为未定义
// 典型用途：获取 ncclLsaBarrierSession<Coop>& 并手动调用 arrive() 或 wait()
// -------------------------------------------------------------------------
// 便捷构造：Rail-only Barrier（仅跨节点同步）
// -------------------------------------------------------------------------
// 手动指定：
//   gin         = present(gin)  —— 保存 gin 对象（outerGinBar 内部会用到）
//   innerLsaBar = Absent()      —— 内层 LSA barrier 不激活
//   outerGinBar = present(...) —— 仅激活外层 GIN barrier
// 适合：节点内 LSA 同步已在其他地方完成，此处只需做跨节点 fence
// -------------------------------------------------------------------------
// 便捷构造：LSA-only Barrier（仅节点内同步）
// -------------------------------------------------------------------------
// 直接调用 ncclBarrierSession_internal<Coop> 构造，手动指定：
//   gin         = Absent()  —— 无 GIN（不使用跨节点网络）
//   outerGinBar = Absent()  —— 外层 GIN barrier 不激活
//   innerLsaBar = present(...) —— 仅激活内层 LSA barrier
// 注意：此版本接受 ncclDevComm const& 而非 ncclGin，因为不需要 GIN 对象
// -------------------------------------------------------------------------
// 便捷构造：World Barrier（全局同步，两层都激活）
// -------------------------------------------------------------------------
// 委托给 Full-featured 构造函数，自动推断：
//   innerTeam  = ncclTeamLsa(gin.comm)      — 节点内 lsa team
//   outerTeam  = ncclTeamRail(gin.comm)     — 跨节点 rail team
//   innerHandle = gin.comm.hybridLsaBarrier       — hybrid lsa barrier 句柄（2.30.3 用 hybrid 版本）
//   outerHandle = gin.comm.hybridRailGinBarrier   — hybrid gin barrier 句柄（2.30.3 用 hybrid 版本）
//   innerMmHandle = gin.comm.lsaMultimem    — MultiCast 句柄（供 multimem=true 时使用）
// -------------------------------------------------------------------------
// 构造函数（Full-featured）：所有参数显式指定
// -------------------------------------------------------------------------
// 使用 present()/Absent() 同时初始化三个 Optional 字段：
//   gin       = present(gin)              —— gin 始终存在（World/Rail 时）
//   innerLsaBar = present(coop, gin.comm, innerTeam, innerHandle, index, multimem, innerMmHandle)
//                  —— 构造 ncclLsaBarrierSession<Coop>
//   outerGinBar = present(coop, gin, outerTeam, outerHandle, index)
//                  —— 构造 ncclGinBarrierSession<Coop>
#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclBarrierSession<Coop>::ncclBarrierSession(
    Coop coop, ncclTeam innerTeam, ncclTeam outerTeam, ncclGin gin,
    ncclLsaBarrierHandle innerHandle, ncclGinBarrierHandle outerHandle,
    uint32_t index, bool multimem, ncclMultimemHandle innerMmHandle
  ):
  ncclBarrierSession_internal<Coop>(coop,
    // gin：直接包装为 Optional<ncclGin>（present=true）
    nccl::utility::present(gin),  // gin = present：GIN barrier 需要 gin 对象
    // innerLsaBar：用 gin.comm 和 innerTeam 构造 ncclLsaBarrierSession
    // gin.comm：Device Communicator（含 lsaBarrier/lsaMultimem 等字段）
    nccl::utility::present(coop, gin.comm, innerTeam, innerHandle, index, multimem, innerMmHandle),
    // outerGinBar：用 gin 和 outerTeam 构造 ncclGinBarrierSession
    nccl::utility::present(coop, gin, outerTeam, outerHandle, index)
  ) {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclBarrierSession<Coop>::ncclBarrierSession(
    Coop coop, ncclTeamTagWorld, ncclGin gin, uint32_t index, bool multimem
  ):
  // [2.30.3 变更] 使用 hybridLsaBarrier / hybridRailGinBarrier（替代 lsaBarrier / railGinBarrier）
  ncclBarrierSession<Coop>(
    coop, ncclTeamLsa(gin.comm), ncclTeamRail(gin.comm), gin,
    gin.comm.hybridLsaBarrier, gin.comm.hybridRailGinBarrier,
    index, multimem, gin.comm.lsaMultimem
  ) {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclBarrierSession<Coop>::ncclBarrierSession(
    Coop coop, ncclTeamTagLsa, ncclDevComm const& comm, uint32_t index, bool multimem
  ):
  ncclBarrierSession_internal<Coop>(coop,
    nccl::utility::Absent(),  // innerLsaBar = Absent：跳过节点内同步
    nccl::utility::present(coop, comm, ncclTeamLsa(comm), comm.hybridLsaBarrier, index, multimem, comm.lsaMultimem),
    nccl::utility::Absent()  // outerGinBar = Absent：跳过跨节点同步
  ) {
}
#endif

// -------------------------------------------------------------------------
// ginBarrier()：获取外层 GIN barrier 的可变引用
// -------------------------------------------------------------------------
// 注意：仅在 outerGinBar.present == true 时调用，否则行为未定义
#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclBarrierSession<Coop>::ncclBarrierSession(
    Coop coop, ncclTeamTagRail, ncclGin gin, uint32_t index
  ):
  ncclBarrierSession_internal<Coop>(coop,
    nccl::utility::present(gin),
    nccl::utility::Absent(),
    nccl::utility::present(coop, gin, ncclTeamRail(gin.comm), gin.comm.hybridRailGinBarrier, index)
  ) {
}
#endif

// -------------------------------------------------------------------------
// sync()：执行完整的组合 barrier 同步
// -------------------------------------------------------------------------
// 内存序拆分策略（当两层 barrier 都存在时）：
//
//   innerLsaBar.sync(coop, releaseOrderOf(ord))：
//     - 对 innerLsaBar 使用 release 语义（下游的 outerGinBar 起 acquire 作用）
//     - 确保节点内所有写操作在 LSA barrier 完成后对外可见
//     - 若 ord=acq_rel，releaseOrderOf 返回 release；若 ord=relaxed，返回 relaxed
//
//   outerGinBar.sync(coop, acquireOrderOf(ord), fence)：
//     - 对 outerGinBar 使用 acquire 语义（上游的 innerLsaBar 起 release 作用）
//     - 确保能读到所有节点的写操作（在 GIN signal 接收后）
//     - 若 ord=acq_rel，acquireOrderOf 返回 acquire；若 ord=relaxed，返回 relaxed
//
// 内存序拆分图示（World barrier，ord=acq_rel）：
//   [写操作] → innerLsaBar.sync(release) → [节点内同步] →
//   outerGinBar.sync(acquire) → [跨节点同步] → [读到所有写]
//
// 单层 barrier 时直接传递原始 ord，不做拆分。
#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclLsaBarrierSession<Coop>& ncclBarrierSession<Coop>::lsaBarrier() {
  // innerLsaBar.thing：Optional 内部存储的 ncclLsaBarrierSession<Coop> 对象
  return this->innerLsaBar.thing;
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>& ncclBarrierSession<Coop>::ginBarrier() {
  // outerGinBar.thing：Optional 内部存储的 ncclGinBarrierSession<Coop> 对象
  return this->outerGinBar.thing;
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE void ncclBarrierSession<Coop>::sync(Coop, cuda::memory_order ord, ncclGinFenceLevel fence) {
  if (this->innerLsaBar.present) {
    this->innerLsaBar.thing.sync(this->coop, this->outerGinBar.present ? nccl::utility::releaseOrderOf(ord) : ord);
  }
  if (this->outerGinBar.present) {
    this->outerGinBar.thing.sync(this->coop, this->innerLsaBar.present ? nccl::utility::acquireOrderOf(ord) : ord, fence);
  }
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclResult_t ncclBarrierSession<Coop>::sync(
    Coop, cuda::memory_order ord, ncclGinFenceLevel fence, uint64_t timeoutCycles) {
  ncclResult_t lsaResult = ncclSuccess, railResult = ncclSuccess;

  // Inner LSA barrier (if present) - detects remote CTA/rank issues
  if (this->innerLsaBar.present) {
    uint64_t startCycle = clock64();
    lsaResult = this->innerLsaBar.thing.sync(
      this->coop,
      this->outerGinBar.present ? nccl::utility::releaseOrderOf(ord) : ord,
      timeoutCycles
    );
    uint64_t elapsed = clock64() - startCycle;
    timeoutCycles -= min(elapsed, timeoutCycles);
    // Because threads within a coop don't synchronize about the timeout condition,
    // we need to invoke the second barrier even if the first one times out,
    // to ensure that all the threads arrive at the coop sync.
  }

  // Outer GIN barrier (if present) - detects remote GPU/network issues
  if (this->outerGinBar.present) {
    railResult = this->outerGinBar.thing.sync(
      this->coop,
      this->innerLsaBar.present ? nccl::utility::acquireOrderOf(ord) : ord,
      fence,
      timeoutCycles
    );
  }
  return lsaResult != ncclSuccess ? lsaResult : railResult;
}
#endif

#endif // _NCCL_DEVICE_BARRIER__FUNCS_H_
