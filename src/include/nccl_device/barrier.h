/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/


// ===== barrier.h：组合 Barrier 接口 =====
// ncclBarrierSession 是对 LSA Barrier 和 GIN Barrier 的高层封装，
// 统一了节点内同步（LSA）和跨节点同步（GIN）的接口。
//
// 两层 Barrier 的分工：
//   innerLsaBar（LSA Barrier）：
//     作用范围：节点内的 lsa team（共享 NVLink 的 GPU 集合）
//     实现：通过 NVLink MultiCast / unicast 写 epoch 计数器
//
//   outerGinBar（GIN Barrier）：
//     作用范围：跨节点的 rail team（不同节点同编号的 GPU）
//     实现：通过 GIN DMA 网络发送/等待 signal
//
// 完整 World Barrier 的执行顺序：
//   1. innerLsaBar.sync(release)  —— 节点内同步，并将写操作 release 到 NVLink
//   2. outerGinBar.sync(acquire)  —— 跨节点同步，acquire 收到对方写操作的结果
//   这两步合起来保证所有节点所有 GPU 完成同步，并提供完整的内存可见性。
//
// 便捷构造标签：
//   ncclTeamTagWorld：全局 barrier（LSA 内同步 + GIN 跨节点同步）
//   ncclTeamTagLsa  ：仅节点内 LSA barrier（无 GIN）
//   ncclTeamTagRail ：仅跨节点 GIN barrier（无 LSA）
#ifndef _NCCL_DEVICE_BARRIER_H_
#define _NCCL_DEVICE_BARRIER_H_
#include "impl/core__types.h"
#include "impl/lsa_barrier__types.h"
#include "impl/gin_barrier__types.h"

#if NCCL_CHECK_CUDACC
// -------------------------------------------------------------------------
// ncclBarrierSession<Coop>：统一 Barrier 会话对象
// -------------------------------------------------------------------------
// 继承 ncclBarrierSession_internal<Coop>（含 innerLsaBar + outerGinBar）
// 根据构造时选择的 tag/参数，决定激活哪个子 barrier。
// 前向声明内部结构体（实现在 impl/barrier__types.h）
template<typename Coop>
struct ncclBarrierSession_internal;

template<typename Coop>
struct ncclBarrierSession: ncclBarrierSession_internal<Coop> {
  // Full featured constructor:

  // --- 便捷构造：Rail-only Barrier（仅跨节点同步）---
  // outerGinBar 激活，innerLsaBar = Absent（不参与 LSA 同步）。
  // 适用场景：节点内已单独同步过，只需做跨节点 fence。

  // --- 便捷构造：LSA-only Barrier（仅节点内同步）---
  // innerLsaBar 激活，outerGinBar = Absent（不参与 GIN 同步）。
  // 适用场景：所有通信均在节点内，无需跨节点同步。
  //   参数 comm：直接传入 ncclDevComm（无需 ncclGin）
  // --- 便捷构造：World Barrier（全局同步）---
  // 自动推断：
  //   innerTeam = ncclTeamLsa(gin.comm)      —— 节点内 lsa team
  //   outerTeam = ncclTeamRail(gin.comm)     —— 跨节点 rail team
  //   innerHandle = gin.comm.lsaBarrier      —— lsa barrier 句柄
  //   outerHandle = gin.comm.railGinBarrier  —— rail gin barrier 句柄
  //   innerMmHandle = gin.comm.lsaMultimem   —— lsa multimem 句柄
  // 两层 barrier 都激活，提供完整的全局同步。
  // --- 完整构造函数（Full-featured）---
  // 所有参数显式指定，支持任意组合的 innerTeam + outerTeam。
  //
  // 参数说明：
  //   coop           — 协作组（参与 barrier 的线程集合）
  //   innerTeam      — 内层 team（LSA 节点内，通常为 ncclTeamLsa）
  //   outerTeam      — 外层 team（GIN 跨节点，通常为 ncclTeamRail）
  //   gin            — ncclGin 对象（含 comm 引用和 GIN 操作）
  //   innerBarHandle — LSA barrier 句柄（含 resource buffer 句柄）
  //   outerBarHandle — GIN barrier 句柄（含 signal0）
  //   index          — barrier 实例编号（0..nBarriers-1）
  //   multimem       — 是否启用 MultiCast 模式（sm_90+）
  //   innerMmHandle  — MultiCast 句柄（multimem=true 时使用）
  NCCL_DEVICE_INLINE ncclBarrierSession(
    Coop, ncclTeam innerTeam, ncclTeam outerTeam, ncclGin,
    ncclLsaBarrierHandle innerBarHandle,
    ncclGinBarrierHandle outerBarHandle,
    uint32_t index,
    bool multimem=false, ncclMultimemHandle innerMmHandle={}
  );
  // Convenience constructors for baked in teams:
  NCCL_DEVICE_INLINE ncclBarrierSession(
    Coop, ncclTeamTagWorld, ncclGin, uint32_t index, bool multimem=false
  );
  NCCL_DEVICE_INLINE ncclBarrierSession(
    Coop, ncclTeamTagLsa, ncclDevComm const&, uint32_t index, bool multimem=false
  );
  NCCL_DEVICE_INLINE ncclBarrierSession(
    Coop, ncclTeamTagRail, ncclGin, uint32_t index
  );

  // Sessions are not copyable（会话含动态状态，禁止复制）
  ncclBarrierSession(ncclBarrierSession const&) = delete; // Sessions are not copyable

  // --- 访问子 barrier ---
  // lsaBarrier()：返回内层 LSA barrier 的引用（仅在 innerLsaBar.present 时调用）
  NCCL_DEVICE_INLINE ncclLsaBarrierSession<Coop>& lsaBarrier();
  // ginBarrier()：返回外层 GIN barrier 的引用（仅在 outerGinBar.present 时调用）
  NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>& ginBarrier();

  // --- sync：执行本次 barrier 的完整同步 ---
  // 根据激活的子 barrier 组合，按正确顺序执行：
  //   case 1: only innerLsaBar → innerLsaBar.sync(coop, ord)
  //   case 2: only outerGinBar → outerGinBar.sync(coop, ord, fence)
  //   case 3: both             → innerLsaBar.sync(release) + outerGinBar.sync(acquire)
  //     内存语义：release 保证 inner 完成后的写对外可见，acquire 保证能读到 outer 方的写
  NCCL_DEVICE_INLINE void sync(Coop, cuda::memory_order, ncclGinFenceLevel);
  NCCL_DEVICE_INLINE ncclResult_t sync(Coop, cuda::memory_order, ncclGinFenceLevel, uint64_t timeoutCycles);
};
#endif

#endif // _NCCL_DEVICE_BARRIER_H_
