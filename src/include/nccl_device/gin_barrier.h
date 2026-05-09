/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/


// ===== gin_barrier.h：GIN 分布式 Barrier 接口 =====
// GIN Barrier 用于在 rail team（跨节点）之间进行同步，底层通过 GIN
//（GPU Interconnect Notification）实现，无需 CPU 介入。
//
// 与 LSA Barrier 的区别：
//   - LSA Barrier：节点内所有 rank，通过 NVLink Symmetric Memory 直接写信号
//   - GIN Barrier：跨节点 rail team，通过 GIN DMA 网络发送信号
//
// 实现原理：
//   每个 rank 向所有其他 team rank 发送 GIN signal（net.signal）
//   然后等待来自所有其他 rank 的 GIN signal（net.waitSignal）
//   使用旋转发送模式（rotating pattern）避免某个 rank 成为热点
//
// 关键设计：
//   signal 编号：handle.signal0 + barrierIndex * nRanks + rank
//     每个 barrier 实例占用 nRanks 个连续 signal slot
//     每个 rank 使用自己的 signal slot 来接收来自其他 rank 的通知
//   shadowPtr：GIN signal 的 shadow 副本，记录已收到的信号次数
//     用于计算下一次 wait 的目标值（waitVal = ++*shadowPtr）
#ifndef _NCCL_DEVICE_GIN_BARRIER_H_
#define _NCCL_DEVICE_GIN_BARRIER_H_
#include "core.h"
#if defined(NCCL_OS_WINDOWS)
#include "gin_win_stub.h"
#else
#include "gin.h"
#endif

struct ncclGinBarrierHandle;

// Host 端创建 GIN barrier 需求：
// comm：NCCL communicator
// team：将要使用该 barrier 的 team（通常是 rail team）
// nBarriers：需要的并发 barrier 实例数
// outHandle：输出 GIN barrier 句柄
// outReq：输出资源需求（插入 resourceRequirementsList）
NCCL_EXTERN_C __host__ ncclResult_t ncclGinBarrierCreateRequirement(ncclComm_t, ncclTeam_t, int nBarriers, ncclGinBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq);

#if NCCL_CHECK_CUDACC
// ncclGinFenceLevel：GIN barrier 的 fence 级别
// 当前只有 Relaxed 一种（仅执行同步，不提供强内存序保证）
// 未来可能新增 Release/Acquire 等级别
enum class ncclGinFenceLevel {
  Relaxed
};

// ncclGinBarrierSession<Coop>：GIN Barrier 的 RAII 会话对象
// Coop：协作组类型
template<typename Coop>
struct ncclGinBarrierSession_internal;

template<typename Coop>
struct ncclGinBarrierSession: ncclGinBarrierSession_internal<Coop> {
  // 构造（通用版本）：
  //   coop：参与 barrier 的协作组
  //   net：ncclGin 对象（包含 comm 和 GIN 操作方法）
  //   team：barrier 所在的逻辑 team
  //   handle：GIN barrier 句柄（signal0 起始编号）
  //   index：本次使用的 barrier 实例编号（0..nBarriers-1）
  NCCL_DEVICE_INLINE ncclGinBarrierSession(Coop, ncclGin, ncclTeam, ncclGinBarrierHandle, uint32_t index);

  // 构造（rail tag 版本）：使用 ncclTeamRail(net.comm) 和 comm.railGinBarrier
  NCCL_DEVICE_INLINE ncclGinBarrierSession(Coop, ncclGin, ncclTeamTagRail, uint32_t index);
  NCCL_DEVICE_INLINE ncclGinBarrierSession(Coop, ncclGin, ncclTeamTagWorld, uint32_t index);

  // 析构：GIN barrier 无需持久化状态（shadow 指针已在 waitSignal 内部更新）
  NCCL_DEVICE_INLINE ~ncclGinBarrierSession();

  ncclGinBarrierSession(ncclGinBarrierSession const&) = delete; // Sessions are not copyable

  // sync：完成一次完整的 GIN barrier（arrive + wait 合并）
  //   ord：内存序（影响 signal 发送的 fence 级别）
  //   fence：GIN fence 级别
  NCCL_DEVICE_INLINE void sync(Coop, cuda::memory_order, ncclGinFenceLevel);
  NCCL_DEVICE_INLINE ncclResult_t sync(Coop, cuda::memory_order, ncclGinFenceLevel, uint64_t timeoutCycles);
};
#endif

#endif // _NCCL_DEVICE_GIN_BARRIER_H_
