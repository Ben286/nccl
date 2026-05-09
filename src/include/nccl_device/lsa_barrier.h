/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// ===== lsa_barrier.h：LSA 分布式 Barrier 接口 =====
// LSA Barrier 用于在 lsa team（或任意指定 team）的所有 rank 之间进行同步。
// 与 CUDA CTA barrier 不同，它跨多个 GPU 的 kernel 工作线程进行同步，
// 完全无需 CPU 介入，通过 Symmetric Memory 中的计数器实现。
//
// 实现原理（两种模式）：
//   1. multimem 模式（需要 arch >= sm_90，即 Hopper）：
//      - arrive：每个 rank 的代表线程向 MultiCast 地址原子加 1
//        （asm "multimem.red.release.sys.add.u32"）
//        MultiCast 写 = 广播到所有 rank 的同一物理地址 → 各 rank 各自看到 +1
//      - wait：等待 mcInbox 计数器达到 epoch + nRanks
//        （每次 arrive 将所有 rank 的 inbox 各加 1）
//
//   2. unicast 模式（通用，适用所有硬件）：
//      - arrive：每个 rank 向其他所有 rank 的 ucInbox[me] 写入 epoch+1
//        （N 个 rank 共需 N*(N-1) 次写操作，用 coop 内多线程并行分担）
//      - wait：等待来自每个 peer 的 ucInbox[peer] 达到 epoch+1
//
// 关键字段：
//   epoch：barrier 轮次计数器，每次 wait 完成后递增
//          multimem 模式递增 nRanks；unicast 模式递增 1
//   index：barrier 实例编号（0..nBarriers-1），支持多个并发 barrier
//
// 析构函数（~ncclLsaBarrierSession）：
//   将 epoch 写回 state[] 保存，下一次创建同一 index 的 session 时恢复
//   这允许跨多次 session 保持 barrier epoch 连续性

#ifndef _NCCL_DEVICE_MEM_BARRIER_H_
#define _NCCL_DEVICE_MEM_BARRIER_H_
#include "impl/core__types.h"

struct ncclLsaBarrierHandle;

// Host 端创建 lsa barrier 需求：
// team：将要使用该 barrier 的 team
// nBarriers：需要的并发 barrier 实例数
// outHandle：输出 barrier 句柄（创建 session 时使用）
// outReq：输出资源需求（插入 ncclDevCommRequirements.resourceRequirementsList）
NCCL_EXTERN_C __host__ ncclResult_t ncclLsaBarrierCreateRequirement(ncclTeam_t, int nBarriers, ncclLsaBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq);

#if NCCL_CHECK_CUDACC
// ncclLsaBarrierSession<Coop>：LSA Barrier 的 RAII 会话对象
// Coop：协作组类型（如 ncclCoopWarp、ncclCoopTile 等）
// 构造时读取 epoch，析构时写回 epoch，确保每次 barrier 的轮次正确连续
template<typename Coop>
struct ncclLsaBarrierSession_internal;

template<typename Coop>
struct ncclLsaBarrierSession: ncclLsaBarrierSession_internal<Coop> {
  // 构造（通用版本）：
  //   coop：参与 barrier 的协作组
  //   comm：DevComm 引用
  //   team：barrier 所在的逻辑 team
  //   handle：barrier 句柄（包含 bufHandle 和 nBarriers）
  //   index：本次使用的 barrier 实例编号
  //   multimem：是否使用 NVLink MultiCast 加速（需要 sm_90+）
  //   mmHandle：MultiCast 句柄（multimem=true 时有效）
  NCCL_DEVICE_INLINE ncclLsaBarrierSession(Coop, ncclDevComm const&, ncclTeam, ncclLsaBarrierHandle, uint32_t index, bool multimem=false, ncclMultimemHandle mmHandle={});

  // 构造（lsa tag 版本）：使用 comm.lsaBarrier 和 ncclTeamLsa(comm)，更便捷
  NCCL_DEVICE_INLINE ncclLsaBarrierSession(Coop, ncclDevComm const&, ncclTeamTagLsa, uint32_t index, bool multimem=false);

  // 析构：将 epoch 写回 state[] 以持久化，供下一次 session 恢复
  NCCL_DEVICE_INLINE ~ncclLsaBarrierSession();

  ncclLsaBarrierSession(ncclLsaBarrierSession const&) = delete; // Sessions are not copyable

  // arrive：通知其他 rank 本 rank 已到达 barrier
  //   内存序 order 决定 release 语义（保证 arrive 之前的写对其他 rank 可见）
  NCCL_DEVICE_INLINE void arrive(Coop, cuda::memory_order);

  // wait：等待所有其他 rank 都 arrive
  //   内存序 order 决定 acquire 语义（保证看到其他 rank arrive 之前的写）
  NCCL_DEVICE_INLINE void wait(Coop, cuda::memory_order);

  // sync：arrive + wait 的组合，完成一次完整的 barrier
  NCCL_DEVICE_INLINE void sync(Coop, cuda::memory_order);
  NCCL_DEVICE_INLINE ncclResult_t wait(Coop, cuda::memory_order, uint64_t timeoutCycles);
  NCCL_DEVICE_INLINE ncclResult_t sync(Coop, cuda::memory_order, uint64_t timeoutCycles);
};
#endif

#endif // _NCCL_DEVICE_MEM_BARRIER_H_
