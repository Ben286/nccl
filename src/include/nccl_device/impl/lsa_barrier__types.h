/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// ===== lsa_barrier__types.h：LSA Barrier 内部数据结构 =====
// 本文件定义了 barrier 的句柄（ncclLsaBarrierHandle）和内部状态（ncclLsaBarrierSession_internal）。
//
// Resource Buffer 布局（每个 handle 指向一段 resource buffer）：
//
//   [0 .. nBarriers-1]           : multimem epoch state（一个 per barrier，共享被所有 rank 看到）
//   [nBarriers .. 2*nBarriers-1] : unicast epoch state（供 unicast 模式读取本 rank 的 epoch）
//   [2*nBarriers .. 3*nBarriers-1]: mcInbox（multimem 模式的接收计数器，每 rank 独立）
//   [3*nBarriers .. 3*nBarriers + nBarriers*nRanks*nRanks - 1]: ucInbox（unicast 模式的接收矩阵）
//
// mcInbox(multimem=true)  → 返回 MultiCast 视图下的接收计数器（用 multimem.red 写入）
// mcInbox(multimem=false) → 返回 unicast 视图下的接收计数器（等待读取时用）
// ucInbox(owner, peer)    → 返回 owner rank 的 resource buffer 中 [3*nBarriers + index*nRanks + peer] 处

#ifndef _NCCL_DEVICE_MEM_BARRIER__TYPES_H_
#define _NCCL_DEVICE_MEM_BARRIER__TYPES_H_
#include "../lsa_barrier.h"
#include "core__types.h"

// ncclLsaBarrierHandle：LSA barrier 的不透明句柄
struct ncclLsaBarrierHandle {
  // bufHandle：resource buffer 的句柄，barrier 的所有状态存放在此 buffer 中
  ncclDevResourceHandle_t bufHandle;
  // nBarriers：该 handle 支持的并发 barrier 实例数
  // state 数组按 nBarriers 分段组织（详见文件头注释）
  int nBarriers;
};

#if NCCL_CHECK_CUDACC
// ncclLsaBarrierSession_internal<Coop>：barrier 会话的内部状态基类
// 被 ncclLsaBarrierSession 继承，提供底层的 inbox 地址计算
template<typename Coop>
struct ncclLsaBarrierSession_internal {
  Coop coop;  // 参与 barrier 的协作组
  ncclDevComm const& comm;  // DevComm 引用（用于 abortFlag 和 resource buffer 寻址）
  ncclTeam team;  // barrier 的逻辑 team（定义 nRanks 和 rank 编号）
  ncclLsaBarrierHandle handle;  // barrier 句柄（bufHandle + nBarriers）
  int index;  // 本 barrier 实例的编号（0..nBarriers-1）
  bool multimem;  // true：使用 NVLink MultiCast 加速（sm_90+）；false：unicast
  ncclMultimemHandle mmHandle;  // MultiCast 句柄（multimem=true 时有效）
  uint32_t epoch;  // 当前 barrier 轮次（构造时从 state 读取，析构时写回）

  // mcInbox(multimem)：返回 MultiCast/unicast 模式下的接收计数器地址
  //   multimem=true → 使用 ncclGetResourceBufferMultimemPointer（MultiCast 视图）
  //     arrive 时向此地址 multimem.red.add → 广播到所有 rank 的同一物理位置
  //   multimem=false → 使用 ncclGetResourceBufferLocalPointer（本 rank 视图）
  //     wait 时轮询此地址等待计数达到 epoch + nRanks
  //   地址 = state_base + 2*nBarriers + index（跳过前两段的 epoch state）
  NCCL_DEVICE_INLINE uint32_t* mcInbox(bool multimem) {
    uint32_t* state;
    if (multimem) { // multicast
      state = (uint32_t*)ncclGetResourceBufferMultimemPointer(comm, handle.bufHandle, mmHandle);
    } else { // unicast
      state = (uint32_t*)ncclGetResourceBufferLocalPointer(comm, handle.bufHandle);
    }
    return state + 2*handle.nBarriers + index;  // 跳过 epoch state 区域
  }

  // ucInbox(owner, peer)：返回 unicast 模式下，owner rank 收到来自 peer 的信号槽
  //   布局：owner rank 的 resource buffer 中，state[3*nBarriers + index*nRanks + peer]
  //   arrive 时：peer 向 owner 的 ucInbox(owner, peer) 写 epoch+1
  //   wait 时：owner 读自己的 ucInbox(myRank, peer) 等待来自 peer 的 epoch+1
  NCCL_DEVICE_INLINE uint32_t* ucInbox(int owner, int peer) {
    uint32_t* state = (uint32_t*)ncclGetResourceBufferPeerPointer(comm, handle.bufHandle, team, owner);
    return state + 3*handle.nBarriers + index*team.nRanks + peer;
  }

  template<bool EnableTimeout>
  NCCL_DEVICE_INLINE ncclResult_t waitInternal(Coop, cuda::memory_order order, uint64_t timeoutCycles);
};
#endif

#endif // _NCCL_DEVICE_MEM_BARRIER__TYPES_H_
