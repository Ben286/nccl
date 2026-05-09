/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// ===== gin_barrier__types.h：GIN Barrier 内部数据结构 =====
// 本文件定义 GIN Barrier 的句柄和会话内部状态结构体。
//
// 关键数据结构层次：
//   ncclGinBarrierHandle        —— 轻量句柄，可跨 host/device 传递
//   ncclGinBarrierSession_internal<Coop> —— 会话运行时状态（device only）
//   ncclGinBarrierSession<Coop>  —— 继承 _internal，提供对外接口（见 gin_barrier.h）

#ifndef _NCCL_DEVICE_GIN_BARRIER__TYPES_H_
#define _NCCL_DEVICE_GIN_BARRIER__TYPES_H_
#include "../gin_barrier.h"
#include "core__types.h"
#include "gin__types.h"

// -------------------------------------------------------------------------
// ncclGinBarrierHandle：GIN Barrier 句柄（host/device 均可持有）
// -------------------------------------------------------------------------
struct ncclGinBarrierHandle {
  // signal0：该 barrier 占用的 GIN signal slot 起始编号。
  //
  // GIN signal slot 分配模型（多 barrier 共存时）：
  //   每个 barrier 实例占用 team.nRanks 个连续 signal slot，
  //   分配给 barrierIndex 的 slot 范围为：
  //     [signal0 + barrierIndex * nRanks,  signal0 + barrierIndex * nRanks + nRanks - 1]
  //
  //   其中 slot signal0 + barrierIndex * nRanks + r 由 rank r "拥有"：
  //     - rank r 向该 slot 发送（signal）来通知自己已到达
  //     - 其他所有 rank 等待（waitSignal）该 slot 达到目标值
  //
  //   示意图（nRanks=3，barrierIndex=1）：
  //     slot 编号：  ... | signal0+3 | signal0+4 | signal0+5 | ...
  //                       rank 0      rank 1      rank 2   （barrier 1 的分配）
  ncclGinSignal_t signal0;

  // unused：预留字段，占位对齐（与 ncclLsaBarrierHandle 保持结构统一）
  ncclDevResourceHandle_t unused;
};

// -------------------------------------------------------------------------
// ncclGinBarrierSession_internal<Coop>：GIN Barrier 会话内部状态
// -------------------------------------------------------------------------
// 由 ncclGinBarrierSession<Coop> 继承，所有字段对子类可见。
// 所有字段在构造时初始化，sync() 期间只读（shadow 计数除外）。
#if NCCL_CHECK_CUDACC
template<typename Coop>
struct ncclGinBarrierSession_internal {
  // coop：参与 barrier 的协作组（负责线程分工和组内同步）
  Coop coop;

  // net：ncclGin 对象，封装所有 GIN 网络操作（signal/waitSignal/getSignalShadowPtr）
  // net.comm：Device Communicator 引用，用于访问 rank/nRanks 等全局信息
  ncclGin net;

  // team：本次 barrier 所在的逻辑 team（通常是 rail team，跨节点同 GPU 编号的 rank 集合）
  // team.rank：本 rank 在 team 内的编号（0..team.nRanks-1）
  // team.nRanks：team 总 rank 数
  // team.stride：world rank 步长（如 lsaSize，即跨多少个 LSA rank 跳一个 rail rank）
  ncclTeam team;

  // handle：GIN barrier 句柄（含 signal0 起始编号）
  ncclGinBarrierHandle handle;

  // index：本次 barrier 实例的编号（0..nBarriers-1）
  // 创建时由调用方指定，用于区分同一 handle 下的不同并发 barrier
  int index;

  // signal：本次 barrier 实例的 signal slot 基地址（= handle.signal0 + barrierIndex * team.nRanks）
  //
  // 每个 rank r 的"收信箱" slot 编号 = signal + r
  //   - 其他 rank 向 signal + team.rank 发信（net.signal）
  //   - 本 rank 等待 signal + peer 收到信号（net.waitSignal）
  //
  // 实际使用时：
  //   发送：net.signal(..., ncclGin_SignalInc{signal + team.rank}, ...)
  //   等待：net.waitSignal(..., signal + peer, waitVal, ...)
  ncclGinSignal_t signal;

  template<bool EnableTimeout>
  NCCL_DEVICE_INLINE ncclResult_t syncInternal(Coop, cuda::memory_order ord, ncclGinFenceLevel fence,
                                               uint64_t timeoutCycles);
};
#endif

#endif // _NCCL_DEVICE_GIN_BARRIER__TYPES_H_
