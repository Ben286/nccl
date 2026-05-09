/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// ===== ll_a2a__types.h：Low-Latency A2A 内部数据结构 =====
// 定义 ncclLLA2AHandle 和 ncclLLA2ASession_internal<Coop>。
//
// Resource Buffer 内存布局（整体视角）：
//
//   handle 的 bufHandle 指向一块 resource buffer，其中按 block 分段：
//   每个 block 的数据区起始偏移为：
//     block * (1 + 2 * nSlots) * sizeof(uint4)
//
//   block 内部布局（uint4 为单位，每个 uint4 = 16 字节 = 128 位）：
//     +-------+----------+----------+---...---+----------+----------+---...---+
//     | [0]   | [1]      | [2]      | ...     | [nSlots] | [nSlots+1] | ...  |
//     | epoch | slot[0]  | slot[1]  |         | slot[0]  | slot[1]  |        |
//     | line  | (even)   | (even)   | (even)  | (odd)   | (odd)    | (odd)  |
//     +-------+----------+----------+---...---+----------+----------+---...---+
//     ↑ 持久化 ↑                                ↑ 奇偶双 buffer（epoch & 1 选择）
//
//   epoch 持久化行（偏移 0）：
//     uint4.x = epoch - 2（下次构造时读取并加 2 恢复）
//     初始化：x=0，即第一次构造后 epoch=2（从 2 开始，避免与"0"初始值混淆）
//
//   slot buffer（偏移 1 .. 2*nSlots）：
//     calcSlotOffset() = block*(1+2*nSlots) + 1 + (epoch & 1) * nSlots
//     即 epoch 偶数时使用前半段，奇数时使用后半段，实现无锁双 buffer。
//
//   每个 slot 存储格式（LL 协议，uint4）：
//     { data_lo, epoch_tag, data_hi, epoch_tag }
//     发送方写：st.v4.u32 [addr], {data_lo, epoch, data_hi, epoch}
//     接收方等：ld.v4.u32，检查 y==epoch && w==epoch
//
// calcSlotOffset()：
//   返回当前 epoch 对应的 slot buffer 起始偏移（相对于 resource buffer 的 uint4 索引）
//   公式：block * (1 + 2*nSlots) + 1 + (epoch & 1) * nSlots
//     block*(1+2*nSlots) — 跳过前面 block 的数据
//     +1                  — 跳过本 block 的 epoch 持久化行
//     +(epoch&1)*nSlots   — 偶数轮：偏移 0（前半），奇数轮：偏移 nSlots（后半）

#ifndef _NCCL_DEVICE_LL_A2A__TYPES_H_
#define _NCCL_DEVICE_LL_A2A__TYPES_H_
#include "../ll_a2a.h"
#include "core__types.h"

// -------------------------------------------------------------------------
// ncclLLA2AHandle：A2A resource buffer 句柄（host/device 均可持有）
// -------------------------------------------------------------------------
struct ncclLLA2AHandle {
  // bufHandle：resource buffer 的 device 句柄，用于 ncclGetResourceBuffer*** 函数族
  // 通过该句柄可获取每个 peer 的 slot buffer 基地址（unicast 或 multimem）
  ncclDevResourceHandle_t bufHandle;

  // nSlots：每个 block 的 slot 总数（= ncclLLA2ACalcSlots(maxElts, maxEltSize)）
  // 决定每个 block 在 resource buffer 中占用的行数（1 + 2*nSlots）
  // 也是 calcSlotOffset 中 pitch 使用的基准
  uint32_t nSlots;
};

// -------------------------------------------------------------------------
// ncclLLA2ASession_internal<Coop>：A2A 会话内部状态
// -------------------------------------------------------------------------
#if NCCL_CHECK_CUDACC
template<typename Coop>
struct ncclLLA2ASession_internal {
  // coop：协作组（多线程协作访问 slot，bcast 时分配发送任务）
  Coop coop;

  // comm：Device Communicator（含 resourceWindow、abortFlag 等）
  // 注意：持有引用（不是副本），要求 comm 生命周期覆盖会话
  ncclDevComm const& comm;

  // team：参与 A2A 的逻辑 team（rank + nRanks + stride）
  // team.rank：本 rank 在 team 内的编号（0..nRanks-1）
  // team.nRanks：team 总 rank 数（= A2A 参与者总数）
  ncclTeam team;

  // handle：A2A 句柄（bufHandle + nSlots）
  ncclLLA2AHandle handle;

  // block：当前 block 编号（0..nBlocks-1）
  // 每个 block 在 resource buffer 中占独立区域，互不干扰
  int block;

  // pitch：单个 slot 的 uint4 行数（= maxElts，即每个 elt 间隔几行）
  // 当 T 较大（sizeof(T) > 8）时，pitch 决定多 uint4 行之间的步长：
  //   slot 内第 u 个 uint4 行地址 = bufBase + slotsOffset + elt + u * pitch
  // 单 elt 时 pitch = maxElts（即下一个 slot 的偏移）
  int pitch;

  // multimem：是否启用 MultiCast 广播模式（bcast 时选择路径）
  bool multimem;

  // mmHandle：MultiCast 句柄（multimem=true 时供 bcast 使用）
  // 通过 ncclGetResourceBufferMultimemPointer 获取 MultiCast 视图地址
  ncclMultimemHandle mmHandle;

  // epoch：当前 A2A epoch 编号（uint32_t）
  // 初始值从 resource buffer 恢复：epoch = line->x + 2
  //   line->x 是上次析构时写入的 epoch - 2
  //   加 2 的原因：保证新 epoch 的 tag 值不等于上一轮还可能残留在 buffer 中的 epoch-1
  //   初始化时 line->x = 0，所以第一次 epoch = 2（从 2 开始）
  //
  // epoch 回绕保护（endEpoch 中）：
  //   若 epoch >= -2u（接近 uint32_t 最大值），则清零 slot buffer，
  //   然后 epoch 从 -1u 跳到 2（+3）而非 0（+1），跳过 0/1 两个"危险值"。
  //   保证 epoch 永远不等于 0（0 是 buffer 初始化值，会导致误判为"已接收"）
  uint32_t epoch;

  // slotsOffset：当前 epoch 对应的 slot buffer 起始偏移（uint4 行索引）
  // 由 calcSlotOffset() 计算，每次 endEpoch() 后更新
  // send/recv 时使用：buf + slotsOffset + elt
  uint32_t slotsOffset;

  // calcSlotOffset()：根据当前 block 和 epoch 计算 slot buffer 偏移
  //   = block * (1 + 2*nSlots) + 1 + (epoch & 1) * nSlots
  // epoch 偶数时：slotsOffset 指向前半段（偶数 buffer）
  // epoch 奇数时：slotsOffset 指向后半段（奇数 buffer）
  // 这样相邻两轮 epoch 使用不同的内存区域，避免读写竞争
  NCCL_DEVICE_INLINE uint32_t calcSlotOffset() const {
    // block*(1 + 2*handle.nSlots)：跳过前面所有 block 的数据区
    // +1：跳过本 block 的 epoch 持久化行
    // +(epoch & 1)*handle.nSlots：偶数 epoch → 前半，奇数 epoch → 后半
    return block*(1 + 2*handle.nSlots) + 1 + (epoch & 1)*handle.nSlots;
  }
};
#endif

#endif // _NCCL_DEVICE_LL_A2A__TYPES_H_
