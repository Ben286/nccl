/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// ===== ll_a2a.h：Low-Latency All-to-All Session 接口 =====
// ncclLLA2ASession 提供低延迟的 All-to-All（A2A）通信语义，
// 用于在 team 内的所有 rank 之间高效交换少量数据（Low-Latency LL 模式）。
//
// 核心设计：LL（Low Latency）协议
//   每个数据槽（slot）以 uint4（128位）为单位存储，格式为：
//     { data_lo, epoch_tag, data_hi, epoch_tag }
//   其中 epoch_tag 是当前 epoch 的奇偶标记（uint32_t）。
//   发送方写 data + epoch，接收方轮询直到读到预期的 epoch_tag。
//   这样无需额外的 barrier，通过 epoch_tag 本身实现生产者-消费者同步。
//
// epoch 奇偶交替机制：
//   内存中的 slot 按 epoch 奇偶交替复用（双 buffer），
//   避免覆盖上一轮 epoch 还未被读取的数据。
//   calcSlotOffset() 根据 epoch & 1 确定当前用哪套 buffer。
//
// Resource Buffer 布局（per block，nSlots 个槽）：
//   每个 block 占用 1 + 2*nSlots 个 uint4 行：
//     偏移 0                      : epoch 持久化行（x=epoch-2，跨核执行恢复用）
//     偏移 1 .. nSlots            : epoch 偶数轮 slot buffer
//     偏移 nSlots+1 .. 2*nSlots  : epoch 奇数轮 slot buffer
//   每个 slot 支持多个 rank 同时写（send 到不同 peer 的相同 slot）。
//
// 关键约束（endEpoch 的使用规则）：
//   每次 epoch 内，对 team 中的每个 peer：
//     1. 至少调用一次 send(peer, ...)    向该 peer 写入数据
//     2. 至少调用一次 recv(slot, ...)    接收该 peer 向自己写入的数据
//   两个条件都满足后才能调用 endEpoch()，否则会产生 epoch 不匹配的竞争条件。
//
// 多播（multimem）模式：
//   若 multimem=true（sm_90+，NVLink MultiCast 可用），
//   bcast() 通过 MultiCast 地址一次写入所有 peer，避免 nRanks-1 次 unicast。

#ifndef _NCCL_DEVICE_LL_A2A_H_
#define _NCCL_DEVICE_LL_A2A_H_
#include "impl/core__types.h"

// 前向声明（实现在 impl/ll_a2a__types.h）
struct ncclLLA2AHandle;

// -------------------------------------------------------------------------
// Host 端 API
// -------------------------------------------------------------------------

// ncclLLA2ACalcSlots：计算支持 maxElts 个元素（最大 maxEltSize 字节）所需的 slot 数。
// slot 数 = divUp(maxElts * divUp(maxEltSize, 8), 1)，每个 slot 存 8 字节数据。
// 返回值用于 ncclLLA2ACreateRequirement 的 nSlots 参数。
NCCL_EXTERN_C __host__ int ncclLLA2ACalcSlots(int maxElts, int maxEltSize);

// ncclLLA2ACreateRequirement：创建 A2A resource buffer 需求。
// 参数：
//   nBlocks   — 同时使用该 A2A handle 的 block 数（每个 block 有独立的 slot buffer）
//   nSlots    — 每个 block 的 slot 数（由 ncclLLA2ACalcSlots 计算）
//   outHandle — 输出 A2A 句柄（含 bufHandle + nSlots）
//   outReq    — 输出资源需求（插入 resourceRequirementsList）
NCCL_EXTERN_C __host__ ncclResult_t ncclLLA2ACreateRequirement(int nBlocks, int nSlots, ncclLLA2AHandle_t* outHandle, ncclDevResourceRequirements_t* outReq);

#if NCCL_CHECK_CUDACC
// -------------------------------------------------------------------------
// ncclLLA2ASession<Coop>：Low-Latency A2A 会话对象（RAII）
// -------------------------------------------------------------------------
// 构造时读取 epoch，析构时写回 epoch（支持跨 kernel 持久化状态）。
// 会话期间在 coop 所有线程间共享状态，每个 block 对应一个独立会话。
// 前向声明 Session 内部结构（实现在 impl/ll_a2a__types.h）
template<typename Coop>
struct ncclLLA2ASession_internal;

template<typename Coop>
struct ncclLLA2ASession: ncclLLA2ASession_internal<Coop> {
  // 构造函数：
  //   coop      — 协作组（通常是 warp 或 block 级）
  //   comm      — Device Communicator（含 abortFlag、resourceWindow 等）
  //   team      — 参与 A2A 的逻辑 team（rank + nRanks + stride）
  //   handle    — A2A 句柄（bufHandle + nSlots）
  //   block     — 当前 block 编号（0..nBlocks-1），用于定位 resource buffer 中的行
  //   maxElts   — 单个 slot 最大支持的元素数（决定 pitch，即 elt 间距）
  //   multimem  — 是否启用 MultiCast 广播模式（sm_90+ NVLink MultiCast）
  //   mmHandle  — MultiCast 句柄（multimem=true 时传入，否则忽略）
  NCCL_DEVICE_INLINE ncclLLA2ASession(Coop, ncclDevComm const&, ncclTeam, ncclLLA2AHandle, uint32_t block, int maxElts, bool multimem=false, ncclMultimemHandle mmHandle={});

  // 析构函数：将当前 epoch-2 写回 resource buffer，供下次构造时恢复
  NCCL_DEVICE_INLINE ~ncclLLA2ASession();

  ncclLLA2ASession(ncclLLA2ASession const&) = delete; // Sessions are not copyable

  // recv<T>：从本地 slot 接收数据（spin-wait 直到 epoch_tag 匹配）
  //   slot — slot 编号
  //   返回：发送方写入的数据 T
  // 实现：委托给 recvUnrolled<1,1>
  // bcast<T>：向所有 peer 的 slot 广播数据（= 对每个 peer 调用 send，或用 multimem 一次写）
  //   slot — slot 编号
  //   data — 广播数据
  // multimem 模式：写一次 MultiCast 地址，硬件自动广播到所有参与 GPU
  // unicast 模式：按循环顺序 send 给所有 peer（含自己）
  // send<T>：向指定 peer 的 slot 写入数据（LL 协议格式：data+epoch_tag）
  //   peer — 目标 peer 在 team 内的 rank
  //   slot — slot 编号（0..nSlots-1）
  //   data — 要发送的数据（类型 T，最大 divUp(sizeof(T),8)*8 字节）
  // 实现：将 T 拆分为 uint32_t 对，写入 {data_lo, epoch, data_hi, epoch} 的 uint4
  template<typename T>
  NCCL_DEVICE_INLINE void send(int peer, int slot, T data);

  template<typename T>
  NCCL_DEVICE_INLINE void bcast(int slot, T data);

  template<typename T>
  NCCL_DEVICE_INLINE T recv(int slot);

  // recvUnrolled<MinEltCount, MaxEltCount, T>：批量接收多个 slot 的数据（展开优化）
  //   eltStart  — 起始 slot 编号
  //   eltCount  — 实际接收的 slot 数（MinEltCount <= eltCount <= MaxEltCount）
  //   eltStride — slot 间步长（通常 1）
  //   vals      — 输出数组 T[MaxEltCount]
  //
  // MinEltCount 用于静态保证至少读取 MinEltCount 个 slot（供编译器展开）：
  //   u < MinEltCount：总是读取（编译时确定，无运行时分支）
  //   u < eltCount   ：运行时判断
  // 两个条件 OR 后确定是否读取第 u 个 slot。
  //
  // spin-wait 机制：
  //   循环读取 tmp[u][v]（uint4），检查 y == epoch && w == epoch，
  //   即两个 epoch_tag 字段都匹配时认为数据有效。
  //   每隔一定次数调用 testAbort 检查 abortFlag（故障容忍）。
  template<int MinEltCount, int MaxEltCount, typename T>
  NCCL_DEVICE_INLINE void recvUnrolled(int eltStart, int eltCount, int eltStride, T(&vals)[MaxEltCount]);

  // recvReduce<Unroll, Elt, EltToAcc, Reduce>：接收并 reduce 多个 slot 的数据
  //   Unroll    — 循环展开因子（模板参数，编译期决定）
  //   Elt       — 原始元素类型
  //   EltToAcc  — Elt -> Acc 的转换函数（如 float -> float2）
  //   Reduce    — (Acc, Acc) -> Acc 的规约函数
  //   eltStart/eltCount/eltStride — slot 范围描述
  //   返回：所有接收元素经 eltToAcc 转换后 reduce 的结果（类型 Acc）
  //
  // 实现：两段循环（整块 Unroll 段 + 尾余段），保证编译器最大展开
  template<int Unroll, typename Elt, typename EltToAcc, typename Reduce>
  NCCL_DEVICE_INLINE auto recvReduce(int eltStart, int eltCount, int eltStride, EltToAcc eltToAcc, Reduce red)
    -> decltype(eltToAcc(nccl::utility::declval<Elt>())) ;

  // End an alltoall region. For every peer in team you must have done both of the
  // following each of which can be accomplished using any thread in coop:
  //  1. Targeted that peer with at least one send().
  //  2. Received from a slot targeted by that peer.
  // endEpoch：结束当前 A2A epoch，为下一轮 send/recv 准备
  //
  // 使用约束：调用前必须已完成对 team 中每个 peer 的 send 和 recv，
  //   否则会破坏下一轮 epoch 的正确性（新 epoch 覆盖未读取的数据）。
  //
  // 实现流程：
  //   1. 若 epoch 接近回绕（>= -2u），清零当前 slot buffer（防止新 epoch 读到旧数据）
  //   2. coop.sync() 保证清零完成后再推进 epoch
  //   3. epoch += 1（或 +3 绕回：-1u → 2u，保证不经过 0）
  //   4. 更新 slotsOffset（指向新的 slot buffer）
  NCCL_DEVICE_INLINE void endEpoch(Coop);
};
#endif

#endif // _NCCL_DEVICE_LL_A2A_H_
