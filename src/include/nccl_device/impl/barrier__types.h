/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// ===== barrier__types.h：组合 Barrier 内部数据结构 =====
// 定义 ncclBarrierSession_internal<Coop>，它是 ncclBarrierSession<Coop> 的基类，
// 持有所有运行时状态字段。
//
// 核心设计：Optional<T>
//   nccl::utility::Optional<T> 是一个可空容器（类似 std::optional<T>）：
//     - present 字段：bool，表示该 barrier 是否参与本次同步
//     - thing 字段：T，存储实际的 barrier session 对象
//   使用 nccl::utility::present(...)  构造 present=true 的 Optional（携带初始化参数）
//   使用 nccl::utility::Absent()      构造 present=false 的 Optional（不占用实际资源）
//
// 三种激活组合：
//   World barrier：innerLsaBar.present=true + outerGinBar.present=true
//   LSA-only：     innerLsaBar.present=true + outerGinBar.present=false
//   Rail-only：    innerLsaBar.present=false + outerGinBar.present=true

#ifndef _NCCL_DEVICE_BARRIER__TYPES_H_
#define _NCCL_DEVICE_BARRIER__TYPES_H_
#include "../barrier.h"
#include "../utility.h"

#if NCCL_CHECK_CUDACC
// -------------------------------------------------------------------------
// ncclBarrierSession_internal<Coop>：组合 Barrier 内部状态
// -------------------------------------------------------------------------
template<typename Coop>
struct ncclBarrierSession_internal {
  // coop：协作组（参与 barrier 的线程集合）
  // 在构造时保存，sync() 中用于线程分工和组内同步
  Coop coop;

  // gin：ncclGin 对象（含 comm 引用）
  // Optional：仅 Rail-only 和 World barrier 时 present=true
  // Rail-only barrier（无 LSA）时同样需要 gin 访问 comm.railGinBarrier
  nccl::utility::Optional<ncclGin> gin;

  // innerLsaBar：内层 LSA barrier session
  // Optional：仅 LSA-only 和 World barrier 时 present=true
  // 激活时：负责节点内 lsa team 的同步（NVLink MultiCast/unicast）
  nccl::utility::Optional<ncclLsaBarrierSession<Coop>> innerLsaBar;

  // outerGinBar：外层 GIN barrier session
  // Optional：仅 Rail-only 和 World barrier 时 present=true
  // 激活时：负责跨节点 rail team 的同步（GIN DMA 信号）
  nccl::utility::Optional<ncclGinBarrierSession<Coop>> outerGinBar;

  // 构造函数模板：
  //   GinInit   — 用于初始化 gin 的参数包（present(...) 或 Absent()）
  //   InnerInit — 用于初始化 innerLsaBar 的参数包
  //   OuterInit — 用于初始化 outerGinBar 的参数包
  //
  // 通过模板参数包转发，避免对每种激活组合都写独立的构造函数重载。
  // Optional<T> 的构造接受 nccl::utility::present(args...) 或 nccl::utility::Absent()。
  template<typename GinInit, typename InnerInit, typename OuterInit>
  NCCL_DEVICE_INLINE ncclBarrierSession_internal(
      Coop coop, GinInit ginInit, InnerInit innerInit, OuterInit outerInit
    ):
    coop(coop), gin{ginInit}, innerLsaBar{innerInit}, outerGinBar{outerInit} {
    // 所有字段通过成员初始化列表直接构造，无额外开销
  }
};
#endif

#endif // _NCCL_DEVICE_BARRIER__TYPES_H_
