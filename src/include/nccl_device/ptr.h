/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// ===== ptr.h：ncclSymPtr<T> — Symmetric Memory 智能指针 =====
// 背景：Symmetric Memory 中每个 rank 的 buffer 在虚拟地址空间中具有固定跨度关系。
// 普通指针只能表示一个具体的地址，而 ncclSymPtr 封装了"window + offset"两个分量，
// 运行时可按需释放为任意 rank 的真实地址。
//
// 核心设计：
//   window：指向 ncclWindow_vidmem 结构，包含 lsaFlatBase/stride4G 等寻址参数
//   offset：该指针在 window 内的字节偏移
//
// 主要操作：
//   +=/+/-=/-：将 offset 按类型 T 的元素大小进行指针算术
//   operator-(a, b)：计算两个 SymPtr 的元素数差（仅计算 offset 差）
//   localPtr()：返回本 rank 的真实地址
//   lsaPtr(peer) / peerPtr(peer) / peerPtr(team, peer)：其他 rank 的地址
//   multimemPtr(mmHandle) / lsaMultimemPtr(comm)：NVLink MultiCast 地址
//
// 注意：localPtr() 特殊情况：window==nullptr 时直接将 offset 当作地址返回，
//   这允许在没有 window 时将普通指针封装为 SymPtr

#ifndef _NCCL_DEVICE_PTR_H_
#define _NCCL_DEVICE_PTR_H_
#include "core.h"
#include <stdint.h>

#if __cplusplus
// ncclSymPtr<T>：Symmetric Memory 智能指针模板类
template<typename T>
struct ncclSymPtr {
  using ElementType = T;  // 元素类型别名，供模板元编程使用
  // window：指向包含该 buffer 的 ncclWindow_vidmem。
  // window==nullptr 表示该 SymPtr 由普通内存地址封装而来
  ncclWindow_t window;

  // offset：该指针在 window 内的字节偏移
  // +=/+/-=/- 都改变 offset，而不改变 window
  size_t offset;

  // 构造函数：绑定 window 和 offset
  NCCL_HOST_DEVICE_INLINE constexpr ncclSymPtr(ncclWindow_t window=nullptr, size_t offset=0);

  // 类型转换运算符：允许将 ncclSymPtr<T> 隐式转换为 ncclSymPtr<U>
  // 仅改变元素类型解释，window 和 offset 不变
  template<typename U>
  NCCL_HOST_DEVICE_INLINE operator ncclSymPtr<U>() const;

  // operator+=：将 offset 按类型 T 的元素大小向前挪 d 个元素
  // 实现：offset = reinterpret_cast<T*>(offset) + d — 将 offset 当作 T* 进行指针算术
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator+=(int d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator+=(unsigned int d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator+=(long d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator+=(unsigned long d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator+=(long long d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator+=(unsigned long long d);

  // operator-=：将 offset 按类型 T 的元素大小向前回 d 个元素
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator-=(int d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator-=(unsigned int d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator-=(long d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator-=(unsigned long d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator-=(long long d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator-=(unsigned long long d);

  #if NCCL_CHECK_CUDACC
  // localPtr()：获取本 rank 的真实地址
  // 当 window!=nullptr：调用 ncclGetLocalPointer(window, offset)
  // 当 window==nullptr：直接将 offset 当作地址返回（普通指针模式）
  NCCL_DEVICE_INLINE T* localPtr() const;

  // lsaPtr(peer)：获取 lsa team 中 rank peer 的真实地址
  // peer 是 0..lsaSize-1 范围内的 lsa rank 编号
  NCCL_DEVICE_INLINE T* lsaPtr(int peer) const;

  // peerPtr(peer)：获取全局 world rank peer 的真实地址
  NCCL_DEVICE_INLINE T* peerPtr(int peer) const;

  // peerPtr(team, peer)：获取 team 中 rank peer 的真实地址
  NCCL_DEVICE_INLINE T* peerPtr(ncclTeam team, int peer) const;

  // multimemPtr(mmHandle)：获取 NVLink MultiCast 地址（写→广播/读→reduce）
  NCCL_DEVICE_INLINE T* multimemPtr(ncclMultimemHandle mmHandle) const;

  // lsaMultimemPtr(comm)：使用 comm.lsaMultimem 的 MultiCast 地址
  NCCL_DEVICE_INLINE T* lsaMultimemPtr(ncclDevComm const&) const;
  #endif
};

// 相等/不等 比较：window 和 offset 必须都相同/不同
// 全局算术运算符：返回新的 SymPtr，window 不变，仅改变 offset
template<typename T, typename Int>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T> operator+(ncclSymPtr<T> p, Int d);
template<typename T, typename Int>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T> operator-(ncclSymPtr<T> p, Int d);

// operator-(a, b)：计算两个 SymPtr 之间的元素数差（仅比较 offset，不验证 window 是否相同）
template<typename T>
NCCL_HOST_DEVICE_INLINE ptrdiff_t operator-(ncclSymPtr<T> a, ncclSymPtr<T> b);

template<typename T, typename Int>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T> operator==(ncclSymPtr<T> a, ncclSymPtr<T> b);
template<typename T, typename Int>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T> operator!=(ncclSymPtr<T> a, ncclSymPtr<T> b);
#endif

#endif
