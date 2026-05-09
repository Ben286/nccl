/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// ===== ptr__funcs.h：ncclSymPtr<T> 所有成员函数和运算符的内联实现 =====
// 关键技术：
//   1. operator+=/-= 的 offset 计算技巧：
//      offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) + d)
//      将 offset 先重解释为 T* 指针，加上 d 个元素，再转回 size_t
//      这样 offset 的增量就是 d * sizeof(T)，正确实现指针算术
//   2. localPtr() 的 fallback 设计：
//      window==nullptr 时，return (T*)offset
//      这允许用户将普通指针封装为 ncclSymPtr：
//      调用 ncclSymPtr<T>(nullptr, (size_t)rawPtr) 就能得到一个可用 localPtr() 的 SymPtr
//   3. 所有 Ptr 获取函数都转发到 ncclGet***Pointer，共享寻址逻辑

#ifndef _NCCL_DEVICE_PTR__FUNCS_H_
#define _NCCL_DEVICE_PTR__FUNCS_H_
#include "ptr__types.h"
#include "core__funcs.h"
#include "comm__types.h"

#if __cplusplus

// lsaMultimemPtr(comm)：使用 comm.lsaMultimem 句柄获取 MultiCast 地址
// 是 multimemPtr 的便捷包装，自动使用 lsa 级别的 MultiCast 句柄
// localPtr()：获取本 rank 对应的真实地址
// 特殊设计：当 window==nullptr 时直接将 offset 当作原始指针返回
//   这允许用 ncclSymPtr<T>(nullptr, (size_t)rawPtr) 将普通指针包装为 SymPtr
//   而 window!=nullptr 时调用 ncclGetLocalPointer 进行 Symmetric Memory 寻址
// operator-=：将 offset 按 T 的元素大小向后移动 d 个元素
// operator+=：将 offset 按 T 的元素大小向前移动 d 个元素
// 技巧：reinterpret_cast<T*>(offset) + d 利用 C++ 指针算术（增量 = d * sizeof(T)）
// 然后将结果转回 size_t 存入 offset
// 类型转换运算符：将 ncclSymPtr<T> 转为 ncclSymPtr<U>
// window 和 offset 不变，仅解释元素类型改变
// 构造函数：直接存储 window 和 offset
template<typename T>
NCCL_HOST_DEVICE_INLINE constexpr ncclSymPtr<T>::ncclSymPtr(ncclWindow_t window, size_t offset):
  window(window), offset(offset) {
}

template<typename T>
template<typename U>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>::operator ncclSymPtr<U>() const {
  return {window, offset};
}

template<typename T>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& ncclSymPtr<T>::operator+=(int d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) + d);
  return *this;
}
// multimemPtr(mmHandle)：获取 NVLink MultiCast 地址
// 内部调用 ncclGetMultimemPointer(window, offset, mmHandle)
// 向该地址写入 → 广播到所有 lsa rank；从该地址读取 → reduce 所有 lsa rank
template<typename T>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& ncclSymPtr<T>::operator+=(unsigned int d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) + d);
  return *this;
}

template<typename T>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& ncclSymPtr<T>::operator+=(long d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) + d);
  return *this;
}
template<typename T>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& ncclSymPtr<T>::operator+=(unsigned long d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) + d);
  return *this;
}

// peerPtr(team, peer)：获取 team 中 rank peer 的真实地址
// 内部调用 ncclGetPeerPointer(window, offset, team, peer)
// 寻址公式：i = lsaRank + (peer - team.rank)*team.stride
template<typename T>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& ncclSymPtr<T>::operator+=(long long d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) + d);
  return *this;
}
template<typename T>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& ncclSymPtr<T>::operator+=(unsigned long long d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) + d);
  return *this;
}

template<typename T>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& ncclSymPtr<T>::operator-=(int d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) - d);
  return *this;
}
// peerPtr(peer)：获取全局 world rank peer 的真实地址
// 内部调用 ncclGetPeerPointer(window, offset, peer)
// peer 是全局 world rank 编号
template<typename T>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& ncclSymPtr<T>::operator-=(unsigned int d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) - d);
  return *this;
}

template<typename T>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& ncclSymPtr<T>::operator-=(long d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) - d);
  return *this;
}
template<typename T>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& ncclSymPtr<T>::operator-=(unsigned long d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) - d);
  return *this;
}

// lsaPtr(peer)：获取 lsa team 中 rank peer 的真实地址
// 内部调用 ncclGetLsaPointer(window, offset, peer)
// peer 是 0..lsaSize-1 范围内的 lsa rank 编号
template<typename T>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& ncclSymPtr<T>::operator-=(long long d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) - d);
  return *this;
}
template<typename T>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& ncclSymPtr<T>::operator-=(unsigned long long d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) - d);
  return *this;
}

#if NCCL_CHECK_CUDACC
// operator!=：window 或 offset 有一个不同即返回 true
template<typename T>
NCCL_DEVICE_INLINE T* ncclSymPtr<T>::localPtr() const {
  if (window) {
    // 通过 window 的 lsaFlatBase + lsaRank*stride4G + offset 计算本 rank 地址
    return (T*)ncclGetLocalPointer(window, offset);
  } else {
    // window==nullptr：offset 本身就是原始指针（普通内存模式）
    return (T*)offset;
  }
}
#endif

#if NCCL_CHECK_CUDACC
template<typename T>
NCCL_DEVICE_INLINE T* ncclSymPtr<T>::lsaPtr(int peer) const {
  return (T*)ncclGetLsaPointer(window, offset, peer);
}
#endif

#if NCCL_CHECK_CUDACC
// operator==：window 和 offset 都相同时返回 true
template<typename T>
NCCL_DEVICE_INLINE T* ncclSymPtr<T>::peerPtr(int peer) const {
  return (T*)ncclGetPeerPointer(window, offset, peer);
}
#endif

#if NCCL_CHECK_CUDACC
template<typename T>
NCCL_DEVICE_INLINE T* ncclSymPtr<T>::peerPtr(ncclTeam team, int peer) const {
  return (T*)ncclGetPeerPointer(window, offset, team, peer);
}
#endif

#if NCCL_CHECK_CUDACC
// 全局 operator-（两个 SymPtr）：计算两者之间的元素数差
// 实现：将 a.offset 和 b.offset 分别解释为 T*，然后计算指针差
// 注意：不检查 window 是否相同，调用方负责确保同一 window
template<typename T>
NCCL_DEVICE_INLINE T* ncclSymPtr<T>::multimemPtr(ncclMultimemHandle mmHandle) const {
  return (T*)ncclGetMultimemPointer(window, offset, mmHandle);
}
#endif

#if NCCL_CHECK_CUDACC
template<typename T>
NCCL_DEVICE_INLINE T* ncclSymPtr<T>::lsaMultimemPtr(ncclDevComm const& comm) const {
  return (T*)ncclGetLsaMultimemPointer(window, offset, comm);
}
#endif

// 全局 operator-（带整数）：返回新的 SymPtr，window 不变，offset 向后移动 d*sizeof(T)
// 全局 operator+：返回新的 SymPtr，window 不变，offset 向前移动 d*sizeof(T)
template<typename T, typename Int>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T> operator+(ncclSymPtr<T> p, Int d) {
  return p += d;
}
template<typename T, typename Int>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T> operator-(ncclSymPtr<T> p, Int d) {
  return p -= d;
}
template<typename T>
NCCL_HOST_DEVICE_INLINE ptrdiff_t operator-(ncclSymPtr<T> a, ncclSymPtr<T> b) {
  return reinterpret_cast<T*>(a.offset) - reinterpret_cast<T*>(b.offset);
}

template<typename T>
NCCL_HOST_DEVICE_INLINE bool operator==(ncclSymPtr<T> a, ncclSymPtr<T> b) {
  return a.window == b.window && a.offset == b.offset;
}
template<typename T>
NCCL_HOST_DEVICE_INLINE bool operator!=(ncclSymPtr<T> a, ncclSymPtr<T> b) {
  return a.window != b.window || a.offset != b.offset;
}

#endif // __cplusplus
#endif // _NCCL_DEVICE_PTR__FUNCS_H_
