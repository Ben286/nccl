/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_COOP_H_
#define _NCCL_DEVICE_COOP_H_
#include "utility.h"

// ncclCoop[Foo]: NCCL's versions of CUDA's Cooperative Groups. They conform
// to just this subset of the CUDA API:
//   int Coop::thread_rank();
//   int Coop::size();
//   int Coop::num_threads(); // same as size()
//   void Coop::sync();

// ===== ncclCoopBcast：Coop 内数据广播 =====
// 将 root 线程的 value 广播给 coop 内所有线程
// entrySync：是否在广播前先同步（默认 true）
//
// ncclCoopTile<N> 版：用 __shfl_sync 按 4 字节一组广播（导和写入任意类型）
//   union 技巧：将 T 分解为 uint32_t 数组，使用 __shfl_sync 翻用所有 32位寄存器广播
//   root 参数是 tile 内的序号（即 lane%N）
//
// ncclCoopLanes 版：支持非连续线程组，root 用位操作计算实际 lane 号
//   root==0 表示 lmask 内第 0 个线程（__ffs(m)-1）
//   root==k 表示 lmask 内第 k 个线程（__fns(m, 0, 1+root)）
//
// ncclCoopWarpSpan 版：需要跨 warp 广播，使用 shared memory 转存
//   stash[id] 是 16 字节的 smem slot，root 线程写入，其他线程读取
//   entrySync 确保进入广播前所有线程已到达
//
// ncclCoopCta 版：与 WarpSpan 类似，使用全局 smem slot
// ===== ncclCoopCoalesced：返回当前操作安全的线程子集 =====
// 无参数版：返回 __activemask() 中当前活跃的所有线程（适合发散控制流内）
// 带参数版 Coop：对 ncclCoopLanes 返回自身，对其他类型返回整个 warp（ncclCoopTile<32>）
// 用途：在 warp 内安全执行 __ballot_sync、__shfl_sync 等指令
// ===== ncclCoopWithinWarp：编译期判断 coop 是否局限于单个 warp 内部 =====
// 返回 true：ncclCoopTile<N>、ncclCoopLanes（均在 warp 内）
// 返回 false：ncclCoopWarpSpan、ncclCoopCta（跨 warp）
// 用途：判断是否可以用 warp 级指令（__shfl_sync、__ballot_sync 等）或需要跨 warp 的 barrier
// ===== ncclCoopIsThread：编译期判断 coop 是否为单线程 =====
// 返回 true：ncclCoopTile<1>（即 ncclCoopThread）
// 返回 false：所有其他类型
// 用途：允许编译器针对单线程内联展开不同分支（配合 NCCL_IF_CONSTEXPR 使用）
// ===== ncclCoopGetLaneMask：获取 coop 对应的 lane mask =====
// ncclCoopTile<N>：调用 coop.laneMask()
// ncclCoopLanes：直接返回 lmask
// ncclCoopWarpSpan / ncclCoopCta：返回 0xFFFFFFFF（跨 warp，全 1 掩码）
// 用途：与 __ballot_sync、__shfl_sync 等 PTX 指令配合使用，确保正确的参与线程集
// ===== ncclCoopCta：整个 CTA（每个多线程验证最大线程组）=====
// thread_rank()：threadIdx.x
// sync()：__syncthreads()，同步整个 block 的所有线程
// 用途：需要所有 warp 协调的操作（如 LSA barrier init、window table 搜索）
// ===== ncclCoopWarpSpan：CTA 内连续多个 warp =====
// 结构：
//   warp0：起始 warp 编号（在 CTA 内）
//   nWarps：warp 数量
//   id：用户提供的唯一 ID 字段 [0..15]，供 barrier.sync 指令使用
// 为什么需要 id：
//   __barrier_sync_count(barrierID, threadCount)：PTX barrier 指令需要唯一的屏障 ID
//   id 对应 barrier slot，不同 WarpSpan 的 id 不能相同（否则一个 barrier 混淆两个 coop）
//   注意：id=0 保留给 CTA 级屏障，所以 id 取值 [1..15]（接口使用 1+id）
// thread_rank()：threadIdx.x - 32*warp0，线程在 span 内的编号
// sync()：__barrier_sync_count(1+id, 32*nWarps)，只同步此 span 内的 warp
// ===== ncclCoopLanes：warp 内根据位掩码选择的线程组 =====
// lmask：32 位中每一位对应一个 lane，1 表示该 lane 属于此 coop
// thread_rank()：返回当前 lane 在 lmask 有效位中的序号
//   算法：__popc(lmask & lanemask_lt())，即在 lmask 中比当前 lane 小的个数
// sync()：__syncwarp(lmask) — 只对 lmask 指定的线程同步
// 常规用途：用 __activemask() 构造 ncclCoopCoalesced()，确保当前活跃线程序列正确
// ncclCoopThread / ncclCoopWarp：最常用的两个别名
// ncclCoopThread：单线程（tile 大小=1），不需要同步，用于单线程操作（如 GFD 提交、counter 读取）
// ncclCoopWarp：整个 warp（tile 大小=32），最常用于 GIN、Reduce 等 warp 级操作
// ===== ncclCoopTile<N>：warp 内的对齐 N 线程组 =====
// 要求：N 必须是 2 的幂且 <= 32（即 warp 内分山）
// thread_rank()： lane() % N，必须确保线程的 lane 对齐到 N 的边界（对齐 tile）
// laneMask()：返回本 tile 内所有线程的 lane mask
//   算法：全 1 掩码叓 N 位，再左移到 tile 起始位置（lane & -N）
// sync()：__syncwarp(laneMask()) —只对 tile 内的线程同步，不影响 tile 外线程
// An aligned pow2 set of threads within the warp.
// ===== ncclCoopAny：运行时多态 Coop（带 vtable 的类型擦除）=====
// 用途：当需要将不同类型的 Coop（如 ncclCoopWarpSpan 和 ncclCoopCta）统一处理时
//         例如：一个接口再需要接受不同大小的 coop 组的场景
// 实现：
//   Storage：16 字节的对齐内联存储空间（足够容纳所有 Impl 类型）
//   VTable：  3 个函数指针（thread_rank、size、sync）
//   构造时通过 placement new 将 Impl 复制进 Storage
//   调用时通过 vtable 间接展开，相比直接调用有一定开销
// ============================================================================
// coop.h — NCCL Cooperative Group 抗象层
// ============================================================================
// 本文件定义 NCCL 自己的 Cooperative Group 类型，是 CUDA Cooperative Groups 的轻量化替代。
//
// 设计动机：
//   标准 CUDA Cooperative Groups（cooperative_groups.h）功能小而库大，编译耗时长。
//   NCCL Device API 只需要 4 个接口：
//     thread_rank()：返回当前线程在 coop 内的编号 [0, size())
//     size()： coop 内总线程数
//     num_threads()：同 size()
//     sync()：组内屏障同步
//
// 类型体系：
//   ncclCoopAny        — 运行时多态派发（内部带 vtable），适用于不能静态确定类型的场景
//   ncclCoopTile<N>    — warp 内的对齐 N 线程写（N 必须是 2 的幂）
//     ncclCoopThread   — ncclCoopTile<1>，单个线程
//     ncclCoopWarp     — ncclCoopTile<32>，整个 warp
//   ncclCoopLanes      — warp 内按位掩码选取的某些线程（活跃线程集合）
//   ncclCoopWarpSpan   — CTA 内连续的多个 warp（需要用户提供唯一 id 用于 barrier）
//   ncclCoopCta        — 整个 CTA（所有线程）
//
// Coop 类型的选择准则：
//   单线程操作（如 GIN warp 的 thread_rank()==0）→ ncclCoopThread
//   warp 内部操作                                     → ncclCoopWarp/ncclCoopTile
//   stage 级别（数个 warp）                           → ncclCoopWarpSpan
//   全内核（所有 warp）                             → ncclCoopCta
// ============================================================================

#if NCCL_CHECK_CUDACC
struct ncclCoopAny {
  struct Storage { alignas(alignof(void*)) char space[16]; };
  struct VTable {
    int(*thread_rank)(void const*);
    int(*size)(void const*);
    void(*sync)(void*);
  };

  template<typename Impl>
  __device__ static int thread_rank(void const* o) {
    return static_cast<Impl const*>(o)->thread_rank();
  }
  template<typename Impl>
  __device__ static int size(void const* o) {
    return static_cast<Impl const*>(o)->size();
  }
  template<typename Impl>
  __device__ static void sync(void* o) {
    static_cast<Impl*>(o)->sync();
  }

  template<typename Impl>
  __device__ static VTable const* get_vtable() {
    static_assert(sizeof(Impl) <= sizeof(Storage), "Incompatible coop type size");
    static_assert(alignof(Impl) <= alignof(Storage), "Incompatible coop type alignment");
    static constexpr VTable v = {
      &thread_rank<Impl>,
      &size<Impl>,
      &sync<Impl>
    };
    return &v;
  }

  Storage storage;
  VTable const* vtable;

  ncclCoopAny(ncclCoopAny const&) = default;
  ncclCoopAny(ncclCoopAny&&) = default;
  ncclCoopAny() = default;

  template<typename Impl>
  __device__ ncclCoopAny(Impl impl) {
    ::new (&this->storage) Impl(impl);
    this->vtable = get_vtable<Impl>();
  }

  __device__ int thread_rank() const { return vtable->thread_rank(&storage); }
  __device__ int size() const { return vtable->size(&storage); }
  __device__ int num_threads() const { return vtable->size(&storage); }
  __device__ void sync() { vtable->sync(&storage); }
};
#endif

#if NCCL_CHECK_CUDACC
template<int nThreadsPow2>
struct ncclCoopTile { // An aligned pow2 set of threads within the warp.
  static_assert(nccl::utility::isPow2(nThreadsPow2) && nThreadsPow2 <= 32, "Condition required");

  NCCL_DEVICE_INLINE int thread_rank() const {
    return nccl::utility::lane() % nThreadsPow2;
  }
  NCCL_DEVICE_INLINE constexpr int size() const { return nThreadsPow2; }
  NCCL_DEVICE_INLINE constexpr int num_threads() const { return nThreadsPow2; }

  NCCL_DEVICE_INLINE uint32_t laneMask() const {
    return (-1u>>(32-nThreadsPow2))<<(nccl::utility::lane() & -nThreadsPow2);
  }
  NCCL_DEVICE_INLINE void sync() {
    if (nThreadsPow2 > 1) __syncwarp(laneMask());
  }
};
#endif

#if NCCL_CHECK_CUDACC
typedef ncclCoopTile<1> ncclCoopThread;
typedef ncclCoopTile<32> ncclCoopWarp;
#endif

#if NCCL_CHECK_CUDACC
struct ncclCoopLanes { // Some lanes of this warp.
  uint32_t lmask;

  NCCL_DEVICE_INLINE constexpr ncclCoopLanes(uint32_t lmask=~0u): lmask(lmask) {}

  NCCL_DEVICE_INLINE int thread_rank() const {
    return __popc(lmask & nccl::utility::lanemask_lt());
  }
  NCCL_DEVICE_INLINE int size() const {
    return __popc(lmask);
  }
  NCCL_DEVICE_INLINE int num_threads() const {
    return __popc(lmask);
  }
  NCCL_DEVICE_INLINE void sync() {
    __syncwarp(lmask);
  }
};
#endif

#if NCCL_CHECK_CUDACC
// A set of consecutive warps that the user has also supplied with a unique
// id from [0..15]. It is an error for two different warp spans with the same
// id to be in a collective concurrently.
struct ncclCoopWarpSpan {
  uint32_t warp0:8, nWarps:8, id:8;

  NCCL_DEVICE_INLINE constexpr ncclCoopWarpSpan(int warp0, int nWarps, int id):
    warp0(warp0), nWarps(nWarps), id(id) {
  }

  NCCL_DEVICE_INLINE int thread_rank() const {
    return threadIdx.x - 32*warp0;
  }
  NCCL_DEVICE_INLINE int size() const {
    return 32*nWarps;
  }
  NCCL_DEVICE_INLINE int num_threads() const {
    return 32*nWarps;
  }

  NCCL_DEVICE_INLINE void sync() {
    //asm volatile("barrier.sync %0, %1;" :: "r"(1+id), "r"(32*nWarps) : "memory");
    __barrier_sync_count(1+id, 32*nWarps);
  }
};
#endif

#if NCCL_CHECK_CUDACC
struct ncclCoopCta {
  NCCL_DEVICE_INLINE int thread_rank() const { return threadIdx.x; }
  NCCL_DEVICE_INLINE int size() const { return blockDim.x; }
  NCCL_DEVICE_INLINE int num_threads() const { return blockDim.x; }
  NCCL_DEVICE_INLINE void sync() { __syncthreads(); }
};
#endif

#if NCCL_CHECK_CUDACC
template<int nThreadsPow2>
NCCL_DEVICE_INLINE uint32_t ncclCoopGetLaneMask(ncclCoopTile<nThreadsPow2> coop) {
  return coop.laneMask();
}
NCCL_DEVICE_INLINE uint32_t ncclCoopGetLaneMask(ncclCoopLanes coop) {
  return coop.lmask;
}
NCCL_DEVICE_INLINE uint32_t ncclCoopGetLaneMask(ncclCoopWarpSpan coop) {
  return -1u;
}
NCCL_DEVICE_INLINE uint32_t ncclCoopGetLaneMask(ncclCoopCta coop) {
  return -1u;
}
#endif

#if NCCL_CHECK_CUDACC
// ncclCoopIsThread:
// At compile time do we know the given coop is a single thread only.
template<int nThreads>
NCCL_DEVICE_INLINE constexpr bool ncclCoopIsThread(ncclCoopTile<nThreads>) {
  return nThreads == 1;
}
NCCL_DEVICE_INLINE constexpr bool ncclCoopIsThread(ncclCoopLanes) { return false; }
NCCL_DEVICE_INLINE constexpr bool ncclCoopIsThread(ncclCoopWarpSpan) { return false; }
NCCL_DEVICE_INLINE constexpr bool ncclCoopIsThread(ncclCoopCta) { return false; }
#endif

#if NCCL_CHECK_CUDACC
template<int nThreads>
NCCL_DEVICE_INLINE constexpr bool ncclCoopWithinWarp(ncclCoopTile<nThreads>) { return true; }
NCCL_DEVICE_INLINE constexpr bool ncclCoopWithinWarp(ncclCoopLanes) { return true; }
NCCL_DEVICE_INLINE constexpr bool ncclCoopWithinWarp(ncclCoopWarpSpan) { return false; }
NCCL_DEVICE_INLINE constexpr bool ncclCoopWithinWarp(ncclCoopCta) { return false; }
#endif

#if NCCL_CHECK_CUDACC
// Pick threads of our warp that are safe to use collectively.
NCCL_DEVICE_INLINE ncclCoopLanes ncclCoopCoalesced() {
  return ncclCoopLanes{__activemask()};  // 返回当前活跃线程（处于相同控制流分支的线程）
}
#endif

#if NCCL_CHECK_CUDACC
// Pick threads of our warp that are safe to use collectively given that this
// is a collective on the provided cooperative group.
template<typename Coop>
NCCL_DEVICE_INLINE ncclCoopTile<32> ncclCoopCoalesced(Coop) {
  return ncclCoopTile<32>();  // 对于跨 warp coop，返回整个 warp
}
NCCL_DEVICE_INLINE ncclCoopLanes ncclCoopCoalesced(ncclCoopLanes coop) {
  return coop;  // Tile 自身就是对齐的安全子集
}
template<int nThreads>
NCCL_DEVICE_INLINE ncclCoopTile<nThreads> ncclCoopCoalesced(ncclCoopTile<nThreads> coop) {
  return coop;
}
#endif

#if NCCL_CHECK_CUDACC
template<int nThreads, typename T>
NCCL_DEVICE_INLINE T ncclCoopBcast(ncclCoopTile<nThreads>, T value, int root, bool entrySync=true) {
  constexpr int n = (sizeof(T)+4-1)/4;
  union { uint32_t u[n]; T v; };
  v = value;
  #pragma unroll
  for (int i=0; i < n; i++) u[i] = __shfl_sync(-1u, u[i], root, nThreads);
  return v;
}
template<typename T>
NCCL_DEVICE_INLINE T ncclCoopBcast(ncclCoopLanes coop, T value, int root, bool entrySync=true) {
  uint32_t m = coop.lmask;
  uint32_t r = root == 0 ? __ffs(m)-1 : __fns(m, 0, 1+root);
  constexpr int n = (sizeof(T)+4-1)/4;
  union { uint32_t u[n]; T v; };
  v = value;
  #pragma unroll
  for (int i=0; i < n; i++) u[i] = __shfl_sync(m, u[i], r);
  return v;
}

NCCL_DEVICE_INLINE ulong2* ncclCoopBcast_WarpSpan_stash() {
  __shared__ ulong2 stash[15];
  return stash;
}

template<typename T>
NCCL_DEVICE_INLINE T ncclCoopBcast(ncclCoopWarpSpan coop, T value, int root, bool entrySync=true) {
  static_assert(sizeof(T) <= sizeof(ncclCoopBcast_WarpSpan_stash()[0]), "Required");
  if (entrySync) coop.sync();
  if (coop.thread_rank() == root) *(T*)&ncclCoopBcast_WarpSpan_stash()[coop.id] = value;
  coop.sync();
  return *(T*)&ncclCoopBcast_WarpSpan_stash()[coop.id];
}

NCCL_DEVICE_INLINE ulong2* ncclCoopBcast_Cta_stash() {
  __shared__ ulong2 stash;
  return &stash;
}

template<typename T>
NCCL_DEVICE_INLINE T ncclCoopBcast(ncclCoopCta coop, T value, int root, bool entrySync=true) {
  static_assert(sizeof(T) <= sizeof(*ncclCoopBcast_Cta_stash()), "Required");
  if (entrySync) coop.sync();
  if (coop.thread_rank() == root) *(T*)ncclCoopBcast_Cta_stash() = value;
  coop.sync();
  return *(T*)ncclCoopBcast_Cta_stash();
}
#endif

#endif
