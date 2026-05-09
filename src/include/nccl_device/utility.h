/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_UTILITY_H_
#define _NCCL_DEVICE_UTILITY_H_

// compiler specific check for __CUDACC__
// ===== 编译器环境检测 =====
// NCCL_CHECK_CUDACC：判断当前是否在 CUDA 编译器（nvcc/clang-cuda）下编译
// 用途：用 #if NCCL_CHECK_CUDACC 包裹只能在 GPU 上编译的代码
// clang 的判断方式与 nvcc 不同（clang 用 #ifdef __CUDACC__，nvcc 用 #if __CUDACC__）
#ifndef NCCL_CHECK_CUDACC
    #if defined(__clang__)
        #ifdef __CUDACC__
            #define NCCL_CHECK_CUDACC 1
        #else
            #define NCCL_CHECK_CUDACC 0
        #endif
    #else
        #if __CUDACC__
            #define NCCL_CHECK_CUDACC 1
        #else
            #define NCCL_CHECK_CUDACC 0
        #endif
    #endif
#endif

// ===== inline 修饰完考虚 =====
// NCCL_DEVICE_INLINE：展开为 __device__ __forceinline__（nvcc）或 __device__ always_inline（clang/hostlib）
//   __forceinline__ 强制内联，避免 GPU 内核中的函数调用开销（无寄存器保存、无转跟需求）
// NCCL_HOST_DEVICE_INLINE：添加 __host__ 之后可在 CPU 和 GPU 上并用
// NCCL_HOSTLIB_ONLY：特殊模式，构建 Host-only 库时使用 always_inline
// __clang_llvm_bitcode_lib__：构建 LLVM bitcode 库时使用 always_inline
#if NCCL_CHECK_CUDACC
  #if defined(NCCL_HOSTLIB_ONLY) || defined(__clang_llvm_bitcode_lib__)
    #define NCCL_DEVICE_INLINE __device__ __attribute__((always_inline))
    #define NCCL_HOST_DEVICE_INLINE __host__ __device__ __attribute__((always_inline))
  #else
    #define NCCL_DEVICE_INLINE __device__ __forceinline__
    #define NCCL_HOST_DEVICE_INLINE __host__ __device__ __forceinline__
  #endif
#else
  #ifndef __host__
    #define __host__  // CPU-only 环境下 __host__ 为空宏
  #endif
  #define NCCL_DEVICE_INLINE  // CPU-only 环境下 没有 __device__ 修饰
  #if defined(NCCL_OS_WINDOWS)
    #define NCCL_HOST_DEVICE_INLINE __forceinline
  #else
    #define NCCL_HOST_DEVICE_INLINE inline __attribute__((always_inline))
  #endif
#endif

// Macro for conditional constexpr support
// NCCL_IF_CONSTEXPR：展开为 constexpr（C++17 以上）或空（C++14 及以下）
// 用途：允许在编译期 if constexpr 分支化，限制檢查居不成立的分支不被编译
#if defined(__cpp_if_constexpr) && __cpp_if_constexpr >= 201606
  #ifndef NCCL_IF_CONSTEXPR
    #define NCCL_IF_CONSTEXPR constexpr
  #endif
#else
  #ifndef NCCL_IF_CONSTEXPR
    #define NCCL_IF_CONSTEXPR  // C++17 以下的平台上：退化为普通 if
  #endif
#endif

// NCCL_EXTERN_C：确保 C++ 代码里的公共 API 用 C 链接符号（避免 name mangling）
// 使得 NCCL Device API 可以被纯 C 代码调用
#if __cplusplus
#define NCCL_EXTERN_C extern "C"
#else
#define NCCL_EXTERN_C
#endif

// NCCL_IR_EXTERN_C：仅在构建 LLVM IR 库（clang bitcode）时起效，其余情况为空
// 用途：确保 IR 库内的函数符号不被 C++ mangling
#ifdef __clang_llvm_bitcode_lib__
#define NCCL_IR_EXTERN_C extern "C"
#else
#define NCCL_IR_EXTERN_C
#endif

#include <stdint.h>
#include <stdbool.h>

#if defined(NCCL_OS_WINDOWS)
#include <intrin.h>
#endif

// ===== testAbort：spin-wait 轮询时的故障检测函数 =====
// 调用者：每次 spin 循环都需传入 steps（当前转数）和 abortFlag
// 返回值：true 表示应中断轮询（礼豆2次响应故障，避免死等）
// 工作原理：
//   - 常规转数 ++steps < maxSteps：直接返回 false（不检测，高性能）
//   - 每转了 maxSteps（10000）次才检查一次 abortFlag
//   - 使用 cuda::atomic_ref（GPU）或 volatile（CPU）读取，确保能看到其他线程/CPU 的写入
#if NCCL_CHECK_CUDACC
#include <cuda/atomic>
#endif

#if __cplusplus
namespace nccl {
namespace utility {

#if NCCL_CHECK_CUDACC
// cuda/atomic header file is included so we can use atomic_ref to load the abortFlag
static NCCL_DEVICE_INLINE bool testAbort(uint32_t* abortFlag, uint32_t& steps) {
  const uint32_t maxSteps = 10000;
  if (++steps < maxSteps) {
    return false;
  } else {  // alignof(T) >= 16时用 ulonglong2（128 位）单次读取
    steps = 0;
    return abortFlag != nullptr && cuda::atomic_ref<uint32_t>{*abortFlag}.load(cuda::memory_order_relaxed) != 0;
  }
}
#else
static NCCL_DEVICE_INLINE bool testAbort(uint32_t* abortFlag, uint32_t& steps) {
  const uint32_t maxSteps = 10000;
  if (++steps < maxSteps) {
    return false;
  } else {
    volatile uint32_t *ptr = (volatile uint32_t*)abortFlag;  // volatile 防止编译器缓存读取
    steps = 0;
    return ptr != nullptr && *ptr != 0;
  }
}
#endif

// declval<T>: 类似 std::declval，不实际构造 T，但可用于 decltype() 推导返回类型
// 用途：在不调用构造函数的情况下推导表达式类型（如 lambda 返回类型）
// 注意：static_assert 确保该函数永远不会被实际调用
template<typename T>
T&& declval() noexcept {
  static_assert(sizeof(T)!=sizeof(T), "You can't evaluate declval.");
}

// always_false<T>：始终为 false 的静态断言，用于 static_assert 中让编译器指向具体类型的错误信息
// 例如：static_assert(always_false<T>::value, "Unsupported type");
template <typename>
struct always_false {
  static constexpr bool value = false;
};

// ValueAsType<T, value>：将编译期常量封装为类型，用于模板元编程传递常量
// 类似 std::integral_constant
template<typename T, T value_>
struct ValueAsType { static constexpr T value = value_; };

// Returns the value zero but the compiler cannot prove that it is zero so it
// is useful to inhibit compiler optimizations.
// opaqueZero()：返回值为 0ï¼但编译器无法证明它一定是 0ï¼因此可商馋编译器优化
// 常规用途：在深度内联中，插入一个 "opaque" 值阻止过度内联展开
// 确保 static int zero 存在于全局内存，__ldg 绵过 L1 cache 直接读取 (texture cache 路径)
#if NCCL_CHECK_CUDACC
template<typename=void>
NCCL_DEVICE_INLINE int opaqueZero() {
  __device__ static int zero = 0;
  return __ldg(&zero);  // __ldg: 使用只读纹理 cache 读取，不进入 L1 cache
}
#endif

// alignUp(x, a)：将 x 向上对齐到 a 字节边界，a 必须是 2 的幂
// 算法：(x + a-1) & ~(a-1)，使用位操作快速对齐（只适用于 2 的幂）
// roundDown(x, y)：将 x 向下圆整到 y 的倍数
// roundUp(x, y)：将 x 向上圆整到 y 的倍数（y 可以不是 2 的幂）
// divUp(x, y)：向上取整除法，等价于 ceil(x/y)
// 算法：(x+y-1)/y，利用整数除圆滚把余数不为 0 的情况向上约
template<typename X, typename Y, typename Z = decltype(X()+Y())>
NCCL_HOST_DEVICE_INLINE constexpr Z divUp(X x, Y y) {
  return (x+y-1)/y;
}

template<typename X, typename Y, typename Z = decltype(X()+Y())>
NCCL_HOST_DEVICE_INLINE constexpr Z roundUp(X x, Y y) {
  return (x+y-1) - (x+y-1)%y;
}
template<typename X, typename Y, typename Z = decltype(X()+Y())>
NCCL_HOST_DEVICE_INLINE constexpr Z roundDown(X x, Y y) {
  return x - x%y;
}

// assumes second argument is a power of 2
template<typename X, typename Y, typename Z = decltype(X()+Y())>
NCCL_HOST_DEVICE_INLINE constexpr Z alignUp(X x, Y a) {
  return (x + a-1) & -Z(a);  // -Z(a) 二进制补码 = ~(a-1)，也就是高位全 1 的掩码
}
// add4G(base, delta4G)：修改 64 位指针的高 32 位（高 4G 部分）
// 用途：Symmetric Memory 中，各 GPU 的 buffer 在物理地址上互相相差一个固定偏移（stride4G 个 4GB）
// 因此只需修改 64 位地址的高 32 位（代表 4GB 单位的偏移），低 32 位保持不变
// 示例：base=0x1_0000_0000, delta4G=1 → 返回 0x2_0000_0000
// 指针版本：只适用于 sizeof(T)==1 （char*）的指针对齐
template<typename T>
NCCL_HOST_DEVICE_INLINE T* alignUp(T* x, size_t a) {
  static_assert(sizeof(T) == 1, "Only single byte types allowed.");
  return reinterpret_cast<T*>((reinterpret_cast<uintptr_t>(x) + a-1) & -uintptr_t(a));
}
template<typename T>
NCCL_HOST_DEVICE_INLINE void* alignUp(void const* x, size_t a) {
  return reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(x) + a-1) & -uintptr_t(a));
}

// assumes second argument is a power of 2
// alignDown(x, a)：将 x 向下对齐到 a 字节边界，a 必须是 2 的幂
// 算法：x & ~(a-1) = x & -a
template<typename X, typename Y, typename Z = decltype(X()+int())>
NCCL_HOST_DEVICE_INLINE constexpr Z alignDown(X x, Y a) {
  return x & -Z(a);
}
template<typename T>
NCCL_HOST_DEVICE_INLINE T* alignDown(T* x, size_t a) {
  static_assert(sizeof(T) == 1, "Only single byte types allowed.");
  return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(x) & -uintptr_t(a));
}
template<typename T>
NCCL_HOST_DEVICE_INLINE void* alignDown(void const* x, size_t a) {
  return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(x) & -uintptr_t(a));
}

template<typename T>
NCCL_HOST_DEVICE_INLINE T add4G(T base, int delta4G) {
  union { uint32_t u32[2]; T tmp; };  // 将 T（指针/uint64）转为两个 uint32 分离操作
  tmp = base;
  u32[1] += delta4G;  // 修改高 32 位（等价于加上 delta4G * 2^32）
  return tmp;
}


// isPow2(x)：判断 x 是否是 2 的幂
// 利用性质：2 的幂 n 的二进制表示为 1000...0，n-1 的表示为 0111...1，二者与 = 0
template<typename Int>
NCCL_HOST_DEVICE_INLINE constexpr bool isPow2(Int x) {
  return (x & (x-1)) == 0;
}

// rollingLessThan(a, b)：a < b（回卷语义）。等价于 !rollingLessEq(b, a)
// ===== Rolling 比较函数 =====
// rollingLessEq(a, b, nBits)：判断 a <= b，其中 a、b 是洋 nBits 比特的回卷计数器值
// 用途：signal/counter 等计数器会回卷（超过 nBits 最大値后回到 0）时的大小比较
// 算法：(b-a) & m 就是从 a 到 b 的正向距离，如果距离 <= m/2，则 a <= b
// 具体原理：对于 nBits 位无符号整数，[a, a+m/2] 区间内的所有 b 都认为 a<=b
// 警告：当 b-a > m/2 时（距离过大），判断可能不准确
template<typename Uint>
NCCL_HOST_DEVICE_INLINE bool rollingLessEq(Uint a, Uint b, int nBits = 8*sizeof(Uint)) {
  static_assert(Uint(0) < Uint(-1), "Uint must be unsigned.");
  Uint m = Uint(-1) >> (8*sizeof(Uint) - nBits);  // 生成 nBits 位就的全 1 掩码
  return ((b-a) & m) <= m>>1;  // 正向距离 <= 半圆则认为 a <= b
}
template<typename Uint>
NCCL_HOST_DEVICE_INLINE bool rollingLessThan(Uint a, Uint b, int nBits = 8*sizeof(Uint)) {
  return !rollingLessEq(b, a, nBits);
}

// Produce the reciprocal of x for use in idivByRcp
// ===== 快速整数除法系统 =====
// 背景：GPU 上的整数除法指令耗时较长（透情延迟可达 20+ 循环）
// 解决方案：预先计算除数的倒数 rcp，然后用乘高位（mulhi）替代除法
// 应用场景：ncclTeam 的 rank/nRanks 计算（如 ncclTeamRail 中 lsaRank = rank / lsaSize）

// idivRcp32(x)：计算 x 的 32 位倒数 rcp，使得 mulhi(n, rcp) 近似于 n/x
// 算法：rcp = floor(2^32 / x) 加 1（如果 x 是 2 的幂）
NCCL_HOST_DEVICE_INLINE constexpr uint32_t idivRcp32(uint32_t x) {
  return uint32_t(-1)/x + isPow2(x);
}
// idivRcp64：64 位版本
NCCL_HOST_DEVICE_INLINE constexpr uint64_t idivRcp64(uint64_t x) {
  return uint64_t(-1)/x + isPow2(x);
}

// mul32hi(a, b)：返回 64位乘积 a*b 的高 32 位
// GPU 上使用 __umulhi 指令（单循环）；CPU 上用 uint64 转换模拟
NCCL_HOST_DEVICE_INLINE uint32_t mul32hi(uint32_t a, uint32_t b) {
#if __CUDA_ARCH__
  return __umulhi(a, b);  // PTX 特殊指令，等价于 (a*b) >> 32
#else
  return uint64_t(a)*b >> 32;
#endif
}
// mul64hi：64 位版本
NCCL_HOST_DEVICE_INLINE uint64_t mul64hi(uint64_t a, uint64_t b) {
#if __CUDA_ARCH__
  return __umul64hi(a, b);  // PTX 指令处理 128 位算术
#elif defined(NCCL_OS_WINDOWS)
  // Use MSVC intrinsic for 64-bit multiplication with high part result
  return __umulh(a, b);
#else
  return (uint64_t)(((unsigned __int128)a)*b >> 64);  // CPU 用 128位整数
#endif
}

// Produce the reciprocal of x*y given their respective reciprocals. This incurs
// no integer division on device.
// imulRcp32(x, xrcp, y, yrcp)：由 x、y 各自的倒数推导 x*y 的倒数
// 不需要实际执行除法指令，张目合编器内联展开
NCCL_HOST_DEVICE_INLINE uint32_t imulRcp32(uint32_t x, uint32_t xrcp, uint32_t y, uint32_t yrcp) {
  if (xrcp == 0) return yrcp;  // x==1 时，rcp(x*y) = rcp(y)
  if (yrcp == 0) return xrcp;  // y==1 时，rcp(x*y) = rcp(x)
  uint32_t rcp = mul32hi(xrcp, yrcp);  // rcp(x*y) 近似为 mulhi(rcp_x, rcp_y)
  uint32_t rem = 0u - x*y*rcp;
  if (x*y <= rem) rcp += 1;
  return rcp;
}
NCCL_HOST_DEVICE_INLINE uint64_t imulRcp64(uint64_t x, uint64_t xrcp, uint64_t y, uint64_t yrcp) {
  if (xrcp == 0) return yrcp;
  if (yrcp == 0) return xrcp;
  uint64_t rcp = mul64hi(xrcp, yrcp);
  uint64_t rem = 0ULL - x*y*rcp;
  if (x*y <= rem) rcp += 1;
  return rcp;
}

// Fast unsigned integer division where divisor has precomputed reciprocal.
// idivFast(x, y, idivRcp(y)) == x/y
// idivmodFast32(quo, rem, x, y, yrcp)：同时返回商和余数（快速除法）
// 算法：1. q = mulhi(x, yrcp) 得到近似商
//         2. r = x - y*q 得到近似余
//         3. 修正：如果 r >= y，则 q+=1, r-=y（最多修正一次）
NCCL_HOST_DEVICE_INLINE void idivmodFast32(uint32_t *quo, uint32_t *rem, uint32_t x, uint32_t y, uint32_t yrcp) {
  uint32_t q = yrcp == 0 ? x : mul32hi(x, yrcp);  // yrcp==0 表示 y==1，x/1=x
  uint32_t r = x - y*q;
  if (r >= y) { q += 1; r -= y; }  // 最多修正一次
  *quo = q;
  *rem = r;
}
NCCL_HOST_DEVICE_INLINE void idivmodFast64(uint64_t *quo, uint64_t *rem, uint64_t x, uint64_t y, uint64_t yrcp) {
  uint64_t q = yrcp == 0 ? x : mul64hi(x, yrcp);
  uint64_t r = x - y*q;
  if (r >= y) { q += 1; r -= y; }
  *quo = q;
  *rem = r;
}

// idivFast32/64：只返回商的快速除法
NCCL_HOST_DEVICE_INLINE uint32_t idivFast32(uint32_t x, uint32_t y, uint32_t yrcp) {
  uint32_t q, r;
  idivmodFast32(&q, &r, x, y, yrcp);
  return q;
}
NCCL_HOST_DEVICE_INLINE uint32_t idivFast64(uint64_t x, uint64_t y, uint64_t yrcp) {
  uint64_t q, r;
  idivmodFast64(&q, &r, x, y, yrcp);
  return (uint32_t)q;
}

// imodFast32/64：只返回余数的快速除法
NCCL_HOST_DEVICE_INLINE uint32_t imodFast32(uint32_t x, uint32_t y, uint32_t yrcp) {
  uint32_t q, r;
  idivmodFast32(&q, &r, x, y, yrcp);
  return r;
}
NCCL_HOST_DEVICE_INLINE uint64_t imodFast64(uint64_t x, uint64_t y, uint64_t yrcp) {
  uint64_t q, r;
  idivmodFast64(&q, &r, x, y, yrcp);
  return r;
}

// ===== GPU 特殊寄存器读取 =====
// lane()：返回当前线程在其 warp 内的编号（0~31）
// 确保使用 PTX 汇编直接读取 laneid 寄存器，避免内联展开后编译器约简丢失此信息
// ===== 原子操作封装：atomicLoad / atomicStore =====
// 问题：cuda::atomic_ref 需要在编译期确定 thread_scope，不能运行时传参
// 解决：switch-case 将运行时 scope 转为编译期分支，每个分支都是编译期确定的 atomic_ref
//
// thread_scope 层次（从小到大）：
//   thread_scope_thread  — 仅限单线程，最弱局性性、最小开销
//   thread_scope_block   — 同一 CTA 内的线程
//   thread_scope_device  — 同一 GPU 内的所有线程
//   thread_scope_system  — 跨 GPU（NVLink 或 PCIe）的全系统，开销最大
// scope 越小就能用越弱的同步指令，性能越好
// ===== Memory Order 辅助函数 =====
// acquireOrderOf / releaseOrderOf：从同步内存序中提取安全的 acquire/release 分量
// 用途：对 arrive/wait 分离写入（release）和读取（acquire）的内存序要求
//
// acquireOrderOf 规则：
//   release → relaxed（release only 时，读方不需要 acquire）
//   acq_rel → acquire（将双向内存序操作分为 acquire 一侧）
//   其他  → 不变
// releaseOrderOf 规则对称
// ===== 预计算倒数表 =====
// idivRcp64_upto64(x)：返回分母 1..64 内的预先计算倒数
// 用途：参数（如 lsaSize、rail.nRanks）在运行时确定但在内核内不变，写入 ncclDevComm 预先计算好
// GPU 读取表项和调用 idivFast64 就能避免性能消耗大的除法指令
#if NCCL_CHECK_CUDACC
// Precomputed integer reciprocoals for denominator values 1..64 inclusive.
// Pass these to idivFast64() for fast division on the GPU.
NCCL_DEVICE_INLINE uint64_t idivRcp64_upto64(int x) {
  static constexpr uint64_t table[65] = {
    idivRcp64(0x01), idivRcp64(0x01), idivRcp64(0x02), idivRcp64(0x03),
    idivRcp64(0x04), idivRcp64(0x05), idivRcp64(0x06), idivRcp64(0x07),
    idivRcp64(0x08), idivRcp64(0x09), idivRcp64(0x0a), idivRcp64(0x0b),
    idivRcp64(0x0c), idivRcp64(0x0d), idivRcp64(0x0e), idivRcp64(0x0f),
    idivRcp64(0x10), idivRcp64(0x11), idivRcp64(0x12), idivRcp64(0x13),
    idivRcp64(0x14), idivRcp64(0x15), idivRcp64(0x16), idivRcp64(0x17),
    idivRcp64(0x18), idivRcp64(0x19), idivRcp64(0x1a), idivRcp64(0x1b),
    idivRcp64(0x1c), idivRcp64(0x1d), idivRcp64(0x1e), idivRcp64(0x1f),
    idivRcp64(0x20), idivRcp64(0x21), idivRcp64(0x22), idivRcp64(0x23),
    idivRcp64(0x24), idivRcp64(0x25), idivRcp64(0x26), idivRcp64(0x27),
    idivRcp64(0x28), idivRcp64(0x29), idivRcp64(0x2a), idivRcp64(0x2b),
    idivRcp64(0x2c), idivRcp64(0x2d), idivRcp64(0x2e), idivRcp64(0x2f),
    idivRcp64(0x30), idivRcp64(0x31), idivRcp64(0x32), idivRcp64(0x33),
    idivRcp64(0x34), idivRcp64(0x35), idivRcp64(0x36), idivRcp64(0x37),
    idivRcp64(0x38), idivRcp64(0x39), idivRcp64(0x3a), idivRcp64(0x3b),
    idivRcp64(0x3c), idivRcp64(0x3d), idivRcp64(0x3e), idivRcp64(0x3f),
    idivRcp64(0x40)
  };
  return table[x];
}
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE uint32_t idivRcp32_upto64(int x) {
  return idivRcp64_upto64(x)>>32;
}
#endif

#if NCCL_CHECK_CUDACC
// wait
NCCL_DEVICE_INLINE cuda::memory_order acquireOrderOf(cuda::memory_order ord) {
  return ord == cuda::memory_order_release ? cuda::memory_order_relaxed :
         ord == cuda::memory_order_acq_rel ? cuda::memory_order_acquire :
         ord;
}

// arrive
NCCL_DEVICE_INLINE cuda::memory_order releaseOrderOf(cuda::memory_order ord) {
  return ord == cuda::memory_order_acquire ? cuda::memory_order_relaxed :
         ord == cuda::memory_order_acq_rel ? cuda::memory_order_release :
         ord;
}
#endif

#if NCCL_CHECK_CUDACC
template<typename T>
NCCL_DEVICE_INLINE T atomicLoad(T* ptr, cuda::memory_order ord, cuda::thread_scope scope) {
  switch (scope) {
  case cuda::thread_scope_thread:
    return cuda::atomic_ref<T, cuda::thread_scope_thread>{*ptr}.load(ord);
  case cuda::thread_scope_block:
    return cuda::atomic_ref<T, cuda::thread_scope_block>{*ptr}.load(ord);
  case cuda::thread_scope_device:
    return cuda::atomic_ref<T, cuda::thread_scope_device>{*ptr}.load(ord);
  case cuda::thread_scope_system:
    return cuda::atomic_ref<T, cuda::thread_scope_system>{*ptr}.load(ord);  // 跨 GPU 内存访问
  default: __builtin_unreachable();
  }
}
#endif

#if NCCL_CHECK_CUDACC
template<typename T>
NCCL_DEVICE_INLINE void atomicStore(T* ptr, T val, cuda::memory_order ord, cuda::thread_scope scope) {
  switch (scope) {
  case cuda::thread_scope_thread:
    cuda::atomic_ref<T, cuda::thread_scope_thread>{*ptr}.store(val, ord);
    break;
  case cuda::thread_scope_block:
    cuda::atomic_ref<T, cuda::thread_scope_block>{*ptr}.store(val, ord);
    break;
  case cuda::thread_scope_device:
    cuda::atomic_ref<T, cuda::thread_scope_device>{*ptr}.store(val, ord);
    break;
  case cuda::thread_scope_system:
    cuda::atomic_ref<T, cuda::thread_scope_system>{*ptr}.store(val, ord);  // 跨 GPU 内存写入
    break;
  default: __builtin_unreachable();
  }
}
#endif

// ===== loadConst：常量内存读取优化 =====
// Load anything, but cache like its constant memory.
// 用途：小结构体（如 ncclWindow_vidmem、ncclDevComm 内的字段）在内核常量起始后不会变化。
// __ldg 告知编译器这个地址指向的数据在 kernel 期间不会被写，帮我走只读缓存优化路径来读
// 当多个 warp 读取同一地址时，__ldg 可能合并成一次 broadcast，进一步提升效率
// 根据 alignof(T) 选择位宽（byte/short/int/long/ulonglong2）以实现向量化读取
#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE int lane() {
  int ret;
  asm("mov.u32 %0, %%laneid;" : "=r"(ret));  // 直接读取 PTX 内建 laneid 寄存器
  return ret;
}
// lanemask_lt()：返回一个位掩码，其中第 0..lane()-1 位为 1
// 用途：计算当前 lane 在 warp 中的偏移（thread_rank）或加前缀和
NCCL_DEVICE_INLINE unsigned int lanemask_lt() {
  unsigned int ret;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(ret));  // PTX 内建 lanemask_lt 寄存器
  return ret;
}
#endif

#if NCCL_CHECK_CUDACC
// Load anything, but cache like its constant memory.
template<typename T>
NCCL_DEVICE_INLINE T loadConst(T const *p) {
  if (alignof(T) == 1) {
    union { uint8_t part[sizeof(T)]; T ret; };
    for (int i=0; i < (int)sizeof(T); i++) part[i] = __ldg((uint8_t const*)p + i);
    return ret;
  } else if (alignof(T) == 2) {
    union { uint16_t part[sizeof(T)/2]; T ret; };
    for (int i=0; i < (int)sizeof(T)/2; i++) part[i] = __ldg((uint16_t const*)p + i);
    return ret;
  } else if (alignof(T) == 4) {
    union { uint32_t part[sizeof(T)/4]; T ret; };
    for (int i=0; i < (int)sizeof(T)/4; i++) part[i] = __ldg((uint32_t const*)p + i);
    return ret;
  } else if (alignof(T) == 8) {
    union { uint64_t part[sizeof(T)/8]; T ret; };
    for (int i=0; i < (int)sizeof(T)/8; i++) part[i] = __ldg((uint64_t const*)p + i);
    return ret;
  } else { // alignof(T) >= 16
    union { ulonglong2 part[sizeof(T)/16]; T ret; };
    for (int i=0; i < (int)sizeof(T)/16; i++) part[i] = __ldg((ulonglong2 const*)p + i);
    return ret;
  }
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Optional<T>: Holds a T that may or may not be constructed. An Optional
// constructed with a Present<Arg...> will have its T constructed via the
// T::T(Arg...) constructor. An Optional constructed with a Absent will not
// have its T constructed.

// IntSeq<vals...>：整数序列类型，用于属标 (index_sequence) 模式索引 Present 的元素
// ===== Optional<T> 模板工具 =====
// Optional<T>：可选性持有一个 T 类型对象，势力 T 可能不被构造
// 背景：GPU 上皘1. 无法使用 std::optional（CUDA 不支持 C++17 STL）
//         2. 需要不调用默认构造器就能尚未初始化地占用内存
//
// 使用方式：
//   Optional<Foo> opt(present(arg1, arg2));  // 构造 T（相当于 T{arg1, arg2}）
//   Optional<Foo> opt(Absent{});             // 不构造 T
//   if (opt.present) opt.thing.doSomething(); // 安全访问
//
// 实现细节：
//   - union { T thing; } 确保 T 平多不会调用默认构造器
//   - ~Optional() 如果 present 则手动调用 ~T()
//   - Present<Arg...> 封装参数列表，用于看传递到 T 的构造函数
template<int ...vals>
struct IntSeq {};

// IntSeqUpTo<n, 0>::Type 生成 IntSeq<0, 1, ..., n-1>
template<int n, int m, int ...i>
struct IntSeqUpTo: IntSeqUpTo<n, m+1, i..., m> {};
template<int n, int ...i>
struct IntSeqUpTo<n, n, i...> { using Type = IntSeq<i...>; };

// Present<Arg...>: Packs a list of arguments together to be passed to Optional<T>.
// Present<Arg...>：将一组参数打包成一个类型传递给 Optional<T> 构造
template<typename ...Arg>
struct Present;
template<>
struct Present<> {};  // 空列表层级递归终止
template<typename H, typename ...T>
struct Present<H, T...> {
  H h;
  Present<T...> t;  // 递归展开，头部 h + 尾部 t

  NCCL_HOST_DEVICE_INLINE Present(H h, Present<T...> t): h(static_cast<H>(h)), t(t) {}
  NCCL_HOST_DEVICE_INLINE Present(Present const& that): h(static_cast<H>(that.h)), t(that.t) {}

  // get(IntSeq<0>) 返回第 0 个元素（头部）
  NCCL_HOST_DEVICE_INLINE H get(IntSeq<0>) {
    return static_cast<H>(h);
  }
  // get(IntSeq<i>) 递归返回第 i 个元素
  template<int i>
  NCCL_HOST_DEVICE_INLINE decltype(auto) get(IntSeq<i>) {
    return t.get(IntSeq<i-1>{});
  }
};

// present(args...)：工厂函数，构造 Present<Arg&&...>
NCCL_HOST_DEVICE_INLINE Present<> present() {
  return Present<>{};
}
template<typename H, typename ...T>
NCCL_HOST_DEVICE_INLINE Present<H&&, T&&...> present(H&& h, T&& ...t) {
  return Present<H&&, T&&...>{static_cast<H&&>(h), present(static_cast<T&&>(t)...)};
}

// Absent：占位符类型，传入 Optional 表示「不构造」
struct Absent {};

template<typename T>
struct Optional {
  bool present; // Is `thing` constructed.
  union { T thing; };  // union 确保 T 不会被默认构造

  // Construct with absent thing:
  NCCL_HOST_DEVICE_INLINE constexpr Optional(): present(false) {}
  NCCL_HOST_DEVICE_INLINE constexpr Optional(Absent): present(false) {}  // 显式不构造

  // Helper constructor
  // 展开 Present<Arg...> 里的参数列表并传递给 T 的构造函数
  template<typename ...Arg, int ...i>
  NCCL_HOST_DEVICE_INLINE Optional(Present<Arg...> args, IntSeq<i...>):
    present(true),
    thing{args.get(IntSeq<i>())...} {  // 参数展开： thing{arg0, arg1, ...}
  }
  // Construct with present thing:
  // IntSeqUpTo 自动生成 0..sizeof...(Arg)-1 的整数序列
  template<typename ...Arg>
  NCCL_HOST_DEVICE_INLINE Optional(Present<Arg...> args):
    Optional(args, typename IntSeqUpTo<sizeof...(Arg), 0>::Type()) {
  }

  // 析构时如果 thing 已构造，则手动调用其析构函数
  NCCL_HOST_DEVICE_INLINE ~Optional() {
    if (present) thing.~T();
  }
};

}}
#endif // __cplusplus
#endif
