/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_SESSION_H_
#define _NCCL_DEVICE_GIN_SESSION_H_
#include "core.h"
#include "gin/gin_device_common.h"

#if NCCL_CHECK_CUDACC
struct ncclGinCtx; // Definition in nccl_device/gin/gin_device_host_common.h
template<unsigned> struct ncclGinCtx_M; // ...

struct ncclGinDescriptorSmem; // A type user allocates in __shared__ memory

// Used as completion actions for ncclGinSession::put

// ============================================================================
// GIN Device API 概述
// ============================================================================
// GIN（GPU Initiated Network）允许 GPU kernel 直接发起 RDMA 操作。
// 该文件定义了 GPU 内核可用的 GIN 高层 C++ 模板 API，是手写 NCCL Symmetric Kernel 的主要工具。
//
// 核心抽象层次：
//   ncclGin_BackendMask<mask>、ncclGin_BackendOne<backend>、ncclGin
//   → 此 3 个类型封装了所有 GIN API（put/signal/flush/wait/counter 等）
//
//   mask 是 支持的 backend 岗位展开式：
//     NCCL_GIN_BACKEND_MASK_ALL = 支持所有 GIN backend（运行时选择）
//     (1u << NCCL_GIN_TYPE_PROXY)  = 仅 Proxy 后端
//     (1u << NCCL_GIN_TYPE_GDAKI)  = 仅 GDAKI 后端
//
//   运行时分发： ncclGinCallImpl()（gin_device_common.h）
//     根据 ginCtx.backend 跳转到 Proxy 或 GDAKI 具体实现
//
// RemoteAction 类型（put 的完成回调）：
//   ncclGin_None：无任何远端操作
//   ncclGin_SignalInc：远端对应信号自增 1
//     注：同一信号的两次 reset() 之间，要么全部用 SignalInc，要么全部不用（不能混用）
//   ncclGin_SignalAdd：远端对应信号加上指定值
//   ncclGin_VASignalInc：远端的虚拟地址内存信号自增（VA = Virtual Address）
//   ncclGin_VASignalAdd：远端虚拟地址信号加上指定值
//
// LocalAction 类型（put 的本地回调）：
//   ncclGin_CounterInc：本地计数器自增 1（退出 PUT 流水线的信用监控）
//
// 典型使用模式（见 all_gather_gin.cuh）：
//   gin.put(rail, nextPeer, dstWin, dstOff, srcWin, srcOff, bytes,
//           ncclGin_SignalInc{signal},    // 远端信号
//           ncclGin_CounterInc{counter}, // 本地出站信用回收
//           ncclCoopWarp{})；           // warp 内协作
//   ...  
//   gin.waitSignal(ncclCoopWarp{}, signal, expected); // 等待远端確认
// ============================================================================
struct ncclGin_None {};

struct ncclGin_VASignalInc { ncclWindow_t signalWindow; size_t signalOffset; };
struct ncclGin_VASignalAdd { ncclWindow_t signalWindow; size_t signalOffset; uint64_t value; };

struct ncclGin_SignalAdd { ncclGinSignal_t signal; uint64_t value; };
// SignalInc: equivalent to SignalAdd{+1} except it may not be mixed with any
// other signal operator without intervening signal reset(). Formally: for a
// given signal, all operations between successive reset()'s of that signal must
// either all be SignalInc or all not SignalInc.
struct ncclGin_SignalInc { ncclGinSignal_t signal; };
// Support deferred:
// struct ncclGin_SignalSet { ncclGinSignal_t signal; uint64_t value; };
struct ncclGin_CounterInc { ncclGinCounter_t counter; };

struct ncclGin_DescriptorSmem { ncclGinDescriptorSmem* descriptor; };

// ============================================================================
// ncclGin_BackendMask<backendMask>： GIN Device API 的主入口类型
//
// 在 GPU kernel 中创建：
//   ncclGin gin(devComm, contextIndex); // 构建：从寄存器加载 GIN 上下文
//
// 主要方法：
//   put()：向远端 peer 发送数据，可选附带信号/计数器回调
//     两种载荷形式： (window/offset/bytes) 或 (ncclSymPtr<T>/nElts)
//   putValue()：将一个小就单协议尜子内存值（最多 8 字节）写入远端
//   signal()：向远端发送信号，不附带数据载荷
//   flush()：等待所有 put 的源缓冲已安全复用（不保证远端已收到）
//   readCounter()/waitCounter()：读取/等待本地计数器
//   readSignal()/waitSignal()：读取/等待远端信号
//   resetCounter()/resetSignal()：重置计数器/信号为 0
//
// ===== givenRelease / requiredRelease 参数详解 =====
//   这两个参数控制内存可见性 scope：
//   givenRelease：调用者保证源数据在哪个 scope 内是可见的
//     - thread_scope_thread：只有本线程写过（最弱）
//     - thread_scope_block：本 CTA 内所有线程写过
//     - thread_scope_device：GPU 上所有线程写过
//     - thread_scope_system：CPU+GPU 都写过（最强，不需要额外 fence）
//   requiredRelease：远端设备（IB HCA）需要的内存可见性 scope
//     默认 thread_scope_device：确保 GPU 所有对源缓冲区的写入对 IB HCA 可见
//   如果 givenRelease < requiredRelease，实现会自动补一次 atomic fence
//
// 信号语义（Signal vs Counter）：
//   Signal：由远端修改，本地读取。用于远端就伪检测
//   Counter：本地 CPU/GPU 修改，用于本地出站流水线信用回收监控
//   VA Signal：信号存在于远端对称内存中（Virtual Address 需要 VMM 支持）
//
// Rolling 比较逐步（防溢出）：
//   waitCounter/waitSignal 使用 rolling comparison：
//   rolling_less_equal(a, b, bits) := ((b-a) & mask) <= (mask>>1)
//   这意味着就算 counter 溢出，等待逻辑仍然正确
//
// ===== ncclGin_BackendMask 的构造开销 =====
//   构造时从全局内存加载 GIN context 到寄存器：
//     _ginHandle：后端 GPU context 指针（ncclGinProxyGpuCtx_t* 的 GPU 副本）
//     _signalShadows：信号 shadow 数组，用于 waitSignalFollowShadow 跟踪最新信号値
//   寄存器负载：每个 ncclGin 对象占用寄存器数量与 context 大小有关
// ============================================================================
template<unsigned backendMask>
struct ncclGin_BackendMask;

template<ncclNetDeviceType backend>
using ncclGin_BackendOne = ncclGin_BackendMask<(1u<<(int)backend)>;

using ncclGin = ncclGin_BackendMask<NCCL_GIN_BACKEND_MASK_ALL>;

#endif

#if NCCL_CHECK_CUDACC
struct ncclGin_C {
  ncclDevComm const& comm;
  uint32_t nConnections:8, connectionId:8, _ginBackend:8;
  uint32_t contextId;
  ncclGinResourceSharingMode resourceSharingMode;

  //////////////////////////////////////////////////////////////////////////////
  // internal:
  void* _ginHandle;
  uint64_t* _signalShadows;
  unsigned backendMask;

  NCCL_DEVICE_INLINE ncclGin_C(ncclDevComm const& comm_, unsigned backendMask_, int contextIndex,
                               ncclGinResourceSharingMode resourceSharingMode_ = NCCL_GIN_RESOURCE_SHARING_GPU);
};

// Helper init function that wraps placement new
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGin_C_init(
  ncclGin_C* net, unsigned backendMask, ncclDevComm const& comm, int contextIndex);

// Helper init function with explicit resource sharing mode.
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGin_C_initWithResourceSharingMode(
  ncclGin_C* net, unsigned backendMask, ncclDevComm const& comm, int contextIndex,
  ncclGinResourceSharingMode resourceSharingMode);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinPut(
  ncclGin_C* net,
  ncclTeam team, int peer,
  ncclWindow_t dstWin, size_t dstOffset,
  ncclWindow_t srcWin, size_t srcOffset, size_t bytes,
  bool isSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
  bool isCounter, ncclGinCounter_t counterId,
  ncclCoopAny coop,
  bool isDescriptor, ncclGinDescriptorSmem* descriptor,
  cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinSignal(
  ncclGin_C* net,
  ncclTeam team, int peer,
  bool isSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
  ncclCoopAny coop,
  bool isDescriptor, ncclGinDescriptorSmem* descriptor,
  cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinPut_v2(
  ncclGin_C* net,
  ncclTeam team, int peer,
  ncclWindow_t dstWin, size_t dstOffset,
  ncclWindow_t srcWin, size_t srcOffset, size_t bytes,
  bool isSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
  bool isCounter, ncclGinCounter_t counterId,
  ncclCoopAny coop,
  bool isDescriptor, ncclGinDescriptorSmem* descriptor,
  cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease,
  uint32_t optFlags);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinSignal_v2(
  ncclGin_C* net,
  ncclTeam team, int peer,
  bool isSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
  ncclCoopAny coop,
  bool isDescriptor, ncclGinDescriptorSmem* descriptor,
  cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease,
  uint32_t optFlags);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinFlush(
  ncclGin_C* net,
  ncclCoopAny coop,
  cuda::memory_order ord);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE uint64_t ncclGinReadCounter(
  ncclGin_C* net,
  ncclGinCounter_t counter,
  int bits,
  cuda::memory_order ord);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinWaitCounter(
  ncclGin_C* net,
  ncclCoopAny coop,
  ncclGinCounter_t counter,
  uint64_t least,
  int bits,
  cuda::memory_order ord);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE uint64_t ncclGinReadSignal(
  ncclGin_C* net,
  ncclGinSignal_t signal,
  int bits,
  cuda::memory_order ord);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinWaitSignal(
  ncclGin_C* net,
  ncclCoopAny coop,
  ncclGinSignal_t signal,
  uint64_t least,
  int bits,
  cuda::memory_order ord);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinResetCounter(
  ncclGin_C* net,
  ncclGinCounter_t counter);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinResetSignal(
  ncclGin_C* net,
  ncclGinSignal_t signal);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinPutValue(
  ncclGin_C* net,
  ncclTeam team, int peer,
  ncclWindow_t dstWin, size_t dstOffset,
  uint64_t value, size_t size,
  bool isSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
  ncclCoopAny coop,
  bool isDescriptor, ncclGinDescriptorSmem* descriptor,
  cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinPutValue_v2(
  ncclGin_C* net,
  ncclTeam team, int peer,
  ncclWindow_t dstWin, size_t dstOffset,
  uint64_t value, size_t size,
  bool isSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
  ncclCoopAny coop,
  bool isDescriptor, ncclGinDescriptorSmem* descriptor,
  cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease,
  uint32_t optFlags);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE uint64_t* ncclGinGetSignalShadowPtr(
  ncclGin_C* net,
  ncclGinSignal_t signal);

template<unsigned backendMask>
struct ncclGin_BackendMask {
  ncclDevComm const& comm;
  uint32_t nConnections:8, connectionId:8, _ginBackend:8;
  uint32_t contextId;
  // Runtime-selected resource sharing mode for this context.
  ncclGinResourceSharingMode resourceSharingMode;

  // Loads GIN context into registers. Each context has one QP per peer.
  NCCL_DEVICE_INLINE ncclGin_BackendMask(
    ncclDevComm const&, int contextIndex,
    ncclGinResourceSharingMode resourceSharingMode_ = NCCL_GIN_RESOURCE_SHARING_GPU);

  template <typename Coop = ncclCoopThread>
  NCCL_DEVICE_INLINE void flushAsync(ncclTeam team, uint32_t peer, ncclGinRequest_t* outRequest,
                                     Coop coop = ncclCoopThread{}, uint32_t optFlags = ncclGinOptFlagsDefault) const;

  template <typename Coop = ncclCoopThread, typename DescriptorSmem = ncclGin_None>
  NCCL_DEVICE_INLINE void wait(ncclGinRequest_t& outRequest,
                               Coop coop = ncclCoopThread{}, DescriptorSmem descriptor = ncclGin_None{},
                               cuda::memory_order ord = cuda::memory_order_acquire) const;

  template<typename Coop = ncclCoopThread,
           typename DescriptorSmem = ncclGin_None>
  NCCL_DEVICE_INLINE void get(
    ncclTeam, int peer,
    ncclWindow_t remoteWnd, size_t remoteOffset,
    ncclWindow_t localWnd, size_t localOffset,
    size_t bytes, Coop coop = ncclCoopThread{},
    DescriptorSmem descriptor = ncclGin_None{},
    uint32_t optFlags = ncclGinOptFlagsDefault) const;

  // =====================================================================
  // put() - 核心数据发送 API
  // =====================================================================
  // 向远端 peer 发送 bytes 字节数据，可选附带信号和计数器回调。
  //
  // 内存序保证：
  //   信号在本次 put 以及到同一 peer 的所有先前 put 的载荷都完全落地后，才会对目标可见
  //   即：一个 context 对同一 peer 的所有 put 是有序的内存划款
  //
  // 参数：
  //   ncclTeam      - 转发团队（rail 或 world）。确定 peer 对应哪个 GIN connection
  //   peer          - 团队内的目标 rank
  //   dstWnd/dstOffset - 远端目标窗口和偏移
  //   srcWnd/srcOffset/bytes - 本地源窗口、偏移和字节数
  //   RemoteAction  - 远端完成后的操作（ncclGin_None/SignalInc/SignalAdd/VASignalInc/VASignalAdd）
  //   LocalAction   - 本地完成后的操作（ncclGin_None/CounterInc）
  //   Coop          - 参与此 put 的线程组（ncclCoopThread/ncclCoopWarpSpan/ncclCoopCta 等）
  //   DescriptorSmem - 可选的 smem 缓冲区（适否则用栈上临时 GFD）
  //   givenRelease  - 调用者保证的内存可见范围（默认 thread）
  //   requiredRelease - 远端设备需要的内存可见范围（默认 device）
  //   optFlags      - 可选优化标志（ncclGinOptFlags）
  template<
    // Action to take on peer when put completes. If a signalling action is used
    // then that signal will be visible only after the payload of this put as well as
    // the payloads of preceding puts on this netContext to the same peer are settled.
    typename RemoteAction = ncclGin_None, // one of ncclGin_{None|SignalInc|SignalAdd|SignalSet}
    // Action to take locally when source has been consumed.
    typename LocalAction = ncclGin_None, // one of ncclGin_{None|CounterInc}
    // Set of threads participating in this put. Must be a subset of Coop.
    typename Coop = ncclCoopThread,
    // Optional smem descriptor space to use. Either ncclGin_{None|DescriptorSmem}
    typename DescriptorSmem = ncclGin_None
  >
  NCCL_DEVICE_INLINE void put(
    ncclTeam, int peer,
    ncclWindow_t dstWnd, size_t dstOffset,
    ncclWindow_t srcWnd, size_t srcOffset, size_t bytes,
    RemoteAction remoteAction = ncclGin_None{},
    LocalAction localAction = ncclGin_None{},
    Coop coop = ncclCoopThread{},
    DescriptorSmem descriptor = ncclGin_None{},
    cuda::thread_scope givenRelease = cuda::thread_scope_thread,
    cuda::thread_scope requiredRelease = cuda::thread_scope_device,
    uint32_t optFlags = ncclGinOptFlagsDefault
  ) const;

  template<
    typename T,
    // Action to take on peer when put completes. If a signalling action is used
    // then that signal will be visible only after the payload of this put as well as
    // the payloads of preceding puts on this context to the same peer are settled.
    typename RemoteAction = ncclGin_None, // one of ncclGin_{None|SignalInc|SignalAdd|SignalSet}
    // Action to take locally when source has been consumed.
    typename LocalAction = ncclGin_None, // one of ncclGin_{None|CounterInc}
    // Set of threads participating in this put. Must be a subset of Coop.
    typename Coop = ncclCoopThread,
    // Optional smem descriptor space to use. Either ncclGin_{None|DescriptorSmem}
    typename DescriptorSmem = ncclGin_None
  >
  NCCL_DEVICE_INLINE void put(
    ncclTeam, int peer,
    ncclSymPtr<T> dstElts, ncclSymPtr<T> srcElts, size_t nElts,
    RemoteAction remoteAction = ncclGin_None{},
    LocalAction localAction = ncclGin_None{},
    Coop coop = ncclCoopThread{},
    DescriptorSmem descriptor = ncclGin_None{},
    cuda::thread_scope givenRelease = cuda::thread_scope_thread,
    cuda::thread_scope requiredRelease = cuda::thread_scope_device,
    uint32_t optFlags = ncclGinOptFlagsDefault
  ) const;

  template<
    typename T, // requires sizeof(T) <= 8
    // See put() for all template arguments.
    typename RemoteAction = ncclGin_None,
    typename Coop = ncclCoopThread,
    typename DescriptorSmem = ncclGin_None
  >
  NCCL_DEVICE_INLINE void putValue(
    ncclTeam, int peer,
    ncclWindow_t dstWnd, size_t dstOffset, T value,
    RemoteAction remoteAction = ncclGin_None{},
    Coop coop = ncclCoopThread{},
    DescriptorSmem descriptor = ncclGin_None{},
    cuda::thread_scope givenRelease = cuda::thread_scope_thread,
    cuda::thread_scope requiredRelease = cuda::thread_scope_device,
    uint32_t optFlags = ncclGinOptFlagsDefault
  ) const;

  template<
    typename T, // requires sizeof(T) <= 8
    // See put() for all template arguments.
    typename RemoteAction = ncclGin_None,
    typename Coop = ncclCoopThread,
    typename DescriptorSmem = ncclGin_None
  >
  NCCL_DEVICE_INLINE void putValue(
    ncclTeam, int peer,
    ncclSymPtr<T> dst, T value,
    RemoteAction remoteAction = ncclGin_None{},
    Coop coop = ncclCoopThread{},
    DescriptorSmem descriptor = ncclGin_None{},
    cuda::thread_scope givenRelease = cuda::thread_scope_thread,
    cuda::thread_scope requiredRelease = cuda::thread_scope_device,
    uint32_t optFlags = ncclGinOptFlagsDefault
  ) const;

  template<typename RemoteAction,
           typename Coop = ncclCoopThread,
           typename DescriptorSmem = ncclGin_None>
  NCCL_DEVICE_INLINE void signal(
    ncclTeam, int peer, RemoteAction remoteAction,
    Coop coop = ncclCoopThread(),
    DescriptorSmem descriptor = ncclGin_None{},
    cuda::thread_scope givenRelease = cuda::thread_scope_thread,
    cuda::thread_scope requiredRelease = cuda::thread_scope_device,
    uint32_t optFlags = ncclGinOptFlagsDefault
  ) const;

  // All source buffers from put's from any thread in this coop will be safe to reuse.
  // Flush does not guarantee that data has settled in remote memory.
  // =====================================================================
  // flush() - 等待本地源缓冲区可安全复用
  // =====================================================================
  // 等待 coop 中所有线程提交的所有 put 的源缓冲区已被 IB HCA 读取完毕（即 CPU 已消费对应 GFD）。
  // 注意：它不保证远端已收到数据！
  // 来确认远端收到需要导用 Signal。
  //
  // 硬件行为（GIN_PROXY）：
  //   等待 cis[pe] >= pis[pe]（rolling 比较），即所有已提交 GFD 都被 CPU 处理。
  //   CPU proxy 在发送完成 IB RDMA work request 后空间 GFD，并递增 ci，
  //   因此 ci >= pi 表示 IB HCA 已发出所有请求（源缓冲区已可复用）
  //
  // 对应 GIN_GDAKI：等待 GPU 直接操作的 QP send queue 被消费完毕
  template<typename Coop>
  NCCL_DEVICE_INLINE void flush(Coop, cuda::memory_order ord = cuda::memory_order_acquire) const;

  // Counter and signal wait use "rolling" comparison logic of a given bit-width
  // such that unsigned overflow does not disturb the property that: x < x+1.
  //
  // bool rolling_less_equal(uint64_t a, uint64_t b, int bits) {
  //   uint64_t m = uint64_t(-1)>>(64-bits);
  //   return ((b-a) & m) <= (m>>1);
  // }
  //
  // The condition waited for is that the supplied value is rolling_less_equal
  // to the internal value.
  //
  // Counters are restricted to using a maximum of 56 bits despite that being fewer
  // than a uint64_t can carry.

  // =====================================================================
  // readCounter() / waitCounter() - 本地计数器读取和等待
  // =====================================================================
  // Counter 由 CPU proxy 递增（每完成一次 GFD 处理）或 GPU 直接递增（GDAKI）。
  // 本地 GPU kernel 通过 readCounter/waitCounter 监控局部流水线发送信用消耗。
  // 典型用法：发送 N 个 PUT 后，waitCounter(…, N) 确保所有 N 次 PUT 的源缓冲区已可复用
  NCCL_DEVICE_INLINE uint64_t readCounter(ncclGinCounter_t counter, int bits=56, cuda::memory_order ord = cuda::memory_order_acquire) const;

  // =====================================================================
  // waitSignalMeetShadow() - 等待信号达到 shadow 値
  // =====================================================================
  // 等价于 waitSignal(…, *getSignalShadowPtr(signal))
  // 当使用者先更新了 shadow（不是远端信号），再调用此函数等待远端达到 shadow，这样可以避免问读 shadow 两次
  template<typename Coop>
  NCCL_DEVICE_INLINE void waitCounter(Coop, ncclGinCounter_t counter, uint64_t least, int bits=56, cuda::memory_order ord = cuda::memory_order_acquire) const;

  // Each signal has a dedicated "shadow" which the user is free to manipulate for
  // any reason. The only calls which manipulate the shadow are `increaseSignalShadow`
  // and `resetSignal`.
  // =====================================================================
  // 信号 Shadow 機制
  // =====================================================================
  // 每个信号有一个对应的 shadow 值，完全由用户程序管理。
  // shadow 的典型用法：记录上一次 kernel 结束时的信号値，下一次 kernel 册从 shadow 恢复，
  //   实现跨 kernel 调用的信号连续性（不需要再次从全局内存读取）
  // 工作流：
  //   kernel A 结束时：*getSignalShadowPtr(sig) = localSignalValue
  //   kernel B 开始时： localSignalValue = *getSignalShadowPtr(sig)
  NCCL_DEVICE_INLINE uint64_t* getSignalShadowPtr(ncclGinSignal_t signal) const;
  NCCL_DEVICE_INLINE void increaseSignalShadow(ncclGinSignal_t signal, uint64_t delta) const;

  // Returns current value of signal with all but bottom bits set to zero.
  // =====================================================================
  // readSignal() / waitSignal() - 远端信号读取和等待
  // =====================================================================
  // Signal 由远端通过 put() 的 RemoteAction 递增（INC）或将其设为判定小值（ADD）。
  // 本地通过 readSignal/waitSignal 读取工作：
  //   GIN_PROXY：信号存在于全局内存 signals[] 数组，GPU 可读，CPU proxy 对远端信号递增后写回到本地
  //   GIN_GDAKI：信号由 GPU NIC 直接写入对应内存
  //
  // 硬件实现：GPU spin wait 读取 signals[signalId]，直到它满足 rolling_less_equal(least, value, bits)
  NCCL_DEVICE_INLINE uint64_t readSignal(ncclGinSignal_t signal, int bits=64, cuda::memory_order ord = cuda::memory_order_acquire) const;

  // Returns current value of VA signal at given window and offset with all but bottom bits set to zero.
  NCCL_DEVICE_INLINE uint64_t readSignal(ncclWindow_t signalWindow, size_t signalOffset, int bits=64, cuda::memory_order ord = cuda::memory_order_acquire) const;

  // Wait for signal to meet or exceed value.
  template<typename Coop>
  NCCL_DEVICE_INLINE void waitSignal(Coop, ncclGinSignal_t signal, uint64_t least, int bits=64, cuda::memory_order ord = cuda::memory_order_acquire) const;

  // Wait for VA signal at given window and offset to meet or exceed value.
  template<typename Coop>
  NCCL_DEVICE_INLINE void waitSignal(Coop, ncclWindow_t signalWindow, size_t signalOffset, uint64_t least, int bits=64, cuda::memory_order ord = cuda::memory_order_acquire) const;

  // Wait for signal to meet or exceed shadow value.
  template<typename Coop>
  NCCL_DEVICE_INLINE void waitSignalMeetShadow(Coop, ncclGinSignal_t signal, int bits=64, cuda::memory_order ord = cuda::memory_order_acquire) const;

  // Wait until signal exceeds shadow by `leastDelta` (typically 1), updates shadow
  // with latest value, and returns with `before` equal to previous shadow value
  // and `delta` equal to difference.
  // =====================================================================
  // waitSignalFollowShadow() - 等待信号崚过 shadow+delta，更新 shadow
  // =====================================================================
  // 这是 ReduceScatter 和 AllGather 中最常用的 GIN 等待模式！
  //
  // 语义：
  //   1. spin 等待到 signal 能超过 *shadow + leastDelta（rolling 比较）
  //   2. 将 *shadow 更新为当前 signal 的最新値
  //   3. 返回 before = 更新前的 shadow，delta = 新值 - 老値
  //
  // 典型用法：
  //   int before, delta;
  //   gin.waitSignalFollowShadow(warp, sig, 1, &before, &delta); // 等待至少增加 1
  //   // 现在 sig 值比上次看到的大 delta，before 是上次小值
  //   // 最常见：delta > 1 表示如有多个信号到达，可以批量处理
  template<typename Coop, typename Uint>
  NCCL_DEVICE_INLINE void waitSignalFollowShadow(Coop, ncclGinSignal_t signal, Uint leastDelta, Uint* before, Uint* delta, int bits=64, cuda::memory_order ord = cuda::memory_order_acquire) const;

  // Sets to zero. May not race with concurrent modifications to counter.
  // =====================================================================
  // resetCounter() / resetSignal() - 将计数器/信号重置为 0
  // =====================================================================
  // 警告：
  //   - 必须在所有修改完毕后才调用 reset（不应解 concurrent 修改 race）
  //   - 对 SignalInc：两次 reset() 之间的所有 put 要么全部用 SignalInc，要么全部不用（不能混用）
  //   - resetSignal 同时将 shadow 重置为 0
  NCCL_DEVICE_INLINE void resetCounter(ncclGinCounter_t counter) const;
  // Sets signal and shadow to zero. May not race with concurrent modifcations to signal.
  NCCL_DEVICE_INLINE void resetSignal(ncclGinSignal_t signal) const;
  // Resets a VA signal at the given window and offset.
  NCCL_DEVICE_INLINE void resetSignal(ncclWindow_t signalWindow, size_t signalOffset) const;

  //////////////////////////////////////////////////////////////////////////////
  // internal:

  void* _ginHandle;
  uint64_t* _signalShadows;

  NCCL_DEVICE_INLINE ncclGinCtx_M<backendMask> _makeCtx() const;
};
#endif

#endif // _NCCL_DEVICE_GIN_SESSION_H_
