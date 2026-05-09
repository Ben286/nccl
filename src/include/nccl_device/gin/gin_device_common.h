/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_GIN_DEVICE_COMMON_H_
#define _NCCL_GIN_DEVICE_COMMON_H_

#include <stdint.h>
#include "../net_device.h"
#include "../utility.h"
#include "gin_device_host_common.h"

#if CUDA_VERSION >= 12080 && __CUDA_ARCH__ >= 900
// SM90（Hopper）+ CUDA 12.8 起，硬件提供了带 acquire/release 语义的 fence PTX 指令
// （fence.acquire.gpu / fence.release.gpu）。低版本只能用 membar.gl / ld.acquire 模拟，开销更高。
#define NCCL_GIN_HAS_FENCE_ACQUIRE_RELEASE_PTX 1
#endif

// ---------------------------------------------------------------------------
// GIN 后端编译开关
// ---------------------------------------------------------------------------

// GIN_PROXY 后端：GPU 把 GFD（128 字节描述符）写到队列，CPU proxy 线程轮询并转发 IB Verbs
// 几乎所有支持 CUDA 的系统都可用（不需要特殊硬件），默认开启。
#ifndef NCCL_GIN_PROXY_ENABLE
#define NCCL_GIN_PROXY_ENABLE 1
#endif

// GIN_GDAKI 后端：GPU 直接操作 DOCA GPUNetIO Verbs QP（完全 bypass CPU）
// 需要：CUDA >= 12.2（提供了 cuda::atomic with system scope 的 PTX 支持）
//         GPU 架构 >= SM70（Volta，引入了独立线程调度和原子操作保证）
//         特殊 NIC 硬件支持（DOCA GPUNetIO capable NIC）
// 注：即使 GDAKI 开启，运行时也会根据 ginType 判断是否真正使用
#ifndef NCCL_GIN_GDAKI_ENABLE
#if CUDA_VERSION >= 12020 && __CUDA_ARCH__ >= 700
#define NCCL_GIN_GDAKI_ENABLE 1
#else
#define NCCL_GIN_GDAKI_ENABLE 0
#endif
#endif

// ---------------------------------------------------------------------------
// GIN put() 可选优化标志
// ---------------------------------------------------------------------------
enum ncclGinOptFlags {
  ncclGinOptFlagsDefault = 0,
  // MaySkipCreditCheck：允许实现跳过信用检查（queue 满时不等待）
  // 适用于调用者已通过其他方式确保不会溢出的场景（如已知队列肯定有空位）
  // 误用此标志可能导致数据丢失！
  ncclGinOptFlagsMaySkipCreditCheck = (1 << 0),
  // AggregateRequests：提示实现可以将多个 put 合并为一个网络请求
  // 适用于多个相邻小 put 的场景，有助于减少描述符数量，降低网络开销
  ncclGinOptFlagsAggregateRequests = (1 << 1),
};

// ---------------------------------------------------------------------------
// NCCL_GIN_BACKEND_MASK_ALL：编译期 backendMask 常量
// ---------------------------------------------------------------------------
// backendMask 是一个 bitmask，每一位对应一个 backend（ncclNetDeviceType enum）
// 例如：NCCL_NET_DEVICE_GIN_PROXY = 2, NCCL_NET_DEVICE_GIN_GDAKI = 3
// 所以 (1u << 2) = 0x4 代表 PROXY 位，(1u << 3) = 0x8 代表 GDAKI 位
// NCCL_GIN_BACKEND_MASK_ALL = 这两位的 OR（取决于编译时哪些后端被开启）
// 用于 ncclGin（ncclGin_BackendMask<NCCL_GIN_BACKEND_MASK_ALL>）以在运行时动态选 backend
#define NCCL_GIN_BACKEND_MASK_ALL                                               \
  (((NCCL_GIN_PROXY_ENABLE) ? 1u : 0u) << (unsigned)NCCL_NET_DEVICE_GIN_PROXY | \
   ((NCCL_GIN_GDAKI_ENABLE) ? 1u : 0u) << (unsigned)NCCL_NET_DEVICE_GIN_GDAKI)

// Resource sharing mode for a given ncclGin/ncclGin_C *instance*.
// This mode is selected at construction time and is carried by the ncclGin
// object, then copied into ncclGinCtx for each call. It is not stored as
// persistent per-context state in the communicator (i.e., different ncclGin
// instantiations that target the same contextIndex may use different modes).
enum ncclGinResourceSharingMode : uint8_t {
  NCCL_GIN_RESOURCE_SHARING_GPU = 0, // 资源在 GPU 范围内共享
  NCCL_GIN_RESOURCE_SHARING_CTA = 1, // 资源在单个 CTA(block) 范围内独占
};

// ---------------------------------------------------------------------------
// ncclGinCtx：GIN 上下文的运行时表示（GPU 寄存器中保存的数据）
// ---------------------------------------------------------------------------
// 每个 ncclGin_BackendMask<M> 对象在构造时从设备内存加载并保存在寄存器中
// 这样的寄存器缓存设计减少了 kernel 执行期间对全局内存的访问
struct ncclGinCtx {
  unsigned backendMask;  // 本 context 支持的 backend 的 bitmask（编译期可见，用于 dispatch）
  ncclNetDeviceType backend;  // 运行时实际使用的 backend（GIN_PROXY 或 GIN_GDAKI）
  int rank;  // 本 rank 在 rail 内的编号（0..nRanks-1）
  int nRanks;  // rail 内的 rank 总数（即能 put 的目标数量）
  void* handle;  // 指向后端 GPU 上下文数组的指针
                           //   GIN_PROXY：指向 ncclGinProxyGpuCtx_t 数组
                           //   GIN_GDAKI：指向 GDAKI 的 GPU context 结构
  int contextId;  // 本 block 使用的 context 索引（round-robin 选择，避免多 block 竞争同一 QP）
  uint8_t resourceSharingMode;
};

// ncclGinCtx_M<M>：backendMask 编译期已知时的特化版本
// 继承 ncclGinCtx，只是标记编译期 mask，用于 ncclGinCall 模板选择分支
// 编译器可以在 backendMask 只有一个 bit 时静态选择 backend，消除运行时 switch
template <unsigned backendMask>
struct ncclGinCtx_M : ncclGinCtx {};

// ncclGinDescriptorSmem：在 __shared__ 内存中分配的 GFD 临时缓冲区
// 大小 = 64 字节（注：2.30.3 的 ncclGinProxyGfd_t 已扩展到 128 字节，
//   proxy 后端的 Put/Get 函数在 hasDescriptor=false 时使用栈上 128 字节的 tmpDesc 作为替代）
// 对齐到 16 字节：便于 CUDA __stwt（PTX st.wt，写穿 L2 直达 host DRAM）的 uint4（16字节）向量化写入
// 使用 smem 而非寄存器的原因：
//   GFD 构建需要顺序写入多个字段，用寄存器需要大量 64bit 寄存器
//   smem 写入然后一次性用 stwt 写到全局内存队列，效率更高
struct ncclGinDescriptorSmem {
  alignas(16) char space[64];  // 64 字节缓冲区
};

// ---------------------------------------------------------------------------
// ncclGinSignalType：GIN 信号的存储方式
// ---------------------------------------------------------------------------
// GIN 信号分为两种，用于区分 put 完成时如何通知目标：
//   NONE      - 不附带任何信号（纯数据传输，接收方只能用 counter 感知完成）
//   VA        - Virtual Address 信号：信号存储在某段虚拟地址（如 symmetric 内存）中
//               使用场景：接收方不在 rail 内，需要直接写对方的 VA 空间
//   INDEXED   - Indexed 信号：信号在 signalSpace 全局池中通过 index 访问
//               使用场景：rail 内标准通信（性能更好，不需要额外的 VA 注册）
enum ncclGinSignalType {
  NCCL_GIN_SIGNAL_TYPE_NONE,
  NCCL_GIN_SIGNAL_TYPE_VA,
  NCCL_GIN_SIGNAL_TYPE_INDEXED,
};

struct ncclGinSignalDescriptor {
  // 信号类型（NONE/VA/INDEXED），决定下面 union 中哪个分支有效
  ncclGinSignalType type;
  union {
    struct {
      // VA 信号：信号存在于这个窗口内的 signalOffset 处
      // signalWindow：GIN 内部窗口句柄（GIN-specific handle）
      // signalOffset：该窗口内的字节偏移
      // ncclWindow：对应的 NCCL symmetric window（用于本地复位信号）
      ncclGinWindow_t signalWindow;
      size_t signalOffset;
      ncclWindow_t ncclWindow;
    } vaSignal;
    struct {
      // Indexed 信号：全局 signalSpace 中的索引（uint32_t）
      // signalId 由 ncclGinAllocSignalsCounters 分配，与 ncclGinSignal_t 类型对应
      ncclGinSignal_t signalId;
    } indexedSignal;
  };
};

#if NCCL_CHECK_CUDACC

template <ncclNetDeviceType backend>
struct ncclGinApi_Wait {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx, ncclGinRequest_t& outRequest, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, cuda::memory_order ord, uint32_t* abortFlag);
};

template <ncclNetDeviceType backend>
struct ncclGinApi_FlushAsync {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx, uint32_t peer, ncclGinRequest_t* outRequest, uint32_t optFlags);
};

template <ncclNetDeviceType backend>
struct ncclGinApi_Get {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx, Coop coop, int peer, ncclGinWindow_t remoteWin, size_t remoteOff,
                                      ncclGinWindow_t localWin, size_t localOff, size_t bytes,
                                      bool hasDescriptor, ncclGinDescriptorSmem* descriptor,
                                      uint32_t optFlags = ncclGinOptFlagsDefault);
};

template <ncclNetDeviceType backend>
struct ncclGinApi_Put {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx, Coop coop, int peer, bool hasWins,
                                      ncclGinWindow_t dstWin, size_t dstOff, ncclGinWindow_t srcWin,
                                      size_t srcOff, size_t bytes,
                                      ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasCounter,
                                      ncclGinCounter_t counterId, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given,
                                      uint32_t optFlags = ncclGinOptFlagsDefault);
};

template <ncclNetDeviceType backend>
struct ncclGinApi_PutValue {
  template <typename Coop, typename T>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx, Coop coop, int peer, ncclGinWindow_t dstWin,
                                      size_t dstOff, T srcData,
                                      ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given,
                                      uint32_t optFlags = ncclGinOptFlagsDefault);
};

template <ncclNetDeviceType backend>
struct ncclGinApi_GetSignalPtr {
  NCCL_DEVICE_INLINE static uint64_t* call(ncclGinCtx, int peer, ncclGinSignal_t signalId);
};

template <ncclNetDeviceType backend>
struct ncclGinApi_GetCounterPtr {
  NCCL_DEVICE_INLINE static uint64_t* call(ncclGinCtx, int peer, ncclGinCounter_t counterId);
};

template <ncclNetDeviceType backend>
struct ncclGinApi_ResetSignal {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx, ncclGinSignalDescriptor signal);
};

template <ncclNetDeviceType backend>
struct ncclGinApi_ResetCounter {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx, ncclGinCounter_t counterId);
};

template <ncclNetDeviceType backend>
struct ncclGinApi_Flush {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx, Coop, cuda::memory_order ord, uint32_t* abortFlag);
};
#endif

#if NCCL_CHECK_CUDACC
// ncclGinCall 重载 1：单纯传入 ctx，自动读取 ctx.backendMask
// ---------------------------------------------------------------------------
// ncclGinCallImpl：GIN 后端运行时分发的核心函数
// ---------------------------------------------------------------------------
// 该函数实现了当 backendMask 包含多个后端时（ncclGin）的运行时分发。
//
// 两种分发路径：
//   1. 动态分发：beMask 有多个位时，根据 ctx.backend 运行时 switch
//   2. 静态分发：beMask 只有一个位时（singleton），编译器直接内联对应的 ApiFn
//     弹：__popc(beMask - 1) = log2(beMask) = backend enum 值，switch case 被静态预测为定不跳
//
// 编译器优化说明：
//   singleton == true 时，编译器能证明 switch 只有一个 case 可达，会直接内联不生成 branch
//   这就是为什么 ncclGin_BackendOne<backend> 比 ncclGin 实例化更快的根本原因
//   当所有节点运行同一个 backend 时，可以用 ncclGin_BackendOne 干营 0 分支开销
template <template <ncclNetDeviceType> typename ApiFn, typename... Arg>
NCCL_DEVICE_INLINE static decltype(auto) ncclGinCallImpl(unsigned beMask, ncclGinCtx ctx, Arg&&... arg) {
  // singleton：如果 beMask 只有一个位，(beMask & (beMask-1))==0
  // 此时用 __popc(beMask-1) 就能直接得到 backend enum 值，跳过运行时 ctx.backend 读取
  bool singleton = (beMask & (beMask - 1)) == 0;  // Only one bit set
  // 如果 singleton：__popc(beMask-1) = 未就是该 backend 的 enum 值（不需要读内存）
  // 如果多 backend：读取 ctx.backend（运行时内存读）来决定跳转目标
  switch (singleton ? __popc(beMask - 1) : (int)ctx.backend) {
#if NCCL_GIN_PROXY_ENABLE
    case (int)NCCL_NET_DEVICE_GIN_PROXY:
      // __builtin_unreachable()：如果运行时到达此 case 但 beMask 中对应位不为 1，则是 bug
      // 这个调用告诉编译器此路径不可到达，帮助内联 + 优化移除此分支的出口代码
      if (!(1 & (beMask >> (int)NCCL_NET_DEVICE_GIN_PROXY))) __builtin_unreachable();
      return ApiFn<NCCL_NET_DEVICE_GIN_PROXY>::call(ctx, static_cast<Arg&&>(arg)...);
#endif
#if NCCL_GIN_GDAKI_ENABLE
    case (int)NCCL_NET_DEVICE_GIN_GDAKI:
      if (!(1 & (beMask >> (int)NCCL_NET_DEVICE_GIN_GDAKI))) __builtin_unreachable();
      return ApiFn<NCCL_NET_DEVICE_GIN_GDAKI>::call(ctx, static_cast<Arg&&>(arg)...);
#endif
    default:
      // 运行时到达此处意味着 ctx.backend 不属于任何已开启的后端，不应该发生
      __builtin_unreachable();
  }
}

template <template <ncclNetDeviceType> typename ApiFn, typename... Arg>
NCCL_DEVICE_INLINE static decltype(auto) ncclGinCall(ncclGinCtx ctx, Arg&&... arg) {
  return ncclGinCallImpl<ApiFn>(ctx.backendMask, ctx, static_cast<Arg&&>(arg)...);
}

// ncclGinCall 重载 2：传入类型信息为 ncclGinCtx_M<beMask>，beMask 编译期已知
// 当 beMask 只有一个 bit 时，编译器可完全静态内联对应 backend 的 call，0 分支开销
template <template <ncclNetDeviceType> typename ApiFn, unsigned beMask, typename... Arg>
NCCL_DEVICE_INLINE static decltype(auto) ncclGinCall(ncclGinCtx_M<beMask> ctx, Arg&&... arg) {
  return ncclGinCallImpl<ApiFn>(beMask, ctx, static_cast<Arg&&>(arg)...);
}
#endif

#endif
