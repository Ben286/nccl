/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_PROXY_H_
#define _NCCL_DEVICE_GIN_PROXY_H_

//#include <config.h>

#include <cstdint>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "nccl.h"
#include "nccl_device/gin/gin_device_common.h"
#include "nccl_device/utility.h"
#include "../gin_device_host_common.h"
#include "gin_proxy_device_host_common.h"

// [2.30.3 新增] CPU Proxy 异步请求结构体
// 用于 flushAsync/wait 异步模式：GPU 端记录 {peer, nextGfdIdx}，
// 后续 wait 时只需等待该 GFD 索引被 CPU 消费完毕即可
struct ncclGinCpuProxyRequest {
  int peer;
  uint32_t nextGfdIdx;
};
static_assert(sizeof(ncclGinCpuProxyRequest) <= sizeof(ncclGinRequest_t),
              "ncclGinCpuProxyRequest must fit in ncclGinRequest_t");

namespace nccl {
namespace gin {
namespace proxy {

// [2.30.3 新增] 单片 GFD 数据传输上限，用于大数据分片
static constexpr size_t DataChunkSize = 1ULL << 30;  // 1 GB

// ---------------------------------------------------------------------------
// [2.30.3 新增] waitForGfdComplete：等待指定 GFD 索引被 CPU 消费完成
// ---------------------------------------------------------------------------
// 从原 flush() 拆出的核心等待逻辑，支持指定等待到某个特定 GFD 索引。
// 使得 flushAsync+wait 异步模式成为可能：
//   flushAsync 时记录当前 pi（即 nextGfdIdx），
//   wait 时调用此函数等待 ci >= nextGfdIdx。
NCCL_DEVICE_INLINE void waitForGfdComplete(ncclGinProxyGpuCtx_t* proxyCtx, uint32_t pe, uint32_t nextGfdIdx, cuda::memory_order ord, uint32_t* abortFlag) {
  using nccl::utility::loadConst;
  using nccl::utility::rollingLessEq;
  using nccl::utility::testAbort;
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> ci(loadConst(&proxyCtx->cis)[pe]);
  uint32_t steps = 0;
  // The PI and CI can keep moving because of concurrent threads posting GFDs to this queue, and the CPU consuming them.
  // Therefore, to prevent overflow issues in the while statement, we need to use a special comparison function.
#pragma unroll 1  // 禁止展开：这是一个 spin wait 循环，展开会浪费寄存器
  while (!rollingLessEq<uint32_t>(nextGfdIdx, ci.load(ord)) && !testAbort(abortFlag, steps)) continue;
}


// ---------------------------------------------------------------------------
// flush：等待指定 peer 的 GFD 队列被 CPU 全部消费（即源缓冲区安全可复用）
// ---------------------------------------------------------------------------
// 此函数不保证远端已收到数据，只保证本地 GPU 的源缓冲区已可安全复用。
// 威胁：GPU DMA 指引 srcOff 的源缓冲区可能还没有被 IB HCA 读完，要确认远端收到应用 signal。
//
// 实现算法：
//   1. 读取当前生产者索引 pi = pis[pe]·
//   2. 循环 spin：等待 ci >= pi（就是等待 CPU 把所有已提交的 GFD 都消费）
//   3. pi/ci 均用 rolling 比较（防溢出，即 uint32 回绕不影响结果）
//
// 参数：
//   proxyCtx  - GPU 上保存的 context 上下文
//   pe        - 指定等待哪个 peer 的队列
//   ord       - 内存序展（通常 acquire），确保 ci 读取不被重排
//   abortFlag - 如果指向非 null 且内容非零，提前退出循环（用于 kernel 异常终止）
NCCL_DEVICE_INLINE void flush(ncclGinProxyGpuCtx_t* proxyCtx, uint32_t pe, cuda::memory_order ord, uint32_t* abortFlag) {
  using nccl::utility::loadConst;
  // cuda::atomic_ref：将普通指针包装为原子访问
  // thread_scope_system：保证 CPU（host）和 GPU（device）指间的内存可见性
  // 注： pis/cis 必须映射到 GPU（可用 UVA 或 cudaMallocManaged），才能用 thread_scope_system
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> pi(loadConst(&proxyCtx->pis)[pe]);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> ci(loadConst(&proxyCtx->cis)[pe]);
  // 读取当前生产者索引，relaxed 足够（只需一个直观的快照）
  uint32_t p = pi.load(cuda::memory_order_relaxed);
  nccl::gin::proxy::waitForGfdComplete(proxyCtx, pe, p, ord, abortFlag);
}

// ---------------------------------------------------------------------------
// postGfd：GPU 一个线程将已构建的 GFD 写入经内存队列，通知 CPU proxy 进行处理
// ---------------------------------------------------------------------------
// 只有 thread_rank()==0 的线程执行实际写入，其他线程进入就直接返回。
// 这是因为 GFD 队列是单个生产者单个消费者结构（GPU 单线程 write + CPU 单线程 read）。
//
// 关键硬件细节：
//   1. fetch_add(1, relaxed)：GPU 端用 cuda::atomic fetch_add 申请一个队列槻位
//      relaxed 足够：序号分配不需要有顺序保证
//   2. 空序循环等待信用：如果队列满（pi-ci >= queueSize）就自旋等待
//      queueSize 必须是 2 的幂次，(idx &= queueSize-1) 实现环形队列寻址
//   3. sizeof(GFD)/sizeof(uint4) 次 uint4 向量写入：用 __stwt（write-through store hint）
//      stwt = PTX st.wt，跳过 L1，写穿 L2 直达 system memory（host DRAM）
//      好处：pinned host memory 的物理位置就在 host DRAM，CPU 直接读 DRAM，
//      不经过 GPU 任何 cache，因此写穿到 DRAM 后 CPU 立刻可见
//      GPU 向量化写入 sizeof(ncclGinProxyGfd_t)/sizeof(uint4) 次 = 8x 16字节 = 128字节（一次完成整个 GFD）
template <typename Coop>
NCCL_DEVICE_INLINE void postGfd(Coop coop, ncclGinProxyGpuCtx_t* proxyCtx, ncclGinProxyGfd_t* gfd,
                                uint32_t pe, uint32_t* gfdIdx = nullptr) {
  using nccl::utility::loadConst;
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> pi(loadConst(&proxyCtx->pis)[pe]);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> ci(loadConst(&proxyCtx->cis)[pe]);
  // 算出指向第 pe 个 peer 的队列起始地址
  ncclGinProxyGfd_t* q = &loadConst(&proxyCtx->queues)[pe * proxyCtx->queueSize];
  uint32_t queueSize = loadConst(&proxyCtx->queueSize);
  if (coop.thread_rank() == 0) {
    // claim a slot in the gfd queue
    // 步骤 1： fetch_add 申请一个队列槻位
    // 返回的 idx 是将6个 slot 的序号，在 queueSize 下取模后就是环形队列的实际下标
    uint32_t idx = pi.fetch_add(1, cuda::memory_order_relaxed);
    if (gfdIdx != nullptr) {
      *gfdIdx = idx;
    }
    // wait for credits
    // 步骤 2：等待信用回收（flow control）
    // 条件：queueSize <= idx - ci，即当前 queueSize 已经填满，没有空间了
    // 简化： queueSize > pi - ci 就是还有剩余容量
    // 注：这里的循环不用 rolling 比较，因为 idx 可以自由前进并不会溢出（uint32 足够大）
    while (queueSize <= idx - ci.load(cuda::memory_order_relaxed)) {
    }
    // 步骤 3：用 2^n 取模得到环形队列的实际下标
    idx &= queueSize - 1;
// 16 byte stores with the write-through cache hint
// 为什么用 stwt（write-through）：
//   默认 st.global 是 write-back：数据可能停留在 GPU L1/L2 cache 中，
//   只有 cache evict 时才写回 system memory，CPU 无法及时看到
//   __stwt = PTX st.wt：跳过 L1，写穿（through）L2 直达 system memory（host DRAM）
//   GFD 队列是 pinned host memory，物理位置就在 host DRAM，CPU 直接读 DRAM 即可见
//   sizeof(ncclGinProxyGfd_t)/sizeof(uint4) 次 uint4 写入 = 128 字节，完成整个 GFD
#pragma unroll
    for (uint8_t i = 0; i < sizeof(ncclGinProxyGfd_t) / sizeof(uint4); i++) {
      __stwt((uint4*)&q[idx] + i, ((uint4*)gfd)[i]);
    }
  }
}

// ---------------------------------------------------------------------------
// buildGfd：在 GPU kernel 内构建一个完整的 GFD（128字节，16 qwords）
// ---------------------------------------------------------------------------
// 将各个参数写入 GFD 的对应 qword 位置。
// 注意：所有 16 个 qword 的最低位需要设为 1（flag），CPU proxy 通过轮询最后一个 qword 的 flag 判断 GFD 是否完全写入
// 注：GPU stwt 写入按顺序完成，CPU 读到最后一个 qword（qword[15]）的 flag=1 就认为整个 GFD 已就绪
template <typename T>
// Descriptor must be at least GWQ_GFD_SIZE bytes and it should be aligned
// Assumes little-endian, which is okay.
__device__ __forceinline__ void buildGfd(ncclGinProxyGfd_t* gfd, ncclGinProxyOp_t op, T srcVal,
                                         bool hasInline, size_t srcOff, ncclGinWindow_t srcHandle,
                                         size_t dstOff, ncclGinWindow_t dstHandle, size_t size,
                                         ncclGinCounter_t counterId, ncclGinSignal_t signalId,
                                         uint64_t signalVal, ncclGinWindow_t signalWindow,
                                         size_t signalOff) {

// [2.30.3 变更] GFD 从 64→128 字节（16 qwords），所有 flag 统一初始化
  for (int i = 0; i < ncclGinProxyGfdQwords; i++) {
    gfd->qword[i].flag.v = 1;
  }

  // [qword 0] header: version + 数据大小（2.30.3: op 移到 headerExt）
  gfd->qword[ncclGinProxyGfdHeader].header.version = (uint64_t)NCCL_GIN_PROXY_GFD_VERSION;
  gfd->qword[ncclGinProxyGfdHeader].header.size = (uint64_t)size;
  // [qword 7] headerExt: op 类型（2.30.3 从 header 移到此处）
  gfd->qword[ncclGinProxyGfdHeaderExt].headerExt.op = (uint16_t)op;

  // [2.30.3 新增] flush 操作只需 header + headerExt，不需要其他字段
  if (op & ncclGinProxyOpFlush) {
    return;
  }

  if (hasInline) {
    // 内嵌小值：将 srcVal 序列化到 srcValBits，按低位到高位拆分到 qword[1][2]
    uint64_t srcValBits = 0;
    memcpy(&srcValBits, &srcVal, sizeof(T));  // 小端（little-endian）安全
    gfd->qword[ncclGinProxyGfdInlineLow].inlineLow.inlineValLow = (uint32_t)srcValBits;  // [0:31]
    if (sizeof(T) > 4)
      gfd->qword[ncclGinProxyGfdInlineLow].inlineLow.inlineValLow2 = (uint64_t)srcValBits >> 32;  // [32:47]
    if (sizeof(T) > 6)
      gfd->qword[ncclGinProxyGfdInlineHigh].inlineHigh.inlineValHigh = (uint64_t)srcValBits >> 48;  // [48:63]
  } else if (op & ncclGinProxyOpVASignal) {
    gfd->qword[ncclGinProxyGfdVASignalOff].vaSignalOff.vaSignalOff = (uint64_t)signalOff;  // 信号内的偏移
    gfd->qword[ncclGinProxyGfdVASignalHandle].vaSignalHandle.vaSignalHandle = (uint64_t)signalWindow;  // 窗口句柄
  } else {
    gfd->qword[ncclGinProxyGfdSrcOff].srcOff.srcOff = (uint64_t)srcOff;  // 源窗口内偏移
    gfd->qword[ncclGinProxyGfdSrcHandle].srcHandle.srcHandle = (uint64_t)srcHandle;  // 源窗口句柄（lkey 来源）
  }

  // [qword 3] dstOff：目标窗口内偏移
  gfd->qword[ncclGinProxyGfdDstOff].dstOff.dstOff = (uint64_t)dstOff;
  // [qword 4] dstHandle：目标窗口句柄（rkey 来源）
  gfd->qword[ncclGinProxyGfdDstHandle].dstHandle.dstHandle = (uint64_t)dstHandle;

  // [qword 5] completion：counterId + signalId + signalValLow
  gfd->qword[ncclGinProxyGfdCompletion].completion.counterId = counterId;  // 本地计数器 ID（0 表示无）
  gfd->qword[ncclGinProxyGfdCompletion].completion.signalId = signalId;  // 远端信号 ID（0 表示无）

  // The signal value is split between two qwords, as the signal value is a full 64 bits
  // signalVal 拆分存入 qword[5]和 qword[6]：
  //   qword[5].completion.signalValLow  = bits [0:15]
  //   qword[6].signalVal.signalValLow2  = bits [16:31]
  //   qword[6].signalVal.signalValHigh  = bits [32:63]
  gfd->qword[ncclGinProxyGfdCompletion].completion.signalValLow = (uint16_t)signalVal;
  gfd->qword[ncclGinProxyGfdSignalVal].signalVal.signalValLow2 = (uint16_t)(signalVal >> 16);
  gfd->qword[ncclGinProxyGfdSignalVal].signalVal.signalValHigh = (uint32_t)(signalVal >> 32);
}

// ---------------------------------------------------------------------------
// constructProxyOp：根据操作参数构造 GFD 的 op bitmask
// ---------------------------------------------------------------------------
// op 是一个 16-bit bitmask，编码到 GFD qword[7].headerExt.op 字段。
// CPU proxy 读取此 bitmask 来决定如何处理这个 GFD。
//
// bitmask 组合规则（各 bit 可 OR 叠加）：
//   ncclGinProxyOpPut(0x01)           — 数据写入（RDMA Write）
//   ncclGinProxyOpWithInline(0x02)    — 源数据内嵌在 GFD 中（不走 DMA）
//   ncclGinProxyOpWithCounter(0x04)   — 完成后递增本地 counter
//   ncclGinProxyOpWithSignalInc(0x08) — 完成后对端 signal++
//   ncclGinProxyOpWithSignalAdd(0x10) — 完成后对端 signal += val
//   ncclGinProxyOpVASignal(0x20)      — VA 信号写入（无 PUT 载荷）
//   ncclGinProxyOpGet(0x40)           — 数据读取（RDMA Read）
//   ncclGinProxyOpFlush(0x80)         — 刷新/同步操作
//
// 互斥关系：
//   Get / Flush / VASignal 三者互斥，设置后直接 return
//   Put 可与 Inline / Counter / SignalInc / SignalAdd 叠加
//
// 典型 put+signal+counter 场景：
//   op = Put | WithSignalInc | WithCounter = 0x01 | 0x08 | 0x04 = 0x0D
__device__ __forceinline__ void constructProxyOp(ncclGinProxyOp_t& op, bool isGet, bool isFlush, bool hasInline,
                                                 ncclGinSignalType signalType, ncclGinSignalOp_t signalOp,
                                                 bool hasCounter) {
  op = (ncclGinProxyOp_t)(0);
  if (isGet) {
    op = static_cast<ncclGinProxyOp_t>(static_cast<uint16_t>(op) | static_cast<uint16_t>(ncclGinProxyOpGet));
    return;
  }

  if (isFlush) {
    op = static_cast<ncclGinProxyOp_t>(static_cast<uint16_t>(op) | static_cast<uint16_t>(ncclGinProxyOpFlush));
    return;
  }

  if (signalType != NCCL_GIN_SIGNAL_TYPE_NONE) {
    switch (signalOp) {
      case ncclGinSignalInc:
        op = static_cast<ncclGinProxyOp_t>(static_cast<uint16_t>(op) | static_cast<uint16_t>(ncclGinProxyOpWithSignalInc));
        break;
      case ncclGinSignalAdd:
        op = static_cast<ncclGinProxyOp_t>(static_cast<uint16_t>(op) | static_cast<uint16_t>(ncclGinProxyOpWithSignalAdd));
        break;
      default:
        __builtin_unreachable();
    }
  }
  if (signalType == NCCL_GIN_SIGNAL_TYPE_VA) {
    op = static_cast<ncclGinProxyOp_t>(static_cast<uint16_t>(op) | static_cast<uint16_t>(ncclGinProxyOpVASignal));
    return;
  }
  op = static_cast<ncclGinProxyOp_t>(static_cast<uint16_t>(op) | static_cast<uint16_t>(ncclGinProxyOpPut));
  if (hasInline)
    op = static_cast<ncclGinProxyOp_t>(static_cast<uint16_t>(op) |
                                       static_cast<uint16_t>(ncclGinProxyOpWithInline));
  if (hasCounter)
    op = static_cast<ncclGinProxyOp_t>(static_cast<uint16_t>(op) |
                                       static_cast<uint16_t>(ncclGinProxyOpWithCounter));
}

// ---------------------------------------------------------------------------
// [2.30.3 新增] get：GIN_PROXY 后端的 RDMA Read 函数
// ---------------------------------------------------------------------------
// 与 put 对称，构建并提交 GFD 完成一次远端内存读取。
// 大数据同样按 DataChunkSize (1GB) 分片传输。
// 注意方向：remoteWnd/remoteOff 是源（远端），localWnd/localOff 是目标（本地）
template <typename Coop>
NCCL_DEVICE_INLINE void get(Coop coop, ncclGinProxyGpuCtx_t* proxyCtx,
                            int peer, ncclGinWindow_t remoteWnd, size_t remoteOff,
                            ncclGinWindow_t localWnd, size_t localOff, size_t bytes,
                            ncclGinProxyGfd_t* desc) {
  using nccl::gin::proxy::DataChunkSize;
  while (bytes > 0) {
    size_t sendSize = min(bytes, DataChunkSize);
    ncclGinProxyOp_t op;
    constructProxyOp(op, /*isGet*/true, /*isFlush*/false, /*hasInline*/false, NCCL_GIN_SIGNAL_TYPE_NONE, ncclGinSignalInc, /*hasCounter*/false);
    nccl::gin::proxy::buildGfd(desc, op, /*srcVal*/0, /*hasInline*/false, remoteOff, remoteWnd,
                               localOff, localWnd, sendSize, /*counterId*/0, /*signalId*/0,
                               /*signalVal*/0, nullptr, 0);
    nccl::gin::proxy::postGfd<Coop>(coop, proxyCtx, desc, peer);
    bytes -= sendSize;
    remoteOff += sendSize;
    localOff += sendSize;
  }
}

// ---------------------------------------------------------------------------
// put：GIN_PROXY 后端的核心发送函数
// ---------------------------------------------------------------------------
// 构建并提交 GFD，完成一次远端内存写入。
// 小数据可以内嵌（hasInline=true），大数据必须用 srcHandle/srcOff 指向已注册的内存窗口。
//
// 为什么 VA 信号必须拆为两个 GFD：
//   GFD 内只有一岁分配给源地址（qword[1][2]），当这两个 slot 已用于 VA 信号地址时，沒有地方存源地址
//   因此新版本需要两次 put：第一次只做 PUT（不带信号），第二次只做 VA 信号
//
// 大数据分片传输：
//   当 bytes > chunkSize（1G）时，拆分为多个 GFD，每次不带信号/计数器，最后一个小片才带
template <typename Coop, typename T>
NCCL_DEVICE_INLINE void put(Coop coop, ncclGinProxyGfd_t* gfd, ncclGinProxyGpuCtx_t* proxyCtx,
                            int peer, ncclGinWindow_t dstWnd, size_t dstOff, T srcVal,
                            bool hasInline, ncclGinWindow_t srcWnd, size_t srcOff, size_t bytes,
                            ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                            uint64_t signalVal, bool hasCounter, ncclGinCounter_t counterId,
                            cuda::thread_scope required, cuda::thread_scope given) {
  // 如果调用者声称的内存可见范围不包含 system，需要补一次 release fence
  // 确保之前对源缓冲区的写入对 IB HCA 可见（IB HCA 是 system-scope 设备）
  if ((int)given > (int)cuda::thread_scope_system) {
    cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
  }

  // 大数据分片处理：单片 GFD 不能超过 DataChunkSize (1GB)
  // 将大 PUT 拆分为多个小 GFD，前面的都不带信号和计数器
  using nccl::gin::proxy::DataChunkSize;
  while (bytes > DataChunkSize) {
    ncclGinProxyOp_t op;
    // 中间片：无信号、无计数器，纯数据传输
    constructProxyOp(op, /*isGet*/false, /*isFlush*/false, /*hasInline*/false, NCCL_GIN_SIGNAL_TYPE_NONE, signalOp, /*hasCounter*/false);
    nccl::gin::proxy::buildGfd(gfd, op, /*srcVal*/0, /*hasInline*/false, srcOff, srcWnd,
                               dstOff, dstWnd, DataChunkSize, /*counterId*/0, /*signalId*/0,
                               /*signalVal*/0, nullptr, 0);
    nccl::gin::proxy::postGfd<Coop>(coop, proxyCtx, gfd, peer);
    bytes -= DataChunkSize;
    srcOff += DataChunkSize;
    dstOff += DataChunkSize;
  }

  // 最后一片（或完整的小 PUT）：目标信号类型决定
  ncclGinSignalType putSignalType;
  uint64_t putSignalVal;
  ncclGinSignal_t putSignalId;
  switch (signal.type) {
    case NCCL_GIN_SIGNAL_TYPE_INDEXED:
      // Indexed 信号：可以内嵌在最后一个 PUT 的 GFD 中
      putSignalType = NCCL_GIN_SIGNAL_TYPE_INDEXED;
      putSignalVal = signalVal;
      putSignalId = signal.indexedSignal.signalId;
      break;
    case NCCL_GIN_SIGNAL_TYPE_VA: // VA signals must be in a separate GFD. Use no signal during first put.
    case NCCL_GIN_SIGNAL_TYPE_NONE:
      // VA 信号需要单独 GFD，所以在数据 PUT 的 GFD 中不付带信号
      putSignalType = NCCL_GIN_SIGNAL_TYPE_NONE;
      putSignalVal = 0;
      putSignalId = 0;
      break;
    default:
      __builtin_unreachable();
  }
  if (hasInline || hasCounter || srcWnd != nullptr || putSignalType != NCCL_GIN_SIGNAL_TYPE_NONE) {
    // 有内嵌小值 / 计数器 / 源窗口 / Indexed 信号 之一：需要提交 GFD
    ncclGinProxyOp_t op;
    constructProxyOp(op, /*isGet*/false, /*isFlush*/false, hasInline, putSignalType, signalOp, hasCounter);
    nccl::gin::proxy::buildGfd(gfd, op, srcVal, hasInline, srcOff, srcWnd, dstOff, dstWnd, bytes,
                              hasCounter ? counterId : 0, putSignalId, putSignalVal, nullptr, 0);
    nccl::gin::proxy::postGfd<Coop>(coop, proxyCtx, gfd, peer);
  }

  // Handle additional GFD for VA signals.
  // VA 信号需要单独提交一个附加的 GFD（只包含信号，无 PUT 载荷）
  if (signal.type == NCCL_GIN_SIGNAL_TYPE_VA) {
    ncclGinProxyOp_t op;
    constructProxyOp(op, /*isGet*/false, /*isFlush*/false, /*hasInline*/false, NCCL_GIN_SIGNAL_TYPE_VA, signalOp, /*hasCounter*/false);
    nccl::gin::proxy::buildGfd(gfd, op, /*srcVal*/0, /*hasInline*/false, 0, nullptr,
                               0, nullptr, 0, 0, 0, signalVal, signal.vaSignal.signalWindow, signal.vaSignal.signalOffset);
    nccl::gin::proxy::postGfd<Coop>(coop, proxyCtx, gfd, peer);
  }
}
}  // namespace proxy
}  // namespace gin
}  // namespace nccl

// [2.30.3 新增] ncclGinApi_Get：RDMA Read API 特化
template <>
struct ncclGinApi_Get<NCCL_NET_DEVICE_GIN_PROXY> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t remoteWin, size_t remoteOff,
                                      ncclGinWindow_t localWin, size_t localOff, size_t bytes,
                                      bool hasDescriptor, ncclGinDescriptorSmem* descriptor,
                                      uint32_t optFlags) {
    ncclGinProxyGfd_t tmpDesc;
    ncclGinProxyGfd_t* desc = hasDescriptor ? (ncclGinProxyGfd_t*)descriptor : &tmpDesc;
    ncclGinProxyGpuCtx_t* proxyCtx = &((ncclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    nccl::gin::proxy::get<Coop>(coop, proxyCtx, peer, remoteWin, remoteOff, localWin, localOff, bytes,desc);
  }
};

// [2.30.3 新增] ncclGinApi_FlushAsync：异步 Flush 第一阶段
// 不阻塞等待，只记录当前 pi 到 outRequest，后续通过 Wait 等待完成
template<>
struct ncclGinApi_FlushAsync<NCCL_NET_DEVICE_GIN_PROXY> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, int peer, ncclGinRequest_t* outRequest, uint32_t optFlags) {
    using nccl::utility::loadConst;
    ncclGinCpuProxyRequest* req = reinterpret_cast<ncclGinCpuProxyRequest*>(outRequest);
    req->peer = peer;
    ncclGinProxyGpuCtx_t* proxyCtx = &((ncclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    cuda::atomic_ref<uint32_t, cuda::thread_scope_system> pi(loadConst(&proxyCtx->pis)[peer]);
    req->nextGfdIdx = pi.load(cuda::memory_order_relaxed);
  }
};
// [2.30.3 新增] ncclGinApi_Wait：等待异步操作完成
// FlushAsync+Wait 构成 split-phase flush：先等待已提交 GFD 被消费，
// 再发一个 flush GFD 到自身 rank 确保 get 数据对 GPU 可见
template <>
struct ncclGinApi_Wait<NCCL_NET_DEVICE_GIN_PROXY> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinRequest_t& request, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, cuda::memory_order ord, uint32_t* abortFlag) {
    ncclGinCpuProxyRequest& req = reinterpret_cast<ncclGinCpuProxyRequest&>(request);
    ncclGinProxyGpuCtx_t* proxyCtx = &((ncclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    nccl::gin::proxy::waitForGfdComplete(proxyCtx, req.peer, req.nextGfdIdx, cuda::memory_order_relaxed, abortFlag);

    // Ensure gets are visible by issuing a local flush
    ncclGinProxyGfd_t tmpGfd;
    ncclGinProxyGfd_t* desc = hasDescriptor ? (ncclGinProxyGfd_t*)descriptor : &tmpGfd;
    ncclGinProxyOp_t op;
    uint32_t flushGfdIdx;
    nccl::gin::proxy::constructProxyOp(op, /*isGet*/false, /*isFlush*/true, /*hasInline*/false, NCCL_GIN_SIGNAL_TYPE_NONE, ncclGinSignalInc, /*hasCounter*/false);
    nccl::gin::proxy::buildGfd(desc, op, /*srcVal*/0, /*hasInline*/false, 0, nullptr,
                               0, nullptr, 0, 0, 0, 0, nullptr, 0);
    nccl::gin::proxy::postGfd(ncclCoopThread(), proxyCtx, desc, ctx.rank, &flushGfdIdx);
    nccl::gin::proxy::waitForGfdComplete(proxyCtx, ctx.rank, flushGfdIdx + 1, ord, abortFlag);
  }
};

template <>
struct ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_PROXY> {
  NCCL_DEVICE_INLINE static uint64_t* call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    ncclGinProxyGpuCtx_t* proxyCtx = &((ncclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    uint64_t* counter = nccl::utility::loadConst(&proxyCtx->counters) + counterId;
    return counter;
  }
};

template <>
struct ncclGinApi_ResetCounter<NCCL_NET_DEVICE_GIN_PROXY> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    ncclGinProxyGpuCtx_t* proxyCtx = &((ncclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    uint64_t* counter = nccl::utility::loadConst(&proxyCtx->counters) + counterId;
    *counter = 0;
  }
};

template <>
struct ncclGinApi_GetSignalPtr<NCCL_NET_DEVICE_GIN_PROXY> {
  NCCL_DEVICE_INLINE static uint64_t* call(ncclGinCtx ctx, ncclGinSignal_t signalId) {
    ncclGinProxyGpuCtx_t* proxyCtx = &((ncclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    uint64_t* signal = nccl::utility::loadConst(&proxyCtx->signals) + signalId;
    return signal;
  }
};

template <>
struct ncclGinApi_ResetSignal<NCCL_NET_DEVICE_GIN_PROXY> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinSignalDescriptor signal) {
    ncclGinProxyGpuCtx_t* proxyCtx = &((ncclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    if (signal.type == NCCL_GIN_SIGNAL_TYPE_VA) {
      uint64_t* signalPtr = (uint64_t*)ncclGetLocalPointer(signal.vaSignal.ncclWindow, signal.vaSignal.signalOffset);
      *signalPtr = 0;
    } else {
      nccl::utility::loadConst(&proxyCtx->signals)[signal.indexedSignal.signalId] = 0;
    }
  }
};

// ---------------------------------------------------------------------------
// ncclGinApi_Flush<GIN_PROXY>：等待所有 peer 的 GFD 队列被 CPU 消费完毕
// ---------------------------------------------------------------------------
// 多线程并行等待：每个线程负责一组 peer（按 stride 分配）。
// 调用 proxy::flush 等待 ci >= pi，即该 peer 的所有已提交 GFD 都已被 CPU 处理。
// 注意：只保证 CPU 已提交 IB 操作，不保证远端已收到数据。
template <>
struct ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_PROXY> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, cuda::memory_order ord, uint32_t* abortFlag) {
    ncclGinProxyGpuCtx_t* proxyCtx = &((ncclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    // wait for all GFDs to be completed
#pragma unroll 1
    for (int pe = coop.thread_rank(); pe < ctx.nRanks; pe += coop.size()) {
      nccl::gin::proxy::flush(proxyCtx, pe, ord, abortFlag);
    }
  }
};

// ---------------------------------------------------------------------------
// ncclGinApi_Put<GIN_PROXY>：Proxy 后端的 Put API 入口
// ---------------------------------------------------------------------------
// 从 ncclGinCall<ncclGinApi_Put> 模板分发到此特化版本。
// 职责：准备 GFD 缓冲区和 proxy context，然后调用 nccl::gin::proxy::put。
//
// 关键参数映射：
//   ctx.handle     → ncclGinProxyGpuCtx_t 数组（每个 context 一个）
//   ctx.contextId  → 当前 block 使用的 context 索引（round-robin）
//   hasDescriptor  → true 时复用调用者提供的 SMEM descriptor（避免栈上分配 128B）
//                    false 时在栈上创建 tmpDesc（128B GFD）
//   srcWin/srcOff  → 源窗口句柄和偏移（对应 IB lkey + local VA）
//   dstWin/dstOff  → 目标窗口句柄和偏移（对应 IB rkey + remote VA）
//   signal         → 信号描述符（Indexed / VA / None）
//   hasCounter     → 完成后是否递增本地 counter
//
// 注意：此入口固定 srcVal=0, hasInline=false（非内嵌模式）
//       内嵌小值走 ncclGinApi_PutValue<GIN_PROXY>
template <>
struct ncclGinApi_Put<NCCL_NET_DEVICE_GIN_PROXY> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, bool hasWins,
                                      ncclGinWindow_t dstWin, size_t dstOff, ncclGinWindow_t srcWin,
                                      size_t srcOff, size_t bytes,
                                      ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasCounter,
                                      ncclGinCounter_t counterId, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given,
                                      uint32_t optFlags = ncclGinOptFlagsDefault) {
    ncclGinProxyGfd_t tmpDesc;
    // 如果调用者提供了 SMEM descriptor，直接复用；否则栈上分配
    ncclGinProxyGfd_t* desc = hasDescriptor ? (ncclGinProxyGfd_t*)descriptor : &tmpDesc;
    // 从 ctx.handle 中按 contextId 索引获取当前 block 的 proxy context
    ncclGinProxyGpuCtx_t* proxyCtx = &((ncclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    // 调用核心 put 函数：srcVal=0, hasInline=false（标准 DMA 模式）
    nccl::gin::proxy::put<Coop, uint64_t>(coop, desc, proxyCtx, peer, dstWin, dstOff, 0, false,
                                          srcWin, srcOff, bytes, signal, signalOp, signalOpArg,
                                          hasCounter, counterId, required, given);
  }
};

// ---------------------------------------------------------------------------
// ncclGinApi_PutValue<GIN_PROXY>：内嵌小值 Put API 入口
// ---------------------------------------------------------------------------
// 与 ncclGinApi_Put 的区别：
//   - hasInline=true：将 srcVal（最多 sizeof(T) 字节）直接编码进 GFD qword[1][2]
//   - srcWnd=nullptr, srcOff=0：不需要源窗口（数据就在 GFD 里面）
//   - bytes=sizeof(T)：传输大小 = 值的字节数
//   - hasCounter=false：PutValue 不支持 counter（语义上是轻量级小值写入）
//
// 典型用途：写入 8 字节 flag/signal 到远端（如 put_signal 的 signal 部分）
// CPU proxy 收到后，从 GFD inline 字段提取值，通过 IB RDMA Write 写到远端
template <>
struct ncclGinApi_PutValue<NCCL_NET_DEVICE_GIN_PROXY> {
  template <typename Coop, typename T>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin,
                                      size_t dstOff, T srcVal,
                                      ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given,
                                      uint32_t optFlags = ncclGinOptFlagsDefault) {
    ncclGinProxyGfd_t tmpDesc;
    ncclGinProxyGfd_t* desc = hasDescriptor ? (ncclGinProxyGfd_t*)descriptor : &tmpDesc;
    ncclGinProxyGpuCtx_t* proxyCtx = &((ncclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    // hasInline=true, srcWnd=nullptr, srcOff=0, bytes=sizeof(T), hasCounter=false
    nccl::gin::proxy::put<Coop, T>(coop, desc, proxyCtx, peer, dstWin,
                                   dstOff, srcVal, true, nullptr, 0, sizeof(T), signal,
                                   signalOp, signalOpArg, false, 0, required, given);
  }
};

#endif
