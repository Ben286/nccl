/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <assert.h>
#include "nccl.h"
#include "gin/gin_host.h"
#include "alloc.h"
#include "checks.h"
#include "gdrwrap.h"
#include "nccl_device/gin/proxy/gin_proxy_device_host_common.h"
#include "compiler.h"

NCCL_PARAM(GinProxyQueueSize, "GIN_PROXY_QUEUE_SIZE", -1);
extern int64_t ncclParamIbDataDirect();
extern int64_t ncclParamDmaBufEnable();

// ---------------------------------------------------------------------------
// ginProxyGfdState：CPU 侧每个 GFD 槽位的执行状态
// ---------------------------------------------------------------------------
// 与 GFD 环形队列等长（nRanks * queueSize 个），一一对应。
// 当 CPU proxy 从队列中取出一个 GFD 并提交 IB 操作后，用这个结构跟踪完成状态。
//
// 生命周期：
//   proxyGinPollGfd → 填写 op/counterId/done=0/request=NULL
//   proxyGinProcessGfd → 提交 IB 操作，填写 request
//   proxyGinPollCompletions → 调用 ginBackend->test(request) 检查完成
//                           → done=1 后更新 counter 并推进 CI
struct ginProxyGfdState {
  ncclGinProxyOp_t op;   // 从 GFD 解析出的操作 bitmask（Put/Get/Flush/Signal 等）
  uint16_t counterId;    // 完成后需要递增的计数器 ID（0 = 无）
  int done;              // IB 操作是否完成（0=进行中, 1=已完成）
  void *request;         // IB Verbs 异步请求句柄（ginBackend->test 的入参）
};

// ---------------------------------------------------------------------------
// ginProxyHostGpuCtx：CPU 侧每个 context 的运行时状态
// ---------------------------------------------------------------------------
// 与 GPU 侧的 ncclGinProxyGpuCtx_t 对应（一个 context 一个），但这个结构只在 CPU 使用。
// 包含了 CPU proxy 轮询和处理 GFD 所需的所有状态。
//
// 三个关键索引（每个 rank 一个）：
//   pis[rank]        — GPU 维护的生产者索引（CPU 不写，只在释放时 free）
//   sis[rank]        — CPU "已看到" 索引（CPU 已轮询到但可能还未完成的 GFD 上界）
//   cisShadow[rank]  — CPU 本地的消费者索引影子（已完成的连续 GFD 上界）
//   cis[rank]        — 与 GPU 共享的消费者索引（cisShadow 写入此处通知 GPU 释放槽位）
//
// 推进顺序：GPU 写 pi → CPU 轮询到 flag=1 时 si++ → IB 完成后 cisShadow++ → cis = cisShadow
struct ginProxyHostGpuCtx {
  int contextId;       // 本 context 的索引（0..nContexts-1）
  size_t queueSize;    // 单个 peer 的 GFD 队列容量（2 的幂次）

  // size = nRanks * queueSize，GFD 环形队列（pinned host memory）
  // GPU 通过 __stwt 写入，CPU 轮询读取
  ncclGinProxyGfd_t *queues;
  void *cisGdrHandle;  // cis 的 GDR 映射句柄（如果 cis 在 GPU 内存上）
  // 生产者索引：GPU 独占写入，CPU 只在销毁时 free
  uint32_t* pis;
  // 消费者索引：CPU 写入，GPU 读取（用于 flow control）
  // 可能在 GPU 内存（支持 GDR 时）或 CPU 内存
  uint32_t *cis;
  // cisShadow：cis 的 CPU 本地影子，减少对 WC/uncacheable 内存的读次数
  // 只在连续完成的 GFD 前沿推进时才写回 cis[rank]
  uint32_t *cisShadow;
  // sis（Seen Indices）：CPU 已轮询到的 GFD 上界
  // sis > cisShadow 时说明有 GFD 已提交 IB 操作但尚未完成
  uint32_t *sis;

  // GFD 执行状态数组（与 queues 等长：nRanks * queueSize）
  struct ginProxyGfdState *states;
  // 内嵌小值缓冲区（与 queues 等长）
  // PutValue 的 inline 数据从 GFD 解析后存入此处，再通过 IB RDMA Write 发出
  uint64_t *inlines;
  // inlines 在 GIN plugin 中注册的 MR handle（提供 lkey）
  void *inlinesMhandle;
  void *inlinesGinHandle;
};

// ---------------------------------------------------------------------------
// ginProxyCtx：CPU proxy 全局上下文
// ---------------------------------------------------------------------------
// 整个 proxy 子系统的顶层状态，包含所有 context 的公共资源。
// 在 ncclGinProxyCreateContext 中分配，在 ncclGinProxyProgress 中使用。
struct ginProxyCtx {
  void *collComm;        // GIN plugin 的 collective communicator 句柄
  int nRanks;            // rail 内的 rank 总数
  ncclNetDeviceHandle_t *devHandle;  // GPU 端可见的设备句柄（含 backend type + GPU ctx 指针）

  // 每个 context 的 CPU 侧状态（数组，长度 = nContexts）
  struct ginProxyHostGpuCtx *hostGpuCtx;

  void *countersGdrHandle;   // counters 的 GDR 映射句柄
  uint64_t *counters;        // CPU 可访问的计数器数组（所有 context 共享连续空间）
  uint64_t *countersDev;     // GPU 端计数器地址（可能与 counters 指向同一物理内存）
  CUmemGenericAllocationHandle signalsCumemhandle;  // signals GPU 内存的 cuMem 句柄
  void *signalsMhandle;      // signals 的 MR handle（IB 注册）
  void *signalsGinHandle;    // signals 的 GIN plugin handle
  uint64_t *signalsDev;      // GPU 端信号数组地址（RDMA 可访问）
  bool hasError;             // 错误标志：任何 IB 操作失败后置 true
  int nContexts;             // context 总数
  int nCountersPerContext;   // 每个 context 的计数器数量
  int nSignalsPerContext;    // 每个 context 的信号数量
  void* ginCtx;              // GIN plugin 层的 context（传给 ginBackend->iput 等）
};

static ncclGin_t* ginBackend;

static ncclResult_t getDmaBufFd(void *addr, size_t length, int *fd,
                                bool forceNonDataDirect = false) {
  if (ncclParamDmaBufEnable() == 0) return ncclInvalidUsage;

#if CUDA_VERSION >= 11070
  static size_t hostPageSize = ncclOsGetPageSize();
  size_t alignedSize = length;
  ALIGN_SIZE(alignedSize, hostPageSize);

#if CUDA_VERSION >= 12080
  if (ncclParamIbDataDirect() && !forceNonDataDirect) {
    CUresult status = pfn_cuMemGetHandleForAddressRange(
      (void *)fd, (CUdeviceptr)addr, alignedSize, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
      CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE);
    if (status == CUDA_SUCCESS) return ncclSuccess;
  }
#endif
  CUresult status = pfn_cuMemGetHandleForAddressRange((void *)fd, (CUdeviceptr)addr, alignedSize,
                                                      CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
  if (status == CUDA_SUCCESS) return ncclSuccess;
#endif

  return ncclInvalidUsage;
}

// ---------------------------------------------------------------------------
// proxyGinPollCompletions：检查已提交的 IB 操作是否完成，并推进 CI
// ---------------------------------------------------------------------------
// 遍历所有 rank 的 [cisShadow, sis) 区间：这些是“已轮询并提交 IB，但尚未完成”的 GFD。
//
// 对每个未完成的 GFD：
//   1. 调用 ginBackend->test(request, &done) 检查 IB 完成
//   2. done=1 后，如果 GFD 有 WithCounter 标志，递增对应的 counter
//   3. 如果当前 GFD 刚好在 cisShadow 前沿（即连续完成的最小索引），
//      推进 cisShadow++ 并写入 cis[rank]，通知 GPU 可以释放槽位
//
// 注意：允许“空洞”——如果 GFD[i] 未完成但 GFD[i+1] 已完成，
//   CI 不会跳过 i 前进，必须等 i 完成后才能推进。
//   这保证了发向同一 peer 的 GFD 按顺序完成（IB ordering）。
static ncclResult_t proxyGinPollCompletions(void *collComm,
                                            struct ginProxyCtx *ctx,
                                            struct ginProxyHostGpuCtx *hostGpuCtx) {
  for (int targetRank = 0; targetRank < ctx->nRanks; targetRank++) {
    // 遍历 [cisShadow, sis) 区间：已提交但未完成的 GFD
    for (uint32_t i = hostGpuCtx->cisShadow[targetRank]; i < hostGpuCtx->sis[targetRank]; i++) {
      uint32_t idx = i & (hostGpuCtx->queueSize - 1);
      struct ginProxyGfdState *state =
        &hostGpuCtx->states[targetRank * hostGpuCtx->queueSize + idx];
      // no need to poll if already done
      if (!state->done) {
        ncclResult_t res = ginBackend->test(collComm, state->request, &state->done);
        if (res != ncclSuccess) {
            ctx->hasError = true;
            WARN("Error on GFD test %d - stateIdx: %lu, request: %p", res, state - hostGpuCtx->states, state->request);
            return res;
        }
        if (state->done) {
          TRACE(NCCL_NET, "GFD completed - contextId: %d, stateIdx: %lu, request: %p", hostGpuCtx->contextId, state - hostGpuCtx->states,
                state->request);
          // update the counter specified in the GFD
          if (state->op & ncclGinProxyOpWithCounter) {
            // 用原子 load/store 保证 volatile 语义（编译器不会优化掉）
            // GPU kernel 不允许在有未完成操作时 reset counter，所以加法本身不需要原子
            int contextId = hostGpuCtx->contextId;
            uint64_t* counterPtr = &ctx->counters[contextId * ctx->nCountersPerContext + state->counterId];
            uint64_t oldValue = COMPILER_ATOMIC_LOAD(counterPtr, std::memory_order_relaxed);
            COMPILER_ATOMIC_STORE(counterPtr, oldValue + 1,
                              std::memory_order_relaxed);
            TRACE(NCCL_NET, "Updated counter %d to %ld for context %d", state->counterId,
                  *counterPtr, contextId);
          }
        }
      }
      // 允许“空洞”解决：只在当前 GFD 是连续完成的前沿时才推进 CI
      // 例如 GFD[3] 完成但 GFD[2] 未完成，CI 停在 2 不动，等 GFD[2] 完成后一口气推到 4
      if (state->done && i == hostGpuCtx->cisShadow[targetRank]) {
        // 写入 cis[rank] 通知 GPU 该槽位已可重用
        COMPILER_ATOMIC_STORE(&hostGpuCtx->cis[targetRank], ++hostGpuCtx->cisShadow[targetRank],
                          std::memory_order_relaxed);
        TRACE(NCCL_NET, "Updated cis[%u] to %u for context %d", targetRank, hostGpuCtx->cisShadow[targetRank], hostGpuCtx->contextId);
      }
    }
  }

  return ncclSuccess;
}

// 从 GFD 中重组 64-bit signalVal（分散在 qword[5] 和 qword[6] 中）
static inline uint64_t extractSignalVal(ncclGinProxyGfd_t *gfd) {
  uint64_t signalVal = gfd->qword[ncclGinProxyGfdCompletion].completion.signalValLow;
  signalVal |= (uint64_t)gfd->qword[ncclGinProxyGfdSignalVal].signalVal.signalValLow2 << 16;
  signalVal |= (uint64_t)gfd->qword[ncclGinProxyGfdSignalVal].signalVal.signalValHigh << 32;
  return signalVal;
}

// 从 GFD qword[7].headerExt 中提取 op bitmask
static ncclGinProxyOp_t extractOp(ncclGinProxyGfd_t *gfd) {
  return (ncclGinProxyOp_t)gfd->qword[ncclGinProxyGfdHeaderExt].headerExt.op;
}

// ---------------------------------------------------------------------------
// proxyGinPollGfd：CPU 轮询指定 rank 的 GFD 队列，检测新的 GFD
// ---------------------------------------------------------------------------
// 返回 1 = 取到新 GFD，0 = 队列为空。
//
// 算法流程：
//   1. 读取 queue[idx].qword[0] 的 flag 位
//      flag=0 → 无新 GFD，返回 0
//      flag=1 → GPU 已开始写入，继续
//   2. 复制第一个 qword（已确认有效）
//   3. 循环等待剩余 15 个 qword 的 flag=1（自旋等待）
//      为什么要等：GPU __stwt 是 16 字节 uint4 写入，先写的 qword 先到达
//      后写的 qword 可能稍微延迟，必须等所有 flag 都为 1 才算完整 GFD
//   4. 复制到本地 gfd 结构
//   5. 复位队列中该槽位（所有 qword 清零），确保下一轮不会重复处理
//   6. 填写 state 结构（op/counterId/done=0/request=NULL）
//   7. sis[rank]++ — 标记 CPU 已看到这个 GFD
static int proxyGinPollGfd(struct ginProxyCtx *ctx, ginProxyHostGpuCtx *hostGpuCtx, int targetRank,
                           ncclGinProxyGfd_t *gfd, struct ginProxyGfdState **state) {
  ncclGinProxyGfd_t *q = hostGpuCtx->queues + targetRank * hostGpuCtx->queueSize;
  uint32_t idx = hostGpuCtx->sis[targetRank] & (hostGpuCtx->queueSize - 1);
  ncclGinProxyQword_t qword;
  // 步骤 1：原子读 qword[0]，检查 flag
  COMPILER_ATOMIC_LOAD_DEST(&q[idx].qword[ncclGinProxyGfdHeader].raw, &qword.raw, std::memory_order_relaxed);
  if (qword.flag.v == 0) {
    return 0;  // 无新 GFD
  }

  // 步骤 2：第一个 qword 已确认有效，复制它
  gfd->qword[ncclGinProxyGfdHeader] = q[idx].qword[ncclGinProxyGfdHeader];
  // 步骤 3：自旋等待剩余 15 个 qword 的 flag=1，然后复制
  for (int k = 1; k < ncclGinProxyGfdQwords; k++) {
    do {
      COMPILER_ATOMIC_LOAD_DEST(&q[idx].qword[k].raw, &qword.raw, std::memory_order_relaxed);
    } while (qword.flag.v == 0);
    gfd->qword[k] = qword;
  }
  // 步骤 4 已在循环中完成：所有 qword 已复制到本地 gfd

  // 步骤 5：复位队列中该槽位（所有 qword 清零），避免下一轮误读
  for (int k = 0; k < ncclGinProxyGfdQwords; k++) {
    COMPILER_ATOMIC_STORE(&q[idx].qword[k].raw, 0ULL, std::memory_order_relaxed);
  }

  // 步骤 6：填写 state 结构，为后续 IB 操作跟踪做准备
  uint32_t stateIdx = targetRank * hostGpuCtx->queueSize + idx;
  *state = &hostGpuCtx->states[stateIdx];
  (*state)->op = extractOp(gfd);
  (*state)->counterId = gfd->qword[ncclGinProxyGfdCompletion].completion.counterId;
  (*state)->done = 0;
  (*state)->request = NULL;

  TRACE(NCCL_NET,
        "GFD on context %d to target PE %d raw idx: %u, idx: %u - op: %#lx, size: %lu, srcOff: %lu, dstOff: %lu, "
        "srcHandle: %lu, dstHandle: %lu, counterId: %u, signalId: %u, stateIdx: %u",
        hostGpuCtx->contextId, targetRank, hostGpuCtx->sis[targetRank], idx, extractOp(gfd),
        gfd->qword[ncclGinProxyGfdHeader].header.size,
        gfd->qword[ncclGinProxyGfdSrcOff].srcOff.srcOff,
        gfd->qword[ncclGinProxyGfdDstOff].dstOff.dstOff,
        gfd->qword[ncclGinProxyGfdSrcHandle].srcHandle.srcHandle,
        gfd->qword[ncclGinProxyGfdDstHandle].dstHandle.dstHandle,
        gfd->qword[ncclGinProxyGfdCompletion].completion.counterId,
        gfd->qword[ncclGinProxyGfdCompletion].completion.signalId, stateIdx);

  // 步骤 7：推进 sis（Seen Index），表示 CPU 已轮询到这个 GFD
  hostGpuCtx->sis[targetRank]++;

  return 1;
}

// 将 GFD 的 op bitmask 中的信号操作位转换为 GIN plugin 的 signalOp 枚举
// 返回 -1 表示无信号操作
static int mapGfdOpToSignalOp(ncclGinProxyGfd_t *gfd) {
  ncclGinProxyOp_t op = extractOp(gfd);
  uint8_t signalOp = op & (ncclGinProxyOpWithSignalInc | ncclGinProxyOpWithSignalAdd);
  switch (signalOp) {
    case ncclGinProxyOpWithSignalInc:
      return NCCL_NET_SIGNAL_OP_INC;
    case ncclGinProxyOpWithSignalAdd:
      return NCCL_NET_SIGNAL_OP_ADD;
    default:
      return -1;
  }
}

// ---------------------------------------------------------------------------
// proxyGinProcessGfd：解析 GFD 并提交对应的 IB Verbs 操作
// ---------------------------------------------------------------------------
// 这是 CPU proxy 的核心处理函数，将 GPU 端构建的 GFD 翻译为实际的 IB 操作。
//
// 根据 GFD 的 op bitmask 分类处理（互斥分支）：
//   1. VASignal：只写信号到远端 VA 地址，无数据 PUT
//   2. Get：RDMA Read，从远端读取数据到本地
//   3. Flush：确保之前的 Get 数据对 GPU 可见
//   4. Put：RDMA Write，将数据写到远端，可附带信号
//
// Put 的两种数据源：
//   a. WithInline：从 GFD qword[1][2] 解析 inline 小值 → 写入 inlines 缓冲区
//      srcHandle = inlinesMhandle，srcOff = 相对偏移
//   b. 标准 DMA：直接使用 GFD 中的 srcOff/srcHandle（指向 GPU 已注册内存）
//
// 异步模式：所有 IB 操作都是非阻塞的（i-前缀：iput/iget/iflush），
//   返回 request 句柄，由 proxyGinPollCompletions 检查完成。
static ncclResult_t proxyGinProcessGfd(struct ginProxyCtx *ctx,
                                       struct ginProxyHostGpuCtx *hostGpuCtx, int targetRank,
                                       ncclGinProxyGfd_t *gfd, struct ginProxyGfdState *state) {
  int signalOp;
  uint64_t signalVal;

  // ===== 分支 1：VA Signal 操作（无数据 PUT，只写信号） =====
  // 从 GFD 中提取 signalOff + signalHandle，调用 ginBackend->iputSignal
  // 此时 srcOff=0, srcHandle=nullptr, size=0（无数据载荷）
  if (extractOp(gfd) & ncclGinProxyOpVASignal) {
    uint64_t signalOff = gfd->qword[ncclGinProxyGfdVASignalOff].vaSignalOff.vaSignalOff;
    void *signalHandle = (void *)(uint64_t)gfd->qword[ncclGinProxyGfdVASignalHandle].vaSignalHandle.vaSignalHandle;
    signalVal = extractSignalVal(gfd);
    signalOp = mapGfdOpToSignalOp(gfd);
    NCCLCHECK(ginBackend->iputSignal(ctx->ginCtx, hostGpuCtx->contextId, 0, nullptr, 0, 0, nullptr,
                                  targetRank, signalOff, signalHandle, signalVal,
                                  signalOp, &state->request));
    return ncclSuccess;
  }

  // ===== 分支 2：Get 操作（RDMA Read） =====
  // srcOff/srcHandle = 远端（源），dstOff/dstHandle = 本地（目标）
  if (extractOp(gfd) & ncclGinProxyOpGet) {
    uint64_t srcOff = gfd->qword[ncclGinProxyGfdSrcOff].srcOff.srcOff;
    void *srcHandle = (void *)(uint64_t)gfd->qword[ncclGinProxyGfdSrcHandle].srcHandle.srcHandle;
    uint64_t dstOff = gfd->qword[ncclGinProxyGfdDstOff].dstOff.dstOff;
    void *dstHandle = (void *)(uint64_t)gfd->qword[ncclGinProxyGfdDstHandle].dstHandle.dstHandle;
    uint64_t size = gfd->qword[ncclGinProxyGfdHeader].header.size;
    if (!ginBackend->iget) {
      WARN("GIN plugin does not support GET");
      return ncclInvalidUsage;
    }
    NCCLCHECK(ginBackend->iget(ctx->ginCtx, hostGpuCtx->contextId, srcOff, srcHandle, size, dstOff, dstHandle,
                              targetRank, &state->request));
    return ncclSuccess;
  }

  // ===== 分支 3：Flush 操作 =====
  // 确保之前的 RDMA Read 数据对 GPU 可见
  // 如果 plugin 不支持 flush，报错
  // 如果 request=NULL，说明 flush 是同步完成的，直接标记 done=1
  if (extractOp(gfd) & ncclGinProxyOpFlush) {
    if (!ginBackend->iflush) {
      WARN("GIN plugin does not support FLUSH");
      return ncclInvalidUsage;
    }
    NCCLCHECK(ginBackend->iflush(ctx->ginCtx, hostGpuCtx->contextId, ctx->signalsGinHandle, targetRank,&state->request));
    if (state->request == NULL) {
      state->done = 1;
    }
    return ncclSuccess;
  }

  // ===== 分支 4：Put 操作（RDMA Write） =====
  uint64_t size = gfd->qword[ncclGinProxyGfdHeader].header.size;
  uint64_t srcOff;
  void *srcHandle;
  if (extractOp(gfd) & ncclGinProxyOpWithInline) {
    // 内嵌小值：从 GFD inline 字段重组数据到 inlines 缓冲区
    // 然后用 inlines 的 MR handle 作为源，让 IB HCA 从这里读取数据
    uint64_t *inlineVal = &hostGpuCtx->inlines[state - hostGpuCtx->states];
    srcOff = (uint64_t)&inlineVal[0] - (uint64_t)hostGpuCtx->inlines;
    // 从 GFD 的 inlineLow/inlineHigh qword 中重组原始值
    *inlineVal = gfd->qword[ncclGinProxyGfdInlineLow].inlineLow.inlineValLow;
    if (size > 4)
      *inlineVal |= (uint64_t)gfd->qword[ncclGinProxyGfdInlineLow].inlineLow.inlineValLow2 << 32;
    if (size > 6)
      *inlineVal |= (uint64_t)gfd->qword[ncclGinProxyGfdInlineHigh].inlineHigh.inlineValHigh << 48;
    srcHandle = hostGpuCtx->inlinesMhandle;  // 使用 inlines 的 MR handle 作为 lkey
  } else {
    // 标准 DMA 模式：直接使用 GFD 中的 srcOff/srcHandle（指向 GPU 已注册内存）
    srcOff = gfd->qword[ncclGinProxyGfdSrcOff].srcOff.srcOff;
    srcHandle = (void *)(uint64_t)gfd->qword[ncclGinProxyGfdSrcHandle].srcHandle.srcHandle;
  }
  uint64_t dstOff = gfd->qword[ncclGinProxyGfdDstOff].dstOff.dstOff;
  void *dstHandle = (void *)(uint64_t)gfd->qword[ncclGinProxyGfdDstHandle].dstHandle.dstHandle;

  ncclGinProxyOp_t op = extractOp(gfd);
  switch (op & ncclGinProxyOpBaseMask) {
    case ncclGinProxyOpPut:
      signalOp = mapGfdOpToSignalOp(gfd);
      if (signalOp == -1) {
        // 无信号：纯 RDMA Write
        NCCLCHECK(ginBackend->iput(ctx->ginCtx, hostGpuCtx->contextId, srcOff, srcHandle, size, dstOff, dstHandle,
                                targetRank, &state->request));
      } else {
        // 有信号：RDMA Write + Signal（一次 IB 操作同时完成数据写入和信号通知）
        // signalOff = signalId 在 signals 数组中的字节偏移（context 内部偏移 + context 间偏移）
        signalVal = extractSignalVal(gfd);
        uint64_t signalOff = (gfd->qword[ncclGinProxyGfdCompletion].completion.signalId +
                              hostGpuCtx->contextId * ctx->nSignalsPerContext) * sizeof(uint64_t);
        // signalsGinHandle 是 signals 全局内存的 GIN handle（提供 rkey）
        NCCLCHECK(ginBackend->iputSignal(ctx->ginCtx, hostGpuCtx->contextId, srcOff, srcHandle, size, dstOff, dstHandle,
                                      targetRank, signalOff, ctx->signalsGinHandle, signalVal,
                                      signalOp, &state->request));
      }
      break;
    default:
      // this error should already have been checked in pollGfd
      assert(0);
  }
  TRACE(NCCL_NET, "GFD submitted into GIN plugin - contextId: %d, stateIdx: %lu, request: %p",
        hostGpuCtx->contextId, state - hostGpuCtx->states, state->request);
  return ncclSuccess;
}

struct ncclGinProxyListenComm {
  int dev;
  void* listenComm;
};

static ncclResult_t ncclGinProxyListen(void* ctx, int dev, void* handle, void** listenComm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclGinProxyListenComm* lComm;
  NCCLCHECK(ncclCalloc(&lComm, 1));
  lComm->dev = dev;
  NCCLCHECKGOTO(ginBackend->listen(ctx, dev, handle, &lComm->listenComm), ret, end);

end:
  if (ret != ncclSuccess) free(lComm);
  else *listenComm = lComm;
  return ret;
}

static ncclResult_t ncclGinProxyCloseListen(void* listenComm) {
  struct ncclGinProxyListenComm* lComm = (struct ncclGinProxyListenComm*)listenComm;
  NCCLCHECK(ginBackend->closeListen(lComm->listenComm));
  free(lComm);
  return ncclSuccess;
}

struct ncclGinProxyCollComm {
  ncclNetProperties_t props;
  int nRanks;
  void* collComm;
};

static ncclResult_t ncclGinProxyConnect(void* ctx, void* handles[], int nranks, int rank,
                                 void* listenComm, void** collComm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclGinProxyCollComm* cComm = NULL;
  struct ncclGinProxyListenComm* lComm = (struct ncclGinProxyListenComm*)listenComm;
  NCCLCHECK(ncclCalloc(&cComm, 1));
  cComm->nRanks = nranks;
  NCCLCHECKGOTO(ginBackend->getProperties(lComm->dev, &cComm->props), ret, end);
  NCCLCHECKGOTO(ginBackend->connect(ctx, handles, nranks, rank, lComm->listenComm, &cComm->collComm), ret, end);

end:
  if (ret != ncclSuccess) free(cComm);
  else *collComm = cComm;
  return ret;
}

static ncclResult_t ncclGinProxyCloseColl(void* collComm) {
  struct ncclGinProxyCollComm* cComm = (struct ncclGinProxyCollComm*)collComm;
  NCCLCHECK(ginBackend->closeColl(cComm->collComm));
  free(cComm);
  return ncclSuccess;
}

// Check if the GIN plugin supports DMA-BUF, if so we can try to get the DMA-BUF handle from CUDA,
// if that fails we fallback to non-DMA-BUF
static ncclResult_t ncclGinProxyRegMrSym(void* ginCtx, void* addr, size_t size, int type,
                                         uint64_t mrFlags, void** mhandle, void **ginHandle) {
  struct ncclGinProxyCollComm* cComm = (struct ncclGinProxyCollComm*)ginCtx;
  if (type == NCCL_PTR_HOST) {
    NCCLCHECK(ginBackend->regMrSym(cComm->collComm, addr, size, type, mrFlags, mhandle, ginHandle));
  } else if (type == NCCL_PTR_CUDA) {
    ncclResult_t dmabufResult = ncclInvalidUsage;
    if (ncclParamDmaBufEnable() && (cComm->props.ptrSupport & NCCL_PTR_DMABUF)) {
      ncclResult_t registrationResult = ncclSuccess;
      int dmabufFd = -1;
      dmabufResult = getDmaBufFd(addr, size, &dmabufFd);
      if (dmabufResult == ncclSuccess) {
        registrationResult = ginBackend->regMrSymDmaBuf(cComm->collComm, addr, size, type, 0, dmabufFd,
                                                     mrFlags, mhandle, ginHandle);
        close(dmabufFd);
      }
      if (registrationResult != ncclSuccess) {
        // This code path assumes if one MR enters this path, all others will too.
        // Mixed usage of DataDirect and non-DataDirect breaks GIN ordering guarantees.
        dmabufFd = -1;
        dmabufResult = getDmaBufFd(addr, size, &dmabufFd, true);
        if (dmabufResult == ncclSuccess) {
          NCCLCHECK(ginBackend->regMrSymDmaBuf(cComm->collComm, addr, size, type, 0, dmabufFd,
                                            mrFlags, mhandle, ginHandle));
          close(dmabufFd);
        }
      }
    }
    // Fallback to non-DMA-BUF if the DMA-BUF handle is not supported
    if (dmabufResult != ncclSuccess) {
      NCCLCHECK(ginBackend->regMrSym(cComm->collComm, addr, size, type, mrFlags, mhandle, ginHandle));
    }
  } else {
    return ncclInvalidUsage;
  }

  return ncclSuccess;
}

static ncclResult_t ncclGinProxyDeregMrSym(void* collComm, void* mhandle) {
  struct ncclGinProxyCollComm* cComm = (struct ncclGinProxyCollComm*)collComm;
  // Deregister the memory region with the GIN plugin
  NCCLCHECK(ginBackend->deregMrSym(cComm->collComm, mhandle));
  return ncclSuccess;
}


static uint64_t isPowerOfTwo(uint64_t n) { return (n > 0) && ((n & (n - 1)) == 0); }

static ncclResult_t ncclGinProxyCreateContext(void* collComm, ncclGinConfig_t* config,
                                       void **outGinCtx, ncclNetDeviceHandle_t **outDevHandle) {
  struct ncclGinProxyCollComm* cComm = (struct ncclGinProxyCollComm*)collComm;
  ncclGinProxyGpuCtx_t *devGpuCtxArray_h = nullptr;

  if (!ncclGdrCopy)
    INFO(NCCL_NET, "GIN Proxy will not be using GDRCopy");

  struct ginProxyCtx *proxyCtx = NULL;
  NCCLCHECK(ncclCalloc(&proxyCtx, 1));

  proxyCtx->collComm = cComm->collComm;
  proxyCtx->nRanks = cComm->nRanks;
  int nContexts = proxyCtx->nContexts = config->nContexts;

  NCCLCHECK(ginBackend->createContext(cComm->collComm, config, &proxyCtx->ginCtx, NULL));

  // Sanitize the queue size
  // queueSize 代表 GPU 最多可以同时塞入多少个未被 CPU 消费的 GFD 描述符
  uint64_t queueSize = ncclParamGinProxyQueueSize();
  uint32_t maxRequests = NCCL_NET_MAX_REQUESTS * cComm->props.maxRecvs;
  if (queueSize == -1) {
    queueSize = maxRequests;
  }
  if (queueSize > maxRequests) {
    INFO(NCCL_NET,
         "NCCL_GIN_PROXY_QUEUE_SIZE is greater than the maximum outstanding requests in the GIN "
         "plugin (%d), using the default/maximum value instead",
         maxRequests);
    queueSize = maxRequests;
  }
  if (queueSize < 1) {
    INFO(NCCL_NET,
         "NCCL_GIN_PROXY_QUEUE_SIZE is less than 1, using the default/maximum value instead");
    queueSize = maxRequests;
  }
  if (!isPowerOfTwo(queueSize)) {
    INFO(
      NCCL_NET,
      "NCCL_GIN_PROXY_QUEUE_SIZE is not a power of two, using the default/maximum value instead");
    queueSize = maxRequests;
  }

  if (config->nCounters) {
    // Allocate the counters on the GPU or CPU depending on GDR
    NCCLCHECK(allocMemCPUAccessible(&proxyCtx->counters, &proxyCtx->countersDev,
                                    config->nCounters * nContexts, CU_MEMHOSTALLOC_WRITECOMBINED,
                                    &proxyCtx->countersGdrHandle, NULL));
  }
  proxyCtx->nCountersPerContext = config->nCounters;

  // Allocate the signals on the GPU and then register the memory region with the GIN plugin.
  // Enforcing strong ordering on the signals mr is vital to ensure ordering between puts and
  // signals.
  if (config->nSignals) {
    size_t signalsBufSize = config->nSignals * nContexts * sizeof(uint64_t);
    NCCLCHECK(ncclCuMemAlloc((void **)&proxyCtx->signalsDev, &proxyCtx->signalsCumemhandle,
                             CU_MEM_HANDLE_TYPE_NONE, signalsBufSize, NULL));
    CUDACHECK(cudaMemset(proxyCtx->signalsDev, 0, signalsBufSize));
    NCCLCHECK(ncclGinProxyRegMrSym(collComm, proxyCtx->signalsDev, signalsBufSize,
                                   NCCL_PTR_CUDA, NCCL_NET_MR_FLAG_FORCE_SO,
                                   &proxyCtx->signalsMhandle, &proxyCtx->signalsGinHandle));
  }
  proxyCtx->nSignalsPerContext = config->nSignals;

  // CPU 侧 context 数组
  NCCLCHECK(ncclCalloc(&proxyCtx->hostGpuCtx, nContexts));
  // 将要 H2D 拷贝的 GPU 侧 context 数组（临时）
  NCCLCHECK(ncclCalloc(&devGpuCtxArray_h, nContexts));
  for (int contextId = 0; contextId < nContexts; contextId++) {
    struct ginProxyHostGpuCtx *hostGpuCtx = proxyCtx->hostGpuCtx + contextId;
    hostGpuCtx->contextId = contextId;
    hostGpuCtx->queueSize = queueSize;
    size_t queuesLength = hostGpuCtx->queueSize * cComm->nRanks;
    // ② states 数组（CPU专用，与 queues 等长）
    // 每个 GFD 槽位对应一个 ginProxyGfdState{op, counterId, done, request}
    // CPU proxy 扫描 GFD 队列时，把解析结果存在对应的 state 里，追踪 ibv_post_send 的完成状态
    NCCLCHECK(ncclCalloc(&hostGpuCtx->states, queuesLength));
    NCCLCHECK(ncclCalloc(&hostGpuCtx->cisShadow, cComm->nRanks));
    NCCLCHECK(ncclCalloc(&hostGpuCtx->sis, cComm->nRanks));
    // ⑤ inlines（CPU专用，与 queues 等长）
    // 小数据内联传输缓冲区：当 put 数据量极小时，GPU 把数据直接嵌入 GFD 描述符而非单独 buffer
    // CPU proxy 读取 GFD 后，把 inline 数据通过 ibv_post_send inline 方式发出（减少一次 PCIe 往返）
    NCCLCHECK(ncclCalloc(&hostGpuCtx->inlines, queuesLength));
    NCCLCHECK(ncclGinProxyRegMrSym(collComm, hostGpuCtx->inlines,
                                   queuesLength * sizeof(uint64_t), NCCL_PTR_HOST, 0,
                                   &hostGpuCtx->inlinesMhandle, &hostGpuCtx->inlinesGinHandle));
    NCCLCHECK(ncclCudaCalloc(&hostGpuCtx->pis, cComm->nRanks, NULL));

    // ⑥ GPU 侧 context 视图（devGpuCtx_h）填充
    ncclGinProxyGpuCtx_t *devGpuCtx_h = devGpuCtxArray_h + contextId;
    devGpuCtx_h->nranks = cComm->nRanks;
    devGpuCtx_h->queueSize = hostGpuCtx->queueSize;
    devGpuCtx_h->counters = proxyCtx->countersDev + contextId * config->nCounters;
    devGpuCtx_h->signals = proxyCtx->signalsDev + contextId * config->nSignals;
    devGpuCtx_h->pis = hostGpuCtx->pis;

    // Allocate the GFD queues, CIs, counters, signals and test/wait variables on the either the CPU
    // or GPU.
    // ⑧ queues：GFD 环形队列（forceHost=true）
    // queues 是 GPU 写、CPU 读的核心通信通道
    // forceHost=true：强制用 pinned host memory（不用 GDRCopy）
    //   原因：CPU proxy 需要高频率轮询读取，CPU 从 host pinned memory 读比从 GPU BAR1 读快得多
    //   代价：GPU 写 host memory 需要经 PCIe，比写显存慢（但 GPU 只写一次，CPU 读多次，值得）
    NCCLCHECK(allocMemCPUAccessible(&hostGpuCtx->queues, &devGpuCtx_h->queues, queuesLength, 0, NULL,
                                    NULL, true /*forceHost*/));
    NCCLCHECK(allocMemCPUAccessible(&hostGpuCtx->cis, &devGpuCtx_h->cis, cComm->nRanks,
                                    CU_MEMHOSTALLOC_WRITECOMBINED, &hostGpuCtx->cisGdrHandle, NULL));
  }

  // 在 GPU 上分配 devGpuCtx_d（nContexts 个 ncclGinProxyGpuCtx_t）
  // 然后把 host 侧填好的 devGpuCtxArray_h 拷贝到 GPU
  ncclGinProxyGpuCtx_t *devGpuCtx_d = NULL;
  NCCLCHECK(ncclCudaCalloc(&devGpuCtx_d, nContexts, NULL));
  // Copy the proxy's devGpuCtx to the GPU
  NCCLCHECK(ncclCudaMemcpy(devGpuCtx_d, devGpuCtxArray_h, nContexts));

  ncclNetDeviceHandle_t *devHandle = NULL;
  NCCLCHECK(ncclCalloc(&devHandle, 1));
  devHandle->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  devHandle->netDeviceVersion = NCCL_GIN_PROXY_VERSION;
  devHandle->handle = (void *)devGpuCtx_d;
  devHandle->size = 0;
  // 必须有 CPU proxy 线程，Proxy 模式不能无 CPU 运行
  devHandle->needsProxyProgress = 1;

  proxyCtx->devHandle = devHandle;

  *outDevHandle = devHandle;
  *outGinCtx = proxyCtx;

  free(devGpuCtxArray_h);

  return ncclSuccess;
}

static ncclResult_t ncclGinProxyDestroyContext(void *ginCtx) {
  if (!ginCtx) return ncclSuccess;
  struct ginProxyCtx *ctx = (struct ginProxyCtx *)ginCtx;

  NCCLCHECK(ginBackend->destroyContext(ctx->ginCtx));

  // Free counters
  if (ctx) {
    if (ctx->counters || ctx->countersGdrHandle)
      NCCLCHECK(freeMemCPUAccessible(ctx->counters, ctx->countersGdrHandle, NULL));

    // Free signals
    if (ctx->collComm && ctx->signalsMhandle)
      ginBackend->deregMrSym(ctx->collComm, ctx->signalsMhandle);
    if (ctx->signalsDev) NCCLCHECK(ncclCudaFree(ctx->signalsDev, NULL));

    // Free hostGpuCtx and its allocations
    if (ctx->hostGpuCtx) {
      for (int contextId = 0; contextId < ctx->nContexts; contextId++) {
        struct ginProxyHostGpuCtx *hostGpuCtx = ctx->hostGpuCtx + contextId;
        if (hostGpuCtx->cisShadow) free(hostGpuCtx->cisShadow);
        if (hostGpuCtx->sis) free(hostGpuCtx->sis);
        if (hostGpuCtx->pis) NCCLCHECK(ncclCudaFree(hostGpuCtx->pis, NULL));
        if (hostGpuCtx->states) free(hostGpuCtx->states);
        if (hostGpuCtx->inlines) free(hostGpuCtx->inlines);
        if (ctx->collComm && hostGpuCtx->inlinesMhandle)
          ginBackend->deregMrSym(ctx->collComm, hostGpuCtx->inlinesMhandle);
        if (hostGpuCtx->queues) NCCLCHECK(freeMemCPUAccessible(hostGpuCtx->queues, NULL, NULL));
        if (hostGpuCtx->cis || hostGpuCtx->cisGdrHandle)
          NCCLCHECK(freeMemCPUAccessible(hostGpuCtx->cis, hostGpuCtx->cisGdrHandle, NULL));
      }
      free(ctx->hostGpuCtx);
    }

    ncclNetDeviceHandle_t *devHandle = (ncclNetDeviceHandle_t *)ctx->devHandle;
    if (devHandle) {
      if (devHandle->handle) NCCLCHECK(ncclCudaFree((void *)devHandle->handle, NULL));
      free(devHandle);
    }

    free(ctx);
  }

  return ncclSuccess;
}

// ---------------------------------------------------------------------------
// ncclGinProxyProgress：CPU proxy 主循环（每次调用执行一轮轮询）
// ---------------------------------------------------------------------------
// 由 NCCL transport 层的 proxy 线程定期调用（tcomm->proxyProgress）。
//
// 每轮轮询的执行顺序：
//   对每个 context：
//     1. proxyGinPollCompletions — 检查已提交 IB 操作的完成状态，推进 CI
//     2. 对每个 rank：
//        a. proxyGinPollGfd   — 轮询队列检测新 GFD
//        b. proxyGinProcessGfd — 解析 GFD 并提交 IB 操作
//     3. ginBackend->ginProgress — 调用 plugin 的自定义进度函数（如果有）
//
// 关键设计：先检查完成（推进 CI释放槽位）再轮询新 GFD，
// 避免队列满时 GPU 端无法写入新 GFD而死锁。
static ncclResult_t ncclGinProxyProgress(void *ginCtx) {
  struct ginProxyCtx *ctx = (struct ginProxyCtx *)ginCtx;

  for (int contextId = 0; contextId < ctx->nContexts; contextId++) {
    struct ginProxyHostGpuCtx *hostGpuCtx = ctx->hostGpuCtx + contextId;
    NCCLCHECK(proxyGinPollCompletions(ctx->collComm, ctx, hostGpuCtx));
    for (int targetRank = 0; targetRank < ctx->nRanks; targetRank++) {
      // Poll on the GFD queue
      ncclGinProxyGfd_t gfd;
      struct ginProxyGfdState *state = NULL;
      if (proxyGinPollGfd(ctx, hostGpuCtx, targetRank, &gfd, &state)) {
        ncclResult_t ret =
          proxyGinProcessGfd(ctx, hostGpuCtx, targetRank, &gfd, state);
        if (ret) ctx->hasError = ret;
        NCCLCHECK(ret);
      }
    }
    if (ginBackend->ginProgress) ginBackend->ginProgress(ctx->ginCtx);
  }

  return ncclSuccess;
}

static ncclResult_t ncclGinProxyQueryLastError(void *ginCtx, bool *hasError) {
  struct ginProxyCtx *ctx = (struct ginProxyCtx *)ginCtx;
  *hasError = ctx->hasError;
  if (ctx->hasError == ncclSuccess && ginBackend->queryLastError)
    NCCLCHECK(ginBackend->queryLastError(ginCtx, hasError));
  return ncclSuccess;
}

// ---------------------------------------------------------------------------
// ncclGinProxy：CPU proxy 的函数表（ncclGin_t 结构）
// ---------------------------------------------------------------------------
// 这个结构体作为 GIN plugin 的中间层：
//   - 部分函数指针直接指向底层 plugin（name/init/devices/getProperties/iput/iputSignal/test/finalize）
//   - 部分函数是 proxy 层自己实现的包装（listen/connect/createContext/regMr/progress 等）
// 这种设计让 proxy 可以在不修改底层 plugin 的情况下，增加 GPU-CPU 队列通信层。
ncclGin_t ncclGinProxy {
  NULL, // Will map directly to the plugin: name
  NULL, // Will map directly to the plugin: init()
  NULL, // Will map directly to the plugin: devices()
  NULL, // Will map directly to the plugin: getProperties()
  ncclGinProxyListen,
  ncclGinProxyConnect,
  ncclGinProxyCreateContext,
  ncclGinProxyRegMrSym,
  NULL, // regMrSymDmaBuf() is not used by upper layer at the moment, hidden in RegMrSym.
  ncclGinProxyDeregMrSym,
  ncclGinProxyDestroyContext,
  ncclGinProxyCloseColl,
  ncclGinProxyCloseListen,
  NULL, // Will map directly to the plugin: iput()
  NULL, // Will map directly to the plugin: iputSignal()
  NULL, // Will map directly to the plugin: iget()
  NULL, // Will map directly to the plugin: iflush()
  NULL, // Will map directly to the plugin: test()
  ncclGinProxyProgress,
  ncclGinProxyQueryLastError,
  NULL  // Will map directly to the plugin: finalize()
};

// ---------------------------------------------------------------------------
// ncclGinProxyInit：初始化 proxy 中间层，拦截并包装底层 plugin
// ---------------------------------------------------------------------------
// 将底层 plugin 的函数指针复制到 ncclGinProxy，再用 proxy 自己的实现覆盖需要拦截的部分。
// 最终把 *proxyGin 指向 ncclGinProxy，调用者透明地使用 proxy 层。
ncclResult_t ncclGinProxyInit(ncclGin_t** proxyGin) {
  // 保存底层 plugin 的指针，以便 proxy 层内部调用真正的 IB 操作
  ginBackend = *proxyGin;
  // 直接代理到底层的函数（无需 proxy 层的额外逻辑）
  ncclGinProxy.name = ginBackend->name;
  ncclGinProxy.init = ginBackend->init;
  ncclGinProxy.devices = ginBackend->devices;
  ncclGinProxy.getProperties = ginBackend->getProperties;
  ncclGinProxy.iput = ginBackend->iput;         // CPU proxy 内部用于提交 IB put
  ncclGinProxy.iputSignal = ginBackend->iputSignal;  // CPU proxy 内部用于提交 IB put+signal
  ncclGinProxy.test = ginBackend->test;          // CPU proxy 内部用于检查 IB 完成
  ncclGinProxy.finalize = ginBackend->finalize;
  // 用 proxy 层的 ncclGinProxy 替换原始 plugin 指针
  *proxyGin = &ncclGinProxy;
  return ncclSuccess;
}
