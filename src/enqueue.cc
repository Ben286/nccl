/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/


// ============================================================================
// enqueue.cc — NCCL collective 入队/调度/发射流程
// ============================================================================
// 这是 NCCL 最核心的“中间层”：负责将用户提交的 ncclAllReduce/AllGather 等操作
// 转化为 GPU kernel 发射和 proxy 操作。
//
// ===== 触发机制：ncclGroupEnd 如何驱动整个流程 =====
//
//  用户调用 ncclGroupStart() ... ncclAllReduce/Send/Recv ... ncclGroupEnd()
//  ncclGroupEnd() 内部通过异步 job 机制（group.cc）驱动：
//    1. ncclPrepareTasksAndCollPreconnectFunc（异步 job）
//       → 调用 ncclPrepareTasks()：分析所有任务，选 algo/proto，symk 分流
//       → 如果需要 runtimeConn，还会调用 ncclCollPreconnect() 建立传输连接
//    2. ncclLaunchPrepare()：将任务打包成 plan（每个 plan = 一次 kernel launch）
//    3. ncclLaunchKernelBefore → ncclLaunchKernel → ncclLaunchKernelAfter
//    4. ncclLaunchFinish()：流同步 + 资源回收
//
// ===== 整体流程：一次 ncclGroup 内的调用 =====
//
//  [API 入口] ncclAllReduce / ncclAllGather / ncclSend / ncclRecv 等
//      ↓
//  ncclEnqueueCheck（入口函数，行 2979）
//    - 参数检查（内存对齐、类型匹配等）
//    - 将任务封装为 ncclTaskColl / ncclTaskP2p，加入 planner 队列
//    - 每个 ncclTaskColl 记录 sendbuff/recvbuff/count/datatype/op 等用户参数
//      ↓
//  ncclGroupEnd 触发 ncclPrepareTasks（行 457）
//    - 先尝试 ncclMakeSymmetricTaskList：将能用 symk 的任务分流到 collSymTaskQueue
//    - 剩余任务按 (func, op, datatype) 分组，调 ncclGetAlgoInfo 选 algo/proto
//    - algo 选项：Ring / Tree / NVLS / NVLS_TREE / CollNet_Chain / CollNet_Direct
//    - proto 选项：LL（Low Latency）/ LL128 / Simple
//    - 为 NVLS 算法的任务做 buffer 注册（ncclRegisterCollNvlsBuffers）
//    - 结果：collTaskQueue（传统 collective）、collSymTaskQueue（symmetric）、
//            collCeTaskQueue（CE copy）、rmaTaskQueue（RMA 单边传输）
//      ↓
//  ncclLaunchPrepare（行 1608） — 将任务分装入 plan
//    - 每个 plan 对应一次 kernel launch
//    - plan 构建优先级：RMA > CE > Sym > Traditional Colls > P2P
//    - 对于 Sym plan：调用 ncclSymmetricTaskScheduler，选择 symk kernel，
//      参数通过 ncclSymkDevWorkArgs 传入，不需要 workFifo
//    - 对于传统 plan：scheduleCollTasksToPlan 分配 channel + chunk，
//      为每个 channel 构建 ncclDevWorkBatch + ncclProxyOp
//    - finishPlan：确定 work 存在哪里（Args 内联 vs Fifo vs Persistent）
//      ↓
//  ncclLaunchKernelBefore_NoUncapturedCuda（行 1765）
//    - uploadWork：如果是 Fifo 模式，将 devWork 写入 workFifo ring buffer
//    - symk/CE/RMA 的 plan 直接跳过（它们的参数内嵌在 kernelArgs 中）
//      ↓
//  ncclLaunchKernel（行 1796） - 实际发射 GPU kernel
//    - 处理 CGA cluster（sm90+）、MemSyncDomain、LaunchCompletionEvent 等
//    - cuLaunchKernelEx（CUDA 11.8+）或普通 cuLaunchKernel
//    - grid = {nChannels, 1, 1}：每个 channel 一个 thread block
//    - symk 和传统 kernel 共用同一套发射路径，区别仅在 smem 和 kernel function
//      ↓
//  ncclLaunchKernelAfter_NoCuda（行 1900）
//    - hostStreamPlanTask：将 proxy op 提交给 proxy 线程
//    - 非 persistent 模式下将 plan 送入 callbackQueue 等待回收
//      ↓
//  ncclLaunchFinish（行 1940）
//    - 设置流同步事件，确保所有 user stream 看到 kernel 完成
//    - 为 workFifo 记录回收事件（KernelFinishCallback 异步更新 workFifoConsumed）
//
// ===== WorkFifo vs WorkArgs vs WorkPersistent =====
//   ncclDevWorkStorageTypeFifo：最传统的方式
//     host 将 ncclDevWorkColl 结构写入 ring buffer（workFifo）
//     kernel 启动后巡检 workFifo 获取工作描述符
//     适合：大量任务，节省 host 内存
//     回收：GPU 完成后通过 KernelFinishCallback 异步推进 workFifoConsumed
//   ncclDevWorkStorageTypeArgs：内联到 kernel launch args
//     小任务可以直接把 work struct 内嵌到 cuLaunchKernelEx 的 args 中
//     分配在 memScoped，常见于 4 个以内小任务
//     finishPlan() 中的判断条件：sizeof(ncclDevKernelArgs) + batchBytes + workBytes <= workArgsBytes
//   ncclDevWorkStorageTypePersistent：持久内存（CUDA Graphs 场景）
//     CUDA Graph capture 时，workFifo 不可用，必须用 persistent 内存
//     通过 cudaMallocAsync 分配设备内存，cudaMemcpyAsync 上传
//
// ===== 新增特性：Symmetric Collective Task（Symmetric kernel 模式）=====
//   ncclPrepareTasks 中会先调 ncclMakeSymmetricTaskList 筛选可用 Symmetric 算法的任务
//   Symmetric 任务不经过普通 channel/ring 调度，而是直接发射 ncclSymkKernel
//   此类 kernel 对 warp 分配策略要求不同，用 ncclDevWorkArgs 外的独立内存分配
//
// ===== 新增特性：CE（Copy Engine）Collective =====
//   collCeTaskQueue：小数据量的 AllGather/Reduce，用 cudaMemcpy 实现
//   附带 NVLS 同步（CE 不能直接操作 NVLS multicast）
//
// ===== 新增特性：RMA（Remote Memory Access）=====
//   rmaTaskQueue： ncclPut/ncclGet 不通过 collective channel，
//   而是选择一个最优传输路径来执行单边传输
// ============================================================================
#include "enqueue.h"
#include "argcheck.h"
#include "coll_net.h"
#include "gdrwrap.h"
#include "bootstrap.h"
#include "channel.h"
#include "cudawrap.h"
#include "profiler.h"
#include "transport.h"
#include "register_inline.h"
#include "ce_coll.h"
#include "nvtx.h"
#include "scheduler.h"
#include "compiler.h"
#include "rma/rma.h"

#include <cstring> // std::memcpy
#include <cinttypes> // PRIx64
#include <cassert>
#include <cfloat> // FLT_MAX

NCCL_PARAM(L1SharedMemoryCarveout, "L1_SHARED_MEMORY_CARVEOUT", 0);
NCCL_PARAM(AllgathervEnable, "ALLGATHERV_ENABLE", 1);
NCCL_PARAM(SymCeThreshold, "SYM_CE_THRESHOLD", 8*1024*1024);

// ncclInitKernelsForDevice：为当前 GPU 初始化所有 NCCL kernel 的属性
// ============================================================================
// 对两类 kernel 分别处理：
//   sym=0：普通 NCCL kernel（ncclDevKernelList）
//     - 包含所有 func x algo x proto 组合的 kernel function pointer
//     - 例如 AllReduce_Ring_Simple、AllGather_NVLS_LL128 等
//   sym=1：Symmetric kernel（ncclSymkKernelList）
//     - 18 种 kernel（AllGather 7 种、AllReduce 5 种、ReduceScatter 6 种）
//     - 每种 kernel 有独立的 smem 需求
// 对每个 kernel：
//   1. 检查是否需要更高版本驱动（krequires[k]），如果不满足则 nullify
//      这是因为某些 kernel 使用了新 PTX 指令（如 TMA），需要新驱动支持
//   2. 读取 cudaFuncAttributes（stack size、smem size）
//   3. 如果用户设置了 NCCL_L1_SHARED_MEMORY_CARVEOUT，应用 carveout
//      carveout 控制 L1 cache 和 shared memory 的比例
//   4. 设置 maxDynamicSharedMemorySize（允许 kernel 使用最大动态共享内存）
//      Symmetric kernel 单独记录到 ncclSymkKernelMaxDynamicSmem[k]
//        因为每个 symk kernel 的 smem 需求不同
//      普通 kernel 取所有 kernel 中的最小值（maxDynamicSmem）
//        因为普通 kernel 共用一个发射函数，发射时无法区分
// 最终校验：
//   ncclShmemDynamicSize(cudaArch) 必须 <= maxDynamicSmem，否则报错
//   这保证所有普通 kernel 都能获得足够的动态共享内存
// ============================================================================
ncclResult_t ncclInitKernelsForDevice(int cudaArch, int maxSharedMem, size_t* maxStackSize) {
  ncclResult_t result = ncclSuccess;

  if (maxStackSize) *maxStackSize = 0;
  int carveout = ncclParamL1SharedMemoryCarveout();
  // maxDynamicSmem 初始化为极大值，后续取所有普通 kernel 的最小值
  int maxDynamicSmem = 1<<30;
  int driverVersion;
  NCCLCHECK(ncclCudaDriverVersion(&driverVersion));

  // sym=0: 普通 kernel（ncclDevKernelList 数组）
  // sym=1: Symmetric kernel（ncclSymkKernelList 数组）
  for (int sym=0; sym <= 1; sym++) {
    int kcount = sym==0 ? ncclDevKernelCount : ncclSymkKernelCount;
    void** kptrs = sym==0 ? ncclDevKernelList : ncclSymkKernelList;
    // krequires[k]: kernel k 所需的最低驱动版本号
    int* krequires = sym==0 ? ncclDevKernelRequirements : ncclSymkKernelRequirements;
    for (int k=0; k < kcount; k++) {
      // 驱动版本不满足 → 将该 kernel 置 null，运行时不可用
      if (kptrs[k] != nullptr && driverVersion < krequires[k]) {
        INFO(NCCL_INIT, "Skipping %skernel %d which requires driver %d",
             sym ? "symmetric " : "", k, krequires[k]);
        kptrs[k] = nullptr;
      }
      void* fn = kptrs[k];
      cudaFuncAttributes attr = {0};
      if (fn == nullptr) continue;

      if (!CUDASUCCESS(cudaFuncGetAttributes(&attr, fn))) continue; // Silently ignore failures

      // 记录所有 kernel 中最大的局部变量（栈）使用量
      if (maxStackSize) {
        if (attr.localSizeBytes > *maxStackSize) *maxStackSize = attr.localSizeBytes;
      }
      // L1/shared memory carveout：用户可通过 NCCL_L1_SHARED_MEMORY_CARVEOUT 控制
      if (carveout) {
        CUDACHECKGOTO(cudaFuncSetAttribute(fn,
          cudaFuncAttributePreferredSharedMemoryCarveout, carveout),
          result, ignore1);
      ignore1:;
      }
      // dynSmem = 设备总共享内存 - kernel 静态共享内存使用量
      // 这是该 kernel 可用的最大动态共享内存
      { int dynSmem = maxSharedMem - attr.sharedSizeBytes;
        if (sym) {
          // symk: 每个 kernel 独立记录，因为 launch 时可以单独指定 smem
          ncclSymkKernelMaxDynamicSmem[k] = dynSmem;
        } else {
          // 普通 kernel: 取所有 kernel 中的最小值
          // 因为发射时用统一的 ncclShmemDynamicSize(cudaArch) 值
          maxDynamicSmem = std::min(maxDynamicSmem, dynSmem);
        }
        // 告诉 CUDA runtime 此 kernel 允许使用多少动态共享内存
        CUDACHECKGOTO(cudaFuncSetAttribute(fn,
          cudaFuncAttributeMaxDynamicSharedMemorySize, dynSmem),
          result, next_kernel);
      }
    next_kernel:;
    }
  }

  // 最终校验：确保普通 kernel 所需的动态 smem 不超过设备限制
  if (ncclShmemDynamicSize(cudaArch) > maxDynamicSmem) {
    WARN("cudaArch %d dynamic smem %d exceeds device/fn maxSharedMem %d",
         cudaArch, ncclShmemDynamicSize(cudaArch), maxDynamicSmem);
    return ncclSystemError;
  }
  return result;
}

////////////////////////////////////////////////////////////////////////////////
// Data movement metrics.

// ncclFuncTrafficPerByte：计算每字节数据产生的总流量倍数，用于调度算法选择
// AllReduce: 每字节经过网络传输 2 次（ReduceScatter + AllGather）
// AllGather、ReduceScatter: 每字节被转发 nRanks 次
// 主要用于估算任务总流量，再由 ncclGetAlgoInfo 选择最适算法
static inline int ncclFuncTrafficPerByte(ncclFunc_t func, int nRanks) {
  switch (func) {
  case ncclFuncAllReduce: return 2;
  case ncclFuncAllGather: return nRanks;
  case ncclFuncReduceScatter: return nRanks;
  default: return 1;
  }
}

/*****************************************************************************/
/*       Launch system : synchronization and CUDA kernel launch              */
/*****************************************************************************/

// 发射系统概述：
// ncclAddProxyOpIfNeeded：如果任务需要网络操作，将 proxyOp 加入当前 plan 的通道队列
// ncclAddWorkBatchToPlan：将一个 devWork 封装到 wipPlan 的对应通道的 workBatch 中
//   workBatch 采用二进制 bitmask 小型表示哪些 slot 有效，最大内嵌 64 个 work
// finishPlan：将 wipPlan 转化为正式 plan，确定 work 的存储方式，建立 proxy op 合并序列
// ncclPrepareTasks：每个 ncclGroup 调用一次，对所有任务进行分析和调度
// scheduleCollTasksToPlan：将 collective 任务分配到 plan
// scheduleP2pTasksToPlan：将 P2P 任务分配到 plan
// ncclLaunchPrepare：建立完整的 plan 列表（每个 plan = 一次 launch）
// ncclLaunchKernel：发射单个 plan 对应的 CUDA kernel
// ncclLaunchFinish：设置流同步，确保 user stream 看到所有 kernel 完成
ncclResult_t ncclAddProxyOpIfNeeded(struct ncclComm* comm, struct ncclKernelPlan* plan, struct ncclProxyOp* op) {
  bool needed = true;
  NCCLCHECK(ncclProxySaveOp(comm, op, &needed));
  if (needed) {
    struct ncclProxyOp* q = ncclMemoryPoolAlloc<struct ncclProxyOp>(&comm->memPool_ncclProxyOp, &comm->memPermanent);
    *q = *op; // C++ struct assignment
    ncclIntruQueueEnqueue(&comm->planner.wipPlan.channels[op->channelId].proxyOpQueue, q);
  }
  return ncclSuccess;
}

NCCL_PARAM(P2pEpochEnable, "P2P_EPOCH_ENABLE", 1);

void ncclAddWorkBatchToPlan(
    struct ncclComm* comm, struct ncclKernelPlan* plan, int channelId,
    enum ncclDevWorkType workType, int devFuncId, uint32_t workOffset,
    int p2pEpoch, int p2pRound, bool newBatch
  ) {
  size_t workSize = ncclDevWorkSize(workType);
  ncclKernelPlanner::WipPlan::Channel* chan = &comm->planner.wipPlan.channels[channelId];
  // Conditions causing us to create a new blank batch.
  newBatch = (chan->workBatchQueue.tail == nullptr);
  struct ncclDevWorkBatch* batch = nullptr;
  if (!newBatch) {
    batch = &chan->workBatchQueue.tail->batch;
    // All of the conditions that prevent us from appending to current batch.
    newBatch |= batch->workType != (uint8_t)workType;
    newBatch |= batch->funcId != devFuncId;
    // The following ensure the device can handle a batch this large. They have to
    // account for all extension batches being fused together which is why
    // wipBatch.workBytes and wipBatch.nP2ps aren't reset to 0 for a new extension
    // batch further down.
    if (workType == ncclDevWorkTypeP2p) {
      if (ncclParamP2pEpochEnable()) newBatch |= chan->wipBatch.p2pEpoch != p2pEpoch;
      // We only allow NCCL_MAX_DEV_WORK_P2P_PER_BATCH ops per batch.
      newBatch |= chan->wipBatch.nP2ps == NCCL_MAX_DEV_WORK_P2P_PER_BATCH;
      for (int i = 0; i < chan->wipBatch.nP2ps; i++) {
        // Do not allow the same round twice in the same batch, it would use the same connection.
        newBatch |= p2pRound == chan->wipBatch.p2pRounds[i];
        // Make sure we only aggregate p2p operations within the same p2p group (one group is NCCL_MAX_DEV_WORK_P2P_PER_BATCH ops).
        // This enforces uniform batching accross ranks in the communicator and prevents hangs.
        newBatch |= (p2pRound / NCCL_MAX_DEV_WORK_P2P_PER_BATCH) != (chan->wipBatch.p2pRounds[i] / NCCL_MAX_DEV_WORK_P2P_PER_BATCH);
      }
    }
    if (workType == ncclDevWorkTypeBcast) {
      int maxitem = ncclMaxDevWorkBatchBytes(comm->cudaArch) / sizeof(ncclDevWorkBcast);
      newBatch |= chan->wipBatch.nBcasts == maxitem;
    } else {
      newBatch |= NCCL_MAX_DEV_WORK_BATCH_BYTES < chan->wipBatch.workBytes + workSize;
    }
  }
  // Conditions causing us to create an extension batch (prev->nextExtends=1)
  uint32_t offset = newBatch ? 0 : (workOffset - batch->offsetBase);
  bool extendBatch = 63*workSize < offset;
  extendBatch |= 0 != offset%workSize;
  if (newBatch || extendBatch) {
    if (!newBatch) batch->nextExtends = extendBatch; // Extending the previous batch.
    struct ncclWorkBatchList* batchNode = ncclMemoryStackAlloc<ncclWorkBatchList>(&comm->memScoped);
    // Coverity thinks that ncclIntruQueueEnqueue will access chan->workBatchQueue->tail, which might
    // be NULL.  But that code is guarded by chan->workBatchQueue->head not being NULL, in which
    // case tail won't be NULL either.
    // coverity[var_deref_model:FALSE]
    ncclIntruQueueEnqueue(&chan->workBatchQueue, batchNode);
    batch = &batchNode->batch;
    batch->nextExtends = 0;
    batch->workType = (uint32_t)workType;
    batch->funcId = devFuncId;
    batch->offsetBase = workOffset;
    batch->offsetBitset = 0;
    offset = 0;
    if (newBatch) {
      // Since extension batches are fused together on the device, and these values
      // account for constraints on the fused batch, we only reset the values on
      // a new batch
      chan->wipBatch.workBytes = 0;
      chan->wipBatch.nP2ps = 0;
      chan->wipBatch.nBcasts = 0;
      // We don't count extension batches since this is used to derive a proxyOpCount,
      // and we wan't all ops which are fused together to have the same value.
      chan->nWorkBatchesP2p += (workType == ncclDevWorkTypeP2p ? 1 : 0);
      chan->nWorkBatchesBcast += (workType == ncclDevWorkTypeBcast ? 1 : 0);
    }
    plan->nWorkBatches += 1;
  }
  batch->offsetBitset |= 1ull<<(offset/workSize);
  chan->wipBatch.workBytes += workSize;
  if (workType == ncclDevWorkTypeP2p) {
    chan->wipBatch.p2pEpoch = p2pEpoch;
    chan->wipBatch.p2pRounds[chan->wipBatch.nP2ps++] = p2pRound;
  }
  if (workType == ncclDevWorkTypeBcast) {
    chan->wipBatch.nBcasts += 1;
  }
}

static void finishPlan(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  ncclKernelPlanner::WipPlan::Channel* wipChannels = comm->planner.wipPlan.channels;
  size_t workBytes = plan->workBytes;
  size_t batchBytes = plan->nWorkBatches*sizeof(struct ncclDevWorkBatch);

  if (plan->isSymColl) return;
  plan->threadPerBlock = std::max(plan->threadPerBlock, NCCL_MIN_NTHREADS);

  // If we can fit everything into the kernel args we do so.
  if (sizeof(ncclDevKernelArgs) + batchBytes + workBytes <= comm->workArgsBytes) {
    plan->workStorageType = ncclDevWorkStorageTypeArgs;
  }
  plan->kernelArgsSize = sizeof(struct ncclDevKernelArgs) + batchBytes;
  plan->kernelArgsSize += (plan->workStorageType == ncclDevWorkStorageTypeArgs) ? workBytes : 0;
  plan->kernelArgsSize = alignUp(plan->kernelArgsSize, 16);
  plan->kernelArgs = (struct ncclDevKernelArgs*)ncclMemoryStackAlloc(&comm->memScoped, plan->kernelArgsSize, /*align=*/16);
  plan->kernelArgs->comm = comm->devComm;
  plan->kernelArgs->channelMask = plan->channelMask;
  plan->kernelArgs->workStorageType = plan->workStorageType;

  // Put batches into the kernel arguments. The first batch for each channel
  // must be located at batchZero[blockIdx.x]. To achieve this we round robin
  // over the channels in ascending order until they're exhausted.
  uint64_t hasBatchMask = plan->channelMask;
  struct ncclDevWorkBatch* batchPrev[MAXCHANNELS] = {}; // {0...}
  struct ncclDevWorkBatch* batchZero = (struct ncclDevWorkBatch*)(plan->kernelArgs+1);
  int batchIx = 0;
  while (hasBatchMask != 0) {
    uint64_t tmpMask = hasBatchMask; // channels with a batch for this round.
    do {
      int c = popFirstOneBit(&tmpMask);
      if (!ncclIntruQueueEmpty(&wipChannels[c].workBatchQueue)) {
        struct ncclWorkBatchList* batchNode = ncclIntruQueueDequeue(&wipChannels[c].workBatchQueue);
        if (batchPrev[c] != nullptr) {
          batchPrev[c]->nextJump = int(&batchZero[batchIx] - batchPrev[c]);
        }
        batchPrev[c] = &batchZero[batchIx];
        batchZero[batchIx++] = batchNode->batch;
      }
      if (ncclIntruQueueEmpty(&wipChannels[c].workBatchQueue)) {
        hasBatchMask ^= 1ull<<c;
      }
    } while (tmpMask != 0);
  }

  // Merge-sort per-channel proxy-op lists by opCount when merging them into plan->proxyOpQueue
  // Phase 1: scan first op of each channel, store opCount in headIds[c].
  uint64_t headIds[MAXCHANNELS];
  int nHeads = 0;
  int channelUbound = 0;
  for (int c=0; c < MAXCHANNELS; c++) {
    struct ncclProxyOp* op = ncclIntruQueueHead(&wipChannels[c].proxyOpQueue);
    headIds[c] = op ? op->opCount : uint64_t(-1);
    if (op) nHeads += 1;
    if (op) plan->hasProxyOps = true;
    if (op) channelUbound = c+1;
  }
  // Phase 2: Dequeue from planner->channels[c], enqueue in merged order to plan
  while (nHeads != 0) {
    int c = -1;
    uint64_t minId = uint64_t(-1);
    // Find channel with least proxy-op id. We store the heads[c]->opCount in
    // headIds[c] to remove indirect loads from this loop.
    for (int c1=0; c1 < channelUbound; c1++) {
      uint64_t id = headIds[c1];
      id = (id>>1 | id<<63); // Move tag bit to order collectives before p2p's
      if (id < minId) { c = c1; minId = id; }
    }
    struct ncclProxyOp* op = ncclIntruQueueDequeue(&wipChannels[c].proxyOpQueue);
    struct ncclProxyOp* opNext = ncclIntruQueueHead(&wipChannels[c].proxyOpQueue);
    headIds[c] = opNext ? opNext->opCount : uint64_t(-1);
    nHeads -= opNext ? 0 : 1;
    ncclIntruQueueEnqueue(&plan->proxyOpQueue, op);
  }
}

NCCL_PARAM(GraphRegister, "GRAPH_REGISTER", 1);

static ncclResult_t calcCollChunking(
  struct ncclComm* comm, struct ncclTaskColl* task, int nChannels, size_t nBytes,
  /*outputs*/uint32_t* outChunkSize, uint32_t* outDirectFlags, struct ncclProxyOp* proxyOp
);

struct ncclKernelPlanBudget {
  ssize_t inArgsBytes; // Space available within kernel args struct
  ssize_t outArgsBytes; // Space available outside of args struct (fifo or persistent buf)
};

bool ncclTestBudget(
    struct ncclKernelPlanBudget* budget, int nWorkBatches, ssize_t workBytes
  ) {
  ssize_t batchBytes = nWorkBatches*sizeof(struct ncclDevWorkBatch);
  bool ok = false;
  ok |= (batchBytes + workBytes <= budget->inArgsBytes);
  ok |= (batchBytes <= budget->inArgsBytes) && (workBytes <= budget->outArgsBytes);
  return ok;
}

// ncclTasksRegAndEnqueue：为所有传统 collective 任务构建 ncclDevWorkColl 结构体
// ============================================================================
// 在 ncclPrepareTasks 之后调用（实际调用在 group.cc 的 ncclLaunchPrepare 前）
// 为每个任务构建一个 ncclDevWorkColl（或 ncclDevWorkCollReg）结构体
// 这个结构体将被传递给 GPU kernel，描述一次 collective 操作的全部参数
//
// 分支逻辑：
//   NVLS/NVLS_TREE 任务：已在 ncclPrepareTasks 第 7 阶段预构建，直接从 tmpCollWorkQueue 取出
//   其他任务：在这里注册 buffer（ncclRegisterCollBuffers）+ 构建 devWork
//
// devWork 关键字段：
//   sendbuff/recvbuff：用户传入的数据指针
//   root：广播/归约操作的根 rank
//   nWarps：分配给这个任务的 warp 数
//   redOpArg：reduction 操作的标量参数（如自定义 op 的参数）
//   regUsed：是否使用了 IPC/NVLS 注册 buffer（GPU 直接访问对方内存）
//   netRegUsed：是否使用了网络注册 buffer（NIC 直接访问 GPU 内存，跳过 bounce buffer）
// ============================================================================
ncclResult_t ncclTasksRegAndEnqueue(struct ncclComm* comm) {
  struct ncclKernelPlanner* planner = &comm->planner;
  struct ncclTaskColl *task;
  task = ncclIntruQueueHead(&planner->collTaskQueue);
  while (task != nullptr) {
    // Build a ncclDevWorkColl[Reg?] struct for each task.
    void* regBufSend[NCCL_MAX_LOCAL_RANKS];
    void* regBufRecv[NCCL_MAX_LOCAL_RANKS];
    bool regNeedConnect = true;
    struct ncclWorkList* workNode = NULL;
    struct ncclDevWorkColl devWork = {};

    // NVLS/NVLS_TREE 任务：devWork 已在 ncclPrepareTasks 中预构建，直接从 tmpCollWorkQueue 取出
    if (task->algorithm == NCCL_ALGO_NVLS_TREE || task->algorithm == NCCL_ALGO_NVLS) {
      workNode = ncclIntruQueueDequeue(&planner->tmpCollWorkQueue);
      goto next;
    }
    // 非 NVLS 任务：尝试注册 buffer（IPC 注册、网络注册等）
    ncclRegisterCollBuffers(comm, task, regBufSend, regBufRecv, &planner->collCleanupQueue, &regNeedConnect);

    // 填充 devWork 结构体：将用户参数转化为 kernel 可读的格式
    devWork.sendbuff = (void*)task->sendbuff;
    devWork.recvbuff = (void*)task->recvbuff;
    devWork.sendbuffOffset = task->sendbuffOffset;
    devWork.recvbuffOffset = task->recvbuffOffset;
    devWork.sendbuffRmtAddrs = task->sendbuffRmtAddrs;
    devWork.recvbuffRmtAddrs = task->recvbuffRmtAddrs;
    devWork.root = task->root;
    devWork.nWarps = task->nWarps;
    devWork.redOpArg = task->opDev.scalarArg;
    devWork.redOpArgIsPtr = task->opDev.scalarArgIsPtr;
    devWork.oneNode = (comm->nNodes == 1);   // 单节点优化标志
    devWork.isOneRPN = comm->isOneRPN;        // 每节点只有一个 rank
    devWork.netRegUsed = devWork.regUsed = 0; // 默认不使用注册 buffer
    devWork.profilerEnabled = ncclProfilerPluginLoaded() && (task->eActivationMask & ncclProfileKernelCh);
    // netRegUsed: NIC 直接读写 GPU 内存（跳过 host bounce buffer，降低延迟）
    if (task->regBufType & NCCL_NET_REG_BUFFER)
      devWork.netRegUsed = 1;
    // regUsed: GPU 直接访问对方 GPU 内存（通过 IPC 或 NVLS）
    if (task->regBufType & (NCCL_IPC_REG_BUFFER | NCCL_NVLS_REG_BUFFER))
      devWork.regUsed = 1;

    // 根据 buffer 注册类型选择 ncclDevWorkCollReg（带注册地址）还是 ncclDevWorkColl（普通）
    // ncclWorkList 是侵入式链表节点，数据紧跟在节点后面（workNode+1）
    if (task->regBufType & NCCL_NVLS_REG_BUFFER) {
      struct ncclDevWorkCollReg workReg = {};
      workReg.coll = devWork; // C++ struct assignment
      /* NVLS only has one send and recv buffer registered */
      workReg.dnInputs[0] = regBufSend[0];
      workReg.dnOutputs[0] = regBufRecv[0];
      workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkCollReg>(&comm->memScoped, 1);
      workNode->workType = ncclDevWorkTypeCollReg;
      workNode->size = sizeof(struct ncclDevWorkCollReg);
      memcpy((void*)(workNode+1), (void*)&workReg, workNode->size);
    } else {
      workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkColl>(&comm->memScoped, 1);
      workNode->workType = ncclDevWorkTypeColl;
      workNode->size = sizeof(struct ncclDevWorkColl);
      memcpy((void*)(workNode+1), (void*)&devWork, workNode->size);
    }
next:
    ncclIntruQueueEnqueue(&planner->collWorkQueue, workNode);
    task = task->next;
  }
  assert(ncclIntruQueueEmpty(&planner->tmpCollWorkQueue));
  return ncclSuccess;
}

// ncclPrepareTasks：每个 ncclGroup 调用一次，对所有用户提交的任务进行分析和调度
// ============================================================================
// 输入：
//   comm->planner 中已经累积了所有通过 ncclEnqueueCheck 入队的任务：
//     - collSorter：所有 collective 任务（按 trafficBytes 排序）
//     - peers[i].sendQueue/recvQueue：P2P 任务
//     - peers[i].bcastQueue：Broadcast 任务
//     - rmaTaskQueue：RMA 任务
// 输出：
//   planner->collTaskQueue：传统 collective 任务（已选好 algo/proto）
//   planner->collSymTaskQueue：symmetric kernel 任务
//   planner->collWorkQueue：对应的 ncclDevWorkColl 结构体
//   algoNeedConnect[]：哪些算法需要延迟建连
//
// 整体流程：
//   1. Broadcast 任务预处理（单 peer 降级为 collective）
//   2. 从 collSorter 取出所有任务（按大小降序）
//   3. ***symk 分流***：ncclMakeSymmetricTaskList 先尝试拿走能用 symk 的任务
//   4. 剩余任务按 (func, op, datatype) 分组
//   5. 对每组调 ncclGetAlgoInfo 选择最优 algo/proto
//   6. 按 collnet/nvls 维度二次分组，拼接成最终的 collTaskQueue
//   7. 再次遍历做 NVLS buffer 注册 + 构建 devWork 结构体
// ============================================================================
// Called once per ncclGroup to organize the user submitted tasks in
// comm->planner so that they can be peeled off into plans.
ncclResult_t ncclPrepareTasks(struct ncclComm* comm, bool* algoNeedConnect, bool* needConnect, ncclSimInfo_t* simInfo) {
  struct ncclKernelPlanner* planner = &comm->planner;
  // 检查是否在 CUDA Graph capture 模式下
  planner->persistent = ncclCudaGraphValid(planner->capturingGraph);

  // ==================== 第 1 阶段：Broadcast 任务预处理 ====================
  // 如果 Broadcast 只有一个 peer，将其降级为普通 collective 任务
  // 这样可以复用 collective 的调度路径（Ring/Tree 等），而不需要单独处理
  // Put bcast tasks into collSorter if there's only one bcast peer
  if (planner->bcast_info.BcastPeers == 1) {
    while (!ncclIntruQueueEmpty(&planner->peers[planner->bcast_info.minBcastPeer].bcastQueue)) {
      struct ncclTaskBcast* bcastTask = ncclIntruQueueDequeue(&planner->peers[planner->bcast_info.minBcastPeer].bcastQueue);
      struct ncclTaskColl *t = ncclMemoryPoolAlloc<struct ncclTaskColl>(&comm->memPool_ncclTaskColl, &comm->memPermanent);
      t->func = ncclFuncBroadcast;
      t->sendbuff = bcastTask->sendbuff;
      t->recvbuff = bcastTask->recvbuff;
      t->count = bcastTask->count;
      t->root = bcastTask->root;
      t->datatype = bcastTask->datatype;
      t->trafficBytes = t->count*ncclFuncTrafficPerByte(t->func, comm->nRanks);
      t->chunkSteps = BROADCAST_CHUNKSTEPS;
      t->sliceSteps = BROADCAST_SLICESTEPS;
      ncclTaskCollSorterInsert(&planner->collSorter, t, t->trafficBytes);
      planner->nTasksColl += 1;
      ncclMemoryPoolFree(&comm->memPool_ncclTaskBcast, bcastTask);
    }
    // reset bcast info
    planner->nTasksBcast = 0;
    planner->bcast_info.BcastPeers = 0;
  }

  // ==================== 第 2 阶段：从 sorter 取出所有 collective 任务 ====================
  // collSorter 是一个堆排序结构，任务按 trafficBytes 降序排出
  // trafficBytes = count * elementSize * ncclFuncTrafficPerByte（考虑算法流量倍数）
  // Tasks from the sorter come out ordered size descending.
  struct ncclTaskColl* task = ncclTaskCollSorterDequeueAll(&planner->collSorter);
  // tasksByFnOpTy: 按 (func, redOp, datatype) 三元组的哈希桶
  // 同一三元组的任务可以共用同一个 kernel function，可以被聚合优化
  // Tasks are assembled by (fn,op,ty) size ascending.
  struct ncclTaskColl* tasksByFnOpTy[ncclNumFuncs*ncclNumDevRedOps*ncclNumTypes];
  memset(tasksByFnOpTy, 0, sizeof(tasksByFnOpTy));
  int fnOpTyIndices[ncclNumFuncs*ncclNumDevRedOps*ncclNumTypes];
  int fnOpTyCount = 0;

  // ==================== 第 3 阶段：Symmetric kernel 分流 ====================
  // 这是 symk 和传统路径的核心分叉点！
  // 条件：
  //   1. comm->symmetricSupport：communicator 支持 symmetric（init 时检查）
  //   2. !comm->p2pCrossClique：不跨 NVLink clique（symk 不支持跨 clique）
  // ncclMakeSymmetricTaskList（定义在 symmetric_sched.cc）将遍历 task 链表：
  //   - 对每个任务调用 ncclSymkPickKernel 尝试匹配 symk kernel
  //   - 匹配成功的任务被转移到 collSymTaskQueue
  //   - 匹配失败的任务留在 task 链表中，继续走传统路径
  // 分流后：
  //   task → 剩余的传统 collective 任务
  //   collSymTaskQueue → symk 任务（将在 ncclLaunchPrepare 中由 ncclSymmetricTaskScheduler 处理）
  // Skip symmetric kernels for cross-clique
  if (comm->symmetricSupport && !comm->p2pCrossClique) {
    NCCLCHECK(ncclMakeSymmetricTaskList(comm, task, &planner->collSymTaskQueue, &task));
  }

  // ==================== 第 4 阶段：剩余任务按 (fn,op,ty) 分组 ====================
  // 将剩余任务按三元组 (func, redOp, datatype) 放入对应的桶中
  // 目的：同一组的任务可以聚合计算 algo/proto，共享同一个 kernel function
  // Walk the size sorted tasks, binning them by (fn,op,ty).
  while (task != nullptr) {
    struct ncclTaskColl* next = task->next;
    int index = ((int)task->func*ncclNumDevRedOps + (int)task->opDev.op)*ncclNumTypes + (int)task->datatype;
    // Add to set of (fn,op,ty) indices on first occurrence
    if (tasksByFnOpTy[index] == nullptr) fnOpTyIndices[fnOpTyCount++] = index;
    // Add to LIFO for this (fn,op,ty)
    task->next = tasksByFnOpTy[index];
    tasksByFnOpTy[index] = task;
    // Next task
    task = next;
  }

  // ==================== 第 5 阶段：按组选择 algo/proto ====================
  // 对每个 (fn,op,ty) 组：
  //   1. 将大小相近的任务聚合（4x 范围内），用聚合后的总流量判断最优算法
  //   2. ncclGetAlgoInfo：核心算法选择函数
  //      输入：总流量、collNet 支持、NVLS 支持、nRanks 等
  //      输出：algorithm（Ring/Tree/NVLS/CollNet）、protocol（LL/LL128/Simple）、
  //             nMaxChannels、nWarps
  //      内部通过 ncclTopoTuneModel 的调度表或插件 tuner 做决策
  //   3. ncclDevFuncId：将 (func, op, datatype, algo, proto) 映射为内核索引
  //   4. 将结果写回每个任务，并按 collnet/nvls 维度分入 collBins[2][2]
  // Walk (fn,op,ty) bins, compute algo and proto etc. Then bin them by their
  // scheduling constraints (collnet x nvls).
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collBins[2][2] = {};
  for (int cursor=0; cursor < fnOpTyCount; cursor++) {
    struct ncclTaskColl* aggBeg = tasksByFnOpTy[fnOpTyIndices[cursor]];
    // 检查该组任务是否支持 CollNet（IB SHARP 硬件 reduction）
    int collNetSupport = 0;
    NCCLCHECK(ncclGetCollNetSupport(comm, aggBeg, &collNetSupport));
    // 检查是否支持 NVLS（NVLink SHARP，节点内 NVSwitch 硬件 multicast）
    int nvlsSupport = comm->nvlsSupport && (ncclNvlsSupported(aggBeg->opDev.op, aggBeg->datatype) || aggBeg->func == ncclFuncAllGather);
    // Crudely estimate number of tasks per channel. This is using the wrong number
    // of channels for NVLS algos, but knowing the algo requires having this value,
    // so either be crude our iterate until fixed point, we chose the former.
    int nTasksPerChannel = divUp(comm->planner.nTasksColl, comm->nChannels);
    do {
      struct ncclTaskColl* aggEnd = aggBeg->next;
      struct ncclTaskColl agg = *aggBeg;
      // 聚合策略：将大小在 4 倍范围内的任务聚合在一起
      // 聚合后使用总 count/trafficBytes 调用 ncclGetAlgoInfo
      // 这样可以让小任务“搭便车”享受大任务的算法选择
      // We aggregate operations that are within 4X size of each other.
      while (aggEnd != nullptr && aggEnd->trafficBytes < 4*aggBeg->trafficBytes) {
        agg.count += aggEnd->count;
        agg.trafficBytes += aggEnd->trafficBytes;
        aggEnd = aggEnd->next;
      }

      // ncclGetAlgoInfo：传统路径的核心算法选择函数
      // 根据聚合后的总流量、nRanks、collNet/NVLS 支持等信息，
      // 返回最优的 algorithm、protocol、nMaxChannels、nWarps
      NCCLCHECK(ncclGetAlgoInfo(comm, &agg, collNetSupport, nvlsSupport, nTasksPerChannel, simInfo));
      // 将 (func, op, datatype, algo, proto) 映射为内核索引
      // 这个索引用于在 ncclDevKernelList 中查找对应的 kernel function pointer
      agg.devFuncId = ncclDevFuncId(agg.func, agg.opDev.op, agg.datatype, agg.algorithm, agg.protocol);

      // 标记算法类型：用于后续按 [collnet][nvls] 维度分组
      int isCollnet=0, isNvls=0;
      switch (agg.algorithm) {
      case NCCL_ALGO_NVLS:
      case NCCL_ALGO_NVLS_TREE:
        isNvls = 1;
        isCollnet = agg.algorithm == NCCL_ALGO_NVLS && comm->nNodes > 1;
        break;
      case NCCL_ALGO_COLLNET_CHAIN:
      case NCCL_ALGO_COLLNET_DIRECT:
        isCollnet = 1;
        break;
      }
      // 将聚合的 algo/proto 结果写回每个独立任务
      // 同时将任务放入 collBins[isCollnet][isNvls] 分组
      // Update the aggregated tasks with the computed values.
      do {
        struct ncclTaskColl* next = aggBeg->next;
        aggBeg->algorithm = agg.algorithm;
        aggBeg->protocol = agg.protocol;
        if (aggBeg->protocol == NCCL_PROTO_LL) aggBeg->trafficBytes *= 4;
        aggBeg->nMaxChannels = agg.nMaxChannels;
        aggBeg->nWarps = agg.nWarps;
        aggBeg->devFuncId = agg.devFuncId;
        aggBeg->isCollnet = isCollnet;
        aggBeg->isNvls = isNvls;
        ncclIntruQueueEnqueue(&collBins[isCollnet][isNvls], aggBeg);
        aggBeg = next;
      } while (aggBeg != aggEnd);
    } while (aggBeg != nullptr);
  }

  // ==================== 第 6 阶段：拼接所有分组到 collTaskQueue ====================
  // 拼接顺序：[0][0]普通 → [0][1]NVLS → [1][0]CollNet → [1][1]NVLS+CollNet
  // CollNet 作为外层维度，因为 CollNet 和非-CollNet 的 channel 分配策略不同
  // Concatenate `collBins[*][*]` together into final list `planner->collTaskQueue`.
  // Collnet is the outer dimension since that affects how we divide over the
  // channels.
  for (int isCollnet=0; isCollnet <= 1; isCollnet++) {
    for (int isNvls=0; isNvls <= 1; isNvls++) {
      ncclIntruQueueTransfer(&planner->collTaskQueue, &collBins[isCollnet][isNvls]);
    }
  }

  // ==================== 第 7 阶段：NVLS buffer 注册 + 构建 devWork 结构体 ====================
  // 再次遍历所有任务，进行三件事：
  //   1. 尝试注册 NVLS buffer（ncclRegisterCollNvlsBuffers）
  //      NVLS 需要将用户 buffer 注册到 NVSwitch multicast group 中
  //   2. 检查是否需要延迟建连（runtimeConn 模式）
  //      新版 NCCL 支持算法级别的延迟连接：只有第一次使用某算法时才建立连接
  //   3. 为 NVLS/NVLS_TREE 任务预构建 ncclDevWorkColl 结构体
  //      放入 tmpCollWorkQueue，后续由 ncclTasksRegAndEnqueue 合并处理
  // Walk tasks again to:
  // 1. Possibly register buffers.
  // 2. Build ncclDevWorkColl structs.
  // 3. Bin the work structs according to the number of valid channels they
  //    may be assigned to {collnet, nvls, standard}
  task = ncclIntruQueueHead(&planner->collTaskQueue);
  while (task != nullptr) {
    // Build a ncclDevWorkColl[Reg?] struct for each task.
    void* regBufSend[NCCL_MAX_LOCAL_RANKS];
    void* regBufRecv[NCCL_MAX_LOCAL_RANKS];
    bool regNeedConnect = true;
    ncclRegisterCollNvlsBuffers(comm, task, regBufSend, regBufRecv, &planner->collCleanupQueue, &regNeedConnect);

    // runtimeConn：延迟连接机制
    // 如果这个算法还没有建立连接，标记 algoNeedConnect
    // 连接将在 ncclCollPreconnect 中建立（发生在 ncclPrepareTasks 之后）
    if (comm->runtimeConn && comm->initAlgoChannels[task->algorithm] == false) {
      if (task->algorithm == NCCL_ALGO_NVLS_TREE && comm->initAlgoChannels[NCCL_ALGO_NVLS] == false && regNeedConnect == true) {
        comm->initAlgoChannels[NCCL_ALGO_NVLS] = true;
        algoNeedConnect[NCCL_ALGO_NVLS] = true;
      }
      if (task->algorithm != NCCL_ALGO_NVLS || regNeedConnect == true) {
        comm->initAlgoChannels[task->algorithm] = true;
        algoNeedConnect[task->algorithm] = true;
        *needConnect = true;
      }
    }

    // 为 NVLS/NVLS_TREE 任务预构建 ncclDevWorkColl 结构体
    // 这些任务的 devWork 需要在这里提前构建，因为它们可能需要 NVLS 注册 buffer 地址
    if (task->algorithm == NCCL_ALGO_NVLS_TREE || task->algorithm == NCCL_ALGO_NVLS) {
      struct ncclDevWorkColl devWork = {};
      // 将用户层面的参数填入 devWork，这些会被传递给 GPU kernel
      devWork.sendbuff = (void*)task->sendbuff;
      devWork.recvbuff = (void*)task->recvbuff;
      devWork.sendbuffOffset = task->sendbuffOffset;
      devWork.recvbuffOffset = task->recvbuffOffset;
      devWork.sendbuffRmtAddrs = task->sendbuffRmtAddrs;
      devWork.recvbuffRmtAddrs = task->recvbuffRmtAddrs;
      devWork.root = task->root;
      devWork.nWarps = task->nWarps;
      devWork.redOpArg = task->opDev.scalarArg;
      devWork.redOpArgIsPtr = task->opDev.scalarArgIsPtr;
      devWork.oneNode = (comm->nNodes == 1);
      devWork.netRegUsed = devWork.regUsed = 0;
      devWork.profilerEnabled = ncclProfilerPluginLoaded() && (task->eActivationMask & ncclProfileKernelCh);
      if (task->regBufType & NCCL_NET_REG_BUFFER)
        devWork.netRegUsed = 1;
      if (task->regBufType & (NCCL_IPC_REG_BUFFER | NCCL_NVLS_REG_BUFFER))
        devWork.regUsed = 1;

      struct ncclWorkList* workNode;
      if (task->regBufType & NCCL_NVLS_REG_BUFFER) {
        struct ncclDevWorkCollReg workReg = {};
        workReg.coll = devWork; // C++ struct assignment
        /* NVLS only has one send and recv buffer registered */
        workReg.dnInputs[0] = regBufSend[0];
        workReg.dnOutputs[0] = regBufRecv[0];
        workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkCollReg>(&comm->memScoped, 1);
        workNode->workType = ncclDevWorkTypeCollReg;
        workNode->size = sizeof(struct ncclDevWorkCollReg);
        memcpy((void*)(workNode + 1), (void*)&workReg, workNode->size);
      } else {
        workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkColl>(&comm->memScoped, 1);
        workNode->workType = ncclDevWorkTypeColl;
        workNode->size = sizeof(struct ncclDevWorkColl);
        memcpy((void*)(workNode + 1), (void*)&devWork, workNode->size);
      }

      ncclIntruQueueEnqueue(&planner->tmpCollWorkQueue, workNode);
    }
    task = task->next;
  }

  // ==================== 第 8 阶段：Broadcast runtimeConn 处理 ====================
  // 对于多 peer broadcast 任务（未降级为 collective 的），
  // 检查是否需要为 Ring 算法建立连接
  // Process broadcast tasks for runtimeConn
  if (comm->runtimeConn && planner->nTasksBcast > 0) {
    for (int peer = planner->bcast_info.minBcastPeer; peer <= planner->bcast_info.maxBcastPeer; peer++) {
      struct ncclTaskBcast* bcastTask = ncclIntruQueueHead(&planner->peers[peer].bcastQueue);
      while (bcastTask != nullptr) {
        if (comm->initAlgoChannels[NCCL_ALGO_RING] == false) {
          comm->initAlgoChannels[NCCL_ALGO_RING] = true;
          algoNeedConnect[NCCL_ALGO_RING] = true;
          *needConnect = true;
        }
        bcastTask = bcastTask->next;
      }
    }
  }

  return ncclSuccess;
}

static ncclResult_t addProfilerProxyOpIfNeeded(struct ncclComm* comm, struct ncclKernelPlan* plan, struct ncclProxyOp* op) {
  int tmp = op->pattern;
  op->pattern = ncclPatternProfiler;
  ncclResult_t ret = ncclAddProxyOpIfNeeded(comm, plan, op);
  op->pattern = tmp;
  return ret;
}

// scheduleCollTasksToPlan：将传统 collective 任务分配到 plan 中
// ============================================================================
// 这是传统路径（非 symk）的核心调度函数。它负责：
//   1. 估算当前 plan 能装多少个任务（受 budget 限制）
//   2. 为每个任务分配 channel（基于流量均衡的 "shortest channel first" 策略）
//   3. 计算 chunk size（calcCollChunking）
//   4. 生成 ncclDevWorkBatch（调用 ncclAddWorkBatchToPlan）
//   5. 生成 ncclProxyOp（调用 ncclAddProxyOpIfNeeded）
//
// channel 分配策略（非 CollNet）：
//   - 根据总流量和 channel 数计算每个 channel 的目标流量
//   - 一个任务可以横跨多个 channel（countLo、countMid、countHi 三段）
//   - Lo = 第一个 channel（可能是前一个任务的尾巴）
//   - Mid = 中间的满 channel（每个处理 countMid 个元素）
//   - Hi = 最后一个 channel（可能未填满）
//
// budget 限制：
//   inArgsBytes：kernel args 中 batch 占用的字节数上限
//   outArgsBytes：work 占用的字节数上限（Fifo 模式下为 workFifo 的一半）
// ============================================================================
static ncclResult_t scheduleCollTasksToPlan(
    struct ncclComm* comm, struct ncclKernelPlan* plan, struct ncclKernelPlanBudget* budget
  ) {
  struct ncclKernelPlanner* planner = &comm->planner;
  // --- 第 1 步：估算能装多少个任务 ---
  // 遍历 collTaskQueue，累加 workBytes 和 batch 数，直到超出 budget
  // nPlanColls = 本次 plan 要处理的任务数
  // trafficBytes[kind] = 每种类型的总流量，用于 channel 分配
  // Estimate number of tasks that will fit in this plan.
  int nPlanColls = 0;
  size_t trafficBytes[2*2] = {0, 0, 0, 0}; // [collnet][nvls]
  int nChannels[2*2] = {0, 0, 0, 0}; // [collnet][nvls]
  int const nMaxChannels[2*2] = {comm->nChannels, comm->nvlsChannels, // [collnet][nvls]
                                 comm->nChannels, std::min(comm->nChannels, comm->nvlsChannels)};
  constexpr size_t MinTrafficPerChannel = 32 << 10; // 32K traffic as minimal
  do {
    size_t workBytes = 0;
    struct ncclTaskColl* task = ncclIntruQueueHead(&planner->collTaskQueue);
    struct ncclWorkList* workNode = ncclIntruQueueHead(&planner->collWorkQueue);
    while (task != nullptr) {
      int nBatches = divUp(nPlanColls, 4); // Rough guess: 4 colls per batch.
      if (!ncclTestBudget(budget, nBatches, workBytes + workNode->size)) goto plan_full;

      nPlanColls += 1;
      workBytes += workNode->size;
      int kind = 2*task->isCollnet + task->isNvls;
      trafficBytes[kind] += std::max(MinTrafficPerChannel, task->trafficBytes);
      nChannels[kind] += task->nMaxChannels;
      nChannels[kind] = std::min(nChannels[kind], nMaxChannels[kind]);
      task = task->next;
      workNode = workNode->next;
    }
  plan_full:;
  } while (0);

  // --- 第 2 步：逐个任务分配 channel 并生成 workBatch + proxyOp ---
  int kindPrev = -1;
  size_t trafficPerChannel = 0;  // 每个 channel 的目标流量
  int channelId = 0;             // 当前走到哪个 channel
  size_t currentTraffic = 0;     // 当前 channel 已累积的流量
  while (nPlanColls!=0 && !ncclIntruQueueEmpty(&planner->collTaskQueue)) {
    struct ncclTaskColl* task = ncclIntruQueueHead(&planner->collTaskQueue);
    struct ncclWorkList* workNode = ncclIntruQueueHead(&planner->collWorkQueue);
    struct ncclDevWorkColl* devWork = (struct ncclDevWorkColl*)(workNode+1);
    size_t elementSize = ncclTypeSize(task->datatype);

    // 当 kind 切换时（如从普通切到 NVLS），重新计算 trafficPerChannel
    // trafficPerChannel = 总流量 / channel 数，16 字节对齐
    int kind = 2*task->isCollnet + task->isNvls;
    if (kind != kindPrev) {
      trafficPerChannel = divUp(trafficBytes[kind] / nChannels[kind], 16) * 16;
      kindPrev = kind;
      channelId = 0;
      currentTraffic = 0;
    }

    if (task->isCollnet) {
      // === CollNet 路径 ===
      // CollNet 任务使用所有可用 channel，每个 channel 处理相同的 count
      // 这是因为 CollNet 通过硬件 SHARP 做 reduction，channel 间不需要分割数据
      int nChannels = task->nMaxChannels;
      // Ensure room for worst case of one new batch per channel
      if (!ncclTestBudget(budget, plan->nWorkBatches + nChannels, plan->workBytes + workNode->size)) {
        return ncclSuccess;
      }

      size_t globalBytesPerElement = elementSize*ncclFuncMaxSendRecvCount(task->func, comm->nRanks, 1);
      struct ncclProxyOp proxyOp;
      uint32_t chunkSize, directFlags=0;
      NCCLCHECK(calcCollChunking(comm, task, nChannels, globalBytesPerElement*task->count, &chunkSize, &directFlags, &proxyOp));
      devWork->channelLo = 0;
      devWork->channelHi = nChannels-1;
      devWork->collnet.count = task->count;
      devWork->collnet.chunkCount = chunkSize/ncclTypeSize(task->datatype);
      devWork->direct = directFlags;

      uint64_t proxyOpId = uint64_t(plan->collOpCount++)<<1 | 0;
      for (int c=devWork->channelLo; c <= (int)devWork->channelHi; c++) {
        proxyOp.channelId = c;
        proxyOp.opCount = proxyOpId;
        proxyOp.task.coll = task;
        proxyOp.rank = comm->rank;
        proxyOp.eActivationMask = task->eActivationMask;
        proxyOp.incWorkCounter = true;
        ncclAddWorkBatchToPlan(comm, plan, c, workNode->workType, task->devFuncId, plan->workBytes);
        // Set pattern to profiler to add a proxy profiler for kernel events
        NCCLCHECK(ncclAddProxyOpIfNeeded(comm, plan, &proxyOp));
        NCCLCHECK(addProfilerProxyOpIfNeeded(comm, plan, &proxyOp));
      }
    } else { // not task->isCollnet
      // === 普通 collective 路径（Ring/Tree/NVLS） ===
      // 基于流量均衡的 channel 分配策略：
      //   trafficPerCell = 每个“数据单元”的流量（考虑算法倍数和协议倍数）
      //   将任务切分为 countLo/countMid/countHi 三段，分配到连续的 channel 上
      int trafficPerByte = ncclFuncTrafficPerByte(task->func, comm->nRanks);
      if (task->protocol == NCCL_PROTO_LL) trafficPerByte *= 4;
      size_t cellSize = divUp(divUp(MinTrafficPerChannel, (size_t)trafficPerByte), 16) * 16;
      int elementsPerCell = cellSize/elementSize;
      size_t cells = divUp(task->count*elementSize, cellSize);
      size_t trafficPerElement = elementSize*trafficPerByte;
      size_t trafficPerCell = cellSize*trafficPerByte;
      size_t cellsPerChannel = std::min(cells, divUp(trafficPerChannel, trafficPerCell));
      size_t cellsLo;
      if (channelId+1 == nMaxChannels[kind]) { // On last channel everything goes to "lo"
        cellsLo = cells;
      } else {
        cellsLo = std::min(cells, divUp((trafficPerChannel-currentTraffic),trafficPerCell));
      }
      int nMidChannels = (cells-cellsLo)/cellsPerChannel;
      size_t cellsHi = (cells-cellsLo)%cellsPerChannel;
      int nChannels = (cellsLo!=0 ? 1 : 0) + nMidChannels + (cellsHi!=0 ? 1 : 0);
      if (nMaxChannels[kind] < channelId + nChannels) { // Overflowed available channels
        nMidChannels = nMaxChannels[kind] - channelId - 2;
        cellsPerChannel = (cells-cellsLo)/(nMidChannels+1);
        cellsHi = cellsPerChannel + (cells-cellsLo)%(nMidChannels+1);
      }
      if (cellsHi == 0 && nMidChannels != 0) {
        cellsHi = cellsPerChannel;
        nMidChannels -= 1;
      }
      if (cellsLo == 0) { // Least channel skipped. Make the next channel the new least.
        channelId += 1;
        if (nMidChannels == 0) { cellsLo = cellsHi; cellsHi = 0; }
        else { cellsLo = cellsPerChannel; nMidChannels -= 1; }
      }
      size_t countMid = nMidChannels!=0 ? cellsPerChannel*elementsPerCell : 0;
      size_t countLo = cellsLo*elementsPerCell;
      size_t countHi = cellsHi*elementsPerCell;
      (countHi != 0 ? countHi : countLo) -= cells*elementsPerCell - task->count;

      nChannels = (countLo!=0 ? 1 : 0) + nMidChannels + (cellsHi!=0 ? 1 : 0);

      // Update number of channels propagated to the profiler
      task->nChannels = (uint8_t)nChannels;

      // Ensure room for worst case of one new batch per channel
      if (!ncclTestBudget(budget, plan->nWorkBatches + nChannels, plan->workBytes + workNode->size)) {
        return ncclSuccess;
      }

      devWork->channelLo = channelId;
      devWork->channelHi = channelId + nChannels-1;
      devWork->cbd.countLo = countLo;
      devWork->cbd.countMid = countMid;
      devWork->cbd.countHi = countHi;

      // calcCollChunking() uses global bytes instead of traffic which differs
      // in that allreduce isn't multiplied by 2.
      size_t globalBytesPerElement = elementSize*ncclFuncMaxSendRecvCount(task->func, comm->nRanks, 1);
      // 为 Lo/Mid/Hi 每段分别计算 chunk size
      // chunk size 决定了 kernel 每次处理多少数据（影响 pipeline 深度）
      // calcCollChunking 考虑：数据量、channel 数、pipeline slot 数、协议 grain size
      struct ncclProxyOp proxyOpLo, proxyOpMid, proxyOpHi;

      uint32_t chunkSize, directFlags=0;
      size_t grainSize = ncclProtoGrainSize(task->protocol);
      if (countLo != 0) {
        NCCLCHECK(calcCollChunking(comm, task, /*nChannels=*/1, globalBytesPerElement*countLo, &chunkSize, &directFlags, &proxyOpLo));
        devWork->cbd.chunkGrainsLo = chunkSize/grainSize;
      }
      if (countHi != 0) {
        NCCLCHECK(calcCollChunking(comm, task, /*nChannels=*/1, globalBytesPerElement*countHi, &chunkSize, &directFlags, &proxyOpHi));
        devWork->cbd.chunkGrainsHi = chunkSize/grainSize;
      }
      if (nMidChannels != 0) {
        NCCLCHECK(calcCollChunking(comm, task, /*nChannels=*/1, globalBytesPerElement*countMid, &chunkSize, &directFlags, &proxyOpMid));
        devWork->cbd.chunkGrainsMid = chunkSize/grainSize;
      }
      devWork->direct = directFlags;

      // Update the current channel and vacant traffic budget.
      if (countHi != 0) {
        channelId += nChannels-1;
        currentTraffic = cellsHi*elementsPerCell*trafficPerElement;
      } else if (nMidChannels != 0) {
        channelId += nChannels;
        currentTraffic = 0;
      } else {
        currentTraffic += cellsLo*elementsPerCell*trafficPerElement;
      }

      if (currentTraffic >= trafficPerChannel && channelId+1 != nMaxChannels[kind]) {
        channelId += 1;
        currentTraffic = 0;
      }

      // 为每个 channel 生成 proxyOp + workBatch
      // proxyOpId: 全局唯一的操作 ID，用于 proxy 线程匹配完成通知
      // 低 1 位 = 0 表示 collective，= 1 表示 p2p
      uint64_t proxyOpId = uint64_t(plan->collOpCount++)<<1 | 0;
      for (int c=devWork->channelLo; c <= (int)devWork->channelHi; c++) {
        struct ncclProxyOp* proxyOp;
        if (c == (int)devWork->channelLo) {
          proxyOp = &proxyOpLo;
          proxyOp->loopOffset = 0;
          proxyOp->channelSize = countLo * elementSize;
        } else if (c == (int)devWork->channelHi) {
          proxyOp = &proxyOpHi;
          proxyOp->loopOffset = (countLo + nMidChannels * countMid) * elementSize;
          proxyOp->channelSize = countHi * elementSize;
        } else {
          proxyOp = &proxyOpMid;
          proxyOp->loopOffset = (countLo + (c - devWork->channelLo - 1) * countMid) * elementSize;
          proxyOp->channelSize = countMid * elementSize;
        }
        proxyOp->channelId = c;
        proxyOp->opCount = proxyOpId;
        proxyOp->task.coll = task;
        proxyOp->rank = comm->rank;
        proxyOp->ringAlgo = NULL;
        if (proxyOp->reg && task->algorithm == NCCL_ALGO_RING && (task->recvNetHandles[c] || task->sendNetHandles[c])) {
          if (task->func == ncclFuncAllGather) {
            proxyOp->ringAlgo = new RingAGAlgorithm(task->sendbuff, task->recvbuff, comm->nRanks, comm->channels[c].ring.userRanks, proxyOp->chunkSteps, proxyOp->sliceSteps, proxyOp->chunkSize, proxyOp->sliceSize, proxyOp->loopOffset, proxyOp->channelSize, elementSize, task->count * elementSize, task->sendNetHandles[c], task->recvNetHandles[c], task->srecvNetHandles[c]);
          } else if (task->func == ncclFuncAllReduce) {
            proxyOp->ringAlgo = new RingARAlgorithm(task->sendbuff, task->recvbuff, comm->nRanks, comm->channels[c].ring.index, proxyOp->chunkSteps, proxyOp->sliceSteps, proxyOp->chunkSize, proxyOp->sliceSize, proxyOp->loopOffset, proxyOp->channelSize, elementSize, task->sendNetHandles[c], task->recvNetHandles[c], task->srecvNetHandles[c]);
          } else if (task->func == ncclFuncBroadcast) {
            proxyOp->ringAlgo = new RingBCAlgorithm(task->sendbuff, task->recvbuff, comm->rank, task->root, comm->nRanks, comm->channels[c].ring.userRanks, proxyOp->chunkSteps, proxyOp->sliceSteps, proxyOp->chunkSize, proxyOp->sliceSize, proxyOp->loopOffset, proxyOp->channelSize, task->sendNetHandles[c], task->recvNetHandles[c], task->srecvNetHandles[c]);
          }
          proxyOp->ringAlgo->incRefCount();
        }
        proxyOp->eActivationMask = task->eActivationMask;
        proxyOp->incWorkCounter = true;
        proxyOp->nChannels = nChannels;
        ncclAddWorkBatchToPlan(comm, plan, c, workNode->workType, task->devFuncId, plan->workBytes);
        // Coverity reports "proxyOp->connection" as being possibly uninitialized.  It's hard to
        // determine if that's actually true but it's also not clear if that would be an issue.
        // coverity[uninit_use_in_call:FALSE]
        NCCLCHECK(ncclAddProxyOpIfNeeded(comm, plan, proxyOp));
        NCCLCHECK(addProfilerProxyOpIfNeeded(comm, plan, proxyOp));
      }
    }

    // 更新 plan 的 channelMask（记录哪些 channel 被使用）
    plan->channelMask |= (2ull<<devWork->channelHi) - (1ull<<devWork->channelLo);
    plan->threadPerBlock = std::max(plan->threadPerBlock, task->nWarps*WARP_SIZE);
    // 选择 kernel function：如果已经有特化的 kernel，就不再更新
    // kernelSpecialized = true 意味着找到了一个可以处理当前所有任务的专用 kernel
    if (!plan->kernelSpecialized) {
      plan->kernelFn = ncclDevKernelForFunc[task->devFuncId];
      plan->kernelSpecialized = ncclDevKernelForFuncIsSpecialized[task->devFuncId];
    }
    // Profiler
    plan->groupApiEventHandle = task->groupApiEventHandle;

    if (comm->rank == 0) {
      INFO(NCCL_TUNING, "%s: %ld Bytes -> Algo %s proto %s channel{Lo..Hi}={%d..%d}",
        ncclFuncToString(task->func), task->count * ncclTypeSize(task->datatype), ncclAlgoToString(task->algorithm),
        ncclProtoToString(task->protocol), devWork->channelLo, devWork->channelHi);

      if (task->isCollnet) {
        TRACE(NCCL_COLL, "Collective %s(%s, %s, %s, %s) count=%ld devFuncId=%d channel{Lo..Hi}={%d..%d} count=%ld chunkCount=%d",
          ncclFuncToString(task->func), ncclDevRedOpToString(task->opDev.op),
          ncclDatatypeToString(task->datatype), ncclAlgoToString(task->algorithm),
          ncclProtoToString(task->protocol),
          (long)task->count, task->devFuncId, devWork->channelLo, devWork->channelHi,
          (long)devWork->collnet.count, devWork->collnet.chunkCount);
      } else {
        TRACE(NCCL_COLL, "Collective %s(%s, %s, %s, %s) count=%ld devFuncId=%d channel{Lo..Hi}={%d..%d} count{Lo,Mid,Hi}={%ld,%ld,%ld} chunkBytes{Lo,Mid,Hi}={%d,%d,%d}",
          ncclFuncToString(task->func), ncclDevRedOpToString(task->opDev.op),
          ncclDatatypeToString(task->datatype), ncclAlgoToString(task->algorithm),
          ncclProtoToString(task->protocol),
          (long)task->count, task->devFuncId, devWork->channelLo, devWork->channelHi,
          (long)devWork->cbd.countLo, (long)devWork->cbd.countMid, (long)devWork->cbd.countHi,
          int(devWork->cbd.chunkGrainsLo*ncclProtoGrainSize(task->protocol)),
          int(devWork->cbd.chunkGrainsMid*ncclProtoGrainSize(task->protocol)),
          int(devWork->cbd.chunkGrainsHi*ncclProtoGrainSize(task->protocol)));
      }
    }

    for (int i=0; i < task->nCleanupQueueElts; i++) {
      ncclIntruQueueEnqueue(&plan->cleanupQueue, ncclIntruQueueDequeue(&planner->collCleanupQueue));
    }
    ncclIntruQueueDequeue(&planner->collTaskQueue);
    ncclIntruQueueDequeue(&planner->collWorkQueue);
    nPlanColls -= 1;
    planner->nTasksColl -= 1;
    ncclIntruQueueEnqueue(&plan->collTaskQueue, task);
    ncclIntruQueueEnqueue(&plan->workQueue, workNode);
    plan->workBytes += workNode->size;
  }
  return ncclSuccess;
}

NCCL_PARAM(P2pLLThreshold, "P2P_LL_THRESHOLD", 16384);
NCCL_PARAM(ChunkSize, "CHUNK_SIZE", 0);

// Put p2p op in plan assuming there is sizeof(ncclDevWorkBatch) in batch budget
// and sizeof(ncclDevWorkP2p) in work budget. "sendRank" and "recvRank" must
// match the corresponding values for this round of the p2p schedule (no -1's).
// No-op's are encoded with a -1 size.
static ncclResult_t addP2pToPlan(
    struct ncclComm* comm, struct ncclKernelPlan* plan,
    int nChannelsMin, int nChannelsMax, int p2pEpoch, int p2pRound,
    int sendRank, void* sendAddr, ssize_t sendBytes,
    int recvRank, void* recvAddr, ssize_t recvBytes,
    const int planTotalTasks[], struct ncclTaskP2p** p2pTasks
  ) {
  ncclResult_t ret = ncclSuccess;
  constexpr int connIndex = 1;
  bool selfSend = (sendRank == comm->rank);
  // recv: dir=0, send: dir=1
  void* addrs[2] = {recvAddr, sendAddr};
  ssize_t bytes[2] = {recvBytes, sendBytes};
  bool protoLL[2] = {!selfSend, !selfSend};
  bool network[2] = {false, false};
  bool proxySameProcess[2] = {true, true};
  void** handles[2] = {NULL, NULL};
  uint8_t base = ncclP2pChannelBaseForRound(comm, p2pRound);
  struct ncclProxyOp proxyOps[2] = {};
  int nProxyOps = selfSend ? 0 : 2;
  if (!selfSend) {
    for (int part=0; part < nChannelsMax; part++) {
      int channelId = ncclP2pChannelForPart(comm->p2pnChannels, base, part);
      struct ncclChannelPeer** channelPeers = comm->channels[channelId].peers;
      for (int dir=0; dir <= 1; dir++) {
        int peerRank = dir ? sendRank : recvRank;
        struct ncclConnector* conn = dir ? &channelPeers[peerRank]->send[connIndex]
                                         : &channelPeers[peerRank]->recv[connIndex];
        protoLL[dir] &= conn->conn.buffs[NCCL_PROTO_LL] != nullptr;
        network[dir] |= conn->transportComm == (dir ? &netTransport.send : &netTransport.recv);
        proxySameProcess[dir] &= conn->proxyConn.sameProcess;
      }
    }
  }

  ssize_t paramChunkSize = ncclParamChunkSize();
  // Arrays indexed by dir where recv=0, send=1:
  int nChannels[2];
  int protocol[2];
  int stepSize[2];
  int chunkSize[2];
  int chunkDataSize[2];
  int chunkDataSize_u32fp8[2];
  bool netRegistered[2] = {false, false};
  bool ipcRegistered[2] = {false, false};

  for (int dir=0; dir < 2; dir++) { // 0=recv, 1=send
    // Assume SIMPLE protocol to start with to determine number of channels
    stepSize[dir] = comm->p2pChunkSize;

    if (bytes[dir] == -1) nChannels[dir] = 0;
    else if (bytes[dir] == 0) nChannels[dir] = 1;
    else {
      ssize_t minPartSize = comm->nNodes > 1 ? stepSize[dir]/2 : stepSize[dir]/8;
      ssize_t maxPartSize = comm->nNodes > 1 ? stepSize[dir]   : stepSize[dir]*32;
      nChannels[dir] = std::min<int>(nChannelsMin, divUp(bytes[dir], minPartSize));
      size_t partSize = std::max(minPartSize, divUp(bytes[dir], nChannels[dir]));
      while (partSize > maxPartSize && nChannels[dir] <= nChannelsMax/2) {
        nChannels[dir] *= 2;
        partSize = divUp(bytes[dir], nChannels[dir]);
      }
    }
    // Update number of channels propagated to the profiler
    if (p2pTasks[dir]) p2pTasks[dir]->nChannels = nChannels[dir];

    // Select protocol (LL vs SIMPLE) used based on payload per channel
    if (bytes[dir] != -1)
      protoLL[dir] &= bytes[dir] <= nChannels[dir] * ncclParamP2pLLThreshold();
    protocol[dir] = protoLL[dir] ? NCCL_PROTO_LL : NCCL_PROTO_SIMPLE;

    stepSize[dir] = comm->buffSizes[protocol[dir]]/NCCL_STEPS;
    if (protocol[dir] == NCCL_PROTO_SIMPLE) stepSize[dir] = comm->p2pChunkSize;
    chunkSize[dir] = stepSize[dir];
    if (paramChunkSize != 0) {
      chunkSize[dir] = paramChunkSize;
    } else if (network[dir]) {
      // Tune chunk size for the network
      if (protocol[dir] == NCCL_PROTO_SIMPLE && bytes[dir] < stepSize[dir]) chunkSize[dir] /= 4;
      else if (bytes[dir] < 8*stepSize[dir]) chunkSize[dir] /= 2;
    }

    chunkDataSize[dir] = chunkSize[dir];
    if (protocol[dir] == NCCL_PROTO_LL) chunkDataSize[dir] /= 2;
    chunkDataSize_u32fp8[dir] = u32fp8Encode(chunkDataSize[dir]);
    chunkDataSize[dir] = u32fp8Decode(chunkDataSize_u32fp8[dir]);
    chunkSize[dir] = chunkDataSize[dir];
    if (protocol[dir] == NCCL_PROTO_LL) chunkSize[dir] *= 2;

    if (p2pTasks[dir] && p2pTasks[dir]->allowUB) {
      if (network[dir]) {
        bool pxnUsed = !ncclPxnDisable(comm) && comm->isAllNvlink && comm->maxLocalRanks > 1;
        if (bytes[dir] > 0 && proxySameProcess[dir] && protocol[dir] == NCCL_PROTO_SIMPLE && (!pxnUsed)) {
          int regFlag = 0;
          NCCLCHECKGOTO(ncclCalloc(&handles[dir], nChannelsMax), ret, cleanup);
          for (int part = 0; part < nChannelsMax; part++) {
            int channelId = ncclP2pChannelForPart(comm->p2pnChannels, base, part);
            struct ncclChannelPeer** channelPeers = comm->channels[channelId].peers;
            int peerRank = dir ? sendRank : recvRank;
            struct ncclConnector* conn = dir ? &channelPeers[peerRank]->send[connIndex]
              : &channelPeers[peerRank]->recv[connIndex];
              if (conn->conn.flags & NCCL_DIRECT_NIC)
                ncclRegisterP2pNetBuffer(comm, addrs[dir], bytes[dir], conn, &regFlag, &handles[dir][part], &plan->cleanupQueue);
              if (!regFlag) break;
          }
          netRegistered[dir] = regFlag ? true : false;
        }
      } else if (bytes[dir] > 0 && addrs[dir] && protocol[dir] == NCCL_PROTO_SIMPLE && !selfSend) {
        int peerRank = dir ? sendRank : recvRank;
        int regFlag = 0;
        int channelId = ncclP2pChannelForPart(comm->p2pnChannels, base, 0);
        struct ncclChannelPeer** channelPeers = comm->channels[channelId].peers;
        struct ncclConnector* conn = dir ? &channelPeers[peerRank]->send[connIndex]
          : &channelPeers[peerRank]->recv[connIndex];
          void* regAddr = NULL;
          if (conn->conn.flags & (NCCL_P2P_WRITE | NCCL_P2P_READ)) {
            // We require users registering buffers on both sides
            NCCLCHECKGOTO(ncclRegisterP2pIpcBuffer(comm, addrs[dir], bytes[dir], peerRank, &regFlag, &regAddr, &plan->cleanupQueue), ret, cleanup);
            if (regFlag) {
              if (dir == 0 && (conn->conn.flags & NCCL_P2P_WRITE)) recvAddr = regAddr;
              else if (dir == 1 && (conn->conn.flags & NCCL_P2P_READ)) sendAddr = regAddr;
            }
          }
          ipcRegistered[dir] = regFlag ? true : false;
      }
    }
  }

  struct ncclWorkList* workNode;
  workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkP2p>(&comm->memScoped, 1);
  workNode->workType = ncclDevWorkTypeP2p;
  workNode->size = sizeof(struct ncclDevWorkP2p);
  ncclIntruQueueEnqueue(&plan->workQueue, workNode);
  uint32_t workOffset;
  workOffset = plan->workBytes;
  plan->workBytes += sizeof(struct ncclDevWorkP2p);

  struct ncclDevWorkP2p* work;
  work = (struct ncclDevWorkP2p*)(workNode+1);
  work->nP2pChannels = comm->p2pnChannels;
  work->channelBase = base;
  work->nSendChannels = nChannels[1];
  work->sendProtoLL = protoLL[1];
  work->sendNetReg = netRegistered[1];
  work->sendIpcReg = ipcRegistered[1];
  work->sendChunkSize_u32fp8 = chunkDataSize_u32fp8[1];
  work->sendRank = sendRank;
  work->sendAddr = sendAddr;
  work->sendBytes = sendBytes==-1 ? 0 : sendBytes;
  work->nRecvChannels = nChannels[0];
  work->recvProtoLL = protoLL[0];
  work->recvNetReg = netRegistered[0];
  work->recvIpcReg = ipcRegistered[0];
  work->recvChunkSize_u32fp8 = chunkDataSize_u32fp8[0];
  work->recvRank = recvRank;
  work->recvAddr = recvAddr;
  work->recvBytes = recvBytes==-1 ? 0 : recvBytes;
  work->profilerEnabled = ncclProfilerPluginLoaded() && ((p2pTasks[0] ? p2pTasks[0] : p2pTasks[1])->eActivationMask & ncclProfileKernelCh);

  for (int dir=0; dir < nProxyOps; dir++) {
    struct ncclProxyOp* op = &proxyOps[dir];
    op->root = dir ? sendRank : recvRank;
    op->sliceSteps = 1;
    op->chunkSteps = 1;
    op->dtype = ncclInt8;
    op->redOp = ncclSum;
    op->protocol = protocol[dir];
    op->pattern = dir ? ncclPatternSend : ncclPatternRecv;
    op->chunkSize = chunkSize[dir];
    op->reg = netRegistered[dir];
    op->coll = p2pTasks[dir] ? p2pTasks[dir]->func : 0;
    op->collAPI = p2pTasks[dir] ? p2pTasks[dir]->collAPI : 0;
    op->task.p2p = p2pTasks[dir];
    op->rank = comm->rank;
    op->eActivationMask = p2pTasks[dir] ? p2pTasks[dir]->eActivationMask : 0;
    // The following are modified per channel part in addWorkToChannels():
    // op->buffer, op->nbytes, op->nsteps = ...;
  }

  nChannelsMax = std::max(nChannels[0], nChannels[1]);
  // Determine how many peers this plan will target concurrently. Make a
  // simplifying assumption that each task targets a different peer.
  // Each task is striped across 'nChannelsMax' of 'p2pnChannels' channels.
  // Each channel runs up to NCCL_MAX_DEV_WORK_P2P_PER_BATCH tasks concurrently.
  int maxConcurrent;
  int concurrentTasks[2];
  maxConcurrent = comm->p2pnChannels / nChannelsMax * NCCL_MAX_DEV_WORK_P2P_PER_BATCH;
  concurrentTasks[0] = std::min(planTotalTasks[0], maxConcurrent);
  concurrentTasks[1] = std::min(planTotalTasks[1], maxConcurrent);
  for (int part=0; part < nChannelsMax; part++) {
    int incWorkCounter = -1;
    int channelId = ncclP2pChannelForPart(comm->p2pnChannels, base, part);
    plan->channelMask |= uint64_t(1)<<channelId;
    // Add batch first.
    ncclAddWorkBatchToPlan(comm, plan, channelId, ncclDevWorkTypeP2p, ncclDevFuncId_P2p(), workOffset, p2pEpoch, p2pRound);
    for (int dir=0; dir < nProxyOps; dir++) {
      // Partition steps across channels.
      int nParts = dir ? work->nSendChannels : work->nRecvChannels;
      void* addr = dir ? work->sendAddr : work->recvAddr;
      size_t bytes = dir ? work->sendBytes : work->recvBytes;

      proxyOps[dir].recvbuff = nullptr;
      if (nParts <= part) {
        proxyOps[dir].nsteps = 0;
      } else if (bytes == 0) {
        proxyOps[dir].nsteps = 1;
        proxyOps[dir].nbytes = 0;
      } else {
        size_t chunkDataSize = u32fp8Decode(dir ? work->sendChunkSize_u32fp8 : work->recvChunkSize_u32fp8);
        size_t partBeg, partEnd;
        ncclP2pPartBounds(nParts, part, bytes, &partBeg, &partEnd);
        if (proxyOps[dir].reg) {
          (dir ? proxyOps[dir].sendbuff : proxyOps[dir].recvbuff) = (uint8_t*)addr + partBeg;
          (dir ? proxyOps[dir].sendMhandle : proxyOps[dir].recvMhandle) = handles[dir][part];
          proxyOps[dir].nbytes = partEnd - partBeg;
          proxyOps[dir].nsteps = DIVUP(proxyOps[dir].nbytes, NCCL_MAX_NET_SIZE);
        } else {
          proxyOps[dir].nsteps = divUp(partEnd-partBeg, chunkDataSize);
          proxyOps[dir].nbytes = std::min(partEnd-partBeg, chunkDataSize);
        }
        if (proxyOps[dir].protocol == NCCL_PROTO_LL) {
          proxyOps[dir].nbytes *= 2;
          proxyOps[dir].nbytes = roundUp(proxyOps[dir].nbytes, sizeof(union ncclLLFifoLine));
        }
      }

      // Increment work counter for <send, recv> pair rather than individual p2p
      if (proxyOps[dir].nsteps && incWorkCounter < 0) {
        proxyOps[dir].incWorkCounter = true;
        incWorkCounter = dir;
      }

      if (proxyOps[dir].nsteps != 0) {
        // Calculate the opCount after adding batch since then the batch count will
        // equal one plus the batch index this p2p settled in.
        proxyOps[dir].channelId = channelId;
        proxyOps[dir].opCount = uint64_t(comm->planner.wipPlan.channels[channelId].nWorkBatchesP2p)<<1 | 1;
        proxyOps[dir].nChannels = nChannels[dir];
        proxyOps[dir].nPeers = concurrentTasks[dir];
        NCCLCHECKGOTO(ncclAddProxyOpIfNeeded(comm, plan, &proxyOps[dir]), ret, cleanup);
        NCCLCHECKGOTO(addProfilerProxyOpIfNeeded(comm, plan, &proxyOps[dir]), ret, cleanup);
      }
    }
  }
cleanup:
  free(handles[0]);
  free(handles[1]);
  return ret;
}

static int calcP2pChannelCount(size_t totalSize, int minChannels, int maxChannels, size_t minSize, size_t maxSize) {
  size_t size = std::max(minSize, divUp(totalSize, minChannels));
  int nChannels = minChannels;
  while (size > maxSize && nChannels <= maxChannels/2) {
    nChannels *= 2;
    size = divUp(totalSize, nChannels);
  }
  return nChannels;
}

static ncclResult_t scheduleP2pTasksToPlan(struct ncclComm* comm, int* p2pEpoch, int* p2pRound, struct ncclKernelPlan* plan, struct ncclKernelPlanBudget* budget) {
  int nRanks = comm->nRanks;
  struct ncclKernelPlanner::Peer* peers = comm->planner.peers;

  plan->threadPerBlock = std::max(plan->threadPerBlock, NCCL_MAX_NTHREADS);
  if (!plan->kernelSpecialized) {
    plan->kernelFn = ncclDevKernelForFunc[ncclDevFuncId_P2p()];
    plan->kernelSpecialized = ncclDevKernelForFuncIsSpecialized[ncclDevFuncId_P2p()];
  }

  // Compute how much to split operations
  // Try to use all channels
  int nChannelsMax = comm->p2pnChannelsPerPeer;
  int nChannelsMin = nChannelsMax;
  // Try to use all channels, but one channel per operation.
  while (nChannelsMin*nRanks > comm->p2pnChannels && nChannelsMin > 1) nChannelsMin /= 2;

  // Save the total count of send/recv tasks in the plan
  int planTotalTasks[2] = {comm->planner.nTasksP2pRecv, comm->planner.nTasksP2pSend};
  while (comm->planner.nTasksP2p != 0) {
    for (; *p2pRound < nRanks; (*p2pRound)++) {
      int sendRank = comm->p2pSchedule[*p2pRound].sendRank;
      int recvRank = comm->p2pSchedule[*p2pRound].recvRank;
      struct ncclTaskP2p* send = ncclIntruQueueHead(&peers[sendRank].sendQueue);
      struct ncclTaskP2p* recv = ncclIntruQueueHead(&peers[recvRank].recvQueue);
      if (send == nullptr && recv == nullptr) continue;

      if (sendRank == comm->rank) {
        if (send != nullptr && recv == nullptr) {
          WARN("Trying to send to self without a matching recv");
          return ncclInvalidUsage;
        }
        if (send == nullptr && recv != nullptr) {
          WARN("Trying to recv to self without a matching send");
          return ncclInvalidUsage;
        }
      }
      ssize_t sendBytes = send ? send->bytes : -1;
      ssize_t recvBytes = recv ? recv->bytes : -1;
      void* sendBuff = send ? send->buff : nullptr;
      void* recvBuff = recv ? recv->buff : nullptr;

      if (sendRank == comm->rank && send->buff == recv->buff) {
        // Skip send to self in-place (we don't need to support this).
        ncclIntruQueueDequeue(&peers[sendRank].sendQueue);
        ncclIntruQueueDequeue(&peers[recvRank].recvQueue);
        ncclMemoryPoolFree(&comm->memPool_ncclTaskP2p, send);
        ncclMemoryPoolFree(&comm->memPool_ncclTaskP2p, recv);
        comm->planner.nTasksP2p -= 2;
        comm->planner.nTasksP2pSend -= 1;
        comm->planner.nTasksP2pRecv -= 1;
      } else {
        // Ensure room for worst case of one new batch per channel.
        if (!ncclTestBudget(budget, plan->nWorkBatches+nChannelsMax, plan->workBytes + sizeof(struct ncclDevWorkP2p))) {
          return ncclSuccess;
        }
        struct ncclTaskP2p* p2pTasks[2] = { recv, send };
        NCCLCHECK(addP2pToPlan(comm, plan, nChannelsMin, nChannelsMax, *p2pEpoch, *p2pRound, sendRank, sendBuff, sendBytes, recvRank, recvBuff, recvBytes, planTotalTasks, p2pTasks));
        if (send != nullptr) {
          ncclIntruQueueDequeue(&peers[sendRank].sendQueue);
          // Profiler - We can overwrite groupAPI event handles here since all operations here belong to the same group
          plan->groupApiEventHandle = send->groupApiEventHandle;
          ncclIntruQueueEnqueue(&plan->p2pTaskQueue, send);
          comm->planner.nTasksP2p -= 1;
          comm->planner.nTasksP2pSend -= 1;
        }
        if (recv != nullptr) {
          ncclIntruQueueDequeue(&peers[recvRank].recvQueue);
          // Profiler - We can overwrite groupAPI event handles here since all operations here belong to the same group
          plan->groupApiEventHandle = recv->groupApiEventHandle;
          ncclIntruQueueEnqueue(&plan->p2pTaskQueue, recv);
          comm->planner.nTasksP2p -= 1;
          comm->planner.nTasksP2pRecv -= 1;
        }
      }
    }
    *p2pRound=0;
    (*p2pEpoch)++;
  }
  return ncclSuccess;
}

// waitWorkFifoAvailable：自旋等待 workFifo 有足够空间
// ============================================================================
// workFifo 是 host 和 device 之间的 ring buffer：
//   workFifoProduced：host 写入的累计字节数（单调递增）
//   workFifoConsumed：GPU 完成后由 KernelFinishCallback 异步更新
//   produced - consumed <= workFifoBytes 时有空间
// 如果没空间，自旋等待（poll event callbacks + yield）
// 注意：这里用的是无符号整数减法，自然处理回绕
// Spin until its safe to increase comm->workFifoProduced to desiredProduced.
static ncclResult_t waitWorkFifoAvailable(struct ncclComm* comm, uint32_t desiredProduced) {
  bool hasRoom = (desiredProduced - comm->workFifoConsumed) <= comm->workFifoBytes;
  if (!hasRoom) {
    while (true) {
      // Check abort flag to break deadlock when abort is signaled
      if (COMPILER_ATOMIC_LOAD(comm->abortFlag, std::memory_order_acquire)) {
        return ncclInternalError;
      }

      NCCLCHECK(ncclCommPollEventCallbacks(comm, /*waitSome=*/true));
      hasRoom = (desiredProduced - comm->workFifoConsumed) <= comm->workFifoBytes;
      if (hasRoom) break;
      std::this_thread::yield();
    }
  }
  return ncclSuccess;
}

namespace {
  struct uploadWork_cleanup_t {
    struct ncclCommEventCallback base;
    void *hostBuf;
  };
  ncclResult_t uploadWork_cleanup_fn(
      struct ncclComm* comm, struct ncclCommEventCallback* cb
    ) {
    struct uploadWork_cleanup_t* me = (struct uploadWork_cleanup_t*)cb;
    ncclOsAlignedFree(me->hostBuf);
    CUDACHECK(cudaEventDestroy(me->base.event));
    free(me);
    return ncclSuccess;
  }
}

// uploadWork：将 devWork 结构体写入目标位置
// ============================================================================
// 根据 plan->workStorageType 有三种目标：
//
//   ncclDevWorkStorageTypeArgs（内联模式）：
//     直接写入 plan->kernelArgs 指向的内存
//     kernel launch 时通过 CU_LAUNCH_PARAM_BUFFER_POINTER 传入
//     不需要 workFifo，最快的路径
//
//   ncclDevWorkStorageTypeFifo（ring buffer 模式）：
//     写入 comm->workFifoBuf（host pinned memory，GPU 可见）
//     写入位置 = fifoCursor & (workFifoBytes-1)（回绕）
//     kernel 通过 kernelArgs->workBuf + kernelArgs->workMask 寻址
//     需要先调用 waitWorkFifoAvailable 确保有空间
//
//   ncclDevWorkStorageTypePersistent（CUDA Graph 模式）：
//     host 分配临时内存 → cudaMallocAsync 分配设备内存 → cudaMemcpyAsync 上传
//     host 临时内存通过 uploadWork_cleanup_fn 异步释放
//
// symk/CE/RMA 的 plan 直接跳过（它们的参数不通过 workFifo）
// ============================================================================
static ncclResult_t uploadWork(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  // symk/CE/RMA plan 不需要 uploadWork，参数已内嵌在 kernelArgs 中
  if (plan->isSymColl || plan->isCeColl || plan->isRma) return ncclSuccess;

  size_t workBytes = plan->workBytes;
  size_t batchBytes = plan->nWorkBatches*sizeof(struct ncclDevWorkBatch);
  void* fifoBufHost;
  uint32_t fifoCursor, fifoMask;

  // 根据存储类型初始化目标 buffer 和光标
  switch (plan->workStorageType) {
  case ncclDevWorkStorageTypeArgs:
    // 内联模式：直接写入 kernelArgs 结构体后面的空间
    plan->kernelArgs->workBuf = nullptr;  // kernel 知道 data 在 args 内
    fifoBufHost = (void*)plan->kernelArgs;
    fifoCursor = sizeof(ncclDevKernelArgs) + batchBytes; // 工作数据在 batch 之后
    fifoMask = ~0u; // 不需要回绕
    break;
  case ncclDevWorkStorageTypeFifo:
    // Fifo 模式：写入共享的 workFifoBuf ring buffer
    fifoBufHost = comm->workFifoBuf;
    fifoCursor = comm->workFifoProduced;  // 当前生产者指针
    fifoMask = comm->workFifoBytes-1;     // 回绕 mask（workFifoBytes 必须是 2 的幂）
    // 等待 GPU 消费足够空间
    NCCLCHECK(waitWorkFifoAvailable(comm, fifoCursor + workBytes));
    // 告诉 kernel workFifo 的设备地址
    plan->kernelArgs->workBuf = comm->workFifoBufDev;
    break;
  case ncclDevWorkStorageTypePersistent:
    // Persistent 模式：分配临时 host 内存，后续通过 cudaMemcpyAsync 上传到设备
    // We rely on 16-byte alignment. Use aligned alloc when available (C++11+ or MSVC with /std:c++11+).
    // MSVC keeps __cplusplus at 199711L
    #if (__cplusplus >= 201103L) || (defined(_MSC_VER) && _MSVC_LANG >= 201103L)
    fifoBufHost = ncclOsAlignedAlloc(16, ROUNDUP(workBytes, 16));
    #else
    static_assert(16 <= alignof(max_align_t), "We rely on 16-byte alignment.");
    fifoBufHost = malloc(workBytes);
    #endif
    fifoCursor = 0;
    fifoMask = ~0u;
    break;
  default:
    return ncclInternalError;
  }
  plan->kernelArgs->workMask = fifoMask;

  // 修正 batch 的 offsetBase：从 plan 内部的零基地址平移到实际地址
  //   Args 模式：offset 基于 kernelArgs 开头
  //   Fifo 模式：offset 基于 workFifo base
  //   Persistent 模式：无需平移（专用 buffer 从零开始）
  // Batches were placed after kernelArgs by finishPlan(). Only thing left to
  // do is translate the work offset from zero based (in plan) to:
  //  ncclDevWorkStorageTypeArgs: offset from beginning of kernel args
  //  ncclDevWorkStorageTypeFifo: offset from base of fifo
  //  ncclDevWorkStorageTypePersistent: no translation since our dedicated buffer will also begin at zero.
  struct ncclDevWorkBatch* batchZero = (struct ncclDevWorkBatch*)(plan->kernelArgs+1);
  for (int b=0; b < plan->nWorkBatches; b++) {
    batchZero[b].offsetBase += fifoCursor;
  }

  // 将 devWork 结构体按 16 字节对齐写入目标 buffer
  // 16 字节 = 一次 GPU 缓存行读取的最小单位，保证原子可见性
  // Write the channel-shared work structs.
  struct ncclWorkList* workNode = ncclIntruQueueHead(&plan->workQueue);
  while (workNode != nullptr) {
    char* dst = (char*)fifoBufHost;
    char* src = (char*)(workNode+1);
    for (int n = workNode->size; n != 0; n -= 16) {
      memcpy(
        COMPILER_ASSUME_ALIGNED(dst + (fifoCursor & fifoMask), 16),
        COMPILER_ASSUME_ALIGNED(src, 16),
        16
      );
      fifoCursor += 16;
      src += 16;
    }
    workNode = workNode->next;
  }

  // 更新生产者指针 + GDR fence
  switch (plan->workStorageType) {
  case ncclDevWorkStorageTypeFifo:
    // 推进 workFifoProduced，下次 uploadWork 从这里继续
    comm->workFifoProduced = fifoCursor;
    // 如果 workFifo 是 GDR 映射的（通过 PCIe BAR 直写 GPU 内存），需要 fence
    if (comm->workFifoBufGdrHandle != nullptr) wc_store_fence();
    break;
  case ncclDevWorkStorageTypePersistent:
    // Persistent 模式：分配设备内存 + 异步上传 + 注册清理回调
    { ncclResult_t result = ncclSuccess;
      struct uploadWork_cleanup_t* cleanup = nullptr;
      cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
      void* fifoBufDev = nullptr;
      cudaStream_t deviceStream;

      CUDACHECKGOTO(cudaThreadExchangeStreamCaptureMode(&mode), result, fail);

      // Acquire deviceStream. Since the user's graph will be launched later and it also
      // acquires the deviceStream, it will observe this upload.
      NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->deviceStream, /*concurrent=*/false, &deviceStream), result, fail);

      CUDACHECKGOTO(cudaMallocAsync(&fifoBufDev, workBytes, comm->memPool, deviceStream), result, fail);
      plan->workBufPersistent = fifoBufDev;
      plan->kernelArgs->workBuf = fifoBufDev;

      // coverity[uninit_use_in_call:FALSE] => fifoBufHost is never NULL
      CUDACHECKGOTO(cudaMemcpyAsync(fifoBufDev, fifoBufHost, workBytes, cudaMemcpyDefault, deviceStream), result, fail);
      cudaEvent_t memcpyDone;
      CUDACHECKGOTO(cudaEventCreateWithFlags(&memcpyDone, cudaEventDisableTiming), result, fail);
      CUDACHECKGOTO(cudaEventRecord(memcpyDone, deviceStream), result, fail);

      NCCLCHECKGOTO(ncclCalloc(&cleanup, 1), result, fail);
      cleanup->base.fn = uploadWork_cleanup_fn;
      cleanup->base.event = memcpyDone;
      cleanup->hostBuf = fifoBufHost;
      ncclIntruQueueEnqueue(&comm->eventCallbackQueue, (struct ncclCommEventCallback *)cleanup);

      NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->deviceStream, /*concurrent=*/false), result, fail);
      NCCLCHECKGOTO(ncclCommPollEventCallbacks(comm, /*waitSome=*/false), result, fail);

    finish_scope:
      if (mode != cudaStreamCaptureModeRelaxed) (void)cudaThreadExchangeStreamCaptureMode(&mode);
      return result;
    fail:
      if (!cleanup) ncclOsAlignedFree(fifoBufHost);
      goto finish_scope;
    } break;
  default: break;
  }
  return ncclSuccess;
}

static int geteActivationMask(struct ncclProxyOp * op) {
  if (ncclFuncSendRecv <= op->coll && op->coll <= ncclFuncRecv) {
    return op->task.p2p->eActivationMask;
  }
  if (op->coll == ncclFuncAllGatherV) {
    return 0;
  }
  return op->task.coll->eActivationMask;
}

static void *gettaskEventHandle(struct ncclProxyOp * op) {
  if (ncclFuncSendRecv <= op->coll && op->coll <= ncclFuncRecv) {
    return op->task.p2p->eventHandle;
  }
  if (op->coll == ncclFuncAllGatherV) {
    return nullptr;
  }
  return op->task.coll->eventHandle;
}

// uploadProxyOps：将 plan 中的 proxyOp 提交给 proxy 线程
// ============================================================================
// proxyOp 描述了一次网络传输操作（IB RDMA send/recv、GIN put/get 等）
// 这个函数：
//   1. 将 plan 内部的零基 opCount 转换为全局唯一 opCount
//      - collective: 基于 comm 的 collOpCount
//      - p2p: 基于每个 channel 的 p2pOpCount
//   2. 调用 ncclProxySaveOp 将 op 写入 proxy 线程的缓冲区
//   3. 恢复原始 opCount（persistent 模式下 plan 可以被多次重放）
// ============================================================================
static ncclResult_t uploadProxyOps(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  uint64_t collOpCount = comm->sharedRes->collOpCount;
  uint64_t p2pOpBump[MAXCHANNELS] = {/*0...*/};
  // Advance comm's collOpCount by number of colls in this plan.
  int hasp2p = 0;
  comm->sharedRes->collOpCount += plan->collOpCount;
  comm->collOpCount += plan->collOpCount;

  struct ncclProxyOp* op = ncclIntruQueueHead(&plan->proxyOpQueue);
  while (op != nullptr) {
    op->profilerContext = comm->profilerContext;
    op->eActivationMask = geteActivationMask(op);
    op->taskEventHandle = gettaskEventHandle(op);
    ncclProfilerAddPidToProxyOp(op);

    uint64_t oldId = op->opCount;
    // Ignoring the bottom tag bit, opCount's are zero-based within plan so
    // translate them to the tip of the comm's history.
    if (oldId & 1) { // p2p
      // opCount is monotonic increasing within a plan's channel so just
      // remember last value to compute max.
      p2pOpBump[op->channelId] = (oldId>>1) + 1; // +1 to ensure next plan doesn't collide
      op->opCount = (comm->sharedRes->p2pOpCount[op->channelId]<<1) + oldId;
      hasp2p = 1;
    } else { // coll
      op->opCount = (collOpCount<<1) + oldId;
    }

    NCCLCHECK(ncclProxySaveOp(comm, op, nullptr));
    op->opCount = oldId; // Restore for next uploadProxyOps()
    op = op->enqNext;
  }

  if (hasp2p) {
    for (int c=0; c < MAXCHANNELS; c++) {
      // Advance channel's p2pOpCount by number of p2p's in this plan channel.
      comm->sharedRes->p2pOpCount[c] += p2pOpBump[c];
    }
  }
  return ncclSuccess;
}

// hostStreamPlanTask：提交 proxy ops 并处理 plan 回收
// ============================================================================
// 调用时机：
//   - 在 ncclLaunchKernelAfter_NoCuda 中直接调用（普通模式）
//   - 或通过 cudaLaunchHostFunc 在 host stream 上异步调用（persistent/blocking 模式）
// 职责：
//   1. 启动 profiler 事件
//   2. uploadProxyOps：提交网络操作给 proxy 线程
//   3. ncclProxyStart：唤醒 proxy 线程开始工作
//   4. 非 persistent 模式：将 plan 送入 callbackQueue 等待回收
//      （plan 不能立即释放，因为 GPU 还在使用它的数据）
// ============================================================================
static ncclResult_t hostStreamPlanTask(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  NCCLCHECK(ncclProfilerStartGroupEvent(plan));
  NCCLCHECK(ncclProfilerStartTaskEvents(plan));
  if (ncclIntruQueueHead(&plan->proxyOpQueue)) {
    NCCLCHECK(uploadProxyOps(comm, plan));
    NCCLCHECK(ncclProxyStart(comm));
  }
  NCCLCHECK(ncclProfilerStopTaskEvents(plan));
  NCCLCHECK(ncclProfilerStopGroupEvent(plan));
  if (!plan->persistent) {
    // Notify main thread of our reclaiming. This will reclaim plan concurrently.
    ncclIntruQueueMpscEnqueue(&comm->callbackQueue, &plan->reclaimer);
  }
  return ncclSuccess;
}

static void CUDART_CB hostStreamPlanCallback(void *plan_) {
  NCCL_NVTX3_FUNC_RANGE;
  struct ncclKernelPlan* plan = (struct ncclKernelPlan*)plan_;
  ncclResult_t result = hostStreamPlanTask(plan->comm, plan);
  if (result != ncclSuccess) {
    WARN("hostStreamPlanCallback() failed : %s", ncclGetErrorString(result));
  }
  return;
}

static ncclResult_t reclaimPlan(struct ncclComm* comm, struct ncclCommCallback* me) {
  struct ncclKernelPlan* plan = (struct ncclKernelPlan*)me; // cast from first member `reclaim`
  if (plan->persistent) {
    comm->sharedRes->persistentRefs -= 1;
    comm->localPersistentRefs -= 1;
    if (plan->workStorageType == ncclDevWorkStorageTypePersistent) {
      cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
      CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
      CUDACHECK(cudaFree(plan->workBufPersistent));
      CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
    }
  }
  if (plan->isSymColl) {
    free(plan->kernelSymArgs);
  }
  // Free coll tasks
  struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);
  while (ct != nullptr) {
    struct ncclTaskColl* ct1 = ct->next;
    free(ct->sendNetHandles);
    free(ct->recvNetHandles);
    free(ct->srecvNetHandles);
    ncclMemoryPoolFree(&comm->memPool_ncclTaskColl, ct);
    ct = ct1;
  }
  // Free p2p tasks
  struct ncclTaskP2p* pt = ncclIntruQueueHead(&plan->p2pTaskQueue);
  while (pt != nullptr) {
    struct ncclTaskP2p* pt1 = pt->next;
    ncclMemoryPoolFree(&comm->memPool_ncclTaskP2p, pt);
    pt = pt1;
  }
  // Free broadcast tasks
  struct ncclTaskBcast* bt = ncclIntruQueueHead(&plan->bcastTaskQueue);
  while (bt != nullptr) {
    struct ncclTaskBcast* bt1 = bt->next;
    ncclMemoryPoolFree(&comm->memPool_ncclTaskBcast, bt);
    bt = bt1;
  }
  // Free proxy ops
  struct ncclProxyOp* q = ncclIntruQueueHead(&plan->proxyOpQueue);
  while (q != nullptr) {
    struct ncclProxyOp* q1 = q->enqNext;
    if (q->ringAlgo && q->ringAlgo->decRefCount() == 0) delete q->ringAlgo;
    ncclMemoryPoolFree(&comm->memPool_ncclProxyOp, q);
    q = q1;
  }
  // Free RMA persistent descriptors (graph mode)
  if (plan->isRma && plan->persistent) {
    NCCLCHECK(ncclRmaProxyReclaimPlan(comm, plan));
  }
  // Run other free callbacks
  ncclResult_t result = ncclSuccess;
  while (!ncclIntruQueueEmpty(&plan->cleanupQueue)) {
    struct ncclCommCallback* cb = ncclIntruQueueDequeue(&plan->cleanupQueue);
    ncclResult_t res1 = cb->fn(comm, cb); // Expect to reclaim memory of cb
    if (res1 != ncclSuccess) result = res1;
  }
  NCCLCHECK(result);
  // Free plan struct
  ncclMemoryPoolFree(&comm->memPool_ncclKernelPlan, plan);
  return ncclSuccess;
}

static void persistentDestructor(void* plans_) {
  struct ncclKernelPlan* plan = (struct ncclKernelPlan*)plans_;
  struct ncclComm* comm = plan->comm;
  while (plan != nullptr) {
    struct ncclKernelPlan* next = plan->next;
    ncclIntruQueueMpscEnqueue(&comm->callbackQueue, &plan->reclaimer);
    plan = next;
  }
}

NCCL_PARAM(LaunchOrderImplicit, "LAUNCH_ORDER_IMPLICIT", 0);

namespace {
  enum ncclImplicitOrder {
    ncclImplicitOrderNone,
    ncclImplicitOrderSerial,
    ncclImplicitOrderLaunch
  };
}

static ncclResult_t getImplicitOrder(enum ncclImplicitOrder *mode, bool capturing, int driver=-1) {
  if (ncclParamLaunchOrderImplicit()) {
    if (driver < 0) { NCCLCHECK(ncclCudaDriverVersion(&driver)); }
    if (capturing && driver < 12090) { *mode = ncclImplicitOrderSerial; return ncclSuccess; }
    *mode = 12030 <= std::min<int>(CUDART_VERSION, driver) ? ncclImplicitOrderLaunch : ncclImplicitOrderSerial;
    return ncclSuccess;
  }
  *mode = ncclImplicitOrderNone;
  return ncclSuccess;
}

// ncclLaunchPrepare：将任务打包成 plan 并准备发射
// ============================================================================
// 这是从“任务分析”到“kernel launch”的桥梁。
// 输入：planner 中已准备好的任务队列（来自 ncclPrepareTasks）
// 输出：planner->planQueue 中的 plan 链表（每个 plan = 一次 kernel launch）
//
// 主循环逻辑（do-while）：
//   每次迭代生成一个 plan，直到所有任务队列排空
//   plan 构建优先级：
//     1. RMA 任务（ncclPut/ncclGet 单边传输）
//     2. CE 任务（Copy Engine，小数据量用 cudaMemcpy）
//     3. Symmetric kernel 任务（collSymTaskQueue）
//     4. 传统 collective + P2P 任务
//
// plan 构建后的后续处理：
//   - 流同步：确保所有 user stream 在 kernel launch 前完成先前操作
//   - proxy ops 提交：如需要，通过 cudaLaunchHostFunc 异步提交 proxy ops
//   - persistent 模式：CUDA Graph 场景下增加引用计数和析构回调
// ============================================================================
ncclResult_t ncclLaunchPrepare(struct ncclComm* comm) {
  ncclResult_t result = ncclSuccess;
  struct ncclKernelPlanner* planner = &comm->planner;
  // persistent: CUDA Graph capture 模式，plan 会存活到 graph 析构
  bool persistent = ncclCudaGraphValid(planner->capturingGraph);
  planner->persistent = persistent;
  // Operations from different plans will not be batched together. A new batch will be created for each new plan that is used to schedule the ops (see ncclAddWorkBatchToPlan).
  // For p2p ops, we further guarantee that ops from different epochs will not be batched together (to avoid hangs).
  // The p2pEpoch value is incremented in scheduleP2pTasksToPlan and its value is carried over from one plan to another (even if not strictly required)
  int nPlans = 0, p2pEpoch=0, p2pRound=0;

  // ==================== plan 构建主循环 ====================
  // 每次迭代生成一个 plan，直到所有任务队列排空
  if (planner->nTasksColl + planner->nTasksP2p + planner->nTasksBcast != 0 ||
      !ncclIntruQueueEmpty(&planner->collSymTaskQueue) ||
      !ncclIntruQueueEmpty(&planner->collCeTaskQueue) ||
      planner->nTasksRma != 0) {
    do {
      // 初始化临时工作区（wipPlan：work-in-progress plan）
      memset(&planner->wipPlan, 0, sizeof(planner->wipPlan));

      // 分配 plan 结构体（每个 plan 对应一次 kernel launch）
      struct ncclKernelPlan* plan = ncclMemoryPoolAlloc<struct ncclKernelPlan>(&comm->memPool_ncclKernelPlan, &comm->memPermanent);
      plan->comm = comm;
      plan->reclaimer.fn = reclaimPlan; // plan 回收时的清理函数
      plan->persistent = persistent;
      // 默认存储方式：persistent 用 Persistent，否则用 Fifo
      // finishPlan() 会在任务足够小时升级为 Args（内联到 kernel args）
      // finishPlan() promotes ncclDevWorkStorageType[Fifo|Persistent]->Args if the work can fit.
      plan->workStorageType = persistent ? ncclDevWorkStorageTypePersistent
                                         : ncclDevWorkStorageTypeFifo;

      // --- 优先级 1：RMA 任务（ncclPut/ncclGet 单边传输） ---
      // RMA 任务不需要 GPU kernel，通过 proxy thread 直接执行 IB RDMA 操作
      if (planner->nTasksRma != 0) {
        NCCLCHECKGOTO(scheduleRmaTasksToPlan(comm, plan), result, failure);
        if (plan->isRma && plan->rmaArgs != NULL && plan->rmaArgs->nRmaTasks > 0) {
          ncclIntruQueueEnqueue(&planner->planQueue, plan);
          nPlans += 1;
        }
      // --- 优先级 2：CE 任务（Copy Engine，小数据量 allgather/reduce） ---
      // 数据量 < SYM_CE_THRESHOLD（8MB）时，用 cudaMemcpy 比 kernel 更快
      // 不走 GPU kernel，而是直接通过 CUDA Copy Engine 执行
      } else if (!ncclIntruQueueEmpty(&planner->collCeTaskQueue)) {
        struct ncclTaskColl* task = ncclIntruQueueHead(&planner->collCeTaskQueue);
        plan->isCeColl = true;
        plan->ceCollArgs = ncclMemoryStackAlloc<struct ncclCeCollArgs>(&comm->memScoped);
        plan->ceCollArgs->rootRank = task->root;
        plan->ceCollArgs->datatype = task->datatype;
        plan->ceCollArgs->nElts = task->count;
        plan->ceCollArgs->eltSize = ncclTypeSize(task->datatype);
        plan->ceCollArgs->sendBuff = (uint8_t*)task->sendbuff;
        plan->ceCollArgs->recvBuff = (uint8_t*)task->recvbuff;
        plan->ceCollArgs->func = task->func;
        plan->ceCollArgs->sendWin = task->sendWin;
        plan->ceCollArgs->recvWin = task->recvWin;
        plan->ceCollArgs->collApiEventHandle = task->collApiEventHandle;

        if (comm->rank == 0) {
          const char* nvlsSync = comm->nvlsSupport ? "; CE synchronization with NVLS" : "";
          INFO(NCCL_TUNING, "%s [Copy Engine]: %ld Bytes -> cudaMemcpy%s",
            ncclFuncToString(task->func), task->count * ncclTypeSize(task->datatype), nvlsSync);
        }

        ncclIntruQueueEnqueue(&planner->planQueue, plan);
        ncclIntruQueueDequeue(&planner->collCeTaskQueue);
        ncclMemoryPoolFree(&comm->memPool_ncclTaskColl, task);
        nPlans += 1;
      } else {
        // --- 优先级 3：Symmetric kernel 任务 ---
        // ncclSymmetricTaskScheduler（定义在 symmetric_sched.cc）：
        //   1. 从 collSymTaskQueue 取出任务
        //   2. 构建 ncclSymkDevWorkArgs 参数（包含 kcomm + channelWorkRange + devWork 数组）
        //   3. 选择对应的 symk kernel function（如 all_gather_gin）
        //   4. 标记 plan->isSymColl = true
        // symk 任务不需要 workFifo，参数直接通过 kernelArgs 传入
        if (!ncclIntruQueueEmpty(&planner->collSymTaskQueue)) {
          NCCLCHECKGOTO(ncclSymmetricTaskScheduler(comm, &planner->collSymTaskQueue, plan), result, failure);
        }
        // --- 优先级 4：传统 collective + P2P 任务 ---
        // 传统 collective 路径：
        //   1. 优先排空 coll 任务（scheduleCollTasksToPlan）
        //      必须先排空 coll，因为 channel 分配是基于流量的
        //      如果先排 P2P，不同 rank 的切分点可能不一致，导致 hang
        //   2. 再排 broadcast 任务（ncclScheduleBcastTasksToPlan）
        //   3. 最后排 P2P 任务（scheduleP2pTasksToPlan）
        else {
          struct ncclKernelPlanBudget budget;
          budget.inArgsBytes = comm->workArgsBytes - sizeof(struct ncclDevKernelArgs);
          // Non-persistent kernels fill up at most half of our fifo per kernel.
          budget.outArgsBytes = plan->persistent ? (1<<30) : comm->workFifoBytes/2;

          // Drain coll tasks first. This is essential since we partition tasks based
          // on the work budget and p2p work isn't collective. If we were to drain p2p
          // first, the place where we cut the kernel could vary by rank which would
          // cause the "shortest channel first" channel picker to have divergent results.
          if (planner->nTasksColl != 0) {
            NCCLCHECKGOTO(scheduleCollTasksToPlan(comm, plan, &budget), result, failure);
          }
          if (planner->nTasksColl == 0 && planner->nTasksBcast != 0) {
            NCCLCHECKGOTO(ncclScheduleBcastTasksToPlan(comm, plan, &budget), result, failure);
          }
          // And only drain p2p tasks once colls are depleted.
          if (planner->nTasksColl == 0 && planner->nTasksBcast == 0 && planner->nTasksP2p != 0) {
            NCCLCHECKGOTO(scheduleP2pTasksToPlan(comm, &p2pEpoch, &p2pRound, plan, &budget), result, failure);
          }
        }

        // finishPlan：为 plan 确定 work 存储方式和构建 batch 布局
        // 如果 totalSize <= workArgsBytes，升级为 Args 模式（内联到 kernel args）
        // 否则保持 Fifo/Persistent 模式
        // 同时构建 batch 链表（round-robin 遍历所有 channel）+ 合并 proxy ops
        finishPlan(comm, plan);
        if (plan->workBytes != 0) {
          ncclIntruQueueEnqueue(&planner->planQueue, plan);
          nPlans += 1;
        }
      }
    } while (planner->nTasksColl + planner->nTasksP2p + planner->nTasksBcast != 0 ||
             !ncclIntruQueueEmpty(&planner->collSymTaskQueue) ||
             !ncclIntruQueueEmpty(&planner->collCeTaskQueue) ||
             planner->nTasksRma != 0);

    struct ncclKernelPlan* planHead = ncclIntruQueueHead(&planner->planQueue);
    planner->unlaunchedPlansHead = planHead;

    if (nPlans == 0) return ncclSuccess;

    // ==================== 流同步 ====================
    // 在发射 kernel 之前，确保所有 user stream 的先前操作都已完成
    cudaStream_t launchStream = planner->streams->stream;
    cudaStream_t deviceStream, launchOrder;
    // 获取 deviceStream（NCCL 内部的强流，保证操作顺序）
    NCCLCHECKGOTO(ncclStrongStreamAcquire(planner->capturingGraph, &comm->sharedRes->deviceStream, /*concurrent=*/false, &deviceStream), result, failure);

    // 多 user stream 同步：如果 ncclGroup 内有多个不同的 stream，
    // 将它们全部同步到 launchStream（第一个 user stream）
    // userStream[0] waits on each userStream[i]...
    for (struct ncclCudaStreamList* l=planner->streams->next; l != nullptr; l = l->next) {
      CUDACHECKGOTO(cudaEventRecord(comm->sharedRes->scratchEvent, l->stream), result, failure);
      CUDACHECKGOTO(cudaStreamWaitEvent(launchStream, comm->sharedRes->scratchEvent, 0), result, failure);
    }
    // launchStream 等待 deviceStream，确保之前的 NCCL 操作都已完成
    // userStream[0] waits on deviceStream
    NCCLCHECKGOTO(ncclStreamWaitStream(launchStream, deviceStream, comm->sharedRes->scratchEvent), result, failure);

    // ==================== 隐式发射顺序（NCCL_LAUNCH_ORDER_IMPLICIT） ====================
    // 当启用时，确保同一 GPU context 上所有 NCCL kernel 的全局发射顺序
    // ncclImplicitOrderLaunch: 使用 LaunchCompletionEvent（CUDA 12.3+）
    // ncclImplicitOrderSerial: 使用完成事件序列化（老驱动 fallback）
    bool capturing = ncclCudaGraphValid(planner->capturingGraph);
    enum ncclImplicitOrder implicitOrder;
    cudaError_t status = cudaSuccess;
    NCCLCHECKGOTO(getImplicitOrder(&implicitOrder, capturing), result, failure);

    if (implicitOrder != ncclImplicitOrderNone) {
      // userStream[0] waits on per-device (context) launchOrder. Concurrent strong stream access is
      // required if this is a graph capture, non-captured cannot be concurrent because that would violate
      // deterministic program order of launches.
      bool concurrent = capturing;
      NCCLCHECKGOTO(ncclStrongStreamAcquire(planner->capturingGraph, &comm->context->launchOrder, concurrent, &launchOrder), result, failure);
      NCCLCHECKGOTO(ncclStreamWaitStream(launchStream, launchOrder, comm->sharedRes->scratchEvent), result, failure);
    }

    // ==================== proxy ops 提交 ====================
    // 如果 plan 有 proxy ops（网络传输操作），需要通过 host stream 异步提交
    // cudaLaunchHostFunc: 在 host stream 上注册回调，当之前的操作完成后触发
    // hostStreamPlanCallback → hostStreamPlanTask → uploadProxyOps + ncclProxyStart
    // 为什么不直接提交：
    //   persistent 模式下必须通过 host stream（CUDA Graph 不支持直接调用）
    //   ncclCudaLaunchBlocking 模式下需要完成后才能继续
    //   有 persistent kernel 还在运行时，用 host stream 防止竞争
    if (!persistent && comm->sharedRes->persistentRefs) status = CUDACLEARERROR(cudaEventQuery(comm->sharedRes->hostStream.serialEvent));
    if (persistent || ncclCudaLaunchBlocking || status == cudaErrorNotReady) {
      // We have to launch host tasks to push proxy args. We are careful to only
      // do this if necessary since host tasks impose a high performance cost in CUDA.
      bool acquired = false;
      cudaStream_t hostStream;
      for (struct ncclKernelPlan* plan=planHead; plan != nullptr; plan = plan->next) {
        if (plan->hasProxyOps) {
          if (!acquired) {
            acquired = true;
            NCCLCHECKGOTO(ncclStrongStreamAcquire(planner->capturingGraph, &comm->sharedRes->hostStream, /*concurrent=*/false, &hostStream), result, failure);
          }
          plan->isHostCbEnq = true;
          CUDACHECKGOTO(cudaLaunchHostFunc(hostStream, hostStreamPlanCallback, plan), result, failure);
        }
      }
      if (acquired) {
        // Make to-be-launched kernels dependent on just-launched host stream tasks.
        NCCLCHECKGOTO(ncclStreamWaitStream(launchStream, hostStream, comm->sharedRes->scratchEvent), result, failure);
        NCCLCHECKGOTO(ncclStrongStreamRelease(planner->capturingGraph, &comm->sharedRes->hostStream, /*concurrent=*/false), result, failure);
      }
    }

    // ==================== CUDA Graph persistent 处理 ====================
    // persistent 模式：plan 的生命周期绑定到 CUDA Graph
    // persistentRefs 记录还有多少 persistent plan 存活
    // persistentDestructor 作为 graph 析构回调，触发 plan 回收
    if (persistent) {
      comm->sharedRes->persistentRefs += nPlans;
      comm->localPersistentRefs += nPlans;
      NCCLCHECKGOTO(ncclCudaGraphAddDestructor(planner->capturingGraph, persistentDestructor, (void*)planHead), result, failure);
    }
  }
failure:
  return result;
}

// ncclLaunchKernelBefore_NoUncapturedCuda：在 kernel launch 之前的最后准备
// ============================================================================
// 调用时机：
//   在 intra-process barrier 之后、kernel launch 之前
//   这个窗口内不允许调用非捕获的 CUDA API（防止不同 rank 的死锁）
// 职责：
//   调用 uploadWork 将 devWork 写入目标位置（Fifo/Args/Persistent）
//   对于 symk/CE/RMA plan，uploadWork 会直接返回（无需操作）
// ============================================================================
ncclResult_t ncclLaunchKernelBefore_NoUncapturedCuda(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  // This code is called after we've checked in to the intra-process barrier
  // but before launching the kernel. We are not allowed to call CUDA unless the
  // kernel launch is captured.
  NCCLCHECK(uploadWork(comm, plan));
  return ncclSuccess;
}

#if CUDART_VERSION >= 12000
// NCCL uses the "Remote" Mem Sync domain by default
NCCL_PARAM(MemSyncDomain, "MEM_SYNC_DOMAIN", cudaLaunchMemSyncDomainRemote);
#endif

// ncclLaunchKernel：将一个 plan 发射为 CUDA kernel
// ============================================================================
// grid = {nChannels, 1, 1}：每个 channel 一个 block（这是 NCCL 的基本执行模型）
// block = {threadPerBlock, 1, 1}：通常为 256 或 512 线程
// smem：普通 kernel 用 ncclShmemDynamicSize(cudaArch)，Symmetric kernel 用 kernelDynSmem
//
// 新增特性解析：
//   CGA（Cooperative Group Array / Thread Block Cluster）：sm90+ 尓用
//     clusterSize = comm->config.cgaClusterSize，必须整除 grid.x
//     内部并行的 CTA 共享 L2 cache 行，相互可看到对方 smem
//     主要用于 NVLS multi-block AllReduce
//   MemSyncDomain：CUDA 12.0+，sm90+
//     NCCL 默认用 "Remote" 内存同步域，优化多个 GPU 间的 flushing 开销
//   LaunchCompletionEvent：CUDA 12.3+
//     发射自动记录一个 event，用于建立全局发射顺序（launchOrder stream）
//   ProgrammaticStreamSerialization：Symmetric kernel 在 sm90+用，默认序列化内核调度
//   NVLinkUtilCentricScheduling： CUDA 13.0+，sm100+，优化 NVLink 带宽利用率
// ============================================================================
// ncclLaunchKernel：将一个 plan 发射为 CUDA kernel
// ============================================================================
// 发射参数：
//   grid = {nChannels, 1, 1}：每个活跃 channel 一个 thread block
//     这是 NCCL 的基本执行模型：每个 block 独立处理一个 channel 的数据
//     channel 数由 ncclGetAlgoInfo 决定，典型值 2-32
//   block = {threadPerBlock, 1, 1}：通常 256 或 512 线程
//     由 nWarps * WARP_SIZE 确定，不同 algo 可能不同
//   smem：动态共享内存大小
//     普通 kernel：ncclShmemDynamicSize(cudaArch)，所有 kernel 统一值
//     Symmetric kernel：plan->kernelDynSmem，每个 symk kernel 独立计算
//
// kernel function：
//   plan->kernelFn：kernel 的函数指针
//     普通 kernel：从 ncclDevKernelForFunc[devFuncId] 查表得到
//     Symmetric kernel：从 ncclSymkKernelList[kernelId] 查表得到
//
// kernel args：
//   通过 CU_LAUNCH_PARAM_BUFFER_POINTER 传入 plan->kernelArgs
//   布局：[ncclDevKernelArgs][WorkBatch数组][Work数据（Args模式）]
//   对于 symk：传入 ncclSymkDevWorkArgs（kcomm + channelWorkRange + devWork）
//
// 特性属性（通过 cuLaunchKernelEx 的 CUlaunchAttribute 设置）：
//   CGA（Thread Block Cluster）：sm90+
//     clusterSize = comm->config.cgaClusterSize，必须整除 grid.x
//     同一 cluster 内的 block 保证并发调度，可直接访问对方 smem
//     主要用于 NVLS multi-block AllReduce
//   MemSyncDomain：CUDA 12.0+，sm90+
//     NCCL 默认用 "Remote" 同步域，降低跨 GPU fence 的开销
//     原理：将 NCCL 的 fence 和用户 kernel 的 fence 分离，避免互相等待
//   LaunchCompletionEvent：CUDA 12.3+
//     kernel 发射时自动记录一个 event，用于建立全局发射顺序
//     比“完成事件”更高效：不需要等 kernel 完成，只需确认已发射
//   ProgrammaticStreamSerialization：CUDA 12.3+，sm90+
//     symk kernel 专用：允许硬件调度器自动序列化同一 stream 上的连续 kernel
//     减少 kernel launch 间的 gap，提高 GPU 利用率
//   NVLinkUtilCentricScheduling：CUDA 13.0+，sm100+
//     优化 NVLink 带宽利用率，硬件级别的 SM 调度优化
// ============================================================================
ncclResult_t ncclLaunchKernel(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  ncclResult_t ret = ncclSuccess;
  struct ncclKernelPlanner* planner = &comm->planner;
  // nChannels: 从 channelMask 中计算活跃 channel 数 = grid 的 block 数
  int nChannels = countOneBits(plan->channelMask);
  // sym: kernel function pointer，后续通过 cudaGetFuncBySymbol 转为 CUfunction
  void* sym = plan->kernelFn;
  dim3 grid = {(unsigned)nChannels, 1, 1};
  dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};
  // symk 有独立的 smem 大小，普通 kernel 统一用 ncclShmemDynamicSize
  int smem = plan->isSymColl ? plan->kernelDynSmem : ncclShmemDynamicSize(comm->cudaArch);
  cudaStream_t launchStream = planner->streams->stream;

  NCCLCHECK(ncclProfilerStartKernelLaunchEvent(plan, launchStream));

  // extra[]: cuLaunchKernel 的额外参数
  // kernelArgs 指向 [ncclDevKernelArgs][batch数组][work数据]
  // kernelArgsSize 告诉 CUDA runtime 总大小
  void* extra[] = {
    CU_LAUNCH_PARAM_BUFFER_POINTER, plan->kernelArgs,
    CU_LAUNCH_PARAM_BUFFER_SIZE, &plan->kernelArgsSize,
    CU_LAUNCH_PARAM_END
  };

  int driverVersion;
  NCCLCHECKGOTO(ncclCudaDriverVersion(&driverVersion), ret, do_return);

  // 获取 CUfunction handle（从 device symbol 转换）
  CUfunction fn;
  CUDACHECKGOTO(cudaGetFuncBySymbol(&fn, sym), ret, do_return);

  // CUDA 11.8+ 使用 cuLaunchKernelEx（支持属性设置）
  if (CUDART_VERSION >= 11080 && driverVersion >= 11080) {
  #if CUDART_VERSION >= 11080
    int compCap = comm->compCap;
    unsigned int clusterSize = (compCap >= 90) ? comm->config.cgaClusterSize : 0;

    CUlaunchConfig launchConfig = {0};
    CUlaunchAttribute launchAttrs[6] = {};
    int attrs = 0;
    /* Cooperative Group Array (CGA)
     * On sm90 and later we have an extra level of hierarchy where we
     * can group together several blocks within the Grid, called
     * Thread Block Clusters.
     * Clusters enable multiple thread blocks running concurrently
     * across multiple SMs to synchronize and collaboratively fetch
     * and exchange data. A cluster of blocks are guaranteed to be
     * concurrently scheduled onto a group of SMs.
     * The maximum value is 8 and it must be divisible into the grid dimensions
     */
    if (clusterSize) {
      // CGA: 同一 cluster 内的 block 保证并发调度，可直接访问对方 smem
      // 必须整除 grid.x，否则降级为 clusterSize=1
      // Grid dimension must be divisible by clusterSize
      if (grid.x % clusterSize) clusterSize = 1;
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      launchAttrs[attrs++].value.clusterDim = {clusterSize, 1, 1};
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
      launchAttrs[attrs++].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
    }
    #if CUDART_VERSION >= 12000
    if (compCap >= 90 && driverVersion >= 12000) {
      // Set the NCCL Mem Sync domain on CUDA 12.0 and later (sm90)
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN;
      launchAttrs[attrs++].value.memSyncDomain = (CUlaunchMemSyncDomain) ncclParamMemSyncDomain();
    }
    #endif
    #if CUDART_VERSION >= 12030
    enum ncclImplicitOrder implicitOrder;
    NCCLCHECKGOTO(getImplicitOrder(&implicitOrder, plan->persistent, driverVersion), ret, do_return);
    if (implicitOrder == ncclImplicitOrderLaunch) {
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT;
      launchAttrs[attrs].value.launchCompletionEvent.event = comm->sharedRes->launchEvent;
      launchAttrs[attrs].value.launchCompletionEvent.flags = 0;
      attrs++;
    }
    if (plan->isSymColl && compCap >= 90 && driverVersion >= 12030) {
      // ProgrammaticStreamSerialization: 允许硬件自动序列化同 stream 的连续 kernel
      // 只对 symk 启用，因为 symk 的 launch 间隔特别短
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
      launchAttrs[attrs].value.programmaticStreamSerializationAllowed = 1;
      attrs++;
    }
    #endif
    #if CUDART_VERSION >= 13000
    if (compCap >= 100 && driverVersion >= 13000) {
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING;
      launchAttrs[attrs].value.nvlinkUtilCentricScheduling = comm->config.nvlinkCentricSched;
      attrs++;
    }
    #endif
    // 填充 launchConfig 结构体，传给 cuLaunchKernelEx
    launchConfig.gridDimX = grid.x;
    launchConfig.gridDimY = grid.y;
    launchConfig.gridDimZ = grid.z;
    launchConfig.blockDimX = block.x;
    launchConfig.blockDimY = block.y;
    launchConfig.blockDimZ = block.z;
    launchConfig.sharedMemBytes = smem;
    launchConfig.attrs = launchAttrs;
    launchConfig.numAttrs = attrs;
    launchConfig.hStream = launchStream;
    // cuLaunchKernelEx: 带属性的 kernel 发射
    // fn: kernel function handle
    // nullptr: 不使用 kernelParams（改用 extra[] 中的 buffer pointer）
    // extra: 包含 kernelArgs 的地址和大小
    CUCHECKGOTO(cuLaunchKernelEx(&launchConfig, fn, nullptr, extra), ret, do_return);
  #endif
  } else {
    // CUDA < 11.8: 使用标准 cuLaunchKernel（不支持 CGA/MemSyncDomain 等属性）
    // Standard kernel launch
    CUCHECKGOTO(cuLaunchKernel(fn, grid.x, grid.y, grid.z, block.x, block.y, block.z, smem, launchStream, nullptr, extra), ret, do_return);
  }

do_return:
  NCCLCHECK(ncclProfilerStopKernelLaunchEvent(plan));
  return ret;
}

// ncclLaunchKernelAfter_NoCuda：kernel launch 后的处理（不允许调用 CUDA API）
// ============================================================================
// 调用时机：kernel launch 之后、intra-process barrier 释放之前
// 为什么不能调用 CUDA：
//   在这个窗口内，如果未捕获 CUDA graph，调用 CUDA API 可能会造成死锁
//   因为其他 rank 可能正在等待 barrier
// 职责：
//   如果 proxy ops 还没通过 host stream 提交（isHostCbEnq == false），
//   直接调用 hostStreamPlanTask 提交它们
//   isHostCbEnq == true 表示已在 ncclLaunchPrepare 中通过 cudaLaunchHostFunc 提交
// ============================================================================
ncclResult_t ncclLaunchKernelAfter_NoCuda(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  if (!plan->isHostCbEnq) {
    // 普通模式：直接在当前线程提交 proxy ops + 注册 plan 回收
    // we are not using the host stream for proxy ops and reclaimation submission, call
    // hostStreamPlanTask directly
    NCCLCHECK(hostStreamPlanTask(comm, plan));
  }
  return ncclSuccess;
}

// KernelFinishCallback：workFifo 回收机制
// 当 GPU 完成 kernel 后，通过 event callback 更新 workFifoConsumed
// 这样 uploadWork 就知道可以覆写 ring buffer 的哪些区域
namespace {
  struct KernelFinishCallback {
    struct ncclCommEventCallback base;
    uint32_t workFifoConsumed;
  };
  ncclResult_t KernelFinishCallback_fn(
      struct ncclComm* comm, struct ncclCommEventCallback* cb
    ) {
    struct KernelFinishCallback* me = (struct KernelFinishCallback*)cb;
    comm->workFifoConsumed = me->workFifoConsumed;
    CUDACHECK(cudaEventDestroy(me->base.event));
    free(me);
    return ncclSuccess;
  }
}

// ncclLaunchFinish：就发射完成后的后续处理
// ============================================================================
//   1. 重置 planQueue（不销毁 plan，让它们通过 callbackQueue 回收）
//   2. 在 launchStream 上记录 finishedEvent
//   3. 如果 workFifo 使用量超过一定阈值，创建 KernelFinishCallback
//      当 GPU 完成后回调 KernelFinishCallback_fn，更新 workFifoConsumed
//   4. deviceStream 等待 launchStream 完成（fast-forward 优化）
//   5. 其他 user stream 等待 finishedEvent
//   6. 将发射事件并入 launchOrder（确保全局发射顺序）
//
// workFifo 回收逻辑：
//   workFifoProduced：每次 upload 一次 work 时 +1（轮内生产者）
//   workFifoConsumed：GPU 完成后 通过 KernelFinishCallback 更新（轮外消费者）
//   两者之差 <= workFifoBytes；当就将山时 uploadWork 会 spin wait
// ============================================================================
// ncclLaunchFinish：发射完成后的后续处理
// ============================================================================
// 在所有 plan 的 kernel 都已发射后调用，负责：
//
//   1. 在 launchStream 上记录 finishedEvent
//      这个 event 代表“所有 kernel 都已发射”（注意：不是完成，是发射）
//
//   2. workFifo 回收：如果 workFifo 使用量超过 1/8，创建 KernelFinishCallback
//      GPU 完成后回调 KernelFinishCallback_fn，更新 workFifoConsumed
//      这样 uploadWork 就知道可以重用 ring buffer 的哪些区域
//
//   3. deviceStream fast-forward：
//      deviceStream 等待 launchStream 完成（用 ncclStreamAdvanceToEvent 优化）
//      这是因为 launchStream 已经 sync 过 deviceStream，可以“快进”
//
//   4. 多 user stream 同步：
//      其他 user stream 等待 finishedEvent，确保用户的后续操作看到所有 NCCL 结果
//
//   5. launchOrder 更新：
//      将发射事件并入全局 launchOrder stream，确保发射顺序
// ============================================================================
ncclResult_t ncclLaunchFinish(struct ncclComm* comm) {
  struct ncclKernelPlanner* planner = &comm->planner;
  if (!ncclIntruQueueEmpty(&planner->planQueue)) {
    // Reset queue to empty without destroying plans since those will be sent
    // back to us for reclaiming via callbackQueue.
    ncclIntruQueueConstruct(&planner->planQueue);

    cudaStream_t launchStream = planner->streams->stream; // First user stream gets launch
    cudaStream_t deviceStream, launchOrder;
    // 在 launchStream 上记录完成事件，代表所有 kernel 都已发射
    cudaEvent_t finishedEvent = comm->sharedRes->scratchEvent;
    CUDACHECK(cudaEventRecord(finishedEvent, launchStream));

    // workFifo 回收：当使用量超过 1/8 时，注册回调异步更新 consumed 指针
    // 不是每次都回收，而是累积到一定量才回收，减少 event 创建开销
    if (comm->workFifoProduced - comm->workFifoProducedLastRecorded > comm->workFifoBytes/8) {
      comm->workFifoProducedLastRecorded = comm->workFifoProduced;
      struct KernelFinishCallback* cb;
      NCCLCHECK(ncclCalloc(&cb, 1));
      cb->base.event = finishedEvent;
      cb->base.fn = KernelFinishCallback_fn;
      cb->workFifoConsumed = comm->workFifoProduced;
      ncclIntruQueueEnqueue(&comm->eventCallbackQueue, &cb->base);
      // We just stole scratchEvent so must create a new one.
      CUDACHECK(cudaEventCreateWithFlags(&comm->sharedRes->scratchEvent, cudaEventDisableTiming));
    }

    // deviceStream waits on userStream[0]
    NCCLCHECK(ncclStrongStreamAcquiredWorkStream(planner->capturingGraph, &comm->sharedRes->deviceStream, /*concurrent=*/false, &deviceStream));

    // We know that deviceStream is strictly behind the launchStream because launchStream
    // synced with it before kernel launch. This allows us to to see deviceStream waiting
    // on launchStream as a fast-forward. When building CUDA graphs fast forwards should
    // be handled specially so as not to create graphs with a blowup in the number of edges.
    // So we could do this:
    //   CUDACHECK(cudaStreamWaitEvent(deviceStream, finishedEvent, 0));
    // But instead we do:
    NCCLCHECK(ncclStreamAdvanceToEvent(planner->capturingGraph, deviceStream, finishedEvent));

    // 多 user stream 同步：其他 user stream 等待 finishedEvent
    // 确保用户的后续操作能看到 NCCL 的所有结果
    // Each userStream[i] waits on userStream[0]
    for (struct ncclCudaStreamList* l=planner->streams->next; l != nullptr; l = l->next) {
      CUDACHECK(cudaStreamWaitEvent(l->stream, finishedEvent, 0));
    }
    // ── 步骤 5：隐式发射顺序更新（与 ncclLaunchPrepare 对称配对） ──
    // 在 ncclLaunchPrepare 中我们 Acquire 了 launchOrder 强流来保证多 comm 按序发射，
    // 这里需要将 kernel 的完成/启动事件注入该顺序链并 Release 强流。
    bool capturing = ncclCudaGraphValid(planner->capturingGraph);
    enum ncclImplicitOrder implicitOrder;
    NCCLCHECK(getImplicitOrder(&implicitOrder, capturing));
    if (implicitOrder != ncclImplicitOrderNone) {
      // As in ncclLaunchPrepare, strong stream can be non-concurrent when non-captured.
      // 非 graph-capture 模式不可并发，以遵守确定性程序顺序
      bool concurrent = capturing;
      // Incorporate launch event into per-device (context) launch order.
      // 获取 launchOrder 强流的工作流，用于注入事件
      NCCLCHECK(ncclStrongStreamAcquiredWorkStream(planner->capturingGraph, &comm->context->launchOrder, concurrent, &launchOrder));
      // If we don't have launch events (requires CUDA 12.3) then just use completion event (serialize execution).
      // 如果有 LaunchEvent（CUDA 12.3+），用 launchEvent 即可（只需排序发射时刻）；
      // 否则退化为 finishedEvent（序列化整个执行，性能较低）
      CUDACHECK(cudaStreamWaitEvent(launchOrder, implicitOrder == ncclImplicitOrderLaunch ? comm->sharedRes->launchEvent : finishedEvent));
      // Release launchOrder as acquired in ncclLaunchPrepare()
      // 释放在 ncclLaunchPrepare 中 Acquire 的 launchOrder 强流
      NCCLCHECK(ncclStrongStreamRelease(planner->capturingGraph, &comm->context->launchOrder, concurrent));
    }
    // Release deviceStream as acquired in ncclLaunchPrepare()
    // 释放在 ncclLaunchPrepare 中 Acquire 的 deviceStream 强流
    NCCLCHECK(ncclStrongStreamRelease(planner->capturingGraph, &comm->sharedRes->deviceStream, /*concurrent=*/false));
  }
  return ncclSuccess;
}

/*****************************************************************************/
/* Enqueueing system : computation of kernel and proxy operations parameters */
/*****************************************************************************/

// 入队系统概述：
// ncclGetCollNetSupport：判断当前 collective 是否可用 CollNet（IB SHARP）
//   需要：所有 channel 都有 CollNet 连接 + 算子/类型支持
// ncclGetAlgoInfo：选择最优算法（Ring/Tree/NVLS/CollNet）和协议（LL/LL128/Simple）
//   根据流量 + 节点数 + nRanks + 算法延迟 做出决策
//   用 ncclTopoTuneModel 的调度表，或插件垄入的 tuner
// ncclEnqueueCheck：每次 ncclAllReduce 等 API 调用的入口
//   内存对齐、类型检查、count=0 忪path、封装成 ncclTaskColl 入队
ncclResult_t ncclGetCollNetSupport(
    struct ncclComm* comm, struct ncclTaskColl* info, int* collNetSupport
  ) {
  // Translate ncclAvg and PreMulSum
  ncclRedOp_t netOp = info->opHost;
  if (info->opDev.op == ncclDevPreMulSum || info->opDev.op == ncclDevSumPostDiv) {
    netOp = ncclSum;
  }
  *collNetSupport = comm->config.collnetEnable;
  switch (info->func) {
  case ncclFuncAllReduce:
  case ncclFuncReduce:
  case ncclFuncReduceScatter:
    *collNetSupport &= comm->collNetSupportMatrix[netOp][info->datatype];
    break;
  default:
    break;
  }
  return ncclSuccess;
}

static void initCollCostTable(float** collCostTable) {
  float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
      table[a][p] = NCCL_ALGO_PROTO_IGNORE;
    }
  }
}

// numPipeOps: number of pipelined ops. Can be greater than 1 in aggregation mode. Used to adjust latency.
static ncclResult_t updateCollCostTable(
    struct ncclComm* comm, struct ncclTaskColl* info, size_t nBytes,
    int collNetSupport, int nvlsSupport, int numPipeOps,
    float** collCostTable) {
  float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;

  if (comm->nRanks == 1) {
    table[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 0.0;
    return ncclSuccess;
  }

  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    if ((a == NCCL_ALGO_COLLNET_DIRECT || a == NCCL_ALGO_COLLNET_CHAIN) && collNetSupport != 1) continue;
    // CollNetDirect is only supported for up to 8 local GPUs
    if (a == NCCL_ALGO_COLLNET_DIRECT && comm->maxLocalRanks > NCCL_MAX_DIRECT_ARITY+1) continue;
    // Disable CollNet Chain for more than 8 local GPUs
    if (a == NCCL_ALGO_COLLNET_CHAIN && comm->maxLocalRanks > NCCL_MAX_DIRECT_ARITY+1) continue;
    if ((a == NCCL_ALGO_NVLS || a == NCCL_ALGO_NVLS_TREE) && (!nvlsSupport || (info->func != ncclFuncAllReduce && comm->localRanks > NCCL_MAX_NVLS_ARITY))) continue;
    if (a == NCCL_ALGO_NVLS && collNetSupport != 1 && comm->nNodes > 1) continue;
    /* Tree reduceScatter doesn't support scaling yet */
    if (a == NCCL_ALGO_PAT && info->func == ncclFuncReduceScatter
        && (info->opDev.op == ncclDevPreMulSum || info->opDev.op == ncclDevSumPostDiv)) continue;
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      NCCLCHECK(ncclTopoGetAlgoTime(comm, info->func, a, p, nBytes, numPipeOps, &table[a][p]));
      // Relegate fp8 reduction trees of sufficient depth that they incur precision loss
      // to be least preferred.
      if (info->datatype == ncclFloat8e4m3 || info->datatype == ncclFloat8e5m2) {
        if (a == NCCL_ALGO_RING && comm->nRanks > 8) {
          table[a][p] *= 1024.0; // Any factor large enough to act as a partition between lossy and non-lossy algos.
        }
      }
    }
  }

  return ncclSuccess;
}

static ncclResult_t topoGetAlgoInfo(
    struct ncclComm* comm, struct ncclTaskColl* info, size_t nBytes,
    float** collCostTable, ncclSimInfo_t* simInfo
  ) {
  float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;

  float minTime = FLT_MAX;
  int algorithm = info->algorithm = NCCL_ALGO_UNDEF;
  int protocol = info->protocol = NCCL_PROTO_UNDEF;
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      if (table[a][p] == NCCL_ALGO_PROTO_IGNORE) continue;
      if (table[a][p] >= 0.0 && table[a][p] < minTime) {
        algorithm = a;
        protocol = p;
        minTime = table[a][p];
      }
    }
  }

  info->algorithm = algorithm;
  info->protocol = protocol;
  float time = minTime;

  // Yes, we are first assigning and then testing if protocol is sane, but that's OK in this case.
  // coverity[check_after_sink]
  if (info->algorithm == NCCL_ALGO_UNDEF || info->protocol == NCCL_PROTO_UNDEF) {
    char ncclAlgoEnvStr[1024] = "";
    char ncclProtoEnvStr[1024] = "";
    const char* algoEnv = ncclGetEnv("NCCL_ALGO");
    if (algoEnv) {
      snprintf(ncclAlgoEnvStr, 1023, " NCCL_ALGO was set to %s.", algoEnv);
    }
    const char* protoEnv = ncclGetEnv("NCCL_PROTO");
    if (protoEnv) {
      snprintf(ncclProtoEnvStr, 1023, " NCCL_PROTO was set to %s.", protoEnv);
    }
    WARN("No algorithm/protocol available for function %s with datatype %s.%s%s", ncclFuncToString(info->func), ncclDatatypeToString(info->datatype), ncclAlgoEnvStr, ncclProtoEnvStr);
    return (algoEnv || protoEnv) ? ncclInvalidUsage : ncclInternalError;
  }
  if (simInfo) simInfo->estimatedTime = time;
  TRACE(NCCL_COLL, "%ld Bytes -> Algo %d proto %d time %f", nBytes, info->algorithm, info->protocol, time);

  int nc = comm->nChannels;
  int nt = comm->maxThreads[info->algorithm][info->protocol];
  int threadThreshold = comm->threadThresholds[info->algorithm][info->protocol];
  if (info->algorithm == NCCL_ALGO_COLLNET_DIRECT) {
    // CollNet channel tuning
    int ncSwitch = 16;
    bool flag = true;
    while (ncSwitch >= 1 && flag) {
      while ((flag = nBytes < nc*nt*comm->channels[0].collnetDirect.nHeads*threadThreshold) && nc > ncSwitch) {
        if (nc == ncSwitch+ncSwitch/2) threadThreshold /= 2;
        nc--;
      }
      ncSwitch /= 2;
    }
  } else if (info->algorithm == NCCL_ALGO_NVLS || info->algorithm == NCCL_ALGO_NVLS_TREE) {
    // NVLS should not need more than 16 channels to get peak BW.
    if (comm->nNodes > 1 && info->algorithm == NCCL_ALGO_NVLS) {
      nc = std::min(comm->nvlsChannels, comm->nChannels);
    } else {
      nc = comm->nvlsChannels;
    }
  } else {
    // Ring/Tree channel tuning
    while (nBytes < nc * nt * threadThreshold) {
      if (nc >= 2) nc--;
      else break;
    }
  }

  if (info->algorithm != NCCL_ALGO_NVLS && info->algorithm != NCCL_ALGO_NVLS_TREE &&
    info->algorithm != NCCL_ALGO_COLLNET_DIRECT) {
    while (nBytes < nc * nt * threadThreshold) {
      if (nt % 128 == 0) nt /= 2;
      else break;
    }
  }
  if (info->protocol == NCCL_PROTO_SIMPLE) {
    if (info->algorithm == NCCL_ALGO_RING) nt += WARP_SIZE; // Extra warp for sync
    // More threads or sync warps needed due to split thread model
    if (info->algorithm == NCCL_ALGO_TREE) nt += 4*WARP_SIZE;
  }
  nt = nt/WARP_SIZE < 3 ? 3*WARP_SIZE : nt;
  if (info->algorithm == NCCL_ALGO_TREE) nt = NCCL_MAX_NTHREADS; // Tree now uses all threads always.
  if (info->algorithm == NCCL_ALGO_PAT) nt = NCCL_MAX_NTHREADS;
  info->nMaxChannels = nc;
  info->nWarps = nt/WARP_SIZE;
  return ncclSuccess;
}

// Use the default topo-based tuner if tuner plugin is not successful.
// Call the plugin first. Let it set algo+proto, and/or nChannels.
// Then, topoGetAlgoInfo will set algo/proto if not set, then nChannels and nThreads based on algo/proto.
// Finally, nChannels will be overriden by the plugin setting.
ncclResult_t ncclGetAlgoInfo(
    struct ncclComm* comm, struct ncclTaskColl* info,
    int collNetSupport, int nvlsSupport, int numPipeOps, ncclSimInfo_t* simInfo/* = NULL*/
  ) {
  size_t elementSize = ncclTypeSize(info->datatype);
  size_t nBytes = elementSize * ncclFuncMaxSendRecvCount(info->func, comm->nRanks, info->count);
  struct ncclReg* regSendBuf = NULL;
  struct ncclReg* regRecvBuf = NULL;
  int regBuff;
  bool isSendValid, isRecvValid;
  size_t sendbuffSize = elementSize * ncclFuncSendCount(info->func, comm->nRanks, info->count);
  size_t recvbuffSize = elementSize * ncclFuncRecvCount(info->func, comm->nRanks, info->count);
  info->algorithm = NCCL_ALGO_UNDEF;
  info->protocol = NCCL_PROTO_UNDEF;
  int nMaxChannels = 0;
  float collCostTable[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  initCollCostTable((float **)collCostTable);
  NCCLCHECK(updateCollCostTable(comm, info, nBytes, collNetSupport, nvlsSupport, numPipeOps, (float **)collCostTable));
  if (comm->tuner != NULL) {
    NCCLCHECK(ncclRegFind(comm, info->sendbuff, sendbuffSize, &regSendBuf));
    NCCLCHECK(ncclRegFind(comm, info->recvbuff, recvbuffSize, &regRecvBuf));
    NCCLCHECK(ncclRegLocalIsValid(regSendBuf, &isSendValid));
    NCCLCHECK(ncclRegLocalIsValid(regRecvBuf, &isRecvValid));
    regBuff = (regSendBuf && regRecvBuf && isSendValid && isRecvValid) || (ncclCudaGraphValid(comm->planner.capturingGraph) && ncclParamGraphRegister());
    NCCLCHECK(comm->tuner->getCollInfo(
          comm->tunerContext, info->func, nBytes,
          numPipeOps, (float **)collCostTable, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS,
          regBuff, &nMaxChannels));
    NCCLCHECK(topoGetAlgoInfo(comm, info, nBytes, (float **)collCostTable, simInfo));
  } else {
    NCCLCHECK(topoGetAlgoInfo(comm, info, nBytes, (float **)collCostTable, simInfo));
    // NCCL_CTA_POLICY_EFFICIENCY requires user (non-symmetric) buffer registration (currently unsupported with MNNVL)
    if ((comm->config.CTAPolicy & NCCL_CTA_POLICY_EFFICIENCY) && ncclGetEnv("NCCL_ALGO") == NULL && ncclGetEnv("NCCL_PROTO") == NULL && !comm->MNNVL) {
      // make algorithm selection based on buffer registration
      // there can be other specialized policies for algorithms and protocols pickup in the future
      NCCLCHECK(ncclRegFind(comm, info->sendbuff, sendbuffSize, &regSendBuf));
      NCCLCHECK(ncclRegFind(comm, info->recvbuff, recvbuffSize, &regRecvBuf));
      NCCLCHECK(ncclRegLocalIsValid(regSendBuf, &isSendValid));
      NCCLCHECK(ncclRegLocalIsValid(regRecvBuf, &isRecvValid));
      regBuff = (regSendBuf && regRecvBuf && isSendValid && isRecvValid) || (ncclCudaGraphValid(comm->planner.capturingGraph) && ncclParamGraphRegister());
      if (regBuff && (info->func == ncclFuncAllGather || info->func == ncclFuncReduceScatter)) {
        if ((comm->nNodes > 1 && collNetSupport && nvlsSupport) || (comm->nNodes == 1 && nvlsSupport)) {
          int recChannels;
          NCCLCHECK(ncclNvlsRegResourcesQuery(comm, info, &recChannels));
          if (recChannels <= info->nMaxChannels) {
            info->algorithm = NCCL_ALGO_NVLS;
            info->protocol = NCCL_PROTO_SIMPLE;
            info->nMaxChannels = recChannels;
            info->nWarps = comm->maxThreads[info->algorithm][info->protocol] / WARP_SIZE;
          }
        }
      }
    }
  }

  info->nMaxChannels = nMaxChannels == 0 ? info->nMaxChannels : nMaxChannels;
  return ncclSuccess;
}

static ncclResult_t calcCollChunking(
    struct ncclComm* comm, struct ncclTaskColl* info, int nChannels, size_t nBytes,
    /*outputs*/uint32_t* outChunkSize, uint32_t* outDirectFlags, struct ncclProxyOp* proxyOp
  ) {
  ncclPattern_t pattern;
  size_t grainSize = ncclProtoGrainSize(info->protocol);

  switch (info->func) {
  case ncclFuncBroadcast:
    pattern = info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeDown : ncclPatternPipelineFrom;
    break;
  case ncclFuncReduce:
    pattern = info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeUp : ncclPatternPipelineTo;
    break;
  case ncclFuncReduceScatter:
    pattern =
      info->algorithm == NCCL_ALGO_PAT ? ncclPatternPatUp :
      info->algorithm == NCCL_ALGO_NVLS ? ncclPatternNvls :
      info->algorithm == NCCL_ALGO_COLLNET_DIRECT ? ncclPatternCollnetDirect :
      ncclPatternRing;
    break;
  case ncclFuncAllGather:
    pattern =
      info->algorithm == NCCL_ALGO_PAT ? ncclPatternPatDown :
      info->algorithm == NCCL_ALGO_NVLS ? ncclPatternNvls :
      info->algorithm == NCCL_ALGO_COLLNET_DIRECT ? ncclPatternCollnetDirect :
      ncclPatternRing;
    break;
  case ncclFuncAllReduce:
    pattern =
      info->algorithm == NCCL_ALGO_NVLS ? ncclPatternNvls :
      info->algorithm == NCCL_ALGO_NVLS_TREE ? ncclPatternNvlsTree :
      info->algorithm == NCCL_ALGO_COLLNET_DIRECT ? ncclPatternCollnetDirect :
      info->algorithm == NCCL_ALGO_COLLNET_CHAIN ? ncclPatternCollnetChain :
      info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeUpDown :
      ncclPatternRingTwice;
    break;
  default:
    WARN("Unknown pattern for collective %d algorithm %d", info->func, info->algorithm);
    return ncclInternalError;
  }

  int nstepsPerLoop, nchunksPerLoop;
  size_t loopOffset = 0;
  int stepSize   = comm->buffSizes[info->protocol]/NCCL_STEPS;
  int chunkSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->chunkSteps : 1;
  int sliceSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->sliceSteps : 1;
  int chunkSize = stepSize*chunkSteps;
  if (info->protocol == NCCL_PROTO_LL) chunkSize /= 2;
  if (info->protocol == NCCL_PROTO_LL128) chunkSize = (chunkSize / NCCL_LL128_LINEELEMS) * NCCL_LL128_DATAELEMS;
  // Buffer-based ceiling; plugins may increase chunk size up to this limit.
  int bufferMaxChunkSize = chunkSize;

  if (info->algorithm == NCCL_ALGO_COLLNET_DIRECT) {
    // Optimize chunkSize / nSteps
    while (nBytes / (nChannels * comm->channels[0].collnetDirect.nHeads * chunkSize) < comm->channels[0].collnetDirect.depth * 64 && chunkSize > 131072) chunkSize /= 2;
    while (nBytes / (nChannels * comm->channels[0].collnetDirect.nHeads * chunkSize) < comm->channels[0].collnetDirect.depth * 8 && chunkSize > 65536) chunkSize /= 2;
    while (nBytes / (nChannels * comm->channels[0].collnetDirect.nHeads * chunkSize) < comm->channels[0].collnetDirect.depth * 8 && chunkSize > 32768) chunkSize /= 2;
  } else if (info->algorithm == NCCL_ALGO_COLLNET_CHAIN) {
    stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;
    chunkSize = std::min(256 * 1024, stepSize * chunkSteps);
    while (nBytes / (nChannels * chunkSize) < comm->channels[0].collnetChain.depth * 64 && chunkSize > 131072) chunkSize /= 2;
    while (nBytes / (nChannels * chunkSize) < comm->channels[0].collnetChain.depth * 8 && chunkSize > 65536) chunkSize /= 2;
    while (nBytes / (nChannels * chunkSize) < comm->channels[0].collnetChain.depth && chunkSize > 32768) chunkSize /= 2;
  } else if (info->algorithm == NCCL_ALGO_NVLS) {
    if ((info->regBufType & NCCL_NVLS_REG_BUFFER) && (info->func == ncclFuncAllGather || info->func == ncclFuncReduceScatter)) {
      chunkSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;
    } else {
      int maxChunkSize = comm->nvlsChunkSize;
      if (comm->nNodes > 1 && comm->bandwidths[ncclFuncAllReduce][NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] < 150) maxChunkSize = 32768;
      if (chunkSize > maxChunkSize) chunkSize = maxChunkSize;
      // Use uint64_t so that concurrentOps*chunkSize*X does not overflow.
      // However, nChannels * comm->channels[0].nvls.nHeads should easily fit in 32 bits.
      // coverity[overflow_before_widen]
      uint64_t concurrentOps = nChannels * comm->channels[0].nvls.nHeads;
      if ((nBytes < (64 * (concurrentOps * chunkSize))) && (chunkSize > 65536)) chunkSize = 65536;
      if ((nBytes < (8 * (concurrentOps * chunkSize))) && (chunkSize > 32768)) chunkSize = 32768;
      if ((nBytes < (2 * (concurrentOps * chunkSize))) && (chunkSize > 16384)) chunkSize = 16384;
    }
  } else if (info->algorithm == NCCL_ALGO_NVLS_TREE) {
    // Use uint64_t so that concurrentOps*chunkSize*X does not overflow.
    // However, nChannels * comm->channels[0].nvls.nHeads should easily fit in 32 bits.
    // coverity[overflow_before_widen]
    uint64_t concurrentOps = nChannels * comm->channels[0].nvls.nHeads;
    chunkSize = std::min(comm->nvlsChunkSize, comm->nvlsTreeMaxChunkSize);
    if ((nBytes < (32 * (concurrentOps * chunkSize))) && (chunkSize > 262144)) chunkSize = 262144;
    if ((nBytes < (16 * (concurrentOps * chunkSize))) && (chunkSize > 131072)) chunkSize = 131072;
    if ((nBytes < (4 * (concurrentOps * chunkSize))) && (chunkSize > 65536)) chunkSize = 65536;
    if ((nBytes < (1 * (concurrentOps * chunkSize))) && (chunkSize > 32768)) chunkSize = 32768;
  } else if (info->algorithm == NCCL_ALGO_TREE && info->protocol == NCCL_PROTO_LL128) {
    int nNodes = comm->nNodes;
    float ppn = comm->nRanks / (float)nNodes;
    float nstepsLL128 = 1+log2i(nNodes) + 0.1*ppn;
    // Yes, we are OK with the division on the left side of the < operand being integer.
    // coverity[integer_division]
    while (nBytes / (nChannels*chunkSize) < nstepsLL128*64/ppn && chunkSize > 131072) chunkSize /= 2;
    // coverity[integer_division]
    while (nBytes / (nChannels*chunkSize) < nstepsLL128*16/ppn && chunkSize > 32768) chunkSize /= 2;
  } else if (info->func == ncclFuncAllGather && info->algorithm == NCCL_ALGO_PAT) {
    while (chunkSize*nChannels*32 > nBytes && chunkSize > 65536) chunkSize /= 2;
  } else if (info->func == ncclFuncReduceScatter && info->algorithm == NCCL_ALGO_PAT) {
    while (chunkSize*nChannels*16 > nBytes && chunkSize > 65536) chunkSize /= 2;
  }

  // Compute directFlags of work struct.
  if (info->algorithm == NCCL_ALGO_COLLNET_DIRECT) {
    *outDirectFlags = NCCL_P2P_WRITE;
  } else {
    *outDirectFlags = 0;
  }

  if (comm->tuner != nullptr && comm->tuner->getChunkSize != nullptr) {
    size_t tunerChunkSize = chunkSize;
    NCCLCHECK(comm->tuner->getChunkSize(comm->tunerContext, info->func, nBytes,
                                        info->algorithm, info->protocol, nChannels, &tunerChunkSize));
    if (tunerChunkSize > (size_t)bufferMaxChunkSize) {
      INFO(NCCL_TUNING, "%s: tuner chunk size %zu exceeds buffer max %d, clamping",
           ncclFuncToString(info->func), tunerChunkSize, bufferMaxChunkSize);
      tunerChunkSize = bufferMaxChunkSize;
    }
    chunkSize = (int)tunerChunkSize;
  }

  // Compute nSteps for proxies
  chunkSize = chunkSize / grainSize * grainSize; // align chunkSize to multiple grainSize
  switch (pattern) {
  case ncclPatternTreeUp:
  case ncclPatternTreeDown:
  case ncclPatternTreeUpDown:
  case ncclPatternPatUp:
  case ncclPatternPatDown:
  case ncclPatternPipelineFrom:
  case ncclPatternPipelineTo:
  case ncclPatternCollnetChain:
    nstepsPerLoop = nchunksPerLoop = 1;
    break;
  case ncclPatternNvls:
    nstepsPerLoop = 1; nchunksPerLoop = comm->channels[0].nvls.nHeads;
    loopOffset = nChannels * chunkSize * comm->channels[0].nvls.headRank;
    break;
  case ncclPatternCollnetDirect:
    nstepsPerLoop = 1; nchunksPerLoop = comm->channels[0].collnetDirect.nHeads;
    loopOffset = nChannels * chunkSize * comm->channels[0].collnetDirect.headRank;
    break;
  case ncclPatternRing:
    nstepsPerLoop = comm->nRanks-1; nchunksPerLoop = comm->nRanks;
    break;
  case ncclPatternRingTwice:
    nstepsPerLoop = 2*(comm->nRanks-1); nchunksPerLoop = comm->nRanks;
    break;
  case ncclPatternNvlsTree:
    nstepsPerLoop = 1; nchunksPerLoop = comm->channels[0].nvls.nHeads;
    break;
  default:
    WARN("Unknown pattern %d", pattern);
    return ncclInternalError;
  }

  // Compute nSteps for proxies
  size_t loopSize = size_t(nChannels)*nchunksPerLoop*chunkSize;
  int nLoops = (int)DIVUP(nBytes, loopSize);
  memset(proxyOp, 0, sizeof(*proxyOp));
  proxyOp->nsteps = nstepsPerLoop * nLoops * chunkSteps;
  proxyOp->sliceSteps = sliceSteps;
  proxyOp->chunkSteps = chunkSteps;
  proxyOp->chunkSize = chunkSize;
  proxyOp->sliceSize = chunkSize / chunkSteps * sliceSteps;
  proxyOp->loopSize = loopSize;
  proxyOp->loopOffset = loopOffset;
  proxyOp->protocol = info->protocol;
  proxyOp->dtype = info->datatype;
  proxyOp->algorithm = info->algorithm;
  if (info->opDev.op == ncclDevPreMulSum || info->opDev.op == ncclDevSumPostDiv) {
    proxyOp->redOp = ncclSum; // Network sees avg as sum
  } else {
    proxyOp->redOp = info->opHost;
  }
  proxyOp->pattern = pattern;
  proxyOp->coll = info->func;
  proxyOp->collAPI = info->func;
  proxyOp->root = info->root;
  proxyOp->isOneRPN = comm->isOneRPN;
  // This is used by P2P to reduce the receive buffer size. We don't use it in collectives
  // because some protocols need to transmit more than the total size, plus they sometimes
  // round up
  proxyOp->nbytes = stepSize*sliceSteps;

  if (info->regBufType & NCCL_NET_REG_BUFFER) {
    proxyOp->reg = 1;
    if (info->algorithm == NCCL_ALGO_COLLNET_DIRECT || info->algorithm == NCCL_ALGO_NVLS || info->algorithm == NCCL_ALGO_COLLNET_CHAIN) {
      if (proxyOp->isOneRPN) {
        proxyOp->nsteps = 1;
        proxyOp->loopOffset = 0;
        proxyOp->sendbuff = (uint8_t*)info->sendbuff;
        proxyOp->sendMhandle = info->sendMhandle;
      } else {
        if (info->func == ncclFuncAllGather || info->func == ncclFuncReduceScatter) {
          proxyOp->nbytes = nBytes / nchunksPerLoop;
          proxyOp->loopSize = proxyOp->loopSize / nchunksPerLoop;
          proxyOp->loopOffset = 0;
          if (info->func == ncclFuncAllGather) {
            proxyOp->sendbuff = (uint8_t*)info->sendbuff;
            proxyOp->sendMhandle = info->sendMhandle;
          }
        } else {
          proxyOp->sendbuff = (uint8_t*)info->recvbuff;
          proxyOp->sendMhandle = info->recvMhandle;
        }
      }
    } else if (info->algorithm == NCCL_ALGO_RING) {
      if (proxyOp->isOneRPN && info->func == ncclFuncAllGather) {
        proxyOp->chunkSize = NCCL_MAX_NET_SIZE;
        proxyOp->sliceSize = NCCL_MAX_NET_SIZE;
        proxyOp->chunkSteps = 1;
        proxyOp->sliceSteps = 1;
        proxyOp->loopSize = size_t(nChannels) * nchunksPerLoop * proxyOp->chunkSize;
        proxyOp->nsteps = DIVUP(nBytes, proxyOp->loopSize) * nstepsPerLoop;
        proxyOp->loopOffset = 0;
      }
    } else {
      WARN("Net registration invalid algorithm %s", ncclAlgoToString(info->algorithm));
      return ncclInternalError;
    }

    proxyOp->recvMhandle = info->recvMhandle;
    proxyOp->recvbuff = (uint8_t*)info->recvbuff;
    proxyOp->nbytes = nBytes;
  } else {
    proxyOp->reg = 0;
  }

  if (pattern == ncclPatternCollnetDirect || pattern == ncclPatternNvls) {
    proxyOp->specifics.collnetDirect.nNodes = comm->nNodes;
    proxyOp->specifics.collnetDirect.node = comm->node;
    if (info->func == ncclFuncAllGather || info->func == ncclFuncReduceScatter) {
      proxyOp->specifics.collnetDirect.sizePerRank = info->count*ncclTypeSize(info->datatype);
    }
  }

  if (pattern == ncclPatternPatUp || pattern == ncclPatternPatDown) {
    proxyOp->nbytes = DIVUP(nBytes, nChannels);
  }

  // Set peer count hints used by network plugin
  switch (proxyOp->pattern) {
  case ncclPatternRing:
  case ncclPatternRingTwice:
  case ncclPatternPipelineFrom:
  case ncclPatternPipelineTo:
  case ncclPatternPatUp:
  case ncclPatternPatDown:
    proxyOp->nPeers = 1;
    break;
  case ncclPatternTreeUp:
  case ncclPatternTreeDown:
  case ncclPatternTreeUpDown:
  case ncclPatternNvlsTree:
    proxyOp->nPeers = (NCCL_MAX_TREE_ARITY - 1) * 2;
    break;
  case ncclPatternCollnetChain:
  case ncclPatternCollnetDirect:
  case ncclPatternNvls:
  case ncclPatternProfiler:
    // Peer count hints unused
    break;
  case ncclPatternSend:
  case ncclPatternRecv:
  default:
    WARN("Unknown pattern %d", pattern);
    return ncclInternalError;
  }

  *outChunkSize = proxyOp->chunkSize;
  return ncclSuccess;
}

static ncclResult_t hostToDevRedOp(
    ncclDevRedOpFull *opFull, ncclRedOp_t op, ncclDataType_t datatype, ncclComm *comm
  ) {
  union {
    int8_t   i8; uint8_t   u8;
    int32_t i32; uint32_t u32;
    int64_t i64; uint64_t u64;
    __half f16; float f32; double f64;
    #if defined(__CUDA_BF16_TYPES_EXIST__)
      __nv_bfloat16 bf16;
    #endif
    #if defined(__CUDA_FP8_TYPES_EXIST__)
      __nv_fp8_storage_t f8;
    #endif
    void *ptr;
  };
  u64 = 0;
  opFull->scalarArgIsPtr = false;
  opFull->proxyOp = op;

  int nbits = 8*ncclTypeSize(datatype);
  if (nbits <= 0) return ncclInvalidArgument;
  uint64_t allBits = uint64_t(-1)>>(64-nbits);
  uint64_t signBit = allBits^(allBits>>1);
  bool datatype_signed = false;

  switch (int(op)) {
  case ncclSum:  opFull->op = ncclDevSum;  break;
  case ncclProd: opFull->op = ncclDevProd; break;
  case ncclMin:
  case ncclMax:
    opFull->op = ncclDevMinMax;
    opFull->scalarArg = 0;
    // The xormask used by ncclFuncMinMax<[u]int> is the XOR of the sign bit
    // for signed (opposed to unsigned) types and all the bits for max (opposed to min).
    if (datatype==ncclInt8 || datatype==ncclInt32 || datatype==ncclInt64) {
      opFull->scalarArg ^= signBit;
    }
    opFull->scalarArg ^= (op == ncclMax) ? allBits : 0;
    break;
  case ncclAvg:
    switch ((int)datatype) {
    case ncclInt8:  case ncclInt32:  case ncclInt64:
      datatype_signed = true;
      // no break, we want to fall through...
    case ncclUint8: case ncclUint32: case ncclUint64:
      opFull->op = ncclDevSumPostDiv;
      u64 = comm->nRanks<<1 | datatype_signed;
      break;
    #if defined(__CUDA_FP8_TYPES_EXIST__)
    case ncclFloat8e4m3:
      opFull->op = ncclDevPreMulSum;
      f8 = __nv_cvt_float_to_fp8(float(1.0/comm->nRanks), __NV_SATFINITE, __NV_E4M3);
      break;
    case ncclFloat8e5m2:
      opFull->op = ncclDevPreMulSum;
      f8 = __nv_cvt_float_to_fp8(float(1.0/comm->nRanks), __NV_SATFINITE, __NV_E5M2);
      break;
    #endif
    case ncclFloat16:
      opFull->op = ncclDevPreMulSum;
      f16 = __float2half(float(1.0/comm->nRanks)); // __double2half not supported pre CUDA 11.x
      break;
    #if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16:
      opFull->op = ncclDevPreMulSum;
      bf16 = __float2bfloat16(float(1.0/comm->nRanks));
      break;
    #endif
    case ncclFloat32:
      opFull->op = ncclDevPreMulSum;
      f32 = float(1.0/comm->nRanks);
      break;
    case ncclFloat64:
      opFull->op = ncclDevPreMulSum;
      f64 = 1.0/comm->nRanks;
      break;
    }
    opFull->scalarArgIsPtr = false;
    opFull->scalarArg = u64;
    break;
  default: // user created
    int ix = int(ncclUserRedOpMangle(comm, op)) - int(ncclNumOps);
    ncclUserRedOp *user = &comm->userRedOps[ix];
    if (datatype != user->datatype) {
      WARN("Data type supplied to user-created ncclRedOp_t does not match type "
           "given to reduction operation");
      return ncclInvalidArgument;
    }
    *opFull = user->opFull;
    break;
  }
  return ncclSuccess;
}

static ncclResult_t ncclPlannerSetCapturingGraph(struct ncclComm* comm, struct ncclInfo* info) {
  struct ncclKernelPlanner *planner = &comm->planner;
  if (info->stream != planner->streamRecent || planner->streams == nullptr) {
    planner->streamRecent = info->stream;
    struct ncclCudaStreamList* l = planner->streams;
    while (true) {
      if (l == nullptr) { // Got to the end, this must be a new stream.
        struct ncclCudaGraph graph;
        NCCLCHECK(ncclCudaGetCapturingGraph(&graph, info->stream, comm->config.graphUsageMode));
        if (planner->streams != nullptr && !ncclCudaGraphSame(planner->capturingGraph, graph)) {
          WARN("Streams given to a communicator within a NCCL group must either be all uncaptured or all captured by the same graph.");
          return ncclInvalidUsage;
        }
        planner->capturingGraph = graph; // C++ struct assignment
        // Add stream to list
        l = ncclMemoryStackAlloc<struct ncclCudaStreamList>(&comm->memScoped);
        l->stream = info->stream;
        l->next = planner->streams;
        planner->streams = l;
        break;
      }
      if (l->stream == info->stream)
        break; // Already seen stream.
      l = l->next;
    }
  }
  return ncclSuccess;
}

static ncclResult_t p2pTaskAppend(
    struct ncclComm* comm,
    struct ncclInfo* info,
    ncclFunc_t coll,
    ncclFunc_t collAPI,
    void* buff,
    size_t count,
    ncclDataType_t datatype,
    int peer, bool allowUB) {
  struct ncclKernelPlanner *planner = &comm->planner;

  // Determine peer and basic parameters.
  ssize_t nBytes = count*ncclTypeSize(datatype);
  bool isSendNotRecv = coll == ncclFuncSend;

  // Must be in thread local group before tasks can be alloc'd in `comm->memScoped`.
  ncclGroupCommJoin(comm, ncclGroupTaskTypeCollective);
  info->coll = coll;
  // Set capturing graph. Called here so that profiler can emit a group API event with this information
  NCCLCHECK(ncclPlannerSetCapturingGraph(comm, info));
  bool isGraphCaptured = ncclCudaGraphValid(planner->capturingGraph);
  NCCLCHECK(ncclProfilerStartGroupApiEvent(info, isGraphCaptured));
  NCCLCHECK(ncclProfilerRecordGroupApiEventState(ncclProfilerGroupStartApiStop));

  NCCLCHECK(ncclProfilerStartP2pApiEvent(info, isGraphCaptured));

  struct ncclTaskP2p* p2p = ncclMemoryPoolAlloc<struct ncclTaskP2p>(&comm->memPool_ncclTaskP2p, &comm->memPermanent);
  p2p->func = coll;
  p2p->collAPI = collAPI;
  p2p->buff = buff;
  p2p->count = count;
  p2p->datatype = datatype;
  p2p->root = peer;
  p2p->bytes = nBytes;
  p2p->allowUB = allowUB;
  p2p->eActivationMask = ncclProfilerApiState.eActivationMask;
  p2p->groupApiEventHandle = ncclProfilerApiState.groupApiEventHandle;
  p2p->p2pApiEventHandle = ncclProfilerApiState.p2pApiEventHandle;
  ncclIntruQueueEnqueue(
    isSendNotRecv ? &planner->peers[peer].sendQueue : &planner->peers[peer].recvQueue,
    p2p);
  planner->nTasksP2p += 1;
  if (isSendNotRecv)
    planner->nTasksP2pSend += 1;
  else
    planner->nTasksP2pRecv += 1;

  // Mark channels that need pre-connect
  if (comm->rank != peer) {
    if (!(isSendNotRecv ? planner->peers[peer].sendSeen : planner->peers[peer].recvSeen)) {
      // planner->peers[peer].send/recvSeen is private to each comm, so we need to set it anyway.
      (isSendNotRecv ? planner->peers[peer].sendSeen : planner->peers[peer].recvSeen) = true;
      int round = 0;
      while (peer != (isSendNotRecv ? comm->p2pSchedule[round].sendRank
                                    : comm->p2pSchedule[round].recvRank)) {
        round += 1;
      }
      uint8_t base = ncclP2pChannelBaseForRound(comm, round);
      for (int c=0; c < comm->p2pnChannelsPerPeer; c++) {
        int channelId = ncclP2pChannelForPart(comm->p2pnChannels, base, c);
        if (isSendNotRecv) {
          if (comm->channels[channelId].peers[peer]->send[1].hasSeen == 0) { // P2P uses only 1 connector
            // the send/recv connector is shared among split shared comms. We need to set hasSeen to
            // 1 in order to avoid duplicate connection setup if user group sendrecv ops with split
            // shared comms together.
            comm->channels[channelId].peers[peer]->send[1].hasSeen = 1;
            comm->channels[channelId].peers[peer]->send[1].p2pOnly = 1;
            comm->connectSend[peer] |= (1ULL<<channelId);
            ncclGroupCommPreconnect(comm);
          }
        } else {
          if (comm->channels[channelId].peers[peer]->recv[1].hasSeen == 0) { // P2P uses only 1 connector
            comm->channels[channelId].peers[peer]->recv[1].hasSeen = 1;
            comm->channels[channelId].peers[peer]->recv[1].p2pOnly = 1;
            comm->connectRecv[peer] |= (1ULL<<channelId);
            ncclGroupCommPreconnect(comm);
          }
        }
      }
    }
  }
  ncclProfilerStopP2pApiEvent();
  return ncclSuccess;
}

static ncclResult_t collTaskAppend(
    struct ncclComm* comm,
    struct ncclInfo* info,
    struct ncclDevRedOpFull opDev) {
  struct ncclKernelPlanner *planner = &comm->planner;

  // Must be in thread local group before tasks can be alloc'd in `comm->memScoped`.
  ncclGroupCommJoin(info->comm, ncclGroupTaskTypeCollective);
  // Set capturing graph. Called here so that profiler can emit a group API event with this information
  NCCLCHECK(ncclPlannerSetCapturingGraph(comm, info));

  bool isGraphCaptured = ncclCudaGraphValid(planner->capturingGraph);
  NCCLCHECK(ncclProfilerStartGroupApiEvent(info, isGraphCaptured));
  NCCLCHECK(ncclProfilerRecordGroupApiEventState(ncclProfilerGroupStartApiStop));
  NCCLCHECK(ncclProfilerStartCollApiEvent(info, isGraphCaptured));

  if (info->coll == ncclFuncBroadcast && ncclParamAllgathervEnable() && !comm->ccEnable) {
    // Must be in thread local group before tasks can be alloc'd in `comm->memScoped`.
    struct ncclTaskBcast* t = ncclMemoryPoolAlloc<struct ncclTaskBcast>(&comm->memPool_ncclTaskBcast, &comm->memPermanent);
    t->func = ncclFuncAllGatherV;
    t->sendbuff = info->sendbuff;
    t->recvbuff = info->recvbuff;
    t->count = info->count * ncclTypeSize(info->datatype);
    t->datatype = ncclInt8;
    t->root = info->root;

    // update bcast min/max peer
    planner->bcast_info.minBcastPeer = std::min(planner->bcast_info.minBcastPeer, info->root);
    planner->bcast_info.maxBcastPeer = std::max(planner->bcast_info.maxBcastPeer, info->root);
    if (ncclIntruQueueEmpty(&planner->peers[info->root].bcastQueue)) {
      planner->bcast_info.BcastPeers += 1;
    }

    // enqueue to peer's bcast queue instead of collSorter
    ncclIntruQueueEnqueue(&planner->peers[info->root].bcastQueue, t);
    planner->nTasksBcast += 1;
  }
  else {
  struct ncclTaskColl* t = ncclMemoryPoolAlloc<struct ncclTaskColl>(&comm->memPool_ncclTaskColl, &comm->memPermanent);
  t->func = info->coll;
  t->sendbuff = info->sendbuff;
  t->recvbuff = info->recvbuff;
  t->count = info->count;
  t->root = info->root;
  t->datatype = info->datatype;
  size_t elementSize = ncclTypeSize(t->datatype);
  if (t->func == ncclFuncAllGather || t->func == ncclFuncBroadcast) {
    t->count *= elementSize;
    t->datatype = ncclInt8;
    elementSize = 1;
  }
  t->trafficBytes = t->count*elementSize*ncclFuncTrafficPerByte(t->func, comm->nRanks);
  t->opHost = info->op;
  t->opDev = opDev; // C++ struct assignment
  t->chunkSteps = info->chunkSteps;
  t->sliceSteps = info->sliceSteps;
  t->eActivationMask = ncclProfilerApiState.eActivationMask;
  t->groupApiEventHandle = ncclProfilerApiState.groupApiEventHandle;
  t->collApiEventHandle = ncclProfilerApiState.collApiEventHandle;

  planner->nTasksColl += 1;
  ncclTaskCollSorterInsert(&planner->collSorter, t, t->trafficBytes);
  }
  ncclProfilerStopCollApiEvent();
  return ncclSuccess;
}

static ncclResult_t ceCollTaskAppend(
    struct ncclComm* comm,
    struct ncclInfo* info,
    struct ncclDevrWindow* sendWin,
    struct ncclDevrWindow* recvWin,
    struct ncclDevRedOpFull opDev) {
  struct ncclKernelPlanner *planner = &comm->planner;

  // Check if CE needs initialization
  if (comm->ceColl.baseUCSymReadyPtr == NULL && ncclIntruQueueEmpty(&comm->ceInitTaskQueue)) {
    struct ncclCeInitTask* ceTask;
    NCCLCHECK(ncclCalloc(&ceTask, 1));
    ceTask->comm = comm;
    ncclIntruQueueEnqueue(&comm->ceInitTaskQueue, ceTask);
    ncclGroupCommJoin(comm, ncclGroupTaskTypeSymRegister);
  }

  // Must be in thread local group before tasks can be alloc'd in `comm->memScoped`.
  ncclGroupCommJoin(info->comm, ncclGroupTaskTypeCollective);
  // Set capturing graph. Called here so that profiler can emit a group API event with this information
  NCCLCHECK(ncclPlannerSetCapturingGraph(comm, info));
  bool isGraphCaptured = ncclCudaGraphValid(planner->capturingGraph);
  NCCLCHECK(ncclProfilerStartGroupApiEvent(info, isGraphCaptured));
  NCCLCHECK(ncclProfilerRecordGroupApiEventState(ncclProfilerGroupStartApiStop));
  NCCLCHECK(ncclProfilerStartCollApiEvent(info, isGraphCaptured));

  struct ncclTaskColl* t = ncclMemoryPoolAlloc<struct ncclTaskColl>(&comm->memPool_ncclTaskColl, &comm->memPermanent);

  t->func = info->coll;
  t->sendbuff = info->sendbuff;
  t->recvbuff = info->recvbuff;
  t->count = info->count;
  t->root = info->root;
  t->datatype = info->datatype;
  size_t elementSize = ncclTypeSize(t->datatype);
  if (t->func == ncclFuncAllGather || t->func == ncclFuncBroadcast) {
    t->count *= elementSize;
    t->datatype = ncclInt8;
    elementSize = 1;
  }
  t->trafficBytes = t->count*elementSize*ncclFuncTrafficPerByte(t->func, comm->nRanks);
  t->opHost = info->op;
  t->opDev = opDev; // C++ struct assignment
  t->chunkSteps = info->chunkSteps;
  t->sliceSteps = info->sliceSteps;
  t->eActivationMask = COMPILER_ATOMIC_LOAD(&ncclProfilerEventMask, std::memory_order_relaxed);
  t->groupApiEventHandle = ncclProfilerApiState.groupApiEventHandle;
  t->collApiEventHandle = ncclProfilerApiState.collApiEventHandle;
  t->sendWin = sendWin;
  t->recvWin = recvWin;

  ncclIntruQueueEnqueue(&planner->collCeTaskQueue, t);

  ncclProfilerStopCollApiEvent();
  return ncclSuccess;
}

static ncclResult_t rmaTaskAppend(
  struct ncclComm* comm,
  struct ncclInfo* info) {
  struct ncclKernelPlanner *planner = &comm->planner;

  void const* srcBuff = info->sendbuff;

  if (!comm->hostRmaSupport) {
    WARN("One sided RMA: host RMA is not supported in this communicator.");
    return ncclInvalidArgument;
  }

  int driverVersion;
  NCCLCHECK(ncclCudaDriverVersion(&driverVersion));
  if (driverVersion < 12050) {
    WARN("One-sided RMA requires CUDA driver 12.5 or later (found %d.%d).",
      driverVersion / 1000, (driverVersion % 1000) / 10);
    return ncclInvalidUsage;
  }

  // Check if context is valid (must be 0 for now)
  if (info->ctx != 0) {
    WARN("Context %d is invalid (must be 0)", info->ctx);
    return ncclInvalidArgument;
  }

  // Check if signal index is valid (must be 0 for now)
  if (info->sigIdx != 0) {
    WARN("Signal index %d is invalid (must be 0)", info->sigIdx);
    return ncclInvalidArgument;
  }

  // Check if flags is valid
  if (info->flags != 0) {
    WARN("Flags %u is invalid (must be 0)", info->flags);
    return ncclInvalidArgument;
  }

  // Initialize window pointers - only needed for Put and Signal
  struct ncclDevrWindow* peerWinHost = NULL;
  struct ncclDevrWindow* srcWinHost = NULL;
  size_t srcWinOffset = 0;

  if (info->coll == ncclFuncPutSignal) {
    // Validate peer window with detailed debugging
    if (info->peerWin == NULL) {
      WARN("ncclPutSignal: peerWin is NULL");
      return ncclInvalidArgument;
    }

    struct ncclWindow_vidmem* peerWinDevHost = NULL;
    NCCLCHECK(ncclShadowPoolToHost(&comm->devrState.shadows, info->peerWin, &peerWinDevHost));
    peerWinHost = (struct ncclDevrWindow*)peerWinDevHost->winHost;

    // Validate source buffer and window
    if (srcBuff == NULL) {
      WARN("ncclPutSignal: srcBuff is NULL");
      return ncclInvalidArgument;
    }
    NCCLCHECK(ncclDevrFindWindow(comm, srcBuff, &srcWinHost));
    if (srcWinHost == NULL || !(srcWinHost->winFlags & NCCL_WIN_COLL_SYMMETRIC)) {
      WARN("ncclPutSignal: srcWinHost is not in a valid symmetric window");
      return ncclInvalidArgument;
    }
    srcWinOffset = (char*)srcBuff - (char*)srcWinHost->userPtr;

    bool isMultiSegment = ncclDevrWindowIsMultiSegment(srcWinHost) || ncclDevrWindowIsMultiSegment(peerWinHost);
    bool hasSysmemSegment = ncclDevrWindowHasSysmemSegment(srcWinHost) || ncclDevrWindowHasSysmemSegment(peerWinHost);

    if (isMultiSegment) {
      WARN("ncclPutSignal currently does not support VAs backed by multiple physical cuMem segments");
      return ncclInvalidArgument;
    }
    if (hasSysmemSegment) {
      WARN("ncclPutSignal currently does not support VAs with host-backed cuMem segments");
      return ncclInvalidArgument;
    }
  }
  else if (info->coll == ncclFuncSignal) {
    // Check if count is valid
    if (info->count != 0) {
      WARN("ncclSignal: count must be 0");
      return ncclInvalidArgument;
    }
  }
  else if (info->coll == ncclFuncWaitSignal) {
    // Check if signalDescs is valid
    if (info->signalDescs == NULL || info->nDesc == 0) {
      WARN("ncclWaitSignal: invalid arguments");
      return ncclInvalidArgument;
    }
    // Validate each descriptor
    for (int i = 0; i < info->nDesc; i++) {
      if (info->signalDescs[i].opCnt <= 0) {
        WARN("ncclWaitSignal: descriptor %d has invalid opCnt %d", i, info->signalDescs[i].opCnt);
        return ncclInvalidArgument;
      }
      if (info->signalDescs[i].sigIdx != 0) {
        WARN("ncclWaitSignal: descriptor %d has invalid sigIdx %d (must be 0)", i, info->signalDescs[i].sigIdx);
        return ncclInvalidArgument;
      }
      if (info->signalDescs[i].ctx != 0) {
        WARN("ncclWaitSignal: descriptor %d has invalid context %d (must be 0)",
             i, info->signalDescs[i].ctx);
        return ncclInvalidArgument;
      }
    }
  }

  // Check if RMA CE needs initialization
  if (!comm->rmaState.rmaCeState.initialized && ncclIntruQueueEmpty(&comm->rmaCeInitTaskQueue)) {
    struct ncclRmaCeInitTask* ceTask;
    NCCLCHECK(ncclCalloc(&ceTask, 1));
    ceTask->comm = comm;
    ncclIntruQueueEnqueue(&comm->rmaCeInitTaskQueue, ceTask);
    ncclGroupCommJoin(comm, ncclGroupTaskTypeSymRegister);
  }

  // Must be in thread local group before tasks can be alloc'd in `comm->memScoped`.
  ncclGroupCommJoin(info->comm, ncclGroupTaskTypeCollective);
  NCCLCHECK(ncclPlannerSetCapturingGraph(comm, info));


  // Handle WaitSignal separately
  if (info->coll == ncclFuncWaitSignal) {
    struct ncclTaskRma* t = ncclMemoryPoolAlloc<struct ncclTaskRma>(&comm->memPool_ncclTaskRma, &comm->memPermanent);

    t->func = ncclFuncWaitSignal;
    t->ctx = 0;
    t->count = 0;
    t->bytes = 0;
    t->srcBuff = NULL;
    t->srcWinOffset = 0;
    t->srcWinHost = NULL;
    t->peer = 0;
    t->peerWinOffset = 0;
    t->peerWinHost = NULL;
    t->signalMode = NCCL_SIGNAL;

    // Convert descriptors to peers and nsignals arrays
    t->npeers = info->nDesc;
    t->peers = ncclMemoryStackAlloc<int>(&comm->memScoped, info->nDesc);
    t->nsignals = ncclMemoryStackAlloc<int>(&comm->memScoped, info->nDesc);

    for (int i = 0; i < info->nDesc; i++) {
      t->peers[i] = info->signalDescs[i].peer;
      t->nsignals[i] = info->signalDescs[i].opCnt;
    }

    t->eActivationMask = COMPILER_ATOMIC_LOAD(&ncclProfilerEventMask, std::memory_order_relaxed);
    planner->nTasksRma++;
    ncclIntruQueueEnqueue(&planner->rmaTaskQueues[t->ctx], t);

  } else if (info->coll == ncclFuncPutSignal || info->coll == ncclFuncSignal) {

    // Calculate total bytes for the operation
    size_t totalBytes = info->count * ncclTypeSize(info->datatype);

    // Define 1GB chunk size for splitting large put operations
    const size_t chunkSize = 1ULL << 30; // 1GB = 1073741824 bytes

    // Determine if we need to split the operation
    int numChunks = 1;
    if (info->coll == ncclFuncPutSignal && totalBytes > chunkSize) {
      numChunks = (totalBytes + chunkSize - 1) / chunkSize;
    }

    // Create tasks for each chunk
    for (int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
      struct ncclTaskRma* t = ncclMemoryPoolAlloc<struct ncclTaskRma>(&comm->memPool_ncclTaskRma, &comm->memPermanent);

      // Calculate chunk-specific size and offsets
      size_t chunkBytes = (chunkIdx == numChunks - 1)
                          ? (totalBytes - chunkIdx * chunkSize)
                          : chunkSize;

      size_t chunkOffset = chunkIdx * chunkSize;

      t->func = info->coll;
      t->srcBuff = (const char*)srcBuff + chunkOffset;
      t->srcWinOffset = srcWinOffset + chunkOffset;
      t->srcWinHost = srcWinHost;
      t->count = chunkBytes / ncclTypeSize(info->datatype);
      t->datatype = info->datatype;
      t->bytes = chunkBytes;
      t->ctx = info->ctx;
      t->peer = info->root;
      t->peerWinOffset = info->peerWinOffset + chunkOffset;
      t->peerWinHost = peerWinHost;

      // Signal handling: only the last chunk gets the signal
      bool isLastChunk = (chunkIdx == numChunks - 1);
      if (isLastChunk) {
        t->signalMode = NCCL_SIGNAL;
      } else {
        // Earlier chunks: no signal
        t->signalMode = NCCL_SIGNAL_NONE;
      }
      t->peers = NULL;
      t->nsignals = NULL;
      t->npeers = 0;

      t->eActivationMask = COMPILER_ATOMIC_LOAD(&ncclProfilerEventMask, std::memory_order_relaxed);

      planner->nTasksRma++;
      // Enqueue the task into the appropriate context queue
      ncclIntruQueueEnqueue(&planner->rmaTaskQueues[t->ctx], t);
    }
  }

  return ncclSuccess;
}

// Converts `info` to a task and adds it to `comm->planner`. The exception is with
// single rank communicators, collectives are issued as `ncclMemcpyAsync`s and
// thus don't need a task.
static ncclResult_t taskAppend(struct ncclComm* comm, struct ncclInfo* info) {
  ncclFunc_t collAPI = info->coll;

  if (info->coll == ncclFuncSend || info->coll == ncclFuncRecv) {
    NCCLCHECK(p2pTaskAppend(comm, info, info->coll, collAPI, (void*)info->recvbuff, info->count, info->datatype, info->root, true));
  } else if (info->coll == ncclFuncPutSignal || info->coll == ncclFuncSignal || info->coll == ncclFuncWaitSignal) {
    NCCLCHECK(rmaTaskAppend(comm, info));
  } else {
    // Empty collectives can be discarded.
    if (info->count == 0) return ncclSuccess;

    if (info->datatype == ncclFloat8e4m3 || info->datatype == ncclFloat8e5m2) {
      if (comm->minCompCap < 90 && info->coll != ncclFuncAllGather && info->coll != ncclFuncBroadcast && info->coll != ncclFuncAlltoAll && info->coll != ncclFuncScatter && info->coll != ncclFuncGather) {
        WARN("FP8 reduction support begins with sm90 capable devices.");
        return ncclInvalidArgument;
      }
    }

    // Copy reduction op state from op handle into info struct here since the
    // op handle may be destroyed before ncclGroupEnd().
    struct ncclDevRedOpFull opDev;
    NCCLCHECK(hostToDevRedOp(&opDev, info->op, info->datatype, comm));

    if (comm->nRanks == 1) {
      NCCLCHECK(ncclLaunchOneRank(info->recvbuff, info->sendbuff, info->count, opDev, info->datatype, info->stream));
      return ncclSuccess;
    } else {
      struct ncclDevrWindow* sendWin;
      struct ncclDevrWindow* recvWin;
      ncclDevrFindWindow(comm, info->sendbuff, &sendWin);
      ncclDevrFindWindow(comm, info->recvbuff, &recvWin);
      // Append CE collective task if CE is supported and requested by user
      ncclSymRegType_t winRegType;
      NCCLCHECK(ncclGetSymRegType(sendWin, recvWin, &winRegType));
      bool ceAvailable = ncclCeAvailable(comm, info->coll, info->op, info->datatype, winRegType);
      bool hasSysmemSegment = ncclDevrWindowHasSysmemSegment(sendWin) || ncclDevrWindowHasSysmemSegment(recvWin);

      if ((comm->config.CTAPolicy & NCCL_CTA_POLICY_ZERO) && ceAvailable && !hasSysmemSegment) {
        NCCLCHECK(ceCollTaskAppend(comm, info, sendWin, recvWin, opDev));
      }
      // Append kernel-based collective
      else {
        // currently legacy sendrecv needs src and dst buffers to be registered
        // we cannot allow UB if alltoall/scatter/gather fallback to legacy sendrecv
        // when src or dst buffers are not registered
        struct ncclReg* sendReg = NULL;
        struct ncclReg* recvReg = NULL;
        bool allowUB = false;
        bool captured = false;
        struct ncclCudaGraph graph;
        // For cuda graph checking
        NCCLCHECK(ncclCudaGetCapturingGraph(&graph, info->stream, comm->config.graphUsageMode));
        captured = ncclCudaGraphValid(graph);
        if (info->coll == ncclFuncAlltoAll) {
          NCCLCHECK(ncclRegFind(comm, info->sendbuff, comm->nRanks * info->count * ncclTypeSize(info->datatype), &sendReg));
          NCCLCHECK(ncclRegFind(comm, info->recvbuff, comm->nRanks * info->count * ncclTypeSize(info->datatype), &recvReg));
          allowUB = captured || (sendReg != NULL && recvReg != NULL);
          for (int r=0; r<comm->nRanks; r++) {
            NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncSend, collAPI, (void*)((char*)info->sendbuff+r*info->count*ncclTypeSize(info->datatype)), info->count, info->datatype, r, allowUB));
            NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncRecv, collAPI, (void*)((char*)info->recvbuff+r*info->count*ncclTypeSize(info->datatype)), info->count, info->datatype, r, allowUB));
          }
        } else if (info->coll == ncclFuncGather){
          size_t offset = 0;
          allowUB = captured;
          NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncSend, collAPI, (void*)info->sendbuff, info->count, info->datatype, info->root, allowUB));
          if (comm->rank == info->root) {
            for (int r=0; r<comm->nRanks; r++) {
              void* buff = (void*)((char*)info->recvbuff + offset);
              NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncRecv, collAPI, buff, info->count, info->datatype, r, allowUB));
              offset += info->count * ncclTypeSize(info->datatype);
            }
          }
        } else if (info->coll == ncclFuncScatter) {
          size_t offset = 0;
          allowUB = captured;
          if (comm->rank == info->root) {
            for (int r = 0; r < comm->nRanks; r++) {
              void* buff = (void*)((char*)info->sendbuff + offset);
              NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncSend, collAPI, buff, info->count, info->datatype, r, allowUB));
              offset += info->count * ncclTypeSize(info->datatype);
            }
          }
          NCCLCHECK(p2pTaskAppend(comm, info, ncclFuncRecv, collAPI, (void*)info->recvbuff, info->count, info->datatype, info->root, allowUB));
        } else if (ceAvailable && comm->symmetricSupport && info->coll == ncclFuncAllGather && info->count > ncclParamSymCeThreshold() && comm->minCompCap >= 100 && comm->isAllDirectNvlink) {
          // Use CE for Allgather on Blackwell with size > 8MB
          NCCLCHECK(ceCollTaskAppend(comm, info, sendWin, recvWin, opDev));
        } else {
          NCCLCHECK(collTaskAppend(comm, info, opDev));
        }
      }
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclEnqueueCheck(struct ncclInfo* info) {
  // Early-out on invalid or revoked communicator
  ncclResult_t ret = CommCheck(info->comm, info->opName, "comm");
  if (ret != ncclSuccess) return ncclGroupErrCheck(ret);
  if (info->comm->revokedFlag) {
    WARN("%s: communicator was revoked", info->opName);
    return ncclGroupErrCheck(ncclInvalidUsage);
  }
  // Profiler - If a group API event has already started, update the profilerGroupDepth so that the depth
  // updates correctly for implicit ncclGroupStartInternal and ncclGroupEndInternal calls
  if (ncclProfilerApiState.profilerGroupDepth > 0) {
    ncclProfilerApiState.profilerGroupDepth++;
  }
  NCCLCHECK(ncclGroupStartInternal());
  ret = ncclSuccess;
  int devOld = -1;
  // Check whether communicator is ready to communicate
  NCCLCHECKGOTO(ncclCommEnsureReady(info->comm), ret, fail);

  if (info->comm->checkMode != ncclCheckModeDefault) {
    CUDACHECKGOTO(cudaGetDevice(&devOld), ret, fail);
    CUDACHECKGOTO(cudaSetDevice(info->comm->cudaDev), ret, fail);
  }
  // If info->comm->checkMode == ncclCheckModeDebugGlobal, ArgsCheck will enqueue info
  // for collectives and the pairs of peers for sendrecv for global check later
  NCCLCHECKGOTO(ArgsCheck(info), ret, fail);

  INFO(NCCL_COLL,"%s: opCount %lx sendbuff %p recvbuff %p count %zu datatype %d op %d root %d comm %p [nranks=%d] stream %p",
        info->opName, info->comm->opCount, info->sendbuff, info->recvbuff, info->count,
        info->datatype, info->op, info->root, info->comm, info->comm->nRanks, info->stream);
  TRACE_CALL("nccl%s(%" PRIx64 ",%" PRIx64 ",%zu,%d,%d,%d,%p,%p)", info->opName, reinterpret_cast<int64_t>(info->sendbuff), reinterpret_cast<int64_t>(info->recvbuff), info->count, info->datatype, info->op, info->root, info->comm, info->stream);

  NCCLCHECKGOTO(taskAppend(info->comm, info), ret, fail);

exit:
  if (devOld != -1) CUDACHECK(cudaSetDevice(devOld));
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  /* if depth is 1, ncclGroupEndInternal() will trigger group ops. The state can change
   * so we have to check state here. */
  if (info->comm && !info->comm->config.blocking) { NCCLCHECK(ncclCommGetAsyncError(info->comm, &ret)); }
  return ret;
fail:
  if (info->comm && !info->comm->config.blocking) (void) ncclCommSetAsyncError(info->comm, ret);
  goto exit;
}

NCCL_API(ncclResult_t, ncclRedOpCreatePreMulSum, ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm);
ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm) {
  NCCLCHECK(CommCheck(comm, "ncclRedOpCreatePreMulSum", "comm"));
  /* join init thread before creating PreMulSum op. */
  NCCLCHECK(ncclCommEnsureReady(comm));

  if (comm->userRedOpFreeHead == comm->userRedOpCapacity) {
    // double capacity and resize
    int cap = 2*comm->userRedOpCapacity;
    if (cap < 4) cap = 4;
    ncclUserRedOp *ops = new ncclUserRedOp[cap];
    if (comm->userRedOpCapacity > 0)
      std::memcpy(ops, comm->userRedOps, comm->userRedOpCapacity*sizeof(ncclUserRedOp));
    for(int ix=comm->userRedOpCapacity; ix < cap; ix++)
      ops[ix].freeNext = ix + 1;
    delete[] comm->userRedOps;
    comm->userRedOps = ops;
    comm->userRedOpCapacity = cap;
  }
  // pop from free list
  int ix = comm->userRedOpFreeHead;
  ncclUserRedOp *user = &comm->userRedOps[ix];
  comm->userRedOpFreeHead = user->freeNext;

  user->freeNext = -1; // allocated
  user->datatype = datatype;
  user->opFull.op = ncclDevPreMulSum;
  if (residence == ncclScalarHostImmediate) {
    int size = ncclTypeSize(datatype);
    if (size < 1) return ncclInternalError;
    user->opFull.scalarArgIsPtr = false;
    std::memcpy(&user->opFull.scalarArg, scalar, size);
  } else {
    user->opFull.scalarArgIsPtr = true;
    user->opFull.scalarArg = reinterpret_cast<uint64_t>(scalar);
  }
  *op = ncclRedOp_t(int(ncclNumOps) + ix);
  *op = ncclUserRedOpMangle(comm, *op);
  TRACE_CALL("ncclRedOpCreatePreMulSum(%d,%p,%d,%d,%p)", *op, scalar, datatype, residence, comm);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclRedOpDestroy, ncclRedOp_t op, ncclComm_t comm);
ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
  if (0 <= int(op) && int(op) < int(ncclNumOps)) {
    WARN("ncclRedOpDestroy : operator is a NCCL builtin.");
    return ncclInvalidArgument;
  }
  // int(ncclMaxRedOp) < int(op) will always be false due to the sizes of
  // the datatypes involved, and that's by design.  We keep the check though
  // just as a reminder.
  // coverity[result_independent_of_operands]
  if (int(op) < 0 || int(ncclMaxRedOp) < int(op)) {
    WARN("ncclRedOpDestroy :  operator is garbage.");
    return ncclInvalidArgument;
  }
  if (comm == NULL) {
    WARN("ncclRedOpDestroy : invalid communicator passed.");
    return ncclInvalidArgument;
  }

  int ix = int(ncclUserRedOpMangle(comm, op)) - int(ncclNumOps);
  if (comm->userRedOpCapacity <= ix || comm->userRedOps[ix].freeNext != -1) {
    WARN("ncclRedOpDestroy : operator unknown to this communicator.");
    return ncclInvalidArgument;
  }
  // push to free list
  comm->userRedOps[ix].freeNext = comm->userRedOpFreeHead;
  comm->userRedOpFreeHead = ix;
  TRACE_CALL("ncclRedOpDestroy(%d,%p)", op, comm);
  return ncclSuccess;
}
