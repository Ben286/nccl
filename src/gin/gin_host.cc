/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/


// ============================================================================
// gin_host.cc — GIN（GPU Initiated Network）主机侧管理实现
// ============================================================================
// 本文件负责 GIN 连接建立、进展线程、内存注册以及上层 API。
//
// ===== GIN 模型回顾 =====
// GIN 允许 GPU kernel 直接发起网络传输，不需要 CPU 中继。
// 有两种后端：
//   NCCL_GIN_TYPE_PROXY  — GPU 写 GFD 队列 → CPU proxy 转发 IB Verbs
//   NCCL_GIN_TYPE_GDAKI  — GPU 直接操作 DOCA GPUNetIO Verbs QP（真正 bypass CPU）
//
// ===== 连接建立流程（ncclGinConnectOnce）=====
//   1. 加载插件，查询 GIN 设备列表
//   2. bootstrapAllGather 同步 ginCommCount（所有 rank 取 min）
//   3. 每个 connection：listen → bootstrapAllGather 共享 handle → connect → createContext
//   4. 如果是 PROXY 后端，启动 ncclGinProgress 进展线程
//   5. 创建 signalSpace/counterSpace 内存分配器
//
// ===== 内存注册（ncclGinRegister）=====
//   将用户的 symmetric buffer 注册到每个 GIN connection
//   得到 ginHostWins（host 内存句柄）和 ginDevWins（device 可用的 window 句柄）
//   GPU kernel 通过 ginDevWins 就能对该内存进行 远端 RDMA write
//
// ===== 信号/计数器分配（ncclGinAllocSignalsCounters）=====
//   signalSpace 和 counterSpace 是两个线性内存分配器，分别管理：
//   - signal：rail 信号数组（每个 block 对每个 peer 一个 slot）
//   - counter：GIN put 完成计数器（每个 block 一个）
// ============================================================================
#include "comm.h"
#include "gin.h"
#include "param.h"
#include "graph.h"
#include "transport.h"
#include "register_inline.h"
#include "gin/gin_host.h"
#include "gin/gin_host_proxy.h"
#include "compiler.h"
#include <cmath>

NCCL_PARAM(GinEnable, "GIN_ENABLE", 1);  // NCCL_GIN_ENABLE=0 可关闭 GIN

// getGlobalGinType：获取全局 GIN 类型
// 需要 comm->globalGinSupport == NCCL_GIN_CONNECTION_FULL（所有 rank 都支持 GIN）
// 否则返回 NCCL_GIN_TYPE_NONE，即使部分 rank 支持 GIN 也不算
ncclResult_t getGlobalGinType(struct ncclComm* comm, ncclGinType_t* ginType) {
  if (comm == nullptr || ginType == nullptr) {
    return ncclInternalError;
  }

  if (comm->globalGinSupport != NCCL_GIN_CONNECTION_FULL) {
    *ginType = NCCL_GIN_TYPE_NONE;
    return ncclSuccess;
  }

  *ginType = comm->sharedRes->ginState.ginType;
  return ncclSuccess;
}

// getGlobalRailedGinType：获取 Rail 范围内的 GIN 类型
// 与 getGlobalGinType 的区别：即使是 NCCL_GIN_CONNECTION_PARTIAL（只有部分 rank 支持 GIN）也不返回 NONE
// Rail 范围内的 AllGather 即使不是全局支持也可以运行，因为只需要 rail 内的 rank 支持 GIN
ncclResult_t getGlobalRailedGinType(struct ncclComm* comm, ncclGinType_t* ginType) {
  if (comm == nullptr || ginType == nullptr) {
    return ncclInternalError;
  }

  if (comm->globalGinSupport == NCCL_GIN_CONNECTION_NONE) {
    *ginType = NCCL_GIN_TYPE_NONE;
    return ncclSuccess;
  }
  *ginType = comm->sharedRes->ginState.ginType;
  return ncclSuccess;
}

ncclResult_t setLocalGinType(struct ncclComm* comm) {
  if (comm == nullptr || comm->sharedRes->ginState.ncclGin == nullptr) {
    return ncclInternalError;
  }
  ncclGinState& ginState = comm->sharedRes->ginState;
  ginState.ginType = NCCL_GIN_TYPE_NONE;

  if (!ncclParamGinEnable()) {
    return ncclSuccess;
  }

  if (comm->compCap < 70) {
    /* GIN only supported for Volta and later */
    INFO(NCCL_INIT, "Compute Capability (%d) is not sufficient to enable GIN.  Require Volta (70) or newer.",comm->compCap);
    return ncclSuccess;
  }

  ncclNetProperties_t props;
  NCCLCHECK(ginState.ncclGin->getProperties(0, &props));
  if (props.netDeviceType == NCCL_NET_DEVICE_GIN_PROXY ||
      props.netDeviceType == NCCL_NET_DEVICE_GIN_GDAKI) {
    // NOTE: The following cast is valid because ncclGinType_t variant values
    // should match NCCL_NET_DEVICE_GIN_* values from `enum ncclNetDeviceType`.
    ginState.ginType = static_cast<ncclGinType_t>(props.netDeviceType);

    if (ginState.ginType == NCCL_GIN_TYPE_PROXY) {
      // Replace ginState->ncclGin by a layer adding host queues
      NCCLCHECK(ncclGinProxyInit(&ginState.ncclGin));
    }
    return ncclSuccess;
  }
  WARN("Cannot get gin type: ncclGin is not null but net device type (%d) is not a gin type",
       props.netDeviceType);
  return ncclInternalError;
}

// ncclGinProgress：GIN PROXY 模式下的进展线程主函数
// ============================================================================
// 当 ginType == NCCL_GIN_TYPE_PROXY 时，GPU kernel 把 GFD（Go Forward Descriptor）写入共享队列，
// 但真正的 IB Verbs 提交需要 CPU 来做。这个线程就是该 CPU 代理。
//
// 状态机：
//   ginProgress == 0 → 初始化中，等待条件变量通知
//   ginProgress == 1 → 正常运行，轮询所有 connection 的 progress
//   ginProgress == -1 → 请求退出（ncclGinHostFinalize 中设置）
//   ginProgress == -2 → 错误状态
//
// 对于 GDAKI 后端：
//   GPU 直接推送，cpu 侧也调用 ginState->ncclGin->ginProgress（内部处理一些完成事件）
//   但实际上 GDAKI 不依赖这个线程，needsProxyProgress == 0 时不会启动线程
// ============================================================================
void* ncclGinProgress(struct ncclGinState* ginState_) {
  struct ncclGinState* ginState = (struct ncclGinState*)ginState_;
  while (1) {
    // 每次循环都持锁检查状态，避免 ginProgress 在检查后被外部修改
    // 注意：ginProgress==1（运行态）时会立刻 unlock 以免持锁时间过长
    std::unique_lock<std::mutex> lock(ginState->mutex);
    if (ginState->ginProgress == 1) {
      struct ncclGinStateDevComm* dc = ginState->devComms;
      while (dc) {
        for (int n=0; n<ginState->ginCommCount; n++) {
          ncclResult_t ret = ginState->ncclGin->ginProgress(dc->ginCtx[n]);
          if (ret != ncclSuccess) {
          // 异步错误：原子写入 asyncResult（memory_order_release 确保其他线程可见）
          // 上层 ncclCommGetAsyncError 通过 ncclGinQueryLastError 检查此值
            COMPILER_ATOMIC_STORE(&ginState->asyncResult, ret, std::memory_order_release);
            INFO(NCCL_ALL,"%s:%d -> %d [GIN Progress Thread]", __FILE__, __LINE__, ret);
            ginState->ginProgress = -2;  // 切换到错误态，防止无限 retry
      // 退出信号：ncclGinHostFinalize 将 ginProgress 设置为 -1 后 notify_one
      // 此处直接返回，线程退出，join 将在 ncclGinHostFinalize 中完成
            return NULL;
          }
        }
        dc = dc->next;
      }
      // 正常运行态：立刻释放锁，进入 busy-poll 循环
      // 不持锁 poll 的原因：progress 调用本身可能很耗时（IB CQ poll），
      // 如果持锁会导致 ncclGinHostFinalize 中的 notify_one 长时间等待
      lock.unlock();
      // yield()：每轮 poll 后主动让出 CPU 时间片
      // 这是 spin-yield 策略：不 sleep（避免毫秒级延迟），也不纯 busy-poll（避免 100% 烧 CPU）。
      // 底层是 sched_yield()：告诉 OS 调度器"我的时间片不要了"，如果没有其他就绪线程会立刻返回。
      // 不需要唤醒机制——OS 调度器会在下一个时间片自然恢复此线程，回到 while(1) 顶部继续 poll。
      std::this_thread::yield();
    } else if (ginState->ginProgress == -1) {
      return NULL;
    } else if (ginState->ginProgress == 0) {
      // 初始化态：等待条件变量通知（连接建立完成时 ncclGinConnectOnce 会设置为 1 并 notify）
      // cond.wait(lock)：原子释放 lock，进入等待；被唤醒后重新持锁
      ginState->cond.wait(lock);
    } else {
      INFO(NCCL_ALL,"%s:%d -> [GIN Progress Thread] state unknown %d", __FILE__, __LINE__, ginState->ginProgress);
      ginState->ginProgress = -2;
      return NULL;
    }
  }
}

NCCL_PARAM(GinNconnections, "GIN_NCONNECTIONS", -2);  // 强制指定 GIN connection 数

ncclResult_t ncclGinConnectOnce(struct ncclComm* comm) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  if (ginState->connected) return ncclSuccess;  // 已连接则直接返回（共享资源随上）

  ncclResult_t ret = ncclSuccess;
  if (ncclParamGinEnable() == 0) {
    WARN("GIN is disabled.");
    return ncclInternalError;
  }

  // Load plugin
  if (ginState->ncclGin == NULL) {
    WARN("GIN not supported.");
    return ncclInvalidUsage;
  }

  ginState->ginConnectionType = comm->globalGinSupport;
  ginState->ginInstance = comm->ginContext;

  int ndev = 0;
  NCCLCHECK(ginState->ncclGin->devices(&ndev));
  if (ndev <= 0) {
    WARN("No GIN-capable devices found.");
    return ncclInternalError;
  }

  if (!comm->symmetricSupport) {
    WARN("Communicator does not support symmetric memory!");
    return ncclInternalError;
  }

  int nLocalGinDevs;
  int localGinDevs[NCCL_TOPO_MAX_NODES];
  NCCLCHECK(ncclTopoGetLocalGinDevs(comm, localGinDevs, &nLocalGinDevs));

  void** handles = NULL;
  char* allHandles = NULL;

  int* ginCommCountHandles = NULL;
  NCCLCHECKGOTO(ncclCalloc(&ginCommCountHandles, comm->nRanks), ret, fail);

  // ginCommCount 初始化策略：
  //   默认值 = nLocalGinDevs（本节点可用的 GIN NIC 数）
  //   v11：不支持 nContextsPerComm > 1，所以用 reqGinContextCount 作为 connection 数
  //   环境变量覆盖：NCCL_GIN_NCONNECTIONS / NCCL_GIN_NCONTEXTS 可以手动指定
  ginState->ginCommCount = nLocalGinDevs;
  if (ginState->ginVersion < 13) {
    // We only support one context per connection, so we better create as many connections as possible.
    ginState->ginCommCount = NCCL_GIN_MAX_CONNECTIONS;
  }

  if (ncclParamGinNconnections() != -2) ginState->ginCommCount = ncclParamGinNconnections();
  // 不超过系统最大允许的 connection 数
  ginState->ginCommCount = std::min<int>(NCCL_GIN_MAX_CONNECTIONS, ginState->ginCommCount);

  // ===== 全局协商 ginCommCount =====
  // 问题：不同节点可能有不同数量的 GIN NIC（硬件配置不一致）
  //   如果 rank A 认为可以建 4 个 connection，而 rank B 只能建 2 个，
  //   则二者 connection 数对应不上，导致 bootstrapAllGather 里的各 handle 数量不一致
  // 解决方案：AllGather 收集所有 rank 的 ginCommCount，取全局最小值
  ginCommCountHandles[comm->rank] = ginState->ginCommCount;  // 将本 rank 的数量填入 handles[myRank]
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, ginCommCountHandles, sizeof(int)), ret, fail);
  for (int r = 0; r < comm->nRanks; r++) {
    // 全局取 min：确保所有 rank 建立相同数量的 connection
    // 这样每个 connection n 的 listen/connect 就能严格对应，不会出现 rank A n=3 而 rank B n=1 的情况
    ginState->ginCommCount = std::min(ginState->ginCommCount, ginCommCountHandles[r]);
  }

  NCCLCHECKGOTO(ncclCalloc(&allHandles, (size_t)comm->nRanks * NCCL_NET_HANDLE_MAXSIZE), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&handles, comm->nRanks), ret, fail);

  // 全局 connection 建立：连接到所有 nRanks（用于 ReduceScatter RailA2A）
  // Rail connection：只连接同一 Rail 内的 rank（数量少，用于 AllGather RailRing）
  int nGinRanks;
  int myGinRank;
  if (ginState->ginConnectionType == NCCL_GIN_CONNECTION_FULL) {
    nGinRanks = comm->nRanks;
    myGinRank = comm->rank;
    for (int r = 0; r < nGinRanks; r++) {
      handles[r] = allHandles + r * NCCL_NET_HANDLE_MAXSIZE;
    }
  } else {
    ncclTeam_t railTeam = ncclTeamRail(comm);
    nGinRanks = railTeam.nRanks;
    myGinRank = railTeam.rank;
    for (int r = 0; r < nGinRanks; r++) {
      int worldRank = ncclTeamRankToWorld(comm, railTeam, r);
      handles[r] = allHandles + worldRank * NCCL_NET_HANDLE_MAXSIZE;
    }
  }

  for (int n = 0; n < ginState->ginCommCount; n++) {
    void* listenComm;
    // ncclSpaceAlloc：从 counterSpace 线性分配器中申请 nCounters 个连续様本
    //   counterSpaceSize：池总大小上限
    //   nCounters：申请的连续 counter 数（即 nBlocks）
    //   对齐幅度 = 1
    //   如果分配失败，还需要回滚已分配的 signals（跳到 fail_signals）
    // ncclSpaceAlloc：从 signalSpace 线性分配器中申请 nSignals 个连续様本
    //   signalSpaceSize：池总大小上限（无法分配到 signalSpaceSize 以外）
    //   nSignals：申请的连续様本数（即 nBlocks * rail.nRanks）
    //   对齐幅度 = 1（无对齐要求，每个 slot 就是一个 uint64_t 信号）
    //   返回：分配到的第一个 slot 的索引（start）
    // 注意：ncclSpaceAlloc 不进行实际 malloc，只是修改分配器的指针（类似 bump allocator）
    // 类比：想象所有 signal 组成一个大数组，ncclSpaceAlloc 返回的 start 就是子数组的起始索引
    // 每个 connection 对应一个本地 GIN 设备（round-robin 选择）
    // listen 就是返回一个 handle，其他 rank 可用该 handle 建立连接
    // allHandles 内存布局：[rank0_handle | rank1_handle | ... | rankN-1_handle]
    // 每个 handle 占 NCCL_NET_HANDLE_MAXSIZE 字节
    NCCLCHECKGOTO(
      ginState->ncclGin->listen(ginState->ginInstance, localGinDevs[n%nLocalGinDevs],
                                allHandles + NCCL_NET_HANDLE_MAXSIZE * comm->rank, &listenComm),
      ret, fail);

    NCCLCHECKGOTO(ginState->ncclGin->getProperties(localGinDevs[n%nLocalGinDevs], ginState->ginProps+n),
      ret, fail);

    // bootstrapAllGather：将本 rank 的 handle 发广到所有 rank
    // 完成后 allHandles[r * NCCL_NET_HANDLE_MAXSIZE] = rank r 的 listen handle
    // 必须每个 loop iteration 都重新 gather，因为 listen 每次返回不同 handle
    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allHandles, NCCL_NET_HANDLE_MAXSIZE), ret,
                  fail);

      // PROXY 模式连接建立流程：
      //   Step 1: ncclGin->connect
      //     - 建立到 nGinRanks 个目标的 IB QP（Queue Pair）
      //     - nContextsPerComm：这个 connection 下共享的 GPU context 数
      //     - ginQueueDepth：GFD 环形队列深度，决定 GPU 最多可以吸纳多少个未完成的 GFD
      //     - 将连接结果存入 ginState->ginComms[n]
      //   Step 2: ncclGinProxyCreateContext
      //     - 分配共享内存映射的 GFD 队列（GPU 写入，CPU 读取）
      //     - 分配 pis/cis 数组（每个 peer 一个 producer/consumer index）
      //     - 分配 signals 和 counters 数组（GPU kernel 可以直接访问）
      //     - 返回 ginCtx[n]（CPU proxy 管理的 context 指针）
      //     - 返回 ginDevHandles[n]（GPU kernel 中直接使用的设备句柄）
    NCCLCHECKGOTO(ginState->ncclGin->connect(comm->ginContext, handles, nGinRanks, myGinRank,
          listenComm, ginState->ginComms + n),
        ret, fail);

    // 关闭 listen comm，释放内部资源（socket 或内存）
    NCCLCHECKGOTO(ginState->ncclGin->closeListen(listenComm), ret, fail);
  }
  free(handles);
  handles = NULL;
  free(allHandles);
  allHandles = NULL;
  free(ginCommCountHandles);
  ginCommCountHandles = NULL;

exit:
  if (ret == ncclSuccess) ginState->connected = true;
  return ret;
fail:
  if (allHandles)
    free(allHandles);
  if (handles)
    free(handles);
  if (ginCommCountHandles)
    free(ginCommCountHandles);
  goto exit;
}

ncclResult_t ncclGinDevCommSetup(struct ncclComm* comm, struct ncclDevCommRequirements const* reqs,
    struct ncclDevComm* devComm) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;

  devComm->ginSignalCount = reqs->ginSignalCount;
  devComm->ginCounterCount = reqs->ginCounterCount;

  // Allocate contexts
  int nContextsTotal = reqs->ginContextCount;
  if (ginState->ginVersion < 13) {
    nContextsTotal = ginState->ginCommCount;  // v11/12：强制 1:1（每 connection 一个 context）
  }
  devComm->ginContextCount = nContextsTotal;
  devComm->ginConnectionCount = ginState->ginCommCount;

  if (!reqs->ginExclusiveContexts) {
    // TODO: check if a shared devComm in the list could match our requirements.
  }

  // 对齐到 ginCommCount 的倍数，确保每个 connection 分到相同数量的 context
  nContextsTotal = ROUNDUP(nContextsTotal, ginState->ginCommCount);
  int nContextsPerComm = nContextsTotal / ginState->ginCommCount;
  INFO(NCCL_INIT, "devCommCreate: creating %d contexts: %d GIN connections with %d contexts each (%d contexts total requested)",
      nContextsTotal, ginState->ginCommCount, nContextsPerComm, reqs->ginContextCount);

  struct ncclGinStateDevComm* ginStateDevComm = NULL;
  NCCLCHECK(ncclCalloc(&ginStateDevComm, 1));
  ginStateDevComm->contextCount = nContextsTotal;
  ncclResult_t ret = ncclSuccess;

  ncclGinConfig_t ginConfig = {
    reqs->ginSignalCount,
    reqs->ginCounterCount,
    nContextsPerComm,
    reqs->ginQueueDepth,
    reqs->ginTrafficClass != NCCL_CONFIG_UNDEF_INT ? reqs->ginTrafficClass : comm->config.trafficClass
  };

  for (int n = 0; n < ginState->ginCommCount; n++) {
    NCCLCHECKGOTO(ginState->ncclGin->createContext(
                    ginState->ginComms[n], &ginConfig, &ginStateDevComm->ginCtx[n], &ginStateDevComm->devHandles[n]),
                  ret, end);
    devComm->ginNetDeviceTypes[n] = ginStateDevComm->devHandles[n]->netDeviceType;
    devComm->ginHandles[n] = ginStateDevComm->devHandles[n]->handle;
    if (ginStateDevComm->devHandles[n]->needsProxyProgress) ginState->needsProxyProgress = 1;
  }

  if (ginState->needsProxyProgress && ginState->ginProgress == 0) {
    // 启动进展线程步骤：
    //   1. 把 ginProgress 设为 1（运行态），确保线程启动后直接进入 poll 循环
    //   2. std::thread 启动线程，入口函数为 ncclGinProgress
    //   3. ncclSetThreadName：给线程设置名字（pthreads 中最长 15 字符）
    //      命名规范："NCCL GIN Progress%2d"，%2d 是 CUDA 设备号
    ginState->ginProgress = 1;
    ginState->thread = std::thread(ncclGinProgress, ginState);
    ncclSetThreadName(ginState->thread, "NCCL GIN Progress%2d", comm->cudaDev);
  }

  // Add devComm context to the list
  {
    std::unique_lock<std::mutex> lock(ginState->mutex);
    struct ncclGinStateDevComm* last = ginState->devComms;
    if (last) {
      while (last->next) last = last->next;
      last->next = ginStateDevComm;
     } else {
      ginState->devComms = ginStateDevComm;
    }
  }

end:
  if (ret != ncclSuccess) {
    for (int n=0; n<ginState->ginCommCount; n++) {
      if (ginStateDevComm->ginCtx[n])
        ginState->ncclGin->destroyContext(ginStateDevComm->ginCtx[n]);
    }
    free(ginStateDevComm);
  }
  return ret;
}

ncclResult_t ncclGinDevCommFree(struct ncclComm* comm, struct ncclDevComm const* devComm) {
  // Find the resource associated with this devComm. Use the gin handle as key.
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  struct ncclGinStateDevComm* dc = ginState->devComms, *prevDc = NULL;
  while (1) {
    if (dc == NULL) {
      WARN("Dev comm not found\n");
      return ncclInternalError;
    }
    if (dc->devHandles[0]->handle == devComm->ginHandles[0]) break;
    prevDc = dc;
    dc = dc->next;
  }

  std::unique_lock<std::mutex> lock(ginState->mutex);
  // Remove from linked list
  if (prevDc) prevDc->next = dc->next;
  else ginState->devComms = dc->next;
  lock.unlock();

  // Free GIN contexts
  for (int n = 0; n < ginState->ginCommCount; n++) {
    NCCLCHECK(ginState->ncclGin->destroyContext(dc->ginCtx[n]));
  }
  free(dc);
  return ncclSuccess;
}

// ncclGinHostFinalize：释放 GIN 所有资源
// 顺序：
//   1. 通知进展线程退出（设置 ginProgress=-1，等待 join）
//   2. 销毁每个 connection 的 context（PROXY: ncclGinProxyDestroyContext，GDAKI: destroyContext）
//   3. 关闭每个 connection(closeColl）
//   4. memset 清零 ginState(避免悬空指针）
ncclResult_t ncclGinHostFinalize(struct ncclComm* comm) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  if (!ginState->connected) return ncclSuccess;

  if (ginState->needsProxyProgress) {
    {
      std::lock_guard<std::mutex> lock(ginState->mutex);
      comm->sharedRes->ginState.ginProgress = -1;
      ginState->cond.notify_one();
    }
    ginState->thread.join();
  }

  for (int n = 0; n < ginState->ginCommCount; n++) {
    if (ginState->ginComms[n] != NULL) {
      NCCLCHECK(ginState->ncclGin->closeColl(ginState->ginComms[n]));
      ginState->ginComms[n] = NULL;
    }
  }
  memset((void*)ginState, 0, sizeof(*ginState));
  return ncclSuccess;
}

// ncclGinRegister：将一块内存注册到所有 GIN connection
// 主要用于 Symmetric Memory 的 alloc 流程：
//   ncclSymmetricAllocate（或内部等价）首先分配 GPU 内存，然后调用此函数注册到 GIN
// 得到：
//   ginHostWins[n]：host 内存 MR 句柄，PROXY 用于描述远端内存
//   ginDevWins[n]：device 内存 window 句柄，GPU kernel 通过它指定远端内存位置
ncclResult_t ncclGinRegister(struct ncclComm* comm, void* address, size_t size,
                             void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS],
                             ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS], int winFlags,
                             bool multiSegment) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  if (multiSegment) {
    // Multi-segment GIN registration requires DMABUF support on all GIN connections
    for (int n = 0; n < ginState->ginCommCount; n++) {
      if (!(ginState->ginProps[n].ptrSupport & NCCL_PTR_DMABUF)) {
        WARN("Window registration of addresses that span multiple physical segments requires DMABUF support with GIN.");
        return ncclInvalidArgument;
      }
    }
  }
  // NCCL_WIN_STRICT_ORDERING：要求内存注册时强制使用 Strict Ordering
  // NCCL_NET_MR_FLAG_FORCE_SO：告知网存插件使用带内存屏障的 MR 注册方式
  //   Strict Ordering 下：每次 IB RDMA write 完成后，远端内存对其他设备可见（类似 PCIe 层的 fence）
  //   非 Strict Ordering：允许 NIC 重新排序，性能更高但内存内容可能乱序到达
  int mrFlags = (winFlags & NCCL_WIN_STRICT_ORDERING) ? NCCL_NET_MR_FLAG_FORCE_SO : 0;
  for (int n = 0; n < ginState->ginCommCount; n++) {
      // GDAKI 模式：调用 regMrSym 直接在 DOCA GPUNetIO QP 上注册
      //   返回的 ginDevWins[n] 可由 GPU kernel 直接传递给 DOCA API
      //   返回的 ginHostWins[n] 主要用于后续注销
    NCCLCHECK(ginState->ncclGin->regMrSym(ginState->ginComms[n], address, size, NCCL_PTR_CUDA, mrFlags,
                                          &ginHostWins[n], &ginDevWins[n]));
    if (ginHostWins[n] == NULL) {
      WARN("rank %d - GIN Symmetric register failed: buff %p, size %ld", comm->rank, address, size);
      return ncclSystemError;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclGinDeregister(struct ncclComm* comm, void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS]) {
  struct ncclGinState* ginState = &comm->sharedRes->ginState;
  for (int n = 0; n < ginState->ginCommCount; n++) {
    NCCLCHECK(ginState->ncclGin->deregMrSym(ginState->ginComms[n], ginHostWins[n]));
  }
  return ncclSuccess;
}

// ncclGinQueryLastError：查询是否有 GIN 异步错误
// 用于 ncclCommGetAsyncError 中检测 GIN 后台进展线程是否运行失败
ncclResult_t ncclGinQueryLastError(struct ncclGinState* ginState, bool* hasError) {
  *hasError = false;
  struct ncclGinStateDevComm* dc = ginState->devComms;
  while (dc) {
    for (int n = 0; n < ginState->ginCommCount; n++) {
      NCCLCHECK(ginState->ncclGin->queryLastError(dc->ginCtx[n], hasError));
      if (*hasError) return ncclSuccess;
    }
    dc = dc->next;
  }
  return ncclSuccess;
}
