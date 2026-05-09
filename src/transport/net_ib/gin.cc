/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common.h"

#include "gin/gin_host.h"
#include "gin.h"

const int NCCL_GIN_IB_ALLGATHER_TAG = 0xa0;
const int NCCL_GIN_IB_ALLTOALL_TAG = 0xa1;

// Check GDR support for GIN. This is run at init, so we don't know yet whether the GPU will support DMA-BUF.
static ncclResult_t ncclGinIbGdrSupport(bool* gdrSupport, bool gdaki) {
  *gdrSupport = true;
  bool peerMemSupport =
     gdaki ? ncclIbPeerMemSupport() == ncclSuccess : // GDAKI does not support nv_peer_mem.
     ncclIbGdrSupport() == ncclSuccess;
  if (peerMemSupport) return ncclSuccess;

  if (ncclIbDmaBufSupport(0) == ncclSuccess) return ncclSuccess;

  *gdrSupport = false;
  INFO(NCCL_NET, "Unable to use GIN: Peermem is not supported, nor DMA-BUF.");
  return ncclSuccess;
}

// Check the current GPU supports GDR for GIN. This is run during connect().
static ncclResult_t ncclGinIbGdrGpuSupport(bool gdaki) {
  bool peerMemSupport =
     gdaki ? ncclIbPeerMemSupport() == ncclSuccess : // GDAKI does not support nv_peer_mem.
     ncclIbGdrSupport() == ncclSuccess;
  if (peerMemSupport) return ncclSuccess;

  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  int dmaBufSupportOnDevice = 1;
  CUCHECK(cuDeviceGetAttribute(&dmaBufSupportOnDevice, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, cudaDev));
  if (dmaBufSupportOnDevice == 1) return ncclSuccess;

  WARN("Unable to use GIN: Peermem is not supported, and device %d does not support DMA-BUF.", cudaDev);
  return ncclInvalidUsage;
}

NCCL_PARAM(GinType, "GIN_TYPE", -1);

static std::mutex ncclGinIbGdakiLockMutex;
static int ncclGinIbGdakiNDevs = -1;
int ncclGinIbGdakiDevIndexes[MAX_IB_DEVS];

ncclResult_t ncclGinIbGdakiInit() {
  std::lock_guard<std::mutex> lock(ncclGinIbGdakiLockMutex);
  if (ncclGinIbGdakiNDevs == -1) {
    int ndevs = 0;
    for (int i = 0; i < ncclNIbDevs; i++) {
      if (ncclIbDevs[i].ibProvider == IB_PROVIDER_MLX5) {
        ncclGinIbGdakiDevIndexes[ndevs] = i;
        ++ndevs;
      }
    }
    ncclGinIbGdakiNDevs = ndevs;
  }
  return ncclSuccess;
}

extern ncclGin_t ncclGinIb;
extern ncclGin_t ncclGinIbGdaki;
extern ncclGin_t ncclGinIbProxy;

// Initlialize GDAKI or PROXY backend. ginType can force a particular backend.
// If provided, overwrite ginIb with the backend (generic ginIb case).
ncclResult_t ncclGinIbInitType(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction, int ginType, ncclGin_t* ginIb) {
  NCCLCHECK(ncclIbInitDevices(logFunction, nullptr));
  if (ncclNIbDevs == 0) return ncclInternalError; // Caught in plugin init code, not propagated to user.

  if (ginType == NCCL_GIN_TYPE_GDAKI) goto try_gdaki;
  if (ginType == NCCL_GIN_TYPE_PROXY) goto try_proxy;
  if (ginType != -1) {
    INFO(NCCL_INIT|NCCL_NET, "NET_IB: no support for GIN type %ld", ncclParamGinType());
    return ncclInternalError;
  }

  bool gdrSupport;

  // First try GDAKI
try_gdaki:
  NCCLCHECK(ncclGinIbGdakiInit());
  if (ncclGinIbGdakiNDevs == 0 && ginType == -1) goto try_proxy;
  NCCLCHECK(ncclGinIbGdrSupport(&gdrSupport, /*gdaki*/ true));
  if (!gdrSupport && ginType == -1) goto try_proxy;
  if (!gdrSupport) return ncclInternalError;
  if (ginIb) memcpy(ginIb, &ncclGinIbGdaki, sizeof(ncclGinIb));
  goto end;

  // Then Proxy
try_proxy:
  NCCLCHECK(ncclGinIbGdrSupport(&gdrSupport, /*gdaki*/ false));
  if (!gdrSupport) return ncclInternalError;
  if (ginIb) memcpy(ginIb, &ncclGinIbProxy, sizeof(ncclGinIb));

end:
  ncclNetCommConfig_t* netCommConfig = nullptr;
  NCCLCHECK(ncclCalloc(&netCommConfig, 1));
  netCommConfig->trafficClass = NCCL_NET_TRAFFIC_CLASS_UNDEF;
  *ctx = netCommConfig;
  return ncclSuccess;
}
ncclResult_t ncclGinIbInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  return ncclGinIbInitType(ctx, commId, logFunction, ncclParamGinType(), &ncclGinIb);
}

// GIN Entry point, which will then morph into either the GDAKI or PROXY backend
ncclGin_t ncclGinIb = {
  "GIN_IB",
  ncclGinIbInit,
  NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL
};

ncclResult_t ncclGinIbFinalize(void *ctx) {
  if (ctx) free(ctx);
  return ncclIbFinalizeDevices();
}

static ncclResult_t ncclGinIbAllGather(struct ncclGinIbCollComm *cComm, void *srcBuf, void *recvBuf, size_t len) {
  ncclResult_t status = ncclSuccess;
  void *rMhandle = NULL, *sMhandle = NULL;
  void *srequest = NULL, *rrequest = NULL;
  int speer;
  int rpeer;
  void *rbuf;
  int tag;
  int done;

  NCCLCHECKGOTO(ncclNetIb.regMr(cComm->recvComm, recvBuf,
                                cComm->nranks * len, NCCL_PTR_HOST,
                                &rMhandle),
                status, out);
  NCCLCHECKGOTO(ncclNetIb.regMr(cComm->sendComm, recvBuf,
                                cComm->nranks * len, NCCL_PTR_HOST,
                                &sMhandle),
                status, out);

  speer = cComm->rank;
  memcpy((void *)((uintptr_t)recvBuf + speer * len), srcBuf, len);
  for (int i = 0; i < cComm->nranks - 1; i++) {
    rpeer = (speer - 1 + cComm->nranks) % cComm->nranks;
    while (srequest == NULL || rrequest == NULL) {
      rbuf = (void *)((uintptr_t)recvBuf + rpeer * len);
      tag = NCCL_GIN_IB_ALLGATHER_TAG;
      if (srequest == NULL)
        NCCLCHECKGOTO(ncclNetIb.isend(cComm->sendComm,
                                      (void *)((uintptr_t)recvBuf + speer * len),
                                      len, tag, sMhandle, NULL, &srequest),
                      status, out);
      if (rrequest == NULL)
        NCCLCHECKGOTO(ncclNetIb.irecv(cComm->recvComm, 1, &rbuf, &len,
                                      &tag, &rMhandle, NULL, &rrequest),
                      status, out);
    }
    while (srequest || rrequest) {
      if (rrequest)
        NCCLCHECKGOTO(ncclNetIb.test(rrequest, &done, NULL),
                      status, out);
      if (done)
        rrequest = NULL;
      if (srequest)
        NCCLCHECKGOTO(ncclNetIb.test(srequest, &done, NULL),
                      status, out);
      if (done)
        srequest = NULL;
    }
    speer = rpeer;
  }

out:
  if (rMhandle)
    ncclNetIb.deregMr(cComm->recvComm, rMhandle);

  if (sMhandle)
    ncclNetIb.deregMr(cComm->sendComm, sMhandle);

  return status;
}

static ncclResult_t ncclGinIbAllToAll(struct ncclGinIbCollComm *cComm, void *src_buf, void *recv_buf, size_t len) {
  ncclResult_t status = ncclSuccess;

  void *tmp_buf = nullptr;
  NCCLCHECK(ncclIbMalloc((void **)&tmp_buf, cComm->nranks * cComm->nranks * len));
  NCCLCHECKGOTO(cComm->allGather(cComm, src_buf, tmp_buf, cComm->nranks * len), status, out);

  for (int i = 0; i < cComm->nranks; i++) {
    memcpy((void *)((uintptr_t)recv_buf + i * len), (void *)((uintptr_t)tmp_buf + i * cComm->nranks * len + cComm->rank * len), len);
  }

out:
  if (tmp_buf)
    free(tmp_buf);

  return status;
}

ncclResult_t ncclGinIbP2PBarrier(struct ncclGinIbCollComm *cComm) {
  // TODO: move allocation to init or use zero-byte allgather
  int *dummy;
  NCCLCHECK(ncclIbMalloc((void **)&dummy, cComm->nranks * sizeof(int)));
  NCCLCHECK(ncclGinIbAllGather(cComm, dummy + cComm->rank, dummy, sizeof(int)));
  free(dummy);
  return ncclSuccess;
}

ncclResult_t ncclGinIbConnect(void *ctx, void *handles[], int nranks, int rank,
                              void *listenComm, void **collComm) {
  struct ncclIbListenComm *lComm = (struct ncclIbListenComm *)listenComm;
  struct ncclGinIbCollComm *cCommArray = nullptr;
  int next;

  *collComm = NULL;
  NCCLCHECK(ncclIbMalloc((void **)&cCommArray, sizeof(*cCommArray)));

  struct ncclGinIbCollComm *cComm = cCommArray;
  cComm->ctx = ctx;
  cComm->nranks = nranks;
  cComm->rank = rank;

    // 向 rank+1 发起连接，接受 rank-1 的连接，构成 ring
  next = (cComm->rank + 1) % nranks;
  do
  {
    if (cComm->sendComm == NULL) {
      NCCLCHECK(ncclNetIb.connect(ctx, lComm->dev, handles[next], &cComm->sendComm, NULL));
    }
    if (cComm->recvComm == NULL)
      NCCLCHECK(ncclNetIb.accept(lComm, &cComm->recvComm, NULL));
  } while (cComm->sendComm == NULL || cComm->recvComm == NULL);

  cComm->getProperties = (ncclResult_t(*)(int dev, void *props))ncclIbGetProperties;
  cComm->allGather = ncclGinIbAllGather;
  cComm->allToAll = ncclGinIbAllToAll;
  cComm->getGidIndex = ncclIbGetGidIndex;
  cComm->dev = lComm->dev;

  cComm->ib.context = ncclIbDevs[cComm->dev].context;
  cComm->ib.pd = ncclIbDevs[cComm->dev].pd;

  *collComm = cCommArray;
  return ncclSuccess;
}

ncclResult_t ncclGinIbCloseColl(void* collComm) {
  struct ncclGinIbCollComm* cCommArray = (struct ncclGinIbCollComm*)collComm;
  if (!cCommArray) return ncclSuccess;

  struct ncclGinIbCollComm *cComm = cCommArray;
  if (cComm->recvComm) {
    NCCLCHECK(ncclNetIb.closeRecv(cComm->recvComm));
    cComm->recvComm = NULL;
  }

  if (cComm->sendComm) {
    NCCLCHECK(ncclNetIb.closeSend(cComm->sendComm));
    cComm->sendComm = NULL;
  }

  memset(cComm, 0, sizeof(*cComm));

  free(cCommArray);
  return ncclSuccess;
}

#include "gdaki/gin_host_gdaki.h"

ncclResult_t ncclGinIbGdakiInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  return ncclGinIbInitType(ctx, commId, logFunction, NCCL_GIN_TYPE_GDAKI, NULL);
}

ncclResult_t ncclGinIbGdakiDevices(int* ndev) {
  std::lock_guard<std::mutex> lock(ncclGinIbGdakiLockMutex);
  *ndev = ncclGinIbGdakiNDevs;
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiGetProperties(int dev, ncclNetProperties_t* props) {
  std::lock_guard<std::mutex> lock(ncclGinIbGdakiLockMutex);
  if (dev >= ncclGinIbGdakiNDevs) {
    WARN("NET/IB : Requested properties for GIN GDAKI NIC %d, only %d GIN GDAKI NICs have been created", dev, ncclGinIbGdakiNDevs);
    return ncclInvalidUsage;
  }
  NCCLCHECK(ncclIbGetPhysProperties(ncclGinIbGdakiDevIndexes[dev], props));
  props->netDeviceType = NCCL_NET_DEVICE_GIN_GDAKI;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiListen(void* ctx, int dev, void* opaqueHandle, void** listenComm) {
  std::lock_guard<std::mutex> lock(ncclGinIbGdakiLockMutex);
  return ncclNetIb.listen(ctx, ncclGinIbGdakiDevIndexes[dev], opaqueHandle, listenComm);
}

ncclResult_t ncclGinIbGdakiConnect(void *ctx, void *handles[], int nranks, int rank,
                                   void *listenComm, void **collComm) {
  // Check the current GPU supports GDR
  NCCLCHECK(ncclGinIbGdrGpuSupport(/*gdaki*/ true));

  NCCLCHECK(
    ncclGinIbConnect(ctx, handles, nranks, rank, listenComm, collComm));

  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)*collComm;
  cComm->getProperties = (ncclResult_t(*)(int dev, void *props))ncclGinIbGdakiGetProperties;
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiCreateContext(void* collComm, ncclGinConfig_v13_t* config, void **ginCtx, ncclNetDeviceHandle_t** devHandle) {
  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)collComm;

  NCCLCHECK(ncclGinGdakiCreateContext(cComm, config->nSignals, config->nCounters, config->nContexts, config->queueDepth, config->trafficClass, ginCtx, devHandle));

  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mr_flags, void** mhandle, void **ginHandle) {
  return ncclGinGdakiRegMrSym((struct ncclGinIbCollComm *)collComm, data, size, type, mr_flags, mhandle, ginHandle);
}

ncclResult_t ncclGinIbGdakiDeregMrSym(void* collComm, void* mhandle) {
  return ncclGinGdakiDeregMrSym((struct ncclGinIbCollComm *)collComm, mhandle);
}

ncclResult_t ncclGinIbGdakiDestroyContext(void* ginCtx) {
  return ncclGinGdakiDestroyContext(ginCtx);
}

ncclResult_t ncclGinIbGdakiProgress(void *collComm)
{
  return ncclGinGdakiProgress(collComm);
}

ncclResult_t ncclGinIbGdakiQueryLastError(void *ginCtx, bool *hasError) {
  return ncclGinGdakiQueryLastError(ginCtx, hasError);
}

ncclGin_t ncclGinIbGdaki = {
  "GIN_IB_GDAKI",
  ncclGinIbGdakiInit,
  ncclGinIbGdakiDevices,
  ncclGinIbGdakiGetProperties,
  ncclGinIbGdakiListen,
  ncclGinIbGdakiConnect,
  ncclGinIbGdakiCreateContext,
  ncclGinIbGdakiRegMrSym,
  NULL, // regMrSymDmaBuf
  ncclGinIbGdakiDeregMrSym,
  ncclGinIbGdakiDestroyContext,
  ncclGinIbCloseColl,
  ncclIbCloseListen,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  ncclGinIbGdakiProgress,
  ncclGinIbGdakiQueryLastError,
  ncclGinIbFinalize
};


struct ncclIbGinProxyMrHandle {
  struct ncclIbMrHandle *mrHandle;
  uintptr_t *base_vas;
  uint32_t *rkeys;
};

ncclResult_t ncclGinIbProxyInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  return ncclGinIbInitType(ctx, commId, logFunction, NCCL_GIN_TYPE_PROXY, NULL);
}

ncclResult_t ncclGinIbProxyGetProperties(int dev, ncclNetProperties_t* props) {
  NCCLCHECK(ncclNetIb.getProperties(dev, props));
  props->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  return ncclSuccess;
}

// =========================================================================
// GIN Proxy Init 阶段 —— 第一步：建立控制面 Ring 拓扑
// =========================================================================
// 调用共享的 ncclGinIbConnect 建立单向 Ring：
//   rank → (rank+1)%nranks 为 sendComm
//   (rank-1+nranks)%nranks → rank 为 recvComm
// 这个 Ring 仅用于 init 阶段的控制面集合操作（allGather/allToAll），
// 例如：交换 listen handle、同步 MR 的 base_va 和 rkey 等。
// 产出物：ncclGinIbCollComm，包含 ring 上的 send/recv comm + allGather/allToAll 函数指针。
//
// 注意：这里传 gdaki=false，proxy 模式不使用 DOCA GPUNetIO，
// 只需标准的 peer_mem 或 DMA-BUF 来实现 GDR（GPU Direct RDMA）。
ncclResult_t ncclGinIbProxyConnect(void *ctx, void *handles[], int nranks, int rank,
                                   void *listenComm, void **collComm) {
  // Check the current GPU supports GDR
  NCCLCHECK(ncclGinIbGdrGpuSupport(/*gdaki*/ false));

  // Connect.
  NCCLCHECK(
    ncclGinIbConnect(ctx, handles, nranks, rank, listenComm, collComm));

  return ncclSuccess;
}

// IB 后端 context —— Proxy 模式的最底层数据面连接状态
// 与 gin_host_proxy.cc 中的 ginProxyCtx（中间层 proxy context）不同：
//   - ginProxyCtx：存储 GFD 队列、计数器、信号等 GPU↔CPU 通信资源
//   - ncclGinIbProxyCtx：存储 full-mesh IB QP 连接，供 iput/iget/iflush 使用
//
// 内存布局：分配为 ncclGinIbProxyCtx[nContexts] 数组，
//   ginProxyCtx[0].nContexts 记录总数（其余元素不重复存储）。
// runtime 时通过 ginCtx[context] 索引到对应 context 的连接集。
struct ncclGinIbProxyCtx {
  void**        fullRecvComm;  // [nranks] 到每个 rank 的接收 comm（用于 iflush 的 RDMA Read）
  void**        fullSendComm;  // [nranks] 到每个 rank 的发送 comm（用于 iput/iget 的 RDMA Write/Read）
  int rank, nranks;            // 本 rank 编号和总 rank 数
  int nContexts;               // context 总数（仅 [0] 有效）
};

// =========================================================================
// GIN Proxy Init 阶段 —— 第二步：建立数据面 Full-Mesh 拓扑
// =========================================================================
// 为每个 context 创建到所有 rank 的 IB send + recv 连接对（全连接网络）。
// 这些连接是 runtime 阶段 CPU proxy 线程发起 RDMA 操作的实际通路：
//   - fullSendComm[rank] → 用于 iput（RDMA Write）和 iget（RDMA Read）
//   - fullRecvComm[rank] → 用于 iflush（需要 recvComm 上的 gpuFlush QP）
//
// 连接建立策略（避免 N 个 rank 同时 connect 同一 peer 导致死锁）：
//   迭代 i = 0..nranks-1：
//     - rank 向 (rank+i)%nranks 发起 connect（建立 sendComm）
//     - 同时 accept 来自 (rank-i)%nranks 的连接（建立 recvComm）
//     - 每对连接建立后做一次 P2P Barrier 同步
//   这保证在每轮 i 中，每个 rank 恰好发起一个 connect + accept 一个连接，
//   不会出现多个 rank 同时等待同一 peer accept 的死锁。
//
// 注意：proxy 模式不支持自定义 queueDepth（GFD 队列深度由上层 gin_host_proxy 管理）。
// devHandle 输出为 NULL —— proxy 模式不提供 GPU device handle，
// GPU 侧资源由上层 ncclGinProxyInit（gin_host_proxy.cc）单独分配。
ncclResult_t ncclGinIbProxyCreateContext(void* collComm, ncclGinConfig_v13_t* config, void** ginCtx, ncclNetDeviceHandle_v11_t** devHandle) {
  ncclResult_t ret = ncclSuccess;
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;
  // Make sure all QP we create use the provided traffic class.
  ncclIbSetTrafficClass(cComm->ctx, config->trafficClass);

  if (config->queueDepth != 0) {
    WARN("GIN_IB_PROXY does not support specifying qp depth");
    return ncclInvalidUsage;
  }

  // --- Step 1: 分配 ncclGinIbProxyCtx[nContexts] 数组 ---
  int nranks;
  struct ncclGinIbProxyCtx* ginProxyCtx = NULL;
  *ginCtx = NULL;
  NCCLCHECK(ncclCalloc(&ginProxyCtx, config->nContexts));
  ginProxyCtx[0].nContexts = config->nContexts;
  ginProxyCtx[0].nranks = nranks = cComm->nranks;

  // --- Step 2: 创建新的 listen 端口，通过控制面 ring 的 allGather 交换所有 rank 的 handle ---
  void *lComm = NULL;
  char* handle = NULL, *handles = NULL;
  NCCLCHECKGOTO(ncclIbMalloc((void**)&handles, NCCL_NET_HANDLE_MAXSIZE*cComm->nranks), ret, end);
  handle = handles + NCCL_NET_HANDLE_MAXSIZE*cComm->rank;

  NCCLCHECKGOTO(ncclNetIb.listen(cComm->ctx, cComm->dev, handle, &lComm), ret, end);
  NCCLCHECKGOTO(cComm->allGather(cComm, handle, handles, NCCL_NET_HANDLE_MAXSIZE), ret, end);

  // --- Step 3: 为每个 context 建立 full-mesh 连接 ---
  for (int c=0; c<config->nContexts; c++) {
    struct ncclGinIbProxyCtx* gc = ginProxyCtx+c;
    NCCLCHECKGOTO(ncclIbMalloc((void**)&gc->fullSendComm, sizeof(void *) * nranks), ret, end);
    NCCLCHECKGOTO(ncclIbMalloc((void**)&gc->fullRecvComm, sizeof(void *) * nranks), ret, end);
    gc->rank = cComm->rank;

    // Full-mesh 全连接：旋转偏移策略避免死锁
    for (int i = 0; i < nranks; i++) {
      int connectPeer = (cComm->rank + i) % nranks;          // 本轮主动连接的目标 rank
      int acceptPeer = (cComm->rank - i + nranks) % nranks;  // 本轮被动接受的来源 rank
      do {
        if (gc->fullSendComm[connectPeer] == NULL)
          NCCLCHECKGOTO(ncclNetIb.connect(cComm->ctx, cComm->dev, handles+NCCL_NET_HANDLE_MAXSIZE*connectPeer, &gc->fullSendComm[connectPeer], NULL), ret, end);
        if (gc->fullRecvComm[acceptPeer] == NULL)
          NCCLCHECKGOTO(ncclNetIb.accept(lComm, &gc->fullRecvComm[acceptPeer], NULL), ret, end);
      } while ((gc->fullSendComm[connectPeer] == NULL) ||
          (gc->fullRecvComm[acceptPeer] == NULL));
      NCCLCHECKGOTO(ncclGinIbP2PBarrier(cComm), ret, end);   // 确保所有 rank 本轮连接完成后再进入下一轮
    }
  }

end:
  free(handles);
  if (lComm) ncclNetIb.closeListen(lComm);  // listen 端口仅用于 init，建完连接即关闭
  if (ret != ncclSuccess) free(ginProxyCtx);
  else *ginCtx = ginProxyCtx;
  return ret;
}

ncclResult_t ncclGinIbProxyDestroyContext(void* ginCtx) {
  struct ncclGinIbProxyCtx* gc = (struct ncclGinIbProxyCtx*)ginCtx;
  int nContexts = gc[0].nContexts;
  int nranks = gc[0].nranks;
  for (int c=0; c<nContexts; c++) {
    if (gc[c].fullRecvComm) {
      for (int i=0; i<nranks; i++) {
        NCCLCHECK(ncclNetIb.closeRecv(gc[c].fullRecvComm[i]));
      }
      free(gc[c].fullRecvComm);
      gc[c].fullRecvComm = NULL;
    }

    if (gc[c].fullSendComm) {
      for (int i=0; i<nranks; i++) {
        NCCLCHECK(ncclNetIb.closeSend(gc[c].fullSendComm[i]));
      }
      free(gc[c].fullSendComm);
      gc[c].fullSendComm = NULL;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, uint64_t mr_flags, void** mhandle, void **ginHandle) {
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;
  struct ncclIbGinProxyMrHandle *ginMrHandle;
  NCCLCHECK(ncclCalloc(&ginMrHandle, 1));

  NCCLCHECKNOWARN(ncclIbRegMrDmaBufInternal(cComm->recvComm, data, size, type, offset, fd, mr_flags, (void **)&ginMrHandle->mrHandle), NCCL_NET);

  NCCLCHECK(ncclCalloc(&ginMrHandle->base_vas, cComm->nranks));
  NCCLCHECK(ncclCalloc(&ginMrHandle->rkeys, cComm->nranks));

  NCCLCHECK(cComm->allGather(cComm, &data, ginMrHandle->base_vas, sizeof(uintptr_t)));
  NCCLCHECK(cComm->allGather(cComm, &ginMrHandle->mrHandle->mrs[0]->rkey, ginMrHandle->rkeys, sizeof(uint32_t)));

  *mhandle = ginMrHandle;
  *ginHandle = ginMrHandle;

  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mr_flags, void** mhandle, void **ginHandle) {
  return ncclGinIbProxyRegMrSymDmaBuf(collComm, data, size, type, 0, -1, mr_flags, mhandle, ginHandle);
}

ncclResult_t ncclGinIbProxyDeregMrSym(void* collComm, void* mhandle) {
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;
  struct ncclIbGinProxyMrHandle *ginMrHandle = (struct ncclIbGinProxyMrHandle *)mhandle;

  NCCLCHECK(ncclNetIb.deregMr(cComm->recvComm, ginMrHandle->mrHandle));
  free(ginMrHandle->base_vas);
  free(ginMrHandle->rkeys);
  free(ginMrHandle);
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyCloseColl(void* collComm) {
  free(collComm);
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyIPut(void *ginCtx, int context, uint64_t srcOff, void *srcMhandle, size_t size,
                                uint64_t dstOff, void *dstMhandle, uint32_t rank,
                                void **request) {
  struct ncclGinIbProxyCtx* ginProxyCtx = &((struct ncclGinIbProxyCtx*)ginCtx)[context];

  struct ncclIbGinProxyMrHandle *srcMrHandle = (struct ncclIbGinProxyMrHandle *)srcMhandle;
  struct ncclIbGinProxyMrHandle *dstMrHandle = (struct ncclIbGinProxyMrHandle *)dstMhandle;

  void *srcPtr = (void *)(srcMrHandle->base_vas[ginProxyCtx->rank] + srcOff);
  void *dstPtr = (void *)(dstMrHandle->base_vas[rank] + dstOff);
  uint32_t lkey = srcMrHandle->mrHandle->mrs[0]->lkey;
  uint32_t rkey = dstMrHandle->rkeys[rank];

  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)ginProxyCtx->fullSendComm[rank];
  struct ncclIbQp *qp = &comm->base.qps[0];

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->ginProxyCtx = ginProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IPUT;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));

  wr.opcode                  = IBV_WR_RDMA_WRITE;
  wr.send_flags              = IBV_SEND_SIGNALED;
  wr.wr_id                   = req - comm->base.reqs;
  wr.next                    = NULL;
  wr.wr.rdma.remote_addr     = (uint64_t)dstPtr;
  wr.wr.rdma.rkey            = rkey;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  sge.addr = (uintptr_t)srcPtr;  // Local buffer address
  sge.length = size;  // Size of the transfer
  sge.lkey = lkey;  // Local key

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(qp->qp, &wr, &bad_wr));
  ncclIbAddEvent(req, qp->devIndex);

  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyIGet(void *ginCtx, int context, uint64_t remoteOffset, void *remoteMhandle,
                                 size_t size, uint64_t localOffset, void *localMhandle, uint32_t rank,
                                 void **request) {
  struct ncclGinIbProxyCtx* ginProxyCtx = &((struct ncclGinIbProxyCtx*)ginCtx)[context];

  struct ncclIbGinProxyMrHandle *remoteMrHandle = (struct ncclIbGinProxyMrHandle *)remoteMhandle;
  struct ncclIbGinProxyMrHandle *localMrHandle = (struct ncclIbGinProxyMrHandle *)localMhandle;

  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)ginProxyCtx->fullSendComm[rank];
  struct ncclIbQp *qp = &comm->base.qps[0];

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->ginProxyCtx = ginProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IGET;
  req->sock = &comm->base.sock;
  req->iget.rank = rank;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  void *remotePtr = (void *)(remoteMrHandle->base_vas[rank] + remoteOffset);
  void *localPtr = (void *)(localMrHandle->base_vas[ginProxyCtx->rank] + localOffset);
  uint32_t rkey = remoteMrHandle->rkeys[rank];
  uint32_t lkey = localMrHandle->mrHandle->mrs[0]->lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));

  wr.opcode                  = IBV_WR_RDMA_READ;
  wr.send_flags              = IBV_SEND_SIGNALED; // TODO: Potentially optimize this?
  wr.wr_id                   = req - comm->base.reqs;
  wr.next                    = NULL;
  wr.wr.rdma.remote_addr     = (uint64_t)remotePtr;
  wr.wr.rdma.rkey            = rkey;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  sge.addr = (uintptr_t)localPtr;
  sge.length = size;
  sge.lkey = lkey;

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(qp->qp, &wr, &bad_wr));
  ncclIbAddEvent(req, qp->devIndex);

  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyIPutSignal(void *ginCtx, int context, uint64_t srcOff, void *srcMhandle,
                                      size_t size, uint64_t dstOff, void *dstMhandle, uint32_t rank,
                                      uint64_t signalOff, void *signalMhandle, uint64_t signalValue,
                                      uint32_t signalOp, void **request) {
  if (signalOp != NCCL_NET_SIGNAL_OP_INC && signalOp != NCCL_NET_SIGNAL_OP_ADD) {
    WARN("ncclGinIbProxyIPutSignal: Unsupported signalOp %u", signalOp);
    return ncclInvalidArgument;
  }

  struct ncclGinIbProxyCtx* ginProxyCtx = &((struct ncclGinIbProxyCtx*)ginCtx)[context];

  struct ncclIbGinProxyMrHandle *srcMrHandle = (struct ncclIbGinProxyMrHandle *)srcMhandle;
  struct ncclIbGinProxyMrHandle *dstMrHandle = (struct ncclIbGinProxyMrHandle *)dstMhandle;
  struct ncclIbGinProxyMrHandle *signalMrHandle = (struct ncclIbGinProxyMrHandle *)signalMhandle;

  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)ginProxyCtx->fullSendComm[rank];
  struct ncclIbQp *qp = &comm->base.qps[0];
  int devIndex = qp->devIndex;

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->ginProxyCtx = ginProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IPUT;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  struct ibv_send_wr wr[2];
  memset(&wr, 0, sizeof(wr));
  struct ibv_sge sge[2];
  memset(&sge, 0, sizeof(sge));

  // If size is 0, we only need to send the signal. srcMrHandle must be non-NULL
  if (size > 0 && dstMrHandle) {
    void *srcPtr = (void *)(srcMrHandle->base_vas[ginProxyCtx->rank] + srcOff);
    void *dstPtr = (void *)(dstMrHandle->base_vas[rank] + dstOff);
    uint32_t lkey = srcMrHandle->mrHandle->mrs[0]->lkey;
    uint32_t rkey = dstMrHandle->rkeys[rank];

    // PUT
    wr[0].opcode                  = IBV_WR_RDMA_WRITE;
    wr[0].send_flags              = 0; // We only need the CQE from the signal
    wr[0].wr_id                   = req - comm->base.reqs;
    wr[0].next                    = &wr[1];
    wr[0].wr.rdma.remote_addr     = (uint64_t)dstPtr;
    wr[0].wr.rdma.rkey            = rkey;
    wr[0].sg_list = &sge[0];
    wr[0].num_sge = 1;

    sge[0].addr = (uintptr_t)srcPtr;  // Local buffer address
    sge[0].length = size;  // Size of the transfer
    sge[0].lkey = lkey;  // Local key
  }

  void *signalPtr = (void *)(signalMrHandle->base_vas[rank] + signalOff);
  uint32_t signalRkey = signalMrHandle->rkeys[rank];

  // SIGNAL
  wr[1].opcode                  = IBV_WR_ATOMIC_FETCH_AND_ADD;
  wr[1].send_flags              = IBV_SEND_SIGNALED;
  wr[1].wr_id                   = req - comm->base.reqs;  // used for matching completions with request
  wr[1].next                    = NULL;
  wr[1].wr.atomic.remote_addr   = (uint64_t)signalPtr;
  wr[1].wr.atomic.compare_add   = signalOp == NCCL_NET_SIGNAL_OP_INC ? 1 : signalValue;
  wr[1].wr.atomic.rkey          = signalRkey;
  wr[1].sg_list = &sge[1];
  wr[1].num_sge = 1;

  sge[1].addr = (uintptr_t)&comm->putSignalScratchpad;
  sge[1].length = sizeof(comm->putSignalScratchpad);
  sge[1].lkey = comm->devs[devIndex].putSignalScratchpadMr->lkey;

  // Send the put and the signal in one go
  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(qp->qp, size > 0 ? &wr[0] : &wr[1], &bad_wr));
  ncclIbAddEvent(req, qp->devIndex);
  *request = req;
  return ncclSuccess;
}

// =============================================================================
// ncclGinIbProxyTest — 检测一个 IB 异步操作是否完成
// =============================================================================
// 调用链：proxyGinPollCompletions → ginBackend->test → 此函数
//
// 核心机制：
//   提交 WQE 时（如 iput/iputSignal/iget/iflush），若设置了 IBV_SEND_SIGNALED，
//   NIC 在该 WQE 执行完成后会向本地 Send CQ 写入一个 CQE（Completion Queue Entry）。
//   本函数通过 ibv_poll_cq 从 CQ 中取出 CQE，以此判断对应 WQE 是否完成。
//
// events 计数器原理：
//   req->events[devIdx] 是"该 request 在 dev devIdx 上还有多少个未完成的 CQE"的计数。
//   每提交一个 IBV_SEND_SIGNALED 的 WQE 时，ncclIbAddEvent(req, devIdx) 会 events[devIdx]++。
//   每收到一个 CQE 时，wcReq->events[devIdx]--。
//   当 events[devIdx] 递减到 0，说明所有 WQE 都完成了 → done=1。
//
// 与 GDAKI 路径的对比：
//   GDAKI 中 GPU 直接通过 poll_cq_at 在 GPU 端轮询 CQ，不经过 CPU；
//   Proxy 路径中 CPU 通过 ibv_poll_cq 从 host 侧轮询 CQ，然后修改 state->done 告知上层。
// =============================================================================
ncclResult_t ncclGinIbProxyTest(void* collComm, void *request, int *done) {
  struct ncclIbRequest* req = (struct ncclIbRequest*)request;
  struct ncclGinIbProxyCtx* ginProxyCtx = (struct ncclGinIbProxyCtx*)req->ginProxyCtx;
  int rank = req->iput.rank;  // 本次 IB 操作的目标 peer rank
  *done = 0;  // 默认未完成

  // === 快速路径：events[0] 已经为 0，说明之前的 poll 已经收到了所有 CQE ===
  // 这种情况发生在：同一个 CQ 上有多个 request 的 WQE 交错完成，
  // 上一次调用 test(reqA) 时 poll_cq 顺带收到了 reqB 的 CQE 并递减了 reqB->events[0]。
  // 所以本次调用 test(reqB) 时，events[0] 已经为 0，无需再 poll。
  if (req->events[0] == 0) {
    *done = 1;
    NCCLCHECK(ncclIbFreeRequest(req));  // 回收 request 到空闲池
    return ncclSuccess;
  }
  int wrDone = 0;       // ibv_poll_cq 返回的本次收到的 CQE 数量
  struct ibv_wc wc[4];  // Work Completion 数组，一次最多取 4 个 CQE
  // ibv_wc 结构体关键字段：
  //   wc.status  — 完成状态（IBV_WC_SUCCESS=0 表示成功）
  //   wc.wr_id   — 提交 WQE 时设置的用户标识，这里存的是 req 在 reqs[] 数组中的索引
  //   wc.opcode  — 完成的操作类型（RDMA_WRITE / ATOMIC 等）

  // === 根据操作类型选择正确的 CQ ===
  // 每个 peer 有独立的 QP → 独立的 Send CQ（GIN full-mesh QP 架构）。
  // iflush 走 RecvComm 的 QP/CQ（因为 flush 本质是对本地注册内存的 RDMA Read），
  // 其他操作（iput/iputSignal/iget）走 SendComm 的 QP/CQ。
  ncclIbNetCommBase* commBase;
  ncclIbNetCommDevBase* devBase;
  if (req->type == NCCL_NET_IB_REQ_FLUSH) {
    struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)ginProxyCtx->fullRecvComm[rank];
    commBase = &comm->base;
    devBase = &comm->devs[0].base;  // devBase->cq 就是 Send CQ
  } else {
    struct ncclIbSendComm* comm = (struct ncclIbSendComm*)ginProxyCtx->fullSendComm[rank];
    commBase = &comm->base;
    devBase = &comm->devs[0].base;
  }

  // === 核心：ibv_poll_cq 从 Send CQ 中非阻塞地取出 CQE ===
  // 语义：从 devBase->cq 中最多取 4 个 CQE，填入 wc[]，实际取到的个数写入 wrDone。
  // 非阻塞：如果 CQ 为空（没有完成的 WQE），wrDone=0 立即返回。
  // CQE 是"破坏性读取"：取出后从 CQ 中删除，不能重复取。
  // 注意：CQ 是 per-QP 的，但一个 CQ 可以绑定多个 QP。
  //       这里每个 peer 有独立的 QP 和 CQ，所以 poll 只会取到发往该 peer 的 WQE 的 CQE。
  NCCLCHECK(wrap_ibv_poll_cq(devBase->cq, 4, wc, &wrDone));

  // === 遍历取到的每个 CQE ===
  for (int i = 0; i < wrDone; i++) {
    // --- 错误检查：CQE status 非 SUCCESS 表示 IB 操作失败 ---
    // 常见错误：IBV_WC_REM_ACCESS_ERR（远端内存权限错误）、IBV_WC_RETRY_EXC_ERR（重试超时）
    if (wc[i].status != IBV_WC_SUCCESS) {
      union ncclSocketAddress addr;
      ncclSocketGetAddr(req->sock, &addr);
      char localGidString[INET6_ADDRSTRLEN] = "";
      char remoteGidString[INET6_ADDRSTRLEN] = "";
      const char* localGidStr = NULL, *remoteGidStr = NULL;
      if (req->devBases[i]->gidInfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
        localGidStr = ibvGetGidStr(&devBase->gidInfo.localGid, localGidString, sizeof(localGidString));
        remoteGidStr = ibvGetGidStr(&commBase->remDevs[i].remoteGid, remoteGidString, sizeof(remoteGidString));
      }

      char line[SOCKET_NAME_MAXLEN+1];
      char *hcaName = devBase->pd->context->device->name;
      WARN("NET/IB/GIN: Got completion from peer %s with status=%d opcode=%d len=%u vendor err %u (%s)%s%s%s%s hca %s",
          ncclSocketToString(&addr, line), wc[i].status, wc[i].opcode, wc[i].byte_len, wc[i].vendor_err, ncclIbReqTypeStr[req->type],
          localGidStr ?  " localGid ":"", localGidString, remoteGidStr ? " remoteGids":"", remoteGidString, hcaName);
      return ncclRemoteError;
    }

    // --- 通过 wc.wr_id 反查该 CQE 对应的 request ---
    // 提交 WQE 时设置 wr.wr_id = req - comm->base.reqs（即 request 在 reqs[] 池中的索引）。
    // NIC 完成后将 wr_id 原样写入 CQE，这里用 reqs + wr_id 反向定位回 request。
    // 关键：poll_cq 取到的 CQE 可能不属于当前正在 test 的 req！
    //   因为同一 CQ 上可能有多个 request 的 WQE 交错提交并完成。
    //   例如 test(req3) 时可能顺带取到 req2 和 req4 的 CQE。
    struct ncclIbRequest* wcReq = commBase->reqs + wc[i].wr_id;

    // --- 递减该 request 的未完成 CQE 计数 ---
    // 场景举例（iputSignal）：
    //   提交时：wr[0]=RDMA_WRITE(unsignaled, events 不增) + wr[1]=ATOMIC_FAA(signaled, events++)。
    //   因此 events[0]=1，只需 1 个 CQE（来自 wr[1]）就完成。
    //
    // 场景举例（iput）：
    //   提交时：wr=RDMA_WRITE(signaled, events++)。events[0]=1，1 个 CQE 即完成。
    wcReq->events[0]--;

    // --- 判断当前正在 test 的 request 是否完成 ---
    // 条件 1：wcReq == req — 这个 CQE 确实属于我们正在查询的 request
    // 条件 2：events[0] == 0 — 该 request 所有 signaled WQE 的 CQE 都已收到
    // 两个条件同时满足 → done=1，回收 request。
    // 注意：即使 wcReq != req，也要递减 wcReq->events[0]（为其他 request 的 test 提前消费 CQE）。
    if (wcReq == req && wcReq->events[0] == 0) {
      *done = 1;
      NCCLCHECK(ncclIbFreeRequest(wcReq));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyIFlush(void *ginCtx, int context, void* mhandle, uint32_t rank, void **request) {
  struct ncclGinIbProxyCtx* ginProxyCtx = &((struct ncclGinIbProxyCtx*)ginCtx)[context];
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)ginProxyCtx->fullRecvComm[rank];
  struct ncclIbGinProxyMrHandle *ginMrHandle = (struct ncclIbGinProxyMrHandle *)mhandle;
  struct ncclIbQp *qp = &comm->devs[0].gpuFlush.qp;

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->type = NCCL_NET_IB_REQ_FLUSH;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  req->ginProxyCtx = ginProxyCtx;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = req - comm->base.reqs;

  void *flushPtr = (void *)(ginMrHandle->base_vas[rank]);
  wr.wr.rdma.remote_addr = (uint64_t)flushPtr;
  wr.wr.rdma.rkey = ginMrHandle->rkeys[rank];
  wr.sg_list = &comm->devs[qp->devIndex].gpuFlush.sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;

  TRACE(NCCL_NET, "NET/IB: %s: Posting a flush request (req=%p, comm=%p, wr_id=%ld)", __func__, req, req->base, wr.wr_id);
  TIME_START(4);
  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(qp->qp, &wr, &bad_wr));
  TIME_STOP(4);

  ncclIbAddEvent(req, qp->devIndex);

  TRACE(NCCL_NET, "NET/IB: %s: Flush request posted (req=%p, comm=%p, wr_id=%ld)", __func__, req, req->base, wr.wr_id);

  *request = req;
  return ncclSuccess;
}

// No support for NCCL_IB_SPLIT_DATA_ON_QPS or NCCL_IB_MERGE_NICS
ncclGin_t ncclGinIbProxy = {
  "GIN_IB_PROXY",
  ncclGinIbProxyInit,
  ncclIbDevices,
  ncclGinIbProxyGetProperties,
  ncclIbListen,
  ncclGinIbProxyConnect,
  ncclGinIbProxyCreateContext,
  ncclGinIbProxyRegMrSym,
  ncclGinIbProxyRegMrSymDmaBuf,
  ncclGinIbProxyDeregMrSym,
  ncclGinIbProxyDestroyContext,
  ncclGinIbCloseColl,
  ncclIbCloseListen,
  ncclGinIbProxyIPut,
  ncclGinIbProxyIPutSignal,
  ncclGinIbProxyIGet,
  ncclGinIbProxyIFlush,
  ncclGinIbProxyTest,
  NULL,
  NULL,
  ncclGinIbFinalize
};
