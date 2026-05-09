/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <alloc.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <mutex>

#include "ibvwrap.h"
#include "mlx5/mlx5dvwrap.h"
#include "gin/gin_host.h"
#include "gin_host_gdaki.h"
#include "plugin/nccl_net.h"
#include "param.h"

#include "doca_gpunetio_host.h"
#include "nccl_device/gin/gdaki/gin_gdaki_device_host_common.h"
#include "../gin.h"

#define DOCACHECK(call)                 \
  do {                                  \
    doca_error_t err = call;            \
    if (err != DOCA_SUCCESS) {          \
      /* Print the back trace*/         \
      WARN("DOCA failure %d", err);     \
      return ncclSystemError;           \
    }                                   \
  } while (0)

#define DOCACHECKGOTO(call, RES, label) \
  do {                                  \
    doca_error_t err = call;            \
    if (err != DOCA_SUCCESS) {          \
      /* Print the back trace*/         \
      WARN("DOCA failure %d", err);     \
      RES = ncclSystemError;            \
      goto label;                       \
    }                                   \
  } while (0)

#define VERBS_TEST_DBR_SIZE (8)
#define MAX_PCI_ADDRESS_LEN 32U

NCCL_PARAM(GinGdakiNicHandler, "GIN_GDAKI_NIC_HANDLER", 0);
NCCL_PARAM(GinGdakiQpDepth, "GIN_GDAKI_QP_DEPTH", 128);
NCCL_PARAM(GinGdakiMaxDestRdAtomic, "GIN_GDAKI_MAX_DEST_RD_ATOMIC", -2);
NCCL_PARAM(GinGdakiMaxQpRdAtomic, "GIN_GDAKI_MAX_QP_RD_ATOMIC", -2);
NCCL_PARAM(GinErrorQuerySec, "GIN_ERROR_QUERY_SEC", 10);
extern int64_t ncclParamIbTimeout();
extern int64_t ncclParamIbRetryCnt();
extern int64_t ncclParamIbPkey();
extern int64_t ncclParamIbSl();
extern int64_t ncclParamIbTc();
extern int64_t ncclParamIbPciRelaxedOrdering();
extern int64_t ncclParamIbDataDirect();
extern int64_t ncclParamDmaBufEnable();

static const int NCCL_IB_SL_DEFAULT = 0;
static const int NCCL_IB_TC_DEFAULT = 0;

static inline bool gdakiRelaxedOrderingEnabled() {
  static bool hasCheckedRelaxedOrdering = false;
  static bool relaxedOrderingEnabled = false;

  static std::mutex lockMutex;
  std::lock_guard<std::mutex> lock(lockMutex);

  if (!hasCheckedRelaxedOrdering) {
    int roMode = ncclParamIbPciRelaxedOrdering();
    ncclResult_t r = ncclInternalError;
    if (roMode == 1 || roMode == 2) {
      // Query IBVERBS_1.8 API - needed for IBV_ACCESS_RELAXED_ORDERING support
      r = wrap_ibv_reg_mr_iova2(NULL, NULL, NULL, 0, 0, 0);
    }

    relaxedOrderingEnabled = (r != ncclInternalError);
    hasCheckedRelaxedOrdering = true;
  }
  return relaxedOrderingEnabled;
}

static ncclResult_t gdakiRegMrDmaBuf(struct ibv_mr **mr, struct ibv_pd *pd, void *addr,
                                     size_t length, int access) {
  int status = 0;
  int dmabuf_fd = -1;

  if (ncclParamDmaBufEnable() == 0) return ncclInvalidUsage;

#if CUDA_VERSION >= 11070
  static size_t host_page_size = sysconf(_SC_PAGESIZE);
  size_t aligned_size = length;
  ALIGN_SIZE(aligned_size, host_page_size);

#if CUDA_VERSION >= 12080
  if (ncclParamIbDataDirect()) {
    status = pfn_cuMemGetHandleForAddressRange((void *)&dmabuf_fd, (CUdeviceptr)addr, aligned_size,
                                               CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                                               CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE);
    if (status) {
      INFO(NCCL_NET,
           "Failed to get DMA-BUF handle for address range with type PCIE, error=%d. Trying a "
           "different method.",
           status);
      goto try_legacy;
    }
    status = wrap_mlx5dv_reg_dmabuf_mr(mr, pd, 0, aligned_size, 0, dmabuf_fd, access,
                                       MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT);
    if (status) {
      INFO(NCCL_NET,
           "Failed to register memory with DMA-BUF and data direct, error=%d. Trying a different "
           "method.",
           status);
      close(dmabuf_fd);
      dmabuf_fd = -1;
    } else
      goto out;
  }
try_legacy:

#endif

  CUCHECK(cuMemGetHandleForAddressRange((void *)&dmabuf_fd, (CUdeviceptr)addr, aligned_size,
                                        CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));
  status = wrap_ibv_reg_dmabuf_mr(mr, pd, 0, aligned_size, 0, dmabuf_fd, access);
  if (status)
    INFO(NCCL_NET, "Failed to register memory with DMA-BUF, error=%d. Trying a different method.",
         status);
#else
  status = ncclInvalidUsage;
#endif

#if CUDA_VERSION >= 12080
out:
#endif
  if (dmabuf_fd >= 0) {
    close(dmabuf_fd);
  }
  return (ncclResult_t)status;
}

static ncclResult_t gdakiRegMr(struct ibv_mr **mr, struct ibv_pd *pd, void *addr, size_t length,
                               int access, bool force_strict_ordering = false) {
  int status = 0;

  if (!force_strict_ordering && gdakiRelaxedOrderingEnabled())
    access |= IBV_ACCESS_RELAXED_ORDERING;

  NOWARN(status = gdakiRegMrDmaBuf(mr, pd, addr, length, access), NCCL_NET);
  if (status == ncclSuccess) return ncclSuccess;

  NCCLCHECK(wrap_ibv_reg_mr_iova2(mr, pd, addr, length, 0, access));
  return ncclSuccess;
}

template <typename T>
class GdakiHostGPUMemHandle {
 private:
  CUmemGenericAllocationHandle cumemhandle;
  unsigned int num_elements;

 public:
  T *host_buf;
  T *gpu_buf;

  ncclResult_t allocate(unsigned int num_elements) {
    this->host_buf = (T *)calloc(num_elements, sizeof(T));
    EQCHECK(this->host_buf, nullptr);

    NCCLCHECK(ncclCuMemAlloc((void **)&this->gpu_buf, &this->cumemhandle, CU_MEM_HANDLE_TYPE_NONE,
                             num_elements * sizeof(T), nullptr));

    this->num_elements = num_elements;

    return ncclSuccess;
  }

  ncclResult_t deallocate() {
    if (this->host_buf != nullptr) {
      free(this->host_buf);
      this->host_buf = nullptr;
    }
    if (this->gpu_buf != nullptr) {
      NCCLCHECK(ncclCuMemFree(this->gpu_buf, nullptr));
      this->gpu_buf = nullptr;
    }
    return ncclSuccess;
  }

  ncclResult_t copy_h_to_d() {
    NCCLCHECK(ncclCudaMemcpy<T>(this->gpu_buf, this->host_buf, this->num_elements));
    return ncclSuccess;
  }

  ncclResult_t copy_d_to_h() {
    NCCLCHECK(ncclCudaMemcpy<T>(this->host_buf, this->gpu_buf, this->num_elements));
    return ncclSuccess;
  }

  GdakiHostGPUMemHandle() : cumemhandle(0), num_elements(0), host_buf(nullptr), gpu_buf(nullptr){};

  ~GdakiHostGPUMemHandle() {
     // Should only be used in error cleanup path as it ignores return code
     this->deallocate();
  }
};

template <typename T>
class GdakiGlobalGPUBufferTable {
 private:
  CUmemGenericAllocationHandle cumemhandle;
  unsigned int num_elements;
  unsigned int next_unused_idx;
  unsigned int num_ranks;
  GdakiHostGPUMemHandle<__be32> rkeys_hd_mhandle;

 public:
  T *gpu_ptr;
  struct ibv_mr *mr;  // 对应的 IB MR（lkey 用于 Atomic 的 local buffer）

  ncclResult_t allocate(unsigned int num_elements, unsigned int num_ranks) {
    this->num_elements = num_elements;
    this->num_ranks = num_ranks;
    this->next_unused_idx = 0;
    if (num_elements == 0) return ncclSuccess;

    NCCLCHECK(ncclCuMemAlloc((void **)&this->gpu_ptr, &this->cumemhandle, CU_MEM_HANDLE_TYPE_NONE,
                             num_elements * sizeof(T), nullptr));
    CUDACHECK(cudaMemset(this->gpu_ptr, 0, num_elements * sizeof(T)));
    NCCLCHECK(this->rkeys_hd_mhandle.allocate(num_ranks));
    return ncclSuccess;
  }

  ncclResult_t deallocate() {
    if (this->gpu_ptr != nullptr) {
      NCCLCHECK(ncclCuMemFree(this->gpu_ptr, nullptr));
      this->gpu_ptr = nullptr;
    }
    return ncclSuccess;
  }

  ncclResult_t register_mr(struct ibv_pd *ib_pd, bool force_strict_ordering = false) {
    if (this->num_elements == 0) return ncclSuccess;
    NCCLCHECK(gdakiRegMr(&this->mr, ib_pd, this->gpu_ptr, this->num_elements * sizeof(T),
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                           IBV_ACCESS_REMOTE_ATOMIC,
                         force_strict_ordering));
    return ncclSuccess;
  }

  ncclResult_t deregister_mr() {
    if (this->mr != nullptr) {
      NCCLCHECK(wrap_ibv_dereg_mr(this->mr));
      this->mr = nullptr;
    }
    return ncclSuccess;
  }

  ncclResult_t exchange_info(struct ncclGinIbCollComm *cComm) {
    if (this->num_elements == 0) return ncclSuccess;
    __be32 rkey = htobe32(this->mr->rkey);
    NCCLCHECK(cComm->allGather(cComm, &rkey, this->rkeys_hd_mhandle.host_buf, sizeof(__be32)));
    NCCLCHECK(this->rkeys_hd_mhandle.copy_h_to_d());
    return ncclSuccess;
  }

  ncclResult_t allocate_elements(unsigned int num_elements, unsigned int *out_start_idx) {
    if (this->next_unused_idx + num_elements > this->num_elements) {
      WARN("Not enough space to get elements");
      return ncclInvalidUsage;
    }

    *out_start_idx = this->next_unused_idx;
    this->next_unused_idx += num_elements;

    return ncclSuccess;
  }

  void free_elements(unsigned int start_idx, unsigned int num_elements) {
    // No op for now as we don't allow reusing elements.
  }

  uint32_t *get_rkeys_d() { return this->rkeys_hd_mhandle.gpu_buf; }

  GdakiGlobalGPUBufferTable()
    : cumemhandle(0), num_elements(0), next_unused_idx(0), gpu_ptr(nullptr), mr(nullptr) {};
  ~GdakiGlobalGPUBufferTable() {
     // Should only be used in error cleanup path as it ignores return codes
     this->deregister_mr();
     this->deallocate();
  }
};

struct gdaki_mem_handle {
  int type;
  struct ibv_mr *mr;
  GdakiHostGPUMemHandle<struct ncclGinGdakiMemHandle> *gdaki_mhandle_hd_mhandle;
  GdakiHostGPUMemHandle<uint32_t> *rkeys_hd_mhandle;
};

struct gdaki_exch_info {
  int lid;
  int qpn;
  union ibv_gid gid;
  struct doca_verbs_gid vgid;
  int gid_index;  // GID 表索引（RoCE v2 需要选择哪个 GID 表项）
};

// ════════════════════════════════════════════════════════════════════════
// gdaki_context：GDAKI 模式的 CPU 侧顶层上下文
// ════════════════════════════════════════════════════════════════════════
// 生命周期：createContext 创建 → 运行时存活 → destroyContext 销毁
// 角色定位：CPU 侧持有的全部 GDAKI 资源的总管（QP、MR、CQ 等）
//          GPU kernel 不直接访问此结构体，GPU 侧用 ncclGinGdakiGPUContext
// 对比 Proxy 模式：Proxy 有 ginProxyCtx（connection 级）+ ginProxyHostGpuCtx（per-context）
//                  GDAKI 只有一个 gdaki_context，内部按数组管理所有 context 的资源
// ════════════════════════════════════════════════════════════════════════
struct gdaki_context {
  // ── GPU 设备信息 ──
  int cuda_id;  // 当前 CUDA 设备 ID（cudaGetDevice 获得）
  struct doca_gpu *gdev;  // DOCA GPU 设备对象（doca_gpu_create 创建）
  // ── IB 设备基础设施 ──
  struct ibv_device *ib_dev;  // IB 设备句柄（未使用，保留字段）
  struct doca_verbs_ah_attr *ah; /* DOCA Verbs address handle */
  struct ibv_device_attr ib_dev_attr;
                                   // 连接不同 peer 时只更新 dlid/dgid，复用同一个 AH 对象
  struct doca_verbs_gid gid;  // DOCA 格式的本地 GID（未使用，由 rgid 替代）

  // ── 网络路由信息 ──
  union ibv_gid rgid;  // 本地 GID（128-bit，RoCE 路由标识）
  struct ibv_port_attr port_attr;  // IB 端口属性（LID/link_layer/MTU/active_width 等）
  uint8_t port_num;  // IB 端口号（mlx5 默认 1）
  int gid_index;

  // ── QP 资源 ──
  uint32_t qp_rq_size;  // Receive Queue 深度（GDAKI 只发不收，恒为 0）
  uint32_t qp_sq_size;  // Send Queue 深度（WQE 槽数，默认 128）
  struct doca_gpu_verbs_qp_group_hl **gqp_groups;  // QP group 数组 [nqps_for_comm]
                                   // 每个 group = {main QP + companion QP}，共享 UAR
  struct doca_gpu_verbs_qp_hl **gqps;  // main QP 数组 [nqps]
                                   // [0..nqps_for_comm-1] = initiator（含 self_rank）
                                   // [nqps_for_comm..nqps-1] = self-loop responder
  struct doca_gpu_verbs_qp_hl **companion_gqps;  // companion QP 数组 [ncompanion_qps]

  // ── GPU 全局表（Counter/Signal）──
  GdakiGlobalGPUBufferTable<uint64_t> *counters_table;  // 全局 counter 表
                                   // GPU 显存数组 + IB MR + 各 rank 的 rkey（已 allGather）
                                   // companion QP 的 Atomic FAA 写入此表
  GdakiGlobalGPUBufferTable<uint64_t> *signals_table;  // 全局 signal 表
                                   // main QP 的 RDMA Atomic FAA 写入此表（远端通知）
  GdakiHostGPUMemHandle<struct ncclGinGdakiGPUContext> *gin_gdaki_gpu_ctx_hd_mhandle;
                                   // GPU Context 双端容器 [ncontexts]
                                   // host_buf（CPU 填充）→ copy_h_to_d → gpu_buf（GPU 访问）
                                   // gpu_buf 最终赋值给 devHandle->handle

  // ── Sink Buffer（黑洞显存）──
  struct {
    void *addr;  // GPU 显存地址（8 字节）
    struct ibv_mr *mr;
    CUmemGenericAllocationHandle mhandle;  // CUDA 分配句柄（销毁时用）
  } sink_buffer;  // Atomic FAA 的 local 目标（读回的旧值丢这里，不关心）
  // ── 杂项 ──
  uint64_t last_error_query_time;  // 上次查询 QP 错误的时间戳（错误检测限频）

  // ── 反向引用 ──
  struct ncclGinIbCollComm *collComm;  // 指回 connect 阶段的 collComm
  ncclNetDeviceHandle_t *devHandle;  // 返回给 NCCL 的设备句柄
  int nContexts;  // context 数量
};

template <typename T>
static inline T gdaki_round_up(T x, T y) {
  return ((x + y - 1) / y) * y;
}

static void gdakiFillExchInfo(struct gdaki_exch_info *exch_info, struct gdaki_context *gdaki_ctx,
                              struct doca_gpu_verbs_qp_hl *gqp) {
  exch_info->lid = gdaki_ctx->port_attr.lid;
  exch_info->qpn = doca_verbs_qp_get_qpn(gqp->qp);
  memcpy(exch_info->gid.raw, gdaki_ctx->rgid.raw, sizeof(union ibv_gid));
  memcpy(exch_info->vgid.raw, gdaki_ctx->rgid.raw, sizeof(union ibv_gid));
  exch_info->gid_index = gdaki_ctx->gid_index;
}

static ncclResult_t gdakiCreateVerbsAh(struct gdaki_context *ctx, struct ibv_context* ib_context, int ib_sl, int ib_tc,
                                       int ib_gid_index) {
  ncclResult_t status = ncclSuccess;

  DOCACHECK(doca_verbs_ah_attr_create(ib_context, &ctx->ah));

  if (ctx->port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
    DOCACHECKGOTO(doca_verbs_ah_attr_set_sl(ctx->ah, ib_sl), status, destroy_verbs_ah);
    DOCACHECKGOTO(doca_verbs_ah_attr_set_addr_type(ctx->ah, DOCA_VERBS_ADDR_TYPE_IB_NO_GRH),
                  status, destroy_verbs_ah);
  } else {
    DOCACHECKGOTO(doca_verbs_ah_attr_set_traffic_class(ctx->ah, ib_tc), status, destroy_verbs_ah);
    DOCACHECKGOTO(doca_verbs_ah_attr_set_addr_type(ctx->ah, DOCA_VERBS_ADDR_TYPE_IPv4),
                  status, destroy_verbs_ah);
  }

  // set_port_num?
  DOCACHECKGOTO(doca_verbs_ah_attr_set_sgid_index(ctx->ah, ib_gid_index), status, destroy_verbs_ah);
  DOCACHECKGOTO(doca_verbs_ah_attr_set_hop_limit(ctx->ah, 255), status, destroy_verbs_ah);

  return ncclSuccess;

destroy_verbs_ah:
  DOCACHECK(doca_verbs_ah_attr_destroy(ctx->ah));
  return status;
}

static ncclResult_t gdakiConnectQp(struct gdaki_context *ctx, struct doca_gpu_verbs_qp_hl *gqp,
                                   struct gdaki_exch_info *exch_info) {
  ncclResult_t status = ncclSuccess;
  struct doca_verbs_qp_attr *verbs_qp_attr = nullptr;
  int max_dest_rd_atomic = ncclParamGinGdakiMaxDestRdAtomic() > 0 ? ncclParamGinGdakiMaxDestRdAtomic() : ctx->ib_dev_attr.max_qp_rd_atom;
  int max_qp_rd_atomic = ncclParamGinGdakiMaxQpRdAtomic() > 0 ? ncclParamGinGdakiMaxQpRdAtomic() : ctx->ib_dev_attr.max_qp_rd_atom;

  // ctx->ah 是 createContext 时创建的同一个 AH 对象。每次连接不同 peer 时，只是原地修改 GID/LID，
  // 然后在 RTR 阶段固化到 QP。不是每个 peer 单独创建 AH——节省内存，
  // 因为 AH 的 SL、TC、hop_limit 等路由属性对所有 peer 相同。
  DOCACHECK(doca_verbs_ah_attr_set_gid(ctx->ah, exch_info->vgid));
  DOCACHECK(doca_verbs_ah_attr_set_dlid(ctx->ah, exch_info->lid));
  DOCACHECK(doca_verbs_qp_attr_create(&verbs_qp_attr));
  DOCACHECKGOTO(
    doca_verbs_qp_attr_set_path_mtu(verbs_qp_attr, DOCA_VERBS_MTU_SIZE_4K_BYTES),
    status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_rq_psn(verbs_qp_attr, 0), status, destroy_verbs_qp_attr);

  DOCACHECKGOTO(doca_verbs_qp_attr_set_sq_psn(verbs_qp_attr, 0), status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_port_num(verbs_qp_attr, ctx->port_num),
                status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_ack_timeout(verbs_qp_attr, ncclParamIbTimeout()),
                status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_retry_cnt(verbs_qp_attr, ncclParamIbRetryCnt()),
                status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_rnr_retry(verbs_qp_attr, 7), status,
                destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_min_rnr_timer(verbs_qp_attr, 12),
                status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(
    doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_INIT),
    status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_allow_remote_write(verbs_qp_attr, 1),
                status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_allow_remote_read(verbs_qp_attr, 1),
                status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_allow_remote_atomic(
                  verbs_qp_attr, DOCA_VERBS_QP_ATOMIC_MODE_IB_SPEC),
                status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_ah_attr(verbs_qp_attr, ctx->ah),
                status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_dest_qp_num(verbs_qp_attr, exch_info->qpn),
                status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_pkey_index(verbs_qp_attr, ncclParamIbPkey()),
                status, destroy_verbs_qp_attr);

  DOCACHECKGOTO(doca_verbs_qp_modify(
                  gqp->qp, verbs_qp_attr,
                  DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE |
                    DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ | DOCA_VERBS_QP_ATTR_PKEY_INDEX |
                    DOCA_VERBS_QP_ATTR_PORT_NUM),
                status, destroy_verbs_qp_attr);


  DOCACHECKGOTO(
    doca_verbs_qp_attr_set_max_dest_rd_atomic(verbs_qp_attr, max_dest_rd_atomic),
    status, destroy_verbs_qp_attr);

  DOCACHECKGOTO(
    doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_RTR),
    status, destroy_verbs_qp_attr);

  DOCACHECKGOTO(doca_verbs_qp_modify(
                  gqp->qp, verbs_qp_attr,
                  DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_RQ_PSN |
                    DOCA_VERBS_QP_ATTR_DEST_QP_NUM | DOCA_VERBS_QP_ATTR_PATH_MTU |
                    DOCA_VERBS_QP_ATTR_AH_ATTR | DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER | DOCA_VERBS_QP_ATTR_MAX_DEST_RD_ATOMIC),
                status, destroy_verbs_qp_attr);

  DOCACHECKGOTO(
    doca_verbs_qp_attr_set_max_rd_atomic(verbs_qp_attr, max_qp_rd_atomic),
    status, destroy_verbs_qp_attr);

  DOCACHECKGOTO(
    doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_RTS),
    status, destroy_verbs_qp_attr);

  DOCACHECKGOTO(doca_verbs_qp_modify(
                  gqp->qp, verbs_qp_attr,
                  DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN |
                    DOCA_VERBS_QP_ATTR_ACK_TIMEOUT | DOCA_VERBS_QP_ATTR_RETRY_CNT |
                    DOCA_VERBS_QP_ATTR_RNR_RETRY | DOCA_VERBS_QP_ATTR_MAX_QP_RD_ATOMIC),
                status, destroy_verbs_qp_attr);

  DOCACHECK(doca_verbs_qp_attr_destroy(verbs_qp_attr));

  return ncclSuccess;

destroy_verbs_qp_attr:
  DOCACHECK(doca_verbs_qp_attr_destroy(verbs_qp_attr));
  return status;
}

// ============================================================================
// NCCL_GDAKI_USE_RELIABLE_DB 环境变量
// ============================================================================
// value=0: VALID_DBR（不尝试 Reliable DB）
// value=1: NO_DBR_HW → SW_EMULATED → 报错
// value=2: NO_DBR_HW → SW_EMULATED → VALID_DBR（全降级链，永不失败）
// Reliable DB 模式下 NIC 不依赖 DBR，直接从 SQ 取 WQE，延迟更低
// ============================================================================
NCCL_PARAM(GinGdakiUseReliableDB, "GDAKI_USE_RELIABLE_DB", 0);

ncclResult_t ncclGinGdakiCreateContext(void *collComm, int nSignals, int nCounters, int nContexts, int queueDepth,
                                       int trafficClass, void **outGinCtx, ncclNetDeviceHandle_t **outDevHandle) {
  ncclResult_t status = ncclSuccess;

  // collComm 是 connect 阶段产物，包含 rank/nranks/fullSendComm 等信息
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;

  char pciBusId[MAX_PCI_ADDRESS_LEN];  // GPU 的 PCI Bus ID 字符串，如 "0000:07:00.0"

  // ════════════════════════════════════════════════════════════════════════
  // QP 数量计算（GDAKI 最关键的设计决策之一）
  // ════════════════════════════════════════════════════════════════════════
  // Proxy 模式：所有 context 共享 nRanks 个 QP（fullSendComm[]）
  // GDAKI 模式：每个 context × 每个 peer = 一个独立 QP
  //   原因：GPU kernel 并发发包时，共享 SQ 需要原子锁，严重影响性能
  //         独立 QP 让每个 block/warp 可以无锁直接写 WQE
  const int rank = cComm->rank;
  const int nranks = cComm->nranks;
  const int ncontexts = nContexts;
  // 每个 peer 分配 ncontexts 个 QP（每 context 一个独立 QP）
  const int nqps_per_rank = ncontexts;
  // initiator QP 总数 = ncontexts × nranks（包含发往自身的 QP！）
  // 包含 self_rank 的原因：GPU kernel 通过 gdqp[peer] 统一索引，
  // peer 可以是 0~nranks-1 中的任何值（MoE token 可能路由回本 rank），
  // 所以 gdqp[self_rank] 必须是一个合法的 QP
  const int nqps_for_comm = nqps_per_rank * nranks;  // Number of QPs for communication
  // companion QP：每个 initiator QP 配一个 companion QP，用于 WAIT + Atomic FAA
  // companion QP 全部自环（不跨 rank），所以每个 initiator 都需要一个 self-loop responder
  // 总数 = nqps_for_comm（initiator）+ nqps_for_comm（responder）= nqps_for_comm × 2
  const int ncompanion_qps = nqps_for_comm * 2;      // Number of companion QPs for communication
                                                     // Double because we connect to self.
  const int nqps =
    nqps_per_rank * (nranks + 1);  // +1 for the local rank.
                                   // The last group is the responder of the local rank.

  // TODO: Take these config parameters from the environment variables or users.
  // 每个 context 需要的 counter/signal 数量（从调用者传入）
  const int num_counters = nCounters;
  const int num_signals = nSignals;
  ncclNetProperties_t props;  // IB 设备属性（延迟、带宽、MTU 等）
  ncclNetDeviceHandle_t *devHandle = nullptr;  // 返回给 NCCL 的设备句柄
  struct gdaki_context *gdaki_ctx = nullptr;  // GDAKI 顶层上下文（本函数核心产出）
  struct gdaki_exch_info *local_exch_info = nullptr;  // 本端 QPN/GID 信息（准备发给对端）
  struct gdaki_exch_info *remote_exch_info = nullptr;  // 对端 QPN/GID 信息（从对端收到）

  // DOCA QP 创建属性结构体，后面填完后传给 doca_gpu_verbs_create_qp_group_hl
  struct doca_gpu_verbs_qp_init_attr_hl qp_init_attr;

  // ════════════════════════════════════════════════════════════════════════
  // sink_buffer：一个 8 字节的"黑洞"显存
  // ════════════════════════════════════════════════════════════════════════
  // 用途：某些 RDMA 操作需要合法的本地内存目标，但不关心内容，就指向这里
  // 例如：RDMA Read 需要一个 local buffer，如果只想测试延迟就用 sink_buffer
  uint64_t *sink_buffer = nullptr;
  struct ibv_mr *sink_buffer_mr = nullptr;  // sink_buffer 对应的 IB MR
  CUmemGenericAllocationHandle sink_buffer_mhandle;  // CUDA 内存分配句柄

  // need_cpu_proxy：最终是否需要 CPU proxy 线程
  // true 的情况：硬件不支持 GPU 直接敲 Doorbell（如跨 NUMA、PCIe P2P 不支持）
  //             GPU 写 cpu_db（共享内存），CPU proxy 轮询后代劳敲 DB
  bool need_cpu_proxy = false;

  // gverbs_qps：临时数组，用于收集 nranks 个 QP 的 GPU 运行时视图
  // 后续传给 doca_gpu_verbs_export_multi_qps_dev 批量导出到 GPU 显存
  struct doca_gpu_verbs_qp **gverbs_qps = nullptr;

  // ════════════════════════════════════════════════════════════════════════
  // GPU Context 容器分配（host_buf + gpu_buf 双端）
  // ════════════════════════════════════════════════════════════════════════
  // GdakiHostGPUMemHandle 是一个模板类，管理 host_buf（CPU 填充）+ gpu_buf（GPU 访问）
  // 构造函数内部：
  //   1. host_buf = calloc(ncontexts, sizeof(ncclGinGdakiGPUContext))  // CPU 内存
  //   2. ncclCuMemAlloc(&gpu_buf, ..., ncontexts * sizeof(...))        // GPU 显存
  // 后续在 host_buf 上填充每个 context 的 QP 地址、counter/signal 切片，
  // 最后 copy_h_to_d() 部署到 GPU 显存
  GdakiHostGPUMemHandle<struct ncclGinGdakiGPUContext> *gin_gdaki_gpu_ctx_hd_mhandle =
    new GdakiHostGPUMemHandle<struct ncclGinGdakiGPUContext>();
  GdakiGlobalGPUBufferTable<uint64_t> *counters_table = new GdakiGlobalGPUBufferTable<uint64_t>();
  GdakiGlobalGPUBufferTable<uint64_t> *signals_table = new GdakiGlobalGPUBufferTable<uint64_t>();

  const int ib_sl = (ncclParamIbSl() != -1) ? ncclParamIbSl() : (trafficClass != NCCL_NET_TRAFFIC_CLASS_UNDEF) ? trafficClass : NCCL_IB_SL_DEFAULT;
  const int ib_tc = (ncclParamIbTc() != -1) ? ncclParamIbTc() : (trafficClass != NCCL_NET_TRAFFIC_CLASS_UNDEF) ? trafficClass : NCCL_IB_TC_DEFAULT;
  // GID Index：RoCE v2 使用 IPv6 GID，需要选择 NIC 的哪个 GID 表项
  int ib_gid_index = 0;

  // 获取 IB 设备属性（延迟、带宽、MTU 等）
  NCCLCHECK(cComm->getProperties(cComm->dev, &props));

  NCCLCHECKGOTO(gin_gdaki_gpu_ctx_hd_mhandle->allocate(ncontexts), status, out);
  NCCLCHECKGOTO(counters_table->allocate(num_counters * ncontexts, nranks), status, out);
  NCCLCHECKGOTO(signals_table->allocate(num_signals * ncontexts, nranks), status, out);

  // ════════════════════════════════════════════════════════════════════════
  // 分配 GDAKI 顶层上下文和各种指针数组
  // ════════════════════════════════════════════════════════════════════════
  gdaki_ctx = (struct gdaki_context *)calloc(1, sizeof(*gdaki_ctx));
  // EQCHECKGOTO(ptr, nullptr, ...) 意思是：如果 ptr == nullptr 则出错跳转
  EQCHECKGOTO(gdaki_ctx, nullptr, status, out);

  devHandle = (ncclNetDeviceHandle_t *)calloc(1, sizeof(*devHandle));
  EQCHECKGOTO(devHandle, nullptr, status, out);

  // gqp_groups：QP group 数组
  // 每个 QP group 包含：main QP（数据传输）+ companion QP（Atomic 操作）
  // 索引规则：gqp_groups[ctx_idx * nranks + peer_rank]
  gdaki_ctx->gqp_groups = (struct doca_gpu_verbs_qp_group_hl **)calloc(
    nqps_for_comm, sizeof(*gdaki_ctx->gqp_groups));
  EQCHECKGOTO(gdaki_ctx->gqp_groups, nullptr, status, out);

  // Main QP
  // gqps：Main QP 数组，用于发送 RDMA Write（数据传输）
  // 索引规则：gqps[ctx_idx * nranks + peer_rank] = context ctx_idx 发往 peer_rank 的 QP
  // 注意：gqps 数组长度是 nqps（包含 self-loop responder）
  gdaki_ctx->gqps = (struct doca_gpu_verbs_qp_hl **)calloc(nqps, sizeof(*gdaki_ctx->gqps));
  EQCHECKGOTO(gdaki_ctx->gqps, nullptr, status, out);

  // Companion QP
  // companion_gqps：Companion QP 数组，用于发送 RDMA Atomic（signal FAA）
  // companion 与 main 共享同一个 QP group，但拥有独立的 SQ
  // 数量是 ncompanion_qps = nqps_for_comm * 2（因为要连 self-loop）
  gdaki_ctx->companion_gqps =
    (struct doca_gpu_verbs_qp_hl **)calloc(ncompanion_qps, sizeof(*gdaki_ctx->companion_gqps));
  EQCHECKGOTO(gdaki_ctx->companion_gqps, nullptr, status, out);

  // local_exch_info：本端 QPN/GID 信息，准备发给对端
  // 结构体内容：{ lid, qpn, gid, vgid, gid_index }
  local_exch_info = (struct gdaki_exch_info *)calloc(nranks, sizeof(*local_exch_info));
  EQCHECKGOTO(local_exch_info, nullptr, status, out);

  // remote_exch_info：从对端收到的 QPN/GID 信息
  remote_exch_info = (struct gdaki_exch_info *)calloc(nranks, sizeof(*remote_exch_info));
  EQCHECKGOTO(remote_exch_info, nullptr, status, out);

  // ════════════════════════════════════════════════════════════════════════
  // DOCA GPU 设备初始化
  // ════════════════════════════════════════════════════════════════════════
  // 获取当前 CUDA 设备 ID 和 PCI Bus ID
  // pciBusId 格式如 "0000:07:00.0"，用于后续 doca_gpu_create
  CUDACHECK(cudaGetDevice(&gdaki_ctx->cuda_id));
  CUDACHECK(cudaDeviceGetPCIBusId(pciBusId, MAX_PCI_ADDRESS_LEN, gdaki_ctx->cuda_id));

  DOCACHECKGOTO(doca_gpu_create(pciBusId, &gdaki_ctx->gdev), status, out);

  NCCLCHECKGOTO(wrap_ibv_query_device(cComm->ib.context, &gdaki_ctx->ib_dev_attr), status, out);

  // Exchange counters and signals with peers
  NCCLCHECKGOTO(counters_table->register_mr(cComm->ib.pd, true), status, out);
  NCCLCHECKGOTO(signals_table->register_mr(cComm->ib.pd, true), status, out);

  // exchange_info(cComm)：allGather 广播自己的 rkey 给所有 rank
  // 内部动作：
  //   1. rkey = htobe32(mr->rkey)  // 转大端序
  //   2. allGather(&rkey, rkeys_host_buf)  // 每个 rank 收集到所有 rank 的 rkey
  //   3. copy_h_to_d()  // 部署到 GPU 显存
  // 结果：counters_table->rkeys[rank] = rank 的 counter 数组 rkey（存在 GPU 显存）
  //       GPU kernel 可以直接查这个表获得对端的 rkey
  NCCLCHECKGOTO(counters_table->exchange_info(cComm), status, out);
  NCCLCHECKGOTO(signals_table->exchange_info(cComm), status, out);

  // ════════════════════════════════════════════════════════════════════════
  // IB 端口和 GID 信息查询
  // ════════════════════════════════════════════════════════════════════════
  // mlx5 设备默认端口 1（多端口 NIC 可能有 port_num=2）
  gdaki_ctx->port_num = 1; // assume 1 for mlx5 devices
  NCCLCHECKGOTO(wrap_ibv_query_port(cComm->ib.context, gdaki_ctx->port_num, &gdaki_ctx->port_attr),
                status, out);

  // Get the GID index
  NCCLCHECKGOTO(cComm->getGidIndex(cComm->ib.context, gdaki_ctx->port_num, &gdaki_ctx->port_attr, &ib_gid_index), status, out);
  gdaki_ctx->gid_index = ib_gid_index;

  NCCLCHECKGOTO(wrap_ibv_query_gid(cComm->ib.context, 1, ib_gid_index, &gdaki_ctx->rgid), status,
                out);

  NCCLCHECKGOTO(gdakiCreateVerbsAh(gdaki_ctx, cComm->ib.context, ib_sl, ib_tc, ib_gid_index), status, out);

  // ════════════════════════════════════════════════════════════════════════
  // QP 深度配置
  // ════════════════════════════════════════════════════════════════════════
  // qp_rq_size = 0：GDAKI 只发不收，不需要 RQ（Receive Queue）
  //   原因：RDMA Write 是 unilateral 操作，对端不需要 post_receive
  gdaki_ctx->qp_rq_size = 0;
  gdaki_ctx->qp_sq_size = queueDepth > 0 ? queueDepth : ncclParamGinGdakiQpDepth();

  // ════════════════════════════════════════════════════════════════════════
  // 填充 QP 创建属性
  // ════════════════════════════════════════════════════════════════════════
  memset(&qp_init_attr, 0, sizeof(qp_init_attr));
  qp_init_attr.gpu_dev = gdaki_ctx->gdev;  // 绑定到哪块 GPU（决定 WQE 内存在哪）
  qp_init_attr.ibpd = cComm->ib.pd;
  qp_init_attr.sq_nwqe = gdaki_ctx->qp_sq_size;  // SQ WQE 数量（会被 round up to power of 2）
  // ════════════════════════════════════════════════════════════════════════
  // nic_handler：GPU 如何敲 Doorbell（最关键的性能决定因素）
  // ════════════════════════════════════════════════════════════════════════
  // 环境变量 NCCL_GIN_GDAKI_NIC_HANDLER，默认 0（AUTO）
  // 可选值（来自 doca_gpunetio.h）：
  //   AUTO(0)      → 自动选择，优先 GPU_SM_DB，不支持则 CPU_PROXY
  //   CPU_PROXY(1) → 强制 CPU proxy 模式（GPU 写共享内存，CPU 代劳敲 DB）
  //   GPU_SM_DB(2) → 强制 GPU 直接敲 DB（要求 PCIe P2P 支持，GPU 与 NIC 同 NUMA）
  //   GPU_SM_BF(3) → 强制 BlueFlame 模式（高端 NIC 支持 WQE+DB 合并写入）
  //   GPU_SM_NO_DBR(4) → GPU 直接敲 DB + 不写 DBR（create_qp 内部自动设置，不由用户直接选择）
  qp_init_attr.nic_handler =
    (enum doca_gpu_dev_verbs_nic_handler)ncclParamGinGdakiNicHandler();
  // mreg_type：内存注册类型，DEFAULT 表示使用常规 ibv_reg_mr 方式
  qp_init_attr.mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT;

  // send_dbr_mode_ext：Doorbell Record 模式扩展
  // VALID_DBR：传统模式，GPU 写 DBR（WQE PI），NIC 轮询 DBR 知道有新 WQE
  // NO_DBR_HW：硬件 Reliable DB，NIC 直接从 SQ 取 WQE，不依赖 DBR（延迟更低）
  // NO_DBR_SW_EMULATED：软件模拟 Reliable DB，用 GDRCopy 实现
  if (ncclParamGinGdakiUseReliableDB())
    qp_init_attr.send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_HW;
  else
  // 创建 self-loop QP（本 rank 内 loopback 的 responder）
    qp_init_attr.send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR;

  // ════════════════════════════════════════════════════════════════════════
  // 循环创建 nqps_for_comm 个 QP group（每个包含 main + companion QP）
  // ════════════════════════════════════════════════════════════════════════
  for (int qp_idx = 0; qp_idx < nqps_for_comm; qp_idx++) {
retry_create_qp_group_hl:
    doca_error_t docaStatus = doca_gpu_verbs_create_qp_group_hl(&qp_init_attr, &gdaki_ctx->gqp_groups[qp_idx]);
    if (docaStatus != DOCA_SUCCESS) {
      // 如果 NO_DBR_HW 模式失败，回退到 SW_EMULATED 模式
      if (qp_init_attr.send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_HW) {
        qp_init_attr.send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED;
        goto retry_create_qp_group_hl;
      }

      if ((qp_init_attr.send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED)
          && ncclParamGinGdakiUseReliableDB() == 2) {
        qp_init_attr.send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR;
        goto retry_create_qp_group_hl;
      }

      WARN("DOCA Error %d", docaStatus);
      status = ncclSystemError;
      goto out;
    }

    gdaki_ctx->gqps[qp_idx] = &gdaki_ctx->gqp_groups[qp_idx]->qp_main;
    gdaki_ctx->companion_gqps[qp_idx] = &gdaki_ctx->gqp_groups[qp_idx]->qp_companion;

    const char *dbr_opt_str =
      qp_init_attr.send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_HW ? "HW"
      : qp_init_attr.send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED
        ? "SW emulation"
        : "disabled";
    INFO(NCCL_NET,
         "[%d] Created a QP group: qp_idx=%d, main_qpn=%#x, companion_qpn=%#x, reliable_db=%s",
         rank, qp_idx, doca_verbs_qp_get_qpn(gdaki_ctx->gqps[qp_idx]->qp),
         doca_verbs_qp_get_qpn(gdaki_ctx->companion_gqps[qp_idx]->qp), dbr_opt_str);
  }

  qp_init_attr.send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR;
  for (int qp_idx = nqps_for_comm; qp_idx < nqps; qp_idx++) {
    DOCACHECKGOTO(doca_gpu_verbs_create_qp_hl(&qp_init_attr, &gdaki_ctx->gqps[qp_idx]),
                  status, out);
    INFO(NCCL_NET, "[%d] Created a self-loop peer QP: qp_idx=%d, qpn=%#x", rank, qp_idx,
         doca_verbs_qp_get_qpn(gdaki_ctx->gqps[qp_idx]->qp));
  }

  // ════════════════════════════════════════════════════════════════════════
  // 创建 self-loop companion QP（用于本 rank 内 Atomic 操作的 loopback）
  // ════════════════════════════════════════════════════════════════════════
  // companion QP 索引 [nqps_for_comm .. ncompanion_qps-1] 是 self-loop companion
  for (int qp_idx = nqps_for_comm; qp_idx < ncompanion_qps; qp_idx++) {
    DOCACHECKGOTO(
      doca_gpu_verbs_create_qp_hl(&qp_init_attr, &gdaki_ctx->companion_gqps[qp_idx]),
      status, out);
    INFO(NCCL_NET, "[%d] Created a self-loop peer companion QP: qp_idx=%d, qpn=%#x", rank, qp_idx,
         doca_verbs_qp_get_qpn(gdaki_ctx->companion_gqps[qp_idx]->qp));
  }

  // ════════════════════════════════════════════════════════════════════════
  // QP 连接阶段：交换 QPN/GID 并完成 INIT → RTR → RTS
  // ════════════════════════════════════════════════════════════════════════
  // 外层循环：遍历每个 context
  // 内层循环：遍历每个 rank（peer）
  for (int ctx_idx = 0; ctx_idx < ncontexts; ctx_idx++) {
    // Prepare information for exchange with peers
    // 步骤 c：用 remote QPN 驱动每个 QP 完成 INIT→RTR→RTS
    // 步骤 a：填写本 context 的 local_exch_info
    // 每个 peer 对应一个 QP，提取其 QPN、GID、LID 等信息
    for (int rank_idx = 0; rank_idx < nranks; rank_idx++) {
      // qp_idx 计算：context ctx_idx 发往 rank_idx 的 QP 在数组中的索引
      int qp_idx = rank_idx + ctx_idx * nranks;
      // gdakiFillExchInfo：从 QP 中提取 {lid, qpn, gid, vgid, gid_index}
      gdakiFillExchInfo(&local_exch_info[rank_idx], gdaki_ctx, gdaki_ctx->gqps[qp_idx]);
    }

    // Exchange information with peers
    // 步骤 b：allToAll 交换 exch_info
    // 每个 rank 把自己对所有 peer 的 QPN 发出去，同时收到对方对自己的 QPN
    // 结果：remote_exch_info[rank_idx] = rank_idx 对自己的 QPN/GID
    NCCLCHECKGOTO(
      cComm->allToAll(cComm, local_exch_info, remote_exch_info, sizeof(struct gdaki_exch_info)),
      status, out);

    for (int rank_idx = 0; rank_idx < nranks; rank_idx++) {
      int qp_idx = rank_idx + ctx_idx * nranks;
      // 特殊情况：rank_idx == rank（自环）
      // 用 self-loop responder QP 的 QPN 作为"对端"
      if (rank_idx == rank)
        gdakiFillExchInfo(&remote_exch_info[rank_idx], gdaki_ctx,
                          gdaki_ctx->gqps[nqps_for_comm + ctx_idx]);

      // gdakiConnectQp：执行 QP 状态机 RESET → INIT → RTR → RTS
      // 内部调用三次 doca_verbs_qp_modify：
      //   1. RESET → INIT（设置 port_num、pkey、access 权限）
      //   2. INIT → RTR（填入对端 QPN、GID、MTU、path_mtu）
      //   3. RTR → RTS（设置 ack_timeout、retry_cnt、sq_psn）
      NCCLCHECKGOTO(gdakiConnectQp(gdaki_ctx, gdaki_ctx->gqps[qp_idx], &remote_exch_info[rank_idx]),
                    status, out);

      INFO(NCCL_NET,
           "[%d] Connected main QP: qp_idx=%d, main_qpn=%#x, remote_rank=%d, remote_qpn=%#x", rank,
           qp_idx, doca_verbs_qp_get_qpn(gdaki_ctx->gqps[qp_idx]->qp), rank_idx,
           remote_exch_info[rank_idx].qpn);
    }
  }

  // ════════════════════════════════════════════════════════════════════════
  // 连接 self-loop main QP（本 rank 发给自己的数据走这些 QP）
  // ════════════════════════════════════════════════════════════════════════
  // gqps[nqps_for_comm + ctx_idx] 是 context ctx_idx 的 self-loop responder
  // 它需要连接到 gqps[ctx_idx * nranks + rank]（context ctx_idx 发往自己的 QP）
  for (int qp_idx = 0; qp_idx < nqps_per_rank; qp_idx++) {
    int peer_qp_idx = nqps_for_comm + qp_idx;
    struct gdaki_exch_info exch_info;
    // 取出发往自己的 QP 的 QPN
    gdakiFillExchInfo(&exch_info, gdaki_ctx, gdaki_ctx->gqps[qp_idx * nranks + rank]);
    // 让 self-loop responder 连接到它
    NCCLCHECKGOTO(gdakiConnectQp(gdaki_ctx, gdaki_ctx->gqps[peer_qp_idx], &exch_info), status, out);
    INFO(NCCL_NET, "[%d] Connected self-loop peer QP: qp_idx=%d, qpn=%#x, main_qpn=%#x", rank,
         peer_qp_idx, doca_verbs_qp_get_qpn(gdaki_ctx->gqps[peer_qp_idx]->qp), exch_info.qpn);
  }

  // ════════════════════════════════════════════════════════════════════════
  // 连接 companion QP（用于 RDMA Atomic / signal FAA）
  // ════════════════════════════════════════════════════════════════════════
  // companion QP 的连接逻辑与 main QP 类似，但需要处理 self-loop
  for (int qp_idx = 0; qp_idx < nqps_for_comm; qp_idx++) {
    int peer_qp_idx = nqps_for_comm + qp_idx;
    struct gdaki_exch_info exch_info;
    // companion_gqps[peer_qp_idx] 是 self-loop companion
    // 让 companion_gqps[qp_idx] 连接到 peer_qp_idx
    gdakiFillExchInfo(&exch_info, gdaki_ctx, gdaki_ctx->companion_gqps[peer_qp_idx]);
    NCCLCHECKGOTO(gdakiConnectQp(gdaki_ctx, gdaki_ctx->companion_gqps[qp_idx], &exch_info), status,
                  out);
    INFO(NCCL_NET,
         "[%d] Connected companion QP: qp_idx=%d, companion_qpn=%#x, peer_companion_qpn=%#x", rank,
         qp_idx, doca_verbs_qp_get_qpn(gdaki_ctx->companion_gqps[qp_idx]->qp), exch_info.qpn);

    // 反向连接：让 self-loop companion 连接到本端 companion
    gdakiFillExchInfo(&exch_info, gdaki_ctx, gdaki_ctx->companion_gqps[qp_idx]);
    NCCLCHECKGOTO(gdakiConnectQp(gdaki_ctx, gdaki_ctx->companion_gqps[peer_qp_idx], &exch_info),
                  status, out);
    INFO(NCCL_NET,
         "[%d] Connected self-loop peer companion QP: qp_idx=%d, peer_companion_qpn=%#x, "
         "companion_qpn=%#x",
         rank, peer_qp_idx, doca_verbs_qp_get_qpn(gdaki_ctx->companion_gqps[peer_qp_idx]->qp),
         exch_info.qpn);
  }

  // ════════════════════════════════════════════════════════════════════════
  // 分配 sink buffer（"黑洞"显存）
  // ════════════════════════════════════════════════════════════════════════
  // 某些 RDMA 操作需要合法的本地内存目标，但不关心内容，就指向 sink_buffer
  // 大小：8 字节（一个 uint64_t）
  NCCLCHECKGOTO(ncclCuMemAlloc((void **)&sink_buffer, &sink_buffer_mhandle, CU_MEM_HANDLE_TYPE_NONE,
                               sizeof(uint64_t), nullptr),
                status, out);

  NCCLCHECKGOTO(gdakiRegMr(&sink_buffer_mr, cComm->ib.pd, sink_buffer, sizeof(uint64_t),
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                             IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC),
                status, out);

  // ════════════════════════════════════════════════════════════════════════
  // 导出 QP 到 GPU 并填充 ncclGinGdakiGPUContext
  // ════════════════════════════════════════════════════════════════════════
  // 这是 GPU-direct 的最终部署环节
  gverbs_qps = (struct doca_gpu_verbs_qp **)calloc(nranks, sizeof(struct doca_gpu_verbs_qp *));
  for (int ctx_idx = 0; ctx_idx < ncontexts; ctx_idx++) {
    // 获取 CPU 侧的 GPU context 槽位（后续会 H2D 拷贝到 GPU 显存）
    struct ncclGinGdakiGPUContext *gin_gdaki_gpu_ctx =
      &gin_gdaki_gpu_ctx_hd_mhandle->host_buf[ctx_idx];

    // 收集本 context 的 nranks 个 main QP 的 GPU 运行时视图
    unsigned int buffer_start;
    // 同理处理 companion QP
    for (int qp_idx = 0; qp_idx < nranks; qp_idx++) {
      // gqps[(ctx_idx * nranks) + qp_idx] 是 context ctx_idx 发往 qp_idx 的 QP
      gverbs_qps[qp_idx] = gdaki_ctx->gqps[(ctx_idx * nranks) + qp_idx]->qp_gverbs;
      // 如果任何一个 QP 需要 CPU proxy，则整体需要
      need_cpu_proxy |= (gverbs_qps[qp_idx]->cpu_proxy);
    }

    // doca_gpu_verbs_export_multi_qps_dev：关键！
    // 把 nranks 个 qp_gverbs->qp_cpu（doca_gpu_dev_verbs_qp）
    // 打包为一个 doca_gpu_dev_verbs_qp *gdqp（GPU 显存里的数组）
    // 内部是 cudaMemcpy：
    //   cudaMemcpy(gdqp + i, qp_list[i]->qp_cpu, sizeof(doca_gpu_dev_verbs_qp), H2D)
    // GPU kernel 通过 gdqp[rank_idx] 选择目标 QP，拿到：
    //   - sq_wqe_daddr（WQE 数组地址）
    //   - sq_db（Doorbell 地址）
    //   - sq_dbrec（DBR 地址）
    DOCACHECKGOTO(doca_gpu_verbs_export_multi_qps_dev(gdaki_ctx->gdev, gverbs_qps, nranks,
                                                      &gin_gdaki_gpu_ctx->gdqp),
                  status, out);

    for (int qp_idx = 0; qp_idx < nranks; qp_idx++) {
      gverbs_qps[qp_idx] = gdaki_ctx->companion_gqps[(ctx_idx * nranks) + qp_idx]->qp_gverbs;
      need_cpu_proxy |= (gverbs_qps[qp_idx]->cpu_proxy);
    }
    DOCACHECKGOTO(doca_gpu_verbs_export_multi_qps_dev(gdaki_ctx->gdev, gverbs_qps, nranks,
                                                      &gin_gdaki_gpu_ctx->companion_gdqp),
                  status, out);

    if (nCounters) {
    // ════════════════════════════════════════════════════════════════════════
    // Counter/Signal 切片分配
    // ════════════════════════════════════════════════════════════════════════
    // allocate_elements(num, &start)：从全局数组中分配 num 个连续元素
    // 返回 start = 分配到的起始索引
      NCCLCHECKGOTO(counters_table->allocate_elements(num_counters, &buffer_start), status, out);
    // 填充 GPU context 的 counters_table 字段：
    //   buffer  = 全局数组起始地址 + 偏移量（本 context 的 counter 起始）
    //   rkeys   = 各 rank 的 counter rkey 数组（GPU 显存）
    //   lkey    = 本地 lkey（大端序，用于 RDMA Read 的 local buffer）
    //   offset  = 在全局数组的偏移（用于计算远端 VA）
      gin_gdaki_gpu_ctx->counters_table.buffer = counters_table->gpu_ptr + buffer_start;
      gin_gdaki_gpu_ctx->counters_table.rkeys = counters_table->get_rkeys_d();
      gin_gdaki_gpu_ctx->counters_table.lkey = htobe32(counters_table->mr->lkey);
      gin_gdaki_gpu_ctx->counters_table.offset = buffer_start;
    }
    if (nSignals) {

    // 同理处理 signals_table
      NCCLCHECKGOTO(signals_table->allocate_elements(num_signals, &buffer_start), status, out);
      gin_gdaki_gpu_ctx->signals_table.buffer = signals_table->gpu_ptr + buffer_start;
      gin_gdaki_gpu_ctx->signals_table.rkeys = signals_table->get_rkeys_d();
      gin_gdaki_gpu_ctx->signals_table.lkey = htobe32(signals_table->mr->lkey);
      gin_gdaki_gpu_ctx->signals_table.offset = buffer_start;
    }

    // sink_buffer_lkey：用于不需要真实 local buffer 的 RDMA 操作
    gin_gdaki_gpu_ctx->sink_buffer_lkey = htobe32(sink_buffer_mr->lkey);
  }

  // ════════════════════════════════════════════════════════════════════════
  // 部署：H2D 拷贝 GPU context 数组到 GPU 显存
  // ════════════════════════════════════════════════════════════════════════
  NCCLCHECKGOTO(gin_gdaki_gpu_ctx_hd_mhandle->copy_h_to_d(), status, out);

  // ════════════════════════════════════════════════════════════════════════
  // 填充 devHandle（返回给 NCCL 的设备句柄）
  // ════════════════════════════════════════════════════════════════════════
  devHandle->netDeviceType = NCCL_NET_DEVICE_GIN_GDAKI;  // 设备类型标识
  devHandle->netDeviceVersion = NCCL_GIN_GDAKI_VERSION;  // 版本号
  // handle 指向 GPU 显存中的 ncclGinGdakiGPUContext[nContexts]
  // GPU kernel 通过这个指针访问所有 context 的 QP 地址、counter/signal 切片
  devHandle->handle = (void *)gin_gdaki_gpu_ctx_hd_mhandle->gpu_buf;
  devHandle->size = 0;
  // needsProxyProgress：是否需要 CPU proxy 线程
  // 如果 need_cpu_proxy=true，NCCL 会启动一个 CPU 线程代劳敲 Doorbell
  devHandle->needsProxyProgress = need_cpu_proxy;

  gdaki_ctx->counters_table = counters_table;
  gdaki_ctx->signals_table = signals_table;
  gdaki_ctx->gin_gdaki_gpu_ctx_hd_mhandle = gin_gdaki_gpu_ctx_hd_mhandle;
  gdaki_ctx->sink_buffer.addr = sink_buffer;
  gdaki_ctx->sink_buffer.mr = sink_buffer_mr;
  gdaki_ctx->sink_buffer.mhandle = sink_buffer_mhandle;
  gdaki_ctx->collComm = cComm;
  gdaki_ctx->devHandle = devHandle;
  gdaki_ctx->nContexts = ncontexts;

  *outDevHandle = devHandle;
  *outGinCtx = gdaki_ctx;

  // 返回结果
out:
  if (status != ncclSuccess) {
    if (gdaki_ctx) {
      // Clean up any allocated GPU memory
      if (gdaki_ctx->gin_gdaki_gpu_ctx_hd_mhandle) {
        for (int ctx_idx = 0; ctx_idx < ncontexts; ctx_idx++) {
          struct ncclGinGdakiGPUContext *gin_gdaki_gpu_ctx =
            &gdaki_ctx->gin_gdaki_gpu_ctx_hd_mhandle->host_buf[ctx_idx];
          if (gin_gdaki_gpu_ctx->gdqp) {
            for (int qp_idx = 0; qp_idx < nranks; qp_idx++) {
              gverbs_qps[qp_idx] = gdaki_ctx->gqps[(ctx_idx * nranks) + qp_idx]->qp_gverbs;
            }
            doca_gpu_verbs_unexport_multi_qps_dev(gdaki_ctx->gdev, gverbs_qps, nranks, gin_gdaki_gpu_ctx->gdqp);
            gin_gdaki_gpu_ctx->gdqp = nullptr;
          }
          if (gin_gdaki_gpu_ctx->companion_gdqp) {
            for (int qp_idx = 0; qp_idx < nranks; qp_idx++) {
              gverbs_qps[qp_idx] = gdaki_ctx->companion_gqps[(ctx_idx * nranks) + qp_idx]->qp_gverbs;
            }
            doca_gpu_verbs_unexport_multi_qps_dev(gdaki_ctx->gdev, gverbs_qps, nranks, gin_gdaki_gpu_ctx->companion_gdqp);
            gin_gdaki_gpu_ctx->companion_gdqp = nullptr;
          }
        }
      }

      for (int qp_idx = 0; qp_idx < nqps_for_comm; qp_idx++) {
        doca_gpu_verbs_destroy_qp_group_hl(gdaki_ctx->gqp_groups[qp_idx]);
        gdaki_ctx->gqp_groups[qp_idx] = nullptr;
      }
      for (int qp_idx = nqps_for_comm; qp_idx < nqps; qp_idx++) {
        doca_gpu_verbs_destroy_qp_hl(gdaki_ctx->gqps[qp_idx]);
        gdaki_ctx->gqps[qp_idx] = nullptr;
      }
      for (int qp_idx = nqps_for_comm; qp_idx < ncompanion_qps; qp_idx++) {
        doca_gpu_verbs_destroy_qp_hl(gdaki_ctx->companion_gqps[qp_idx]);
        gdaki_ctx->companion_gqps[qp_idx] = nullptr;
      }

      if (gdaki_ctx->gqp_groups) free(gdaki_ctx->gqp_groups);
      if (gdaki_ctx->gqps) free(gdaki_ctx->gqps);
      if (gdaki_ctx->companion_gqps) free(gdaki_ctx->companion_gqps);

      if (gdaki_ctx->gdev) doca_gpu_destroy(gdaki_ctx->gdev);
    }

    if (devHandle) free(devHandle);

    if (sink_buffer_mr) wrap_ibv_dereg_mr(sink_buffer_mr);
    if (sink_buffer) ncclCuMemFree(sink_buffer, nullptr);

    if (gin_gdaki_gpu_ctx_hd_mhandle) delete gin_gdaki_gpu_ctx_hd_mhandle;
    if (counters_table) delete counters_table;
    if (signals_table) delete signals_table;

    if (gdaki_ctx) {
      memset(gdaki_ctx, 0, sizeof(*gdaki_ctx));
      free(gdaki_ctx);
    }
  }

  if (local_exch_info) free(local_exch_info);

  if (remote_exch_info) free(remote_exch_info);

  if (gverbs_qps) free(gverbs_qps);

  return status;
}

ncclResult_t ncclGinGdakiDestroyContext(void *ginCtx) {
  if (!ginCtx) return ncclInvalidArgument;

  struct gdaki_context *gdaki_ctx = (struct gdaki_context *)ginCtx;
  struct ncclGinIbCollComm *cComm = gdaki_ctx->collComm;
  const int nranks = cComm->nranks;
  const int ncontexts = gdaki_ctx->nContexts;
  const int nqps_per_rank = ncontexts;
  const int nqps_for_comm = nqps_per_rank * nranks;  // Number of QPs for communication
  const int ncompanion_qps = nqps_for_comm * 2;      // Number of companion QPs for communication
                                                     // Double because we connect to self.
  const int nqps =
    nqps_per_rank * (nranks + 1);  // +1 for the local rank.
                                   // The last group is the responder of the local rank.

  if (gdaki_ctx->gin_gdaki_gpu_ctx_hd_mhandle) {
    struct doca_gpu_verbs_qp **gverbs_qps = (struct doca_gpu_verbs_qp **)calloc(nranks, sizeof(struct doca_gpu_verbs_qp *));
    for (int ctx_idx = 0; ctx_idx < ncontexts; ctx_idx++) {
      struct ncclGinGdakiGPUContext *gin_gdaki_gpu_ctx =
        &gdaki_ctx->gin_gdaki_gpu_ctx_hd_mhandle->host_buf[ctx_idx];
      if (gin_gdaki_gpu_ctx->gdqp) {
        for (int qp_idx = 0; qp_idx < nranks; qp_idx++) {
          gverbs_qps[qp_idx] = gdaki_ctx->gqps[(ctx_idx * nranks) + qp_idx]->qp_gverbs;
        }
        DOCACHECK(doca_gpu_verbs_unexport_multi_qps_dev(gdaki_ctx->gdev, gverbs_qps, nranks, gin_gdaki_gpu_ctx->gdqp));
        gin_gdaki_gpu_ctx->gdqp = nullptr;
      }
      if (gin_gdaki_gpu_ctx->companion_gdqp) {
        for (int qp_idx = 0; qp_idx < nranks; qp_idx++) {
          gverbs_qps[qp_idx] = gdaki_ctx->companion_gqps[(ctx_idx * nranks) + qp_idx]->qp_gverbs;
        }
        DOCACHECK(doca_gpu_verbs_unexport_multi_qps_dev(gdaki_ctx->gdev, gverbs_qps, nranks, gin_gdaki_gpu_ctx->companion_gdqp));
        gin_gdaki_gpu_ctx->companion_gdqp = nullptr;
      }
    }
    free(gverbs_qps);
    NCCLCHECK(gdaki_ctx->gin_gdaki_gpu_ctx_hd_mhandle->deallocate());
    delete gdaki_ctx->gin_gdaki_gpu_ctx_hd_mhandle;
  }

  for (int qp_idx = 0; qp_idx < nqps_for_comm; qp_idx++) {
    DOCACHECK(doca_gpu_verbs_destroy_qp_group_hl(gdaki_ctx->gqp_groups[qp_idx]));
    gdaki_ctx->gqp_groups[qp_idx] = nullptr;
  }
  for (int qp_idx = nqps_for_comm; qp_idx < nqps; qp_idx++) {
    DOCACHECK(doca_gpu_verbs_destroy_qp_hl(gdaki_ctx->gqps[qp_idx]));
    gdaki_ctx->gqps[qp_idx] = nullptr;
  }
  for (int qp_idx = nqps_for_comm; qp_idx < ncompanion_qps; qp_idx++) {
    DOCACHECK(doca_gpu_verbs_destroy_qp_hl(gdaki_ctx->companion_gqps[qp_idx]));
    gdaki_ctx->companion_gqps[qp_idx] = nullptr;
  }

  if (gdaki_ctx->gqp_groups) free(gdaki_ctx->gqp_groups);
  if (gdaki_ctx->gqps) free(gdaki_ctx->gqps);
  if (gdaki_ctx->companion_gqps) free(gdaki_ctx->companion_gqps);

  if (gdaki_ctx->counters_table) {
    NCCLCHECK(gdaki_ctx->counters_table->deregister_mr());
    NCCLCHECK(gdaki_ctx->counters_table->deallocate());
    delete gdaki_ctx->counters_table;
  }
  if (gdaki_ctx->signals_table) {
    NCCLCHECK(gdaki_ctx->signals_table->deregister_mr());
    NCCLCHECK(gdaki_ctx->signals_table->deallocate());
    delete gdaki_ctx->signals_table;
  }

  if (gdaki_ctx->sink_buffer.mr) NCCLCHECK(wrap_ibv_dereg_mr(gdaki_ctx->sink_buffer.mr));
  if (gdaki_ctx->sink_buffer.addr) NCCLCHECK(ncclCuMemFree(gdaki_ctx->sink_buffer.addr, nullptr));

  if (gdaki_ctx->ah) {
    DOCACHECK(doca_verbs_ah_attr_destroy(gdaki_ctx->ah));
  }

  if (gdaki_ctx->gdev) {
    DOCACHECK(doca_gpu_destroy(gdaki_ctx->gdev));
  }
  if (gdaki_ctx->devHandle) free(gdaki_ctx->devHandle);

  memset(gdaki_ctx, 0, sizeof(*gdaki_ctx));
  free(gdaki_ctx);

  return ncclSuccess;
}

ncclResult_t ncclGinGdakiRegMrSym(void *collComm, void *data, size_t size, int type, uint64_t mr_flags, void **mhandle,
                                  void **ginHandle) {
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;
  ncclResult_t status = ncclSuccess;

  struct ibv_mr *mr = nullptr;
  GdakiHostGPUMemHandle<struct ncclGinGdakiMemHandle> *gdaki_mhandle_hd_mhandle =
    new GdakiHostGPUMemHandle<struct ncclGinGdakiMemHandle>();
  GdakiHostGPUMemHandle<__be32> *rkeys_hd_mhandle =
    new GdakiHostGPUMemHandle<__be32>();
  __be32 rkey;

  struct gdaki_mem_handle *gdaki_mhandle = nullptr;
  gdaki_mhandle = (struct gdaki_mem_handle *)calloc(1, sizeof(*gdaki_mhandle));
  EQCHECK(gdaki_mhandle, nullptr);

  bool force_strict_ordering = (mr_flags & NCCL_NET_MR_FLAG_FORCE_SO);
  NCCLCHECK(gdakiRegMr(&mr, cComm->ib.pd, data, size,
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                         IBV_ACCESS_REMOTE_ATOMIC, force_strict_ordering));

  rkey = htobe32(mr->rkey);
  NCCLCHECKGOTO(rkeys_hd_mhandle->allocate(cComm->nranks), status, out);
  NCCLCHECKGOTO(cComm->allGather(cComm, &rkey, rkeys_hd_mhandle->host_buf, sizeof(__be32)), status, out);
  NCCLCHECKGOTO(rkeys_hd_mhandle->copy_h_to_d(), status, out);

  NCCLCHECKGOTO(gdaki_mhandle_hd_mhandle->allocate(1), status, out);
  gdaki_mhandle_hd_mhandle->host_buf->rkeys = rkeys_hd_mhandle->gpu_buf;
  gdaki_mhandle_hd_mhandle->host_buf->lkey = htobe32(mr->lkey);
  NCCLCHECKGOTO(gdaki_mhandle_hd_mhandle->copy_h_to_d(), status, out);

  gdaki_mhandle->type = type;
  gdaki_mhandle->mr = mr;
  gdaki_mhandle->gdaki_mhandle_hd_mhandle = gdaki_mhandle_hd_mhandle;
  gdaki_mhandle->rkeys_hd_mhandle = rkeys_hd_mhandle;

  INFO(NCCL_NET, "[%d] Registered MR: data=%p, size=%zu, lkey(be32)=%#x, rkey(be32)=%#x",
       cComm->rank, data, size, htobe32(mr->lkey), htobe32(mr->rkey));

  *mhandle = (void *)gdaki_mhandle;
  *ginHandle = (void *)gdaki_mhandle_hd_mhandle->gpu_buf;

out:
  if (status != ncclSuccess) {
    delete gdaki_mhandle_hd_mhandle;
    delete rkeys_hd_mhandle;
  }
  return status;
}

ncclResult_t ncclGinGdakiDeregMrSym(void *collComm, void *mhandle) {
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;
  struct gdaki_mem_handle *gdaki_mhandle = (struct gdaki_mem_handle *)mhandle;
  struct ibv_mr *mr = gdaki_mhandle->mr;

  INFO(NCCL_NET, "[%d] Unregistering MR: lkey(be32)=%#x, rkey(be32)=%#x", cComm->rank,
       htobe32(mr->lkey), htobe32(mr->rkey));

  NCCLCHECK(wrap_ibv_dereg_mr(mr));

  NCCLCHECK(gdaki_mhandle->gdaki_mhandle_hd_mhandle->deallocate());
  delete gdaki_mhandle->gdaki_mhandle_hd_mhandle;
  NCCLCHECK(gdaki_mhandle->rkeys_hd_mhandle->deallocate());
  delete gdaki_mhandle->rkeys_hd_mhandle;

  memset(gdaki_mhandle, 0, sizeof(*gdaki_mhandle));

  free(gdaki_mhandle);

  return ncclSuccess;
}

// ---------------------------------------------------------------------------
// ncclGinGdakiProgress：GDAKI 后端的 CPU proxy progress 函数
// ---------------------------------------------------------------------------
// 由 GIN progress thread 周期性调用（gin_host.cc ncclGinProgress 主循环）。
//
// 背景：正常 GDAKI 模式下，GPU kernel 直接敲 NIC Doorbell 提交 WQE。
//       但当硬件不支持 GPU 直接访问 UAR（如跨 NUMA、PCIe P2P 限制）时，
//       GPU 将 Producer Index 写入 cpu_db（CPU-GPU 共享内存），
//       CPU proxy 从 cpu_db 读取后代劳敲 NIC Doorbell。
//
// QP 遍历顺序：gqps[nContexts * nranks]（main QP）+ companion_gqps 各一轮
//   main QP：数据面 QP（PUT/GET/Signal），每 context 每 peer 一个
//   companion QP：counter QP（WAIT/CQ_UPDATE），与 main QP 一一对应
//
// 轮询策略：repeat-until-idle
//   只要本轮有任何 QP 取得进展（progressed=true），就再轮询一遍
//   直到所有 QP 都无新 WQE，才返回（减少延迟，避免漏处理）
ncclResult_t ncclGinGdakiProgress(void *ctx) {
  struct gdaki_context *gdakiCtx = (struct gdaki_context *)ctx;
  const int ncontexts = gdakiCtx->nContexts;
  const int nranks = gdakiCtx->collComm->nranks;
  const int nqpsPerRank = ncontexts;
  const int nqpsForComm = nqpsPerRank * nranks;  // main/companion QP 总数
  bool has_progressed = true;
  bool progressed;

  while (has_progressed) {
    has_progressed = false;
    for (int qpIdx = 0; qpIdx < nqpsForComm; qpIdx++) {
      // 处理 main QP：检查 GPU 是否写了新的 Producer Index，如果是则代敲 Doorbell
      struct doca_gpu_verbs_qp *qp = gdakiCtx->gqps[qpIdx]->qp_gverbs;
      if (qp->cpu_proxy) {
        DOCACHECK(doca_gpu_verbs_cpu_proxy_progress(qp, &progressed));
        has_progressed |= progressed;
      }

      // 处理 companion QP：同理，companion QP 的 WAIT/CQ_UPDATE WQE 也可能需要代敲
      qp = gdakiCtx->companion_gqps[qpIdx]->qp_gverbs;
      if (qp->cpu_proxy) {
        DOCACHECK(doca_gpu_verbs_cpu_proxy_progress(qp, &progressed));
        has_progressed |= progressed;
      }
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclGinGdakiQueryLastError(void *ginCtx, bool *hasError) {
  struct gdaki_context *gdakiCtx = (struct gdaki_context *)ginCtx;
  bool hasError_ = false;
  const int ncontexts = gdakiCtx->nContexts;
  const int nranks = gdakiCtx->collComm->nranks;
  const int nqpsPerRank = ncontexts;
  const int nqpsForComm = nqpsPerRank * nranks;  // Number of QPs for communication

  // We throttle the frequency of these queries since they can easily take 250us.
  uint64_t now = clockNano();
  if ((now - gdakiCtx->last_error_query_time) / 1e9 < ncclParamGinErrorQuerySec()) {
    goto exit;
  }
  gdakiCtx->last_error_query_time = now;

  for (int qpIdx = 0; qpIdx < nqpsForComm; qpIdx++) {
    struct doca_gpu_verbs_qp *qp = gdakiCtx->gqps[qpIdx]->qp_gverbs;
    struct doca_gpu_verbs_qp_error_info errorInfo;
    DOCACHECK(doca_gpu_verbs_query_last_error(qp, &errorInfo));
    hasError_ |= errorInfo.has_error;
    if (hasError_) break;

    qp = gdakiCtx->companion_gqps[qpIdx]->qp_gverbs;
    DOCACHECK(doca_gpu_verbs_query_last_error(qp, &errorInfo));
    hasError_ |= errorInfo.has_error;
    if (hasError_) break;
  }
exit:
  *hasError = hasError_;
  return ncclSuccess;
}
