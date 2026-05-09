/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file doca_gpunetio_dev_verbs_cq.cuh
 * @brief GDAKI CUDA device functions for CQ management
 *
 * @{
 */
#ifndef DOCA_GPUNETIO_DEV_VERBS_CQ_H
#define DOCA_GPUNETIO_DEV_VERBS_CQ_H

#include <errno.h>

#include "doca_gpunetio_dev_verbs_common.cuh"

/**
 * @brief Return device CQ SQ pointer from a device QP
 *
 * @param[in] qp - Dev QP pointer
 *
 * @return Dev CQ pointer
 */
__device__ static __forceinline__ struct doca_gpu_dev_verbs_cq *doca_gpu_dev_verbs_qp_get_cq_sq(
    struct doca_gpu_dev_verbs_qp *qp) {
    return &(qp->cq_sq);
}

/**
 * @brief Increament and round up CQE id
 *
 * @param[in] cqe_idx - cqe idx
 * @param[in] increment - cqe idx increment
 *
 * @return cqe incremented idx
 */
__device__ static __forceinline__ uint32_t doca_gpu_dev_verbs_cqe_idx_inc_mask(uint32_t cqe_idx,
                                                                               uint32_t increment) {
    return (cqe_idx + increment) & DOCA_GPUNETIO_VERBS_CQE_CI_MASK;
}

#if DOCA_GPUNETIO_VERBS_ENABLE_DEBUG == 1
/**
 * @brief Print error CQE values
 *
 * @param[in] cqe64 - erroneous cqe
 *
 * @return
 */
__device__ static __forceinline__ void doca_gpu_dev_verbs_cq_print_cqe_err(
    struct doca_gpunetio_ib_mlx5_cqe64 *cqe64) {
    struct doca_gpunetio_ib_mlx5_err_cqe_ex *err_cqe =
        (struct doca_gpunetio_ib_mlx5_err_cqe_ex *)cqe64;

    printf(
        "got completion with err: "
        "syndrome=%#x, vendor_err_synd=%#x, "
        "hw_err_synd=%#x, hw_synd_type=%#x, wqe_counter=%u\n",
        err_cqe->syndrome, err_cqe->vendor_err_synd, err_cqe->hw_err_synd, err_cqe->hw_synd_type,
        doca_gpu_dev_verbs_bswap16(err_cqe->wqe_counter));
}
#endif

/**
 * @brief [Internal] Poll the Completion Queue (CQ) at a specific index.
 * This function does not update the SW consumer index nor guarantees the ordering.
 * It also does not wait for the completion to arrive.
 *
 * @param qp - Queue Pair (QP)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ int doca_priv_gpu_dev_verbs_poll_one_cq_at(
    struct doca_gpu_dev_verbs_cq *cq, uint64_t cons_index) {
    uint8_t *cqe = (uint8_t *)__ldg((uintptr_t *)&cq->cqe_daddr);
    const uint32_t cqe_num = __ldg(&cq->cqe_num);
    const uint64_t cqe_rsvd = __ldg(&cq->cqe_rsvd);
    uint64_t cons_index_in_cq = cons_index + cqe_rsvd;
    uint32_t idx = cons_index_in_cq & (cqe_num - 1);
    struct doca_gpunetio_ib_mlx5_cqe64 *cqe64 =
        (struct doca_gpunetio_ib_mlx5_cqe64 *)(cqe + (idx * DOCA_GPUNETIO_VERBS_CQE_SIZE));

    uint64_t cqe_ci = doca_gpu_dev_verbs_load_relaxed<resource_sharing_mode>(&cq->cqe_ci);

    if (cons_index < cqe_ci) return 0;
    if (cons_index >= cqe_ci + cqe_num) return EBUSY;

    uint8_t opown;
    uint8_t opcode;
    bool observed_completion;

#if __CUDA_ARCH__ >= 900
    opown = doca_gpu_dev_verbs_load_relaxed_sys_global((uint8_t *)&cqe64->op_own);

    observed_completion =
        !((opown & DOCA_GPUNETIO_IB_MLX5_CQE_OWNER_MASK) ^ !!(cons_index_in_cq & cqe_num));
#else
    uint32_t cqe_chunk;
    uint16_t wqe_counter;

    cqe_chunk = doca_gpu_dev_verbs_load_relaxed_sys_global((uint32_t *)&cqe64->wqe_counter);
    cqe_chunk = doca_gpu_dev_verbs_bswap32(cqe_chunk);
    wqe_counter = cqe_chunk >> 16;
    opown = cqe_chunk & 0xff;

    observed_completion =
        !((opown & DOCA_GPUNETIO_IB_MLX5_CQE_OWNER_MASK) ^ !!(cons_index_in_cq & cqe_num)) &&
        (wqe_counter == ((uint32_t)cons_index & 0xffff));
#endif

    if (!observed_completion) return EBUSY;

    opcode = opown >> DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT;

#if DOCA_GPUNETIO_VERBS_ENABLE_DEBUG == 1
    if (opcode == DOCA_GPUNETIO_IB_MLX5_CQE_REQ_ERR) doca_gpu_dev_verbs_cq_print_cqe_err(cqe64);
#endif
    return (opcode == DOCA_GPUNETIO_IB_MLX5_CQE_REQ_ERR) * -EIO;
}

/**
 * @brief Poll the Completion Queue (CQ) at a specific index. This function does
 * not wait for the completion to arrive.
 *
 * @param qp - Queue Pair (QP)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 * @return On success, doca_gpu_dev_verbs_poll_one_cq_at() returns 0. If the completion is
 * not available, returns EBUSY. If it is a completion with error, returns a
 * negative value.
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ int doca_gpu_dev_verbs_poll_one_cq_at(
    struct doca_gpu_dev_verbs_cq *cq, uint64_t cons_index) {
    int status =
        doca_priv_gpu_dev_verbs_poll_one_cq_at<resource_sharing_mode, qp_type>(cq, cons_index);
    if (status == 0) {
        doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
        doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode>(&cq->cqe_ci, cons_index + 1);
    }
    return status;
}

// ════════════════════════════════════════════════════════════════════════════
// doca_priv_gpu_dev_verbs_poll_cq_at（阻塞版 - 内部实现）
// ════════════════════════════════════════════════════════════════════════════
// 自旋等待 CQ 中第 cons_index 号 CQE 到达。这是 wait_until_slot_available
// 的核心引擎——通过检测 CQE 来确认 NIC 已完成对应 WQE。
//
// ★ CQ 环形缓冲区布局：
//   cqe_daddr ──▶ ┌─────────┐ CQE[0]  (64 bytes)
//                  ├─────────┤ CQE[1]
//                  │  ...    │
//                  ├─────────┤ CQE[cqe_num-1]
//                  └─────────┘
//   NIC 按顺序向 CQ 写入 CQE，每写一个就翻转该槽位的 ownership bit。
//
// ★ 算法：
//   1. 计算 CQE 在环形缓冲区中的物理位置：
//      cons_index_in_cq = cons_index + cqe_rsvd（初始偏移量）
//      idx = cons_index_in_cq & (cqe_num - 1)（取模）
//
//   2. 快速路径检测：如果 cons_index < cqe_ci（已被其他线程消费），直接返回成功
//
//   3. 自旋检测 CQE 到达（do-while 循环）：
//      - 读取 CQE 的 op_own 字段（NIC DMA 写入 GPU 显存）
//      - 检查 ownership bit 翻转：
//        (opown & OWNER_MASK) ^ !!(cons_index_in_cq & cqe_num)
//        CQ 每绕一圈，ownership 期望值翻转一次。
//        如果实际 ownership == 期望值 → NIC 已写入该 CQE
//      - 在旧架构上还检查 wqe_counter 字段是否匹配 cons_index
//        （确认不是旧 CQE 的残留数据）
//
//   4. 检测到 CQE 后检查 opcode：
//      如果是 CQE_REQ_ERR → 返回 -EIO（WQE 执行出错）
//      否则返回 0（成功）
//
// ★ 为什么用 load_relaxed_sys_global 读 CQE？
//   CQE 是 NIC 通过 PCIe DMA 写入 GPU 显存的，必须用 .sys 作用域
//   才能看到跨设备（NIC→GPU）的最新写入。
// ════════════════════════════════════════════════════════════════════════════
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ int doca_priv_gpu_dev_verbs_poll_cq_at(
    struct doca_gpu_dev_verbs_cq *cq, uint64_t cons_index) {
    struct doca_gpunetio_ib_mlx5_cqe64 *cqe =
        (struct doca_gpunetio_ib_mlx5_cqe64 *)__ldg((uintptr_t *)&cq->cqe_daddr);
    const uint32_t cqe_num = __ldg(&cq->cqe_num);
    const uint64_t cqe_rsvd = __ldg(&cq->cqe_rsvd);
    uint64_t cons_index_in_cq = cons_index + cqe_rsvd;
    uint32_t idx = cons_index_in_cq & (cqe_num - 1);
    struct doca_gpunetio_ib_mlx5_cqe64 *cqe64 = &cqe[idx];
    uint8_t opown;
    uint8_t opcode;
    uint64_t cqe_ci;
#if __CUDA_ARCH__ >= 900
    do {
        cqe_ci = doca_gpu_dev_verbs_load_relaxed<resource_sharing_mode>(&cq->cqe_ci);
        [[unlikely]] if (cons_index < cqe_ci) return 0;
        opown = doca_gpu_dev_verbs_load_relaxed_sys_global((uint8_t *)&cqe64->op_own);
    } while ((cons_index >= cqe_ci + cqe_num) ||
             ((cqe_ci <= cons_index) &&
              ((opown & DOCA_GPUNETIO_IB_MLX5_CQE_OWNER_MASK) ^ !!(cons_index_in_cq & cqe_num))));
#else
    uint32_t cqe_chunk;
    uint16_t wqe_counter;

    do {
        cqe_ci = doca_gpu_dev_verbs_load_relaxed<resource_sharing_mode>(&cq->cqe_ci);
        [[unlikely]] if (cons_index < cqe_ci) return 0;
        cqe_chunk = doca_gpu_dev_verbs_load_relaxed_sys_global((uint32_t *)&cqe64->wqe_counter);
        cqe_chunk = doca_gpu_dev_verbs_bswap32(cqe_chunk);
        wqe_counter = cqe_chunk >> 16;
        opown = cqe_chunk & 0xff;
    } while ((cons_index >= cqe_ci + cqe_num) ||
             ((cqe_ci <= cons_index) &&
              (((opown & DOCA_GPUNETIO_IB_MLX5_CQE_OWNER_MASK) ^ !!(cons_index_in_cq & cqe_num)) ||
               (wqe_counter != ((uint32_t)cons_index & 0xffff)))));
#endif

    opcode = opown >> DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT;

#if DOCA_GPUNETIO_VERBS_ENABLE_DEBUG == 1
    if (opcode == DOCA_GPUNETIO_IB_MLX5_CQE_REQ_ERR) doca_gpu_dev_verbs_cq_print_cqe_err(cqe64);
#endif
    return (opcode == DOCA_GPUNETIO_IB_MLX5_CQE_REQ_ERR) * -EIO;
}

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_poll_cq_at（阻塞版 - 公开接口）
// ════════════════════════════════════════════════════════════════════════════
// 等待第 cons_index 号 CQE 到达，成功后：
//   1. fence.acquire.sys → 确保后续读取能看到 NIC DMA 写入的数据
//   2. atomic_max(cqe_ci, cons_index + 1) → 推进 CQ 软件消费索引
//      使用 atomic_max 是因为多线程可能并发 poll 不同 CQE，
//      只需保证 cqe_ci 单调递增（不回退）
//
// 被 wait_until_slot_available 调用：
//   poll_cq_at(&qp->cq_sq, wqe_idx - nwqes)
//   → 等待第 (wqe_idx - nwqes) 号 WQE 的 CQE 到达
//   → 确认该 WQE 已完成 → 对应物理槽位可以重用
// ════════════════════════════════════════════════════════════════════════════
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ int doca_gpu_dev_verbs_poll_cq_at(
    struct doca_gpu_dev_verbs_cq *cq, uint64_t cons_index) {
    int status = doca_priv_gpu_dev_verbs_poll_cq_at<resource_sharing_mode, qp_type>(cq, cons_index);
    if (status == 0) {
        doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
        doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode>(&cq->cqe_ci, cons_index + 1);
    }
    return status;
}

/**
 * @brief [Internal] Poll the Collapsed Completion Queue (CQ) at a specific index. This function
 * waits for the completion to arrive.
 *
 * @param cq - Collapsed Completion Queue (CQ)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 * @param new_cqe_ci - New CQ Consumer Index
 * @return On success, doca_priv_gpu_dev_verbs_poll_cq_collapsed_at() returns 0. If it is a
 * completion with error, returns a negative value.
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ int doca_priv_gpu_dev_verbs_poll_cq_collapsed_at(
    struct doca_gpu_dev_verbs_qp *qp, struct doca_gpu_dev_verbs_cq *cq, uint64_t cons_index,
    uint64_t *new_cqe_ci) {
    struct doca_gpunetio_ib_mlx5_cqe64 *cqe64 = (struct doca_gpunetio_ib_mlx5_cqe64 *)__ldg((uintptr_t *)&cq->cqe_daddr);
    const uint32_t cqe_num = __ldg(&cq->cqe_num);
    uint8_t opown;
    uint8_t opcode;
    uint64_t cqe_ci;
    uint32_t cqe_chunk;
    uint16_t wqe_counter;

    // If idx is a lot greater than cons_idx, we might get incorrect result due
    // to wqe_counter wraparound. We need to check prod_idx to be sure that idx
    // has already been submitted.
    while (doca_gpu_dev_verbs_atomic_read<uint64_t, resource_sharing_mode>(
               &qp->sq_wqe_pi) < cons_index);
    doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();

    do {
        cqe_ci = doca_gpu_dev_verbs_load_relaxed<resource_sharing_mode>(&cq->cqe_ci);
        [[unlikely]] if (cons_index < cqe_ci) return 0;
        cqe_chunk = doca_gpu_dev_verbs_load_relaxed_sys_global((uint32_t *)&cqe64->wqe_counter);
        cqe_chunk = doca_gpu_dev_verbs_bswap32(cqe_chunk);
        wqe_counter = cqe_chunk >> 16;
        opown = cqe_chunk & 0xff;
        opcode = opown >> DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT;
    }
    // NOTE: This while loop is part of do while above.
    // wqe_counter is the HW consumer index. However, we always maintain index
    // + 1 in SW. To be able to compare with idx, we need to use wqe_counter +
    // 1. Because wqe_counter is uint16_t, it may wraparound. Still we know for
    // sure that if idx - wqe_counter - 1 < ncqes, wqe_counter + 1 is less than
    // idx, and thus we need to wait. We don't need to wait when idx ==
    // wqe_counter + 1. That's why we use - (uint16_t)2 here to make this case
    // wraparound.
    while ((opcode == DOCA_GPUNETIO_IB_MLX5_CQE_INVALID) ||
           ((cqe_ci <= cons_index) &&
            ((uint16_t)((uint16_t)cons_index - wqe_counter - (uint16_t)2) < cqe_num)));

    ++wqe_counter;
    *new_cqe_ci = (cons_index & ~(0xFFFFULL) | wqe_counter) +
                  (((uint16_t)cons_index > wqe_counter) ? 0x10000ULL : 0x0);

#if DOCA_GPUNETIO_VERBS_ENABLE_DEBUG == 1
    if (opcode == DOCA_GPUNETIO_IB_MLX5_CQE_REQ_ERR) doca_gpu_dev_verbs_cq_print_cqe_err(cqe64);
#endif

    return ((opcode == DOCA_GPUNETIO_IB_MLX5_CQE_REQ_ERR) * -EIO);
}

/**
 * @brief Poll the Completion Queue (CQ) at a specific index. This function waits for the completion
 * to arrive.
 *
 * @param qp - Queue Pair (QP)
 * @param cons_index - Index of the Completion Queue (CQ) to be polled
 * @return On success, doca_gpu_dev_verbs_poll_cq_collapsed_at() returns 0. If it is a completion
 * with error, returns a negative value.
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ int doca_gpu_dev_verbs_poll_cq_collapsed_at(
    struct doca_gpu_dev_verbs_qp *qp, uint64_t cons_index) {
    uint64_t new_cqe_ci = 0;
    int status = doca_priv_gpu_dev_verbs_poll_cq_collapsed_at<resource_sharing_mode, qp_type>(
        qp, doca_gpu_dev_verbs_qp_get_cq_sq(qp), cons_index, &new_cqe_ci);
    if (status == 0) {
        doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
        doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode>(
            &(doca_gpu_dev_verbs_qp_get_cq_sq(qp)->cqe_ci), new_cqe_ci);
    }
    return status;
}

/**
 * @brief Poll the Completion Queue (CQ). This function waits for the completion to arrive.
 *
 * @param qp - Queue Pair (QP)
 * @param count - Number of completions to poll
 * @return On success, doca_gpu_dev_verbs_poll_cq() returns 0. If it is a completion with
 * error, returns a negative value.
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ int doca_gpu_dev_verbs_poll_cq(struct doca_gpu_dev_verbs_cq *cq,
                                                                 uint32_t count) {
    [[unlikely]] if (count == 0) return 0;
    uint64_t cons_index =
        doca_gpu_dev_verbs_load_relaxed<resource_sharing_mode>(&cq->cqe_ci) + count - 1;
    return doca_gpu_dev_verbs_poll_cq_at<resource_sharing_mode, qp_type>(cq, cons_index);
}

/**
 * @brief Increment CQ DBREC
 *
 * @param[in] cq - GPU Completion Queue
 * @param[in] cqe_num - CQE num to increment
 *
 * @return new CQE consumer index
 */
template <bool is_overrun>
__device__ static __forceinline__ uint32_t
doca_gpu_dev_verbs_cq_update_dbrec(struct doca_gpu_dev_verbs_cq *cq, uint32_t cqe_num) {
    uint32_t cqe_ci = DOCA_GPUNETIO_VOLATILE(cq->cqe_ci);

    cqe_ci = (cqe_ci + cqe_num) & DOCA_GPUNETIO_VERBS_CQE_CI_MASK;
    if (is_overrun == false) {
        asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;"
                     :
                     : "l"(cq->dbrec), "r"(doca_gpu_dev_verbs_bswap32(cqe_ci)));
    }

    DOCA_GPUNETIO_VOLATILE(cq->cqe_ci) = cqe_ci;

    return cqe_ci;
}

#endif /* DOCA_GPUNETIO_DEV_VERBS_CQ_H */

/** @} */
