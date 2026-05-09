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
 * @file doca_gpunetio_dev_verbs_qp.cuh
 * @brief GDAKI CUDA device functions for QP management
 *
 * @{
 */
#ifndef DOCA_GPUNETIO_DEV_VERBS_QP_H
#define DOCA_GPUNETIO_DEV_VERBS_QP_H

#include <cuda/atomic>
#include "doca_gpunetio_dev_verbs_cq.cuh"

/* *********** WQE UTILS *********** */

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_store_wqe_seg
// ════════════════════════════════════════════════════════════════════════════
// 将 16 字节（2 × uint64_t）原子写入 WQE 缓冲区的一个 segment 位置。
//
// 使用 PTX 汇编指令 st.weak.cs.v2.b64：
//   st     = store（写入）
//   .weak  = 不保证 ordering（不是 release store），性能最优
//   .cs    = cache streaming hint（绕过 L1 cache，直接写到 L2/显存）
//            对 MMIO 映射的 WQE 缓冲区来说避免 cache 污染
//   .v2    = vector of 2（一次写两个 64-bit 值 = 16 字节）
//   .b64   = 每个元素 64 bit
//
// 为什么是 16 字节一写？
//   WQE 每个 segment（cseg/rseg/dseg）恰好 16 字节 = 128 bit。
//   一次 128-bit store 保证该 segment 的写入对 NIC 而言是原子可见的，
//   不会出现 NIC 读到半写的 segment。
//
// 参数：
//   ptr - 目标地址：WQE 缓冲区中某个 segment 的起始指针（16 字节对齐）
//   val - 源数据：要写入的 16 字节内容（如 cseg、rseg、dseg 的内存表示）
// ════════════════════════════════════════════════════════════════════════════
__device__ static __forceinline__ void doca_gpu_dev_verbs_store_wqe_seg(uint64_t *ptr,
                                                                        uint64_t *val) {
    asm volatile("st.weak.cs.v2.b64 [%0], {%1, %2};" : : "l"(ptr), "l"(val[0]), "l"(val[1]));
}

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_get_wqe_ptr
// ════════════════════════════════════════════════════════════════════════════
// 获取 SQ 环形缓冲区中第 wqe_idx 个 WQE 的指针。
//
// SQ（Send Queue）在 GPU 显存中的布局：
//   sq_wqe_daddr ──▶ ┌───────────┐ WQE[0]   (64 bytes)
//                    ├───────────┤ WQE[1]   (64 bytes)
//                    ├───────────┤ WQE[2]
//                    │   ...     │
//                    ├───────────┤ WQE[N-1]  ← sq_wqe_num 个槽位
//                    └───────────┘
//                    （环形：WQE[N] 回绕到 WQE[0]）
//
// 地址计算：
//   idx = wqe_idx & sq_wqe_mask       ← 取模（环形回绕，mask = sq_wqe_num - 1）
//   addr = sq_wqe_daddr + (idx << 6)  ← 每个 WQE 64 字节 = 2^6，SQ_SHIFT = 6
//
// __ldg() 是 CUDA 的只读 cache load（通过 texture cache），
// 对 QP 元数据这类不变量使用只读 cache 避免 L1 竞争。
//
// 参数：
//   qp      - QP 结构体，包含 SQ 的基址和大小
//   wqe_idx - 全局递增的 WQE 索引（可能 > sq_wqe_num，自动取模）
// 返回：
//   该 WQE 在 GPU 显存中的指针
// ════════════════════════════════════════════════════════════════════════════
__device__ static __forceinline__ struct doca_gpu_dev_verbs_wqe *doca_gpu_dev_verbs_get_wqe_ptr(
    struct doca_gpu_dev_verbs_qp *qp, uint16_t wqe_idx) {
    const uint16_t nwqes_mask = __ldg(&qp->sq_wqe_mask);  // sq_wqe_num - 1（例 127 = 0x7F）
    const uintptr_t wqe_addr = __ldg((uintptr_t *)&qp->sq_wqe_daddr);  // SQ 基址
    const uint16_t idx = wqe_idx & nwqes_mask;  // 环形取模
    return (struct doca_gpu_dev_verbs_wqe *)(wqe_addr +
                                             (idx << DOCA_GPUNETIO_IB_MLX5_WQE_SQ_SHIFT));
    // DOCA_GPUNETIO_IB_MLX5_WQE_SQ_SHIFT = 6 → 每个 WQE = 2^6 = 64 字节
}

/* *********** WQE SHARING *********** */

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_wait_until_slot_available
// ════════════════════════════════════════════════════════════════════════════
// 等待 SQ 环形缓冲区中第 wqe_idx 个槽位可用（即 NIC 已消费完该位置的旧 WQE）。
//
// ★ 为什么需要等待？
//   SQ 是固定大小的环形缓冲区（sq_wqe_num 个槽位，典型值 128）。
//   wqe_idx 是全局递增的（0, 1, 2, ...），当 wqe_idx >= sq_wqe_num 时，
//   该槽位在物理缓冲区中会回绕到 wqe_idx % sq_wqe_num，覆盖旧 WQE。
//   如果旧 WQE 还没被 NIC 执行完就覆盖，会导致数据损坏。
//
// ★ 为什么要 poll CQ？
//   IB 规范中，只有设置了 signaled 标志（fm_ce_se 中的 CQ_UPDATE）的 WQE
//   完成后才会在本地 Send CQ 中产生 CQE；未设置 signaled 的 WQE 完成时无通知。
//
//   在 DOCA GPUNetIO 的实现中，所有通过 put_signal_counter / put_counter 等
//   API 提交的 WQE 都设置了 CQ_UPDATE 标志，因此每个 WQE 完成后都会产生
//   一个 CQE，使得 WQE 索引与 CQE 索引形成一一对应关系。
//
//   poll_cq_at(cq, wqe_idx - nwqes) 的含义是：
//     等待 CQ 中出现第 (wqe_idx - nwqes) 号 WQE 对应的 CQE，
//     即该旧 WQE 已完成 → 它占用的物理槽位已释放，可以被新 WQE 覆盖。
//
//   ★ 前提条件：此机制依赖每个 WQE 都是 signaled (CQ_UPDATE) 的！
//     如果某个 WQE 没设置 CQ_UPDATE，它完成后不产生 CQE，
//     poll_cq_at 就无法检测到它完成，会导致死等。
//
// 示例（sq_wqe_num = 128）：
//   wqe_idx = 130 → 物理槽位 = 130 & 127 = 2 → 需要确认旧 WQE[2] 已完成
//   poll_cq_at(cq, 130 - 128 = 2) → 自旋等待 CQE[2] 出现
//
// 如果 wqe_idx < nwqes，说明还没用完一圈，无需等待。
// ════════════════════════════════════════════════════════════════════════════
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ void doca_gpu_dev_verbs_wait_until_slot_available(
    struct doca_gpu_dev_verbs_qp *qp, uint64_t wqe_idx) {
    const uint16_t nwqes = __ldg(&qp->sq_wqe_num);  // SQ 容量（如 128）
    // 只有当 wqe_idx 超过一整圈时才需等待（环形缓冲区满了）
    // [[likely]] = C++20 属性，提示编译器此分支大概率成立（通常 SQ 在第一圈就会用完）
    [[likely]] if (wqe_idx >= nwqes) doca_gpu_dev_verbs_poll_cq_at<resource_sharing_mode, qp_type>(
        &(qp->cq_sq), wqe_idx - nwqes);
    // cq_sq：QP 的 SQ 关联 CQ，NIC 每完成一个 WQE 就在这里写一个 CQE
}

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_reserve_wq_slots
// ════════════════════════════════════════════════════════════════════════════
// 在 SQ 中原子预留 count 个连续的 WQE 槽位，返回第一个槽位的全局索引。
//
// ★ 核心机制：sq_rsvd_index
//   sq_rsvd_index 是一个 GPU 显存中的 uint64_t 计数器，初始为 0。
//   每次预留时 atomic_add(sq_rsvd_index, count)，返回旧值作为起始索引。
//   多个 GPU 线程并发调用时，atomic_add 保证每个线程拿到不重叠的索引范围。
//
//   例：3 个线程同时各预留 3 个槽位：
//     线程 A：atomic_add(0, 3) → 返回 0，预留 [0,1,2]
//     线程 B：atomic_add(3, 3) → 返回 3，预留 [3,4,5]
//     线程 C：atomic_add(6, 3) → 返回 6，预留 [6,7,8]
//
// ★ 为什么 atomic_add 就能 reserve？
//   因为 SQ 是纯生产者-消费者模型：
//   - 生产者（GPU 线程）：只需要原子递增 sq_rsvd_index 就能"占位"
//   - 消费者（NIC 硬件）：通过 Doorbell 得知新 WQE 后按索引顺序消费
//   只要占位不重叠，线程就可以并行填充各自的 WQE，互不干扰。
//
// ★ wait_until_slot_available 的作用？
//   如果 SQ 已满（预留的索引超过了 NIC 消费速度），需要等旧 WQE 完成释放槽位。
//   SKIP_AVAILABILITY_CHECK 标志可跳过此检查（调用者自行保证不满）。
// ════════════════════════════════════════════════════════════════════════════
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ uint64_t
doca_gpu_dev_verbs_reserve_wq_slots(struct doca_gpu_dev_verbs_qp *qp, uint32_t count,
                                    uint32_t code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT) {
    // 原子加：获取当前 sq_rsvd_index 的旧值，并将其推进 count
    uint64_t wqe_idx =
        doca_gpu_dev_verbs_atomic_add<uint64_t, resource_sharing_mode>(&qp->sq_rsvd_index, count);
    // 检查预留的最后一个槽位是否可用（环形缓冲区可能满了）
    if (!(code_opt & DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_AVAILABILITY_CHECK))
        doca_gpu_dev_verbs_wait_until_slot_available<resource_sharing_mode>(qp,
                                                                            wqe_idx + count - 1);
    return wqe_idx;  // 返回第一个预留槽位的全局索引
}

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_common_mark_wqes_ready
// ════════════════════════════════════════════════════════════════════════════
// 标记 WQE[from_wqe_idx..to_wqe_idx] 为"可消费"状态。
//
// ★ 三级流水线：reserve → ready → submit
//   1. reserve_wq_slots：线程原子占位，拿到索引范围
//   2. (线程填充 WQE 内容)
//   3. mark_wqes_ready：告诉系统"我的 WQE 填好了"  ← 就是这个函数
//   4. submit（敲 Doorbell）：通知 NIC 开始处理
//
// ★ 为什么需要 ready_index？
//   多线程并发时，线程 A 可能占了 [0,1,2]，线程 B 占了 [3,4,5]。
//   如果 B 先填完就先 mark ready，NIC 可能看到 [3,4,5] 已 ready 但 [0,1,2] 还没填好。
//   NIC 是按序消费的，所以必须保证 ready_index 严格递增：
//     B 必须等 A 先 mark [0..2] ready 后，才能把 ready_index 推进到 6。
//
// 实现方式（GPU 模式 + CAS）：
//   1. fence.release → 确保线程填写的 WQE 数据对其他线程可见
//   2. atomicCAS(ready_index, from_wqe_idx, to_wqe_idx + 1)
//      CAS 的含义：如果 ready_index == from_wqe_idx（轮到我了），
//      则设为 to_wqe_idx + 1；否则自旋等待前面的线程完成
//   3. fence.acquire → 确保后续操作看到最新状态
// ════════════════════════════════════════════════════════════════════════════
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_qp_ready_mode ready_mode = DOCA_GPUNETIO_VERBS_READY_MODE_DEFAULT>
__device__ static __forceinline__ void doca_gpu_dev_common_mark_wqes_ready(uint64_t &ready_index,
                                                                           uint64_t from_wqe_idx,
                                                                           uint64_t to_wqe_idx) {
    if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE)
        ready_index = to_wqe_idx + 1;
    else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA) {
        if (ready_mode == DOCA_GPUNETIO_VERBS_READY_MODE_ATOMIC_CAS) {
            doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA>();
            while (atomicCAS_block((unsigned long long int *)&ready_index,
                                   (unsigned long long int)from_wqe_idx,
                                   (unsigned long long int)(to_wqe_idx + 1)) !=
                   (unsigned long long int)from_wqe_idx)
                continue;
            doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA>();
        } else {
            doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA>();
            cuda::atomic_ref<uint64_t, cuda::thread_scope_block> ready_index_aref(ready_index);
            while (ready_index_aref.load(cuda::memory_order_relaxed) != from_wqe_idx) continue;
            doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA>();
            ready_index_aref.store(to_wqe_idx + 1, cuda::memory_order_relaxed);
        }
    } else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU) {
        if (ready_mode == DOCA_GPUNETIO_VERBS_READY_MODE_DEFAULT ||
            ready_mode == DOCA_GPUNETIO_VERBS_READY_MODE_ATOMIC_CAS) {
            doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
            while (atomicCAS((unsigned long long int *)&ready_index,
                             (unsigned long long int)from_wqe_idx,
                             (unsigned long long int)(to_wqe_idx + 1)) !=
                   (unsigned long long int)from_wqe_idx)
                continue;
            doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
        } else {
            doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
            cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ready_index_aref(ready_index);
            while (ready_index_aref.load(cuda::memory_order_relaxed) != from_wqe_idx) continue;
            doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
            ready_index_aref.store(to_wqe_idx + 1, cuda::memory_order_relaxed);
        }
    }
}

/**
 * @brief Mark the WQEs in the range [from_wqe_idx, to_wqe_idx] as ready.
 *
 * @param qp - Queue Pair (QP)
 * @param from_wqe_idx - Starting WQE index
 * @param to_wqe_idx - Ending WQE index
 */
template <
    enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
    enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ,
    enum doca_gpu_dev_verbs_qp_ready_mode ready_mode = DOCA_GPUNETIO_VERBS_READY_MODE_ATOMIC_CAS>
__device__ static __forceinline__ void doca_gpu_dev_verbs_mark_wqes_ready(
    struct doca_gpu_dev_verbs_qp *qp, uint64_t from_wqe_idx, uint64_t to_wqe_idx) {
    doca_gpu_dev_common_mark_wqes_ready<resource_sharing_mode, ready_mode>(
        qp->sq_ready_index, from_wqe_idx, to_wqe_idx);
}

/* *********** QP DBR/DB *********** */

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_prepare_dbr
// ════════════════════════════════════════════════════════════════════════════
// 构造 32-bit DBR（Doorbell Record）值。
//
// DBR 是 NIC 在 GPU/CPU 内存中映射的一块 4 字节区域，NIC 通过 DMA 读取。
// 当 NIC 触发"missed doorbell"恢复路径时，会去读 DBR 获取最新的 prod_index。
//
// DBR 格式（MLX5 PRM 定义）：
//   bits [15:0] = prod_index 的低 16 位（大端序）
//   bits [31:16] = 0（保留）
//
// 实现：
//   1. mask1 = 0xFFFF → 取 prod_index 的低 16 位
//   2. and.b32 → dbrec_head_16b = prod_index & 0xFFFF
//   3. prmt.b32 + mask2=0x123 → 字节序翻转为大端（等价于 HTOBE32）
//      prmt 的 mask 0x123：选择字节顺序 [1][2][3][?]（实际只有低 2 字节有效）
//
// 为什么不直接用 bswap32？
//   因为 DBR 只取低 16 位然后大端存储，prmt 一条指令完成 mask + swap，
//   比先 AND 再 bswap 少一条指令。
// ════════════════════════════════════════════════════════════════════════════
__device__ static __forceinline__ __be32 doca_gpu_dev_verbs_prepare_dbr(uint32_t prod_index) {
    __be32 dbrec_val;

    // This is equivalent to
    // HTOBE32(dbrec_head & 0xffff);
    asm volatile(
        "{\n\t"
        ".reg .b32 mask1;\n\t"
        ".reg .b32 dbrec_head_16b;\n\t"
        ".reg .b32 ign;\n\t"
        ".reg .b32 mask2;\n\t"
        "mov.b32 mask1, 0xffff;\n\t"
        "mov.b32 mask2, 0x123;\n\t"
        "and.b32 dbrec_head_16b, %1, mask1;\n\t"      // 取低 16 位
        "prmt.b32 %0, dbrec_head_16b, ign, mask2;\n\t" // 字节重排为大端
        "}"
        : "=r"(dbrec_val)
        : "r"(prod_index));

    return dbrec_val;
}

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_common_update_dbr
// ════════════════════════════════════════════════════════════════════════════
// 将 DBR 值写入 NIC 的 Doorbell Record 内存地址。
//
// dbrec 是 NIC 通过 DMA 可达的 GPU/CPU 内存地址（init 时注册的 BAR 映射或 host memory）。
// 使用 cuda::atomic_ref<thread_scope_system> + relaxed store：
//   - thread_scope_system：确保写入对 NIC（PCIe 设备）可见
//   - relaxed：不保证 ordering（调用者在外层已做 fence_release）
//
// 为什么不用 st.mmio？
//   DBR 不在 MMIO BAR 中，它在 host 或 GPU 显存中（NIC 通过 DMA 读取），
//   所以用普通的 system scope store 即可。MMIO 只用于 BlueFlame Doorbell 寄存器。
// ════════════════════════════════════════════════════════════════════════════
__device__ static __forceinline__ void doca_gpu_dev_common_update_dbr(uint32_t *dbrec,
                                                                      uint32_t prod_index) {
    uint32_t dbrec_val = doca_gpu_dev_verbs_prepare_dbr(prod_index);

    // system scope relaxed store → NIC 可通过 DMA 读取到此值
    cuda::atomic_ref<uint32_t, cuda::thread_scope_system> dbrec_ptr_aref(*dbrec);
    dbrec_ptr_aref.store(dbrec_val, cuda::memory_order_relaxed);
}

// ════════════════════════════════════════════════════════════════════════════
// doca_priv_gpu_dev_verbs_update_dbr
// ════════════════════════════════════════════════════════════════════════════
// 从 QP 结构体获取 DBR 地址并更新。
// __ldg 加载 sq_dbrec（DBR 内存地址，init 时设置，运行时不变）。
// 这是内部函数（priv），不保证 ordering — 调用者负责 fence。
// ════════════════════════════════════════════════════════════════════════════
template <enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ void doca_priv_gpu_dev_verbs_update_dbr(
    struct doca_gpu_dev_verbs_qp *qp, uint32_t prod_index) {
    doca_gpu_dev_common_update_dbr((uint32_t *)__ldg((uintptr_t *)&qp->sq_dbrec), prod_index);
}

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_update_dbr（公开版本，带 fence）
// ════════════════════════════════════════════════════════════════════════════
// 包含 fence_release 的 DBR 更新。用于单独调用场景（不在 lock 内时需要自带 fence）。
// submit_db_multi_qps 内部调用的是 priv 版本（因为外层已有 fence）。
// ════════════════════════════════════════════════════════════════════════════
template <enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
          enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ void doca_gpu_dev_verbs_update_dbr(
    struct doca_gpu_dev_verbs_qp *qp, uint32_t prod_index,
    uint32_t code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT) {
    __be32 dbrec_val = doca_gpu_dev_verbs_prepare_dbr(prod_index);
    __be32 *dbrec_ptr = (__be32 *)__ldg((uintptr_t *)&qp->sq_dbrec);

#ifdef DOCA_GPUNETIO_VERBS_HAS_ASYNC_STORE_RELEASE
    if (code_opt & DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_ASYNC_STORE_RELEASE) {
        // Hopper+ 合并 fence + store 为单条异步指令
        doca_gpu_dev_verbs_async_store_release(dbrec_ptr, dbrec_val);
    } else
#endif
    {
        // 标准路径：先 fence 确保 WQE 可见，再写 DBR
        doca_gpu_dev_verbs_fence_release<sync_scope>();
        doca_priv_gpu_dev_verbs_update_dbr<qp_type>(qp, prod_index);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_common_prepare_db
// ════════════════════════════════════════════════════════════════════════════
// 构造 64-bit Doorbell (DB) 值。
//
// DB 值的格式（写入 BlueFlame MMIO 寄存器的 64 位数据）：
//   这实际上是 WQE 控制段 (ctrl_seg) 的前 8 字节，NIC 只检查其中的：
//   - qpn_ds：QP 编号（告诉 NIC 是哪个 QP 有新 WQE）
//   - opmod_idx_opcode 中的 idx 字段：prod_index（告诉 NIC 处理到第几个 WQE）
//
// 编码方式：
//   ctrl_seg.qpn_ds = qpn_ds（已预计算好的 QPN << 8 | ds_count 大端值）
//   ctrl_seg.opmod_idx_opcode = bswap32(prod_index << WQE_IDX_SHIFT)
//     WQE_IDX_SHIFT = 8 → prod_index 放在 bits[23:8]
//
// 返回：64-bit 值 = *(uint64_t*)&ctrl_seg = { opmod_idx_opcode, qpn_ds }
//   直接写入 BlueFlame MMIO 地址即可触发 NIC 开始处理新 WQE
// ════════════════════════════════════════════════════════════════════════════
__device__ static __forceinline__ __be64 doca_gpu_dev_common_prepare_db(uint32_t qpn_ds,
                                                                        uint64_t prod_index) {
    struct doca_gpunetio_ib_mlx5_wqe_ctrl_seg ctrl_seg = {0};

    // The only ctrl segment fields that are inspected while ringing
    // the DB are QP number and WQE index
    ctrl_seg.qpn_ds = qpn_ds;
    ctrl_seg.opmod_idx_opcode =
        doca_gpu_dev_verbs_bswap32((prod_index << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT));

    return *(uint64_t *)&ctrl_seg;  // 前 8 字节 = DB 值
}

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_common_write_db
// ════════════════════════════════════════════════════════════════════════════
// 将 64-bit DB 值写入 BlueFlame MMIO 地址。
//
// 这是所有 ring_db 的核心 "最后一跳"：
//   fence_release → store_relaxed_mmio（或 async_store_release）
//
// ★ BlueFlame 是什么？
//   MLX5 NIC 的一个特性：通过 PCIe BAR（MMIO 映射区域）直接写入 WQE 控制段，
//   NIC 硬件立即解析并开始处理。比传统 Doorbell 方式快 ~100ns。
//   传统 Doorbell：GPU 写 DB → NIC 检测 → NIC 去 GPU 显存 DMA 读 WQE
//   BlueFlame：GPU 直接把 ctrl_seg 数据推送到 NIC → NIC 立即有数据
//
// 为什么 fence_release 在 store 之前？
//   确保 GPU 先前写入的 WQE 内容（store_wqe_seg 的 st.weak.cs）
//   在 NIC 解析 DB 值时已经可见。否则 NIC 可能看到旧 WQE 数据。
// ════════════════════════════════════════════════════════════════════════════
template <enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>
__device__ static __forceinline__ void doca_gpu_dev_common_write_db(
    uint64_t *db_ptr, uint64_t db_val,
    uint32_t code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT) {
#ifdef DOCA_GPUNETIO_VERBS_HAS_ASYNC_STORE_RELEASE
    if (code_opt & DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_ASYNC_STORE_RELEASE) {
        // Hopper+ 单指令完成 fence + MMIO store
        doca_gpu_dev_verbs_async_store_release((uint64_t *)db_ptr, db_val);
    } else
#endif
#ifdef DOCA_GPUNETIO_VERBS_HAS_STORE_RELAXED_MMIO
    {
        doca_gpu_dev_verbs_fence_release<sync_scope>();
        // st.mmio.relaxed.sys.global：写 PCIe BAR 地址（BlueFlame 寄存器）
        doca_gpu_dev_verbs_store_relaxed_mmio(db_ptr, db_val);
    }
#else
    {
        // 回退：通过 cuda::atomic_ref<sys> relaxed store 写 MMIO
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> db_ptr_aref(*((uint64_t *)db_ptr));
        doca_gpu_dev_verbs_fence_release<sync_scope>();
        db_ptr_aref.store(db_val, cuda::memory_order_relaxed);
    }
#endif
}

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_common_ring_db
// ════════════════════════════════════════════════════════════════════════════
// 组合操作：prepare_db + write_db = 构造 DB 值并写入 MMIO。
// 是 ring_db 的底层实现（接收原始 qpn_ds + prod_index）。
// ════════════════════════════════════════════════════════════════════════════
template <enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>
__device__ static __forceinline__ void doca_gpu_dev_common_ring_db(
    uint64_t *db_ptr, uint32_t qpn_ds, uint64_t prod_index,
    uint32_t code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT) {
    uint64_t db_val = doca_gpu_dev_common_prepare_db(qpn_ds, prod_index);
    doca_gpu_dev_common_write_db<sync_scope>(db_ptr, db_val, code_opt);
}

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_prepare_db
// ════════════════════════════════════════════════════════════════════════════
// 从 QP 结构体获取 qpn_ds（预计算的 QPN 大端值）并构造 DB 值。
// sq_num_shift8_be = QPN << 8 的大端表示（init 时预计算，避免运行时每次 bswap）。
// ════════════════════════════════════════════════════════════════════════════
__device__ static __forceinline__ __be64
doca_gpu_dev_verbs_prepare_db(struct doca_gpu_dev_verbs_qp *qp, uint64_t prod_index) {
    uint32_t qpn_ds = __ldg(&qp->sq_num_shift8_be);  // 预计算的 QPN 大端值
    return doca_gpu_dev_common_prepare_db(qpn_ds, prod_index);
}

/* *************************** Ring Doorbell *************************** */

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_ring_db
// ════════════════════════════════════════════════════════════════════════════
// 从 QP 结构体获取 BlueFlame MMIO 地址并敲 Doorbell。
// 完整流程：prepare_db(QPN, prod_index) → write_db(sq_db MMIO 地址, DB 值)
//
// sq_db 指向 NIC 的 BlueFlame UAR（User Access Region）MMIO 地址：
//   这是 NIC PCIe BAR 中映射到 GPU 地址空间的一个 8 字节寄存器。
//   GPU 向此地址写入 DB 值后，NIC 硬件立即接收并开始处理对应 QP 的新 WQE。
// ════════════════════════════════════════════════════════════════════════════
template <enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>
__device__ static __forceinline__ void doca_gpu_dev_verbs_ring_db(
    struct doca_gpu_dev_verbs_qp *qp, uint64_t prod_index,
    uint32_t code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT) {
    uint64_t db_val = doca_gpu_dev_verbs_prepare_db(qp, prod_index);
    // __ldg 加载 sq_db（BlueFlame MMIO 地址），写入 DB 值
    doca_gpu_dev_common_write_db<sync_scope>((uint64_t *)__ldg((uintptr_t *)&qp->sq_db), db_val,
                                             code_opt);
}

#ifdef DOCA_GPUNETIO_VERBS_HAS_TMA_COPY
/**
 * @brief Ring the BF (BlueFlame). Requires shared memory.
 *
 * @param qp - Queue Pair (QP)
 * @param wqe - WQE to be ringed. This buffer must be in shared memory.
 */
template <enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>
__device__ static __forceinline__ void doca_gpu_dev_verbs_ring_bf(
    struct doca_gpu_dev_verbs_qp *qp, struct doca_gpu_dev_verbs_wqe *wqe_ptr) {
    void *bf_ptr = (void *)__ldg((uintptr_t *)&qp->sq_db);
    uint64_t *wqe = (uint64_t *)wqe_ptr;

    doca_gpu_dev_verbs_fence_release<sync_scope>();
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 64;"
                 :
                 : "l"(bf_ptr), "l"(wqe));
}
#endif

/**
 * @brief Ring the BF (BlueFlame). Requires at least 8 threads in the warp.
 *
 * @param qp - Queue Pair (QP)
 * @param wqe - WQE to be ringed
 */
template <enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>
__device__ static __forceinline__ void doca_gpu_dev_verbs_ring_bf_warp(
    struct doca_gpu_dev_verbs_qp *qp, struct doca_gpu_dev_verbs_wqe *wqe_ptr) {
    unsigned int lane_id = doca_gpu_dev_verbs_get_lane_id();
    uint64_t *bf_ptr = (uint64_t *)qp->sq_db;
    uint64_t *wqe = (uint64_t *)wqe_ptr;

    if (lane_id == 0) doca_gpu_dev_verbs_fence_release<sync_scope>();
    __syncwarp();

    if (lane_id < 8) {
        bf_ptr[lane_id] = wqe[lane_id];
    }
}

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_ring_proxy
// ════════════════════════════════════════════════════════════════════════════
// CPU Proxy 模式下的 "敲门" 操作。
//
// 在 proxy 模式中，sq_db 指向的不是 NIC 的 MMIO BAR，而是一块
// CPU↔GPU 共享的内存地址。GPU 线程将 prod_idx 写到这个地址，
// CPU 上运行的 proxy 线程持续轮询此地址，发现更新后代替 GPU 敲 NIC Doorbell。
//
// 两种模式：
//   EXCLUSIVE（单线程独占 QP）：
//     直接 store prod_idx（用 WRITE_ONCE 确保编译器不优化掉）
//   GPU（多线程共享 QP）：
//     fetch_max(prod_idx) — 原子取最大值
//     因为多线程可能同时写，取最大值保证 CPU proxy 看到的是最新进度
//     不会因为旧线程的小值覆盖新线程的大值
// ════════════════════════════════════════════════════════════════════════════
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode>
__device__ static __forceinline__ void doca_gpu_dev_verbs_ring_proxy(
    struct doca_gpu_dev_verbs_qp *qp, uint64_t prod_idx) {
    // sq_db 在 proxy 模式下 = CPU↔GPU 共享的 prod_index 通信地址
    uint64_t *proxy_ptr = (uint64_t *)__ldg((uintptr_t *)&qp->sq_db);
    cuda::atomic_ref<uint64_t, cuda::thread_scope_system> proxy_ptr_aref(*proxy_ptr);

    if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE) {
        // 单线程：直接写即可，无竞争
        proxy_ptr_aref.store(prod_idx, cuda::memory_order_relaxed);
        WRITE_ONCE(*proxy_ptr, prod_idx);  // 编译器 barrier，防止优化掉
    } else {
        // 多线程：原子取最大值，保证 CPU proxy 看到的是所有线程中的最大 prod_idx
        proxy_ptr_aref.fetch_max(prod_idx, cuda::memory_order_relaxed);
    }
}

/**
 * @brief Submit a work request to the NIC using the DB protocol without updating DBREC.
 *
 * @param qp - Queue Pair (QP)
 * @param prod_index - Producer index
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
          enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static inline void doca_gpu_dev_verbs_submit_db_no_dbr(
    struct doca_gpu_dev_verbs_qp *qp, uint64_t prod_index,
    uint32_t code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT) {
    if (!(code_opt & DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_DB_RINGING)) {
        uint64_t old_prod_index =
            doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode, true>(&qp->sq_wqe_pi,
                                                                                 prod_index);
        if (old_prod_index < prod_index)
            doca_gpu_dev_verbs_ring_db<sync_scope>(qp, prod_index, code_opt);
    }
}

/**
 * @brief Submit a work request to the NIC using the DB protocol.
 *
 * @param qp - Queue Pair (QP)
 * @param prod_index - Producer index
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
          enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ void doca_gpu_dev_verbs_submit_db(
    struct doca_gpu_dev_verbs_qp *qp, uint64_t prod_index,
    uint32_t code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT) {
    if (!(code_opt & DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_DB_RINGING)) {
        doca_gpu_dev_verbs_lock<resource_sharing_mode>(&qp->sq_lock);

        uint64_t old_prod_index =
            doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode, true>(&qp->sq_wqe_pi,
                                                                                 prod_index);
        if (old_prod_index < prod_index) {
            // Early rining of the DB to push WQEs to the NIC ASAP.
            doca_gpu_dev_verbs_ring_db<sync_scope>(qp, prod_index, code_opt);

            // In case the recovery path is triggered, the later DB ringing will cover for
            // correctness.
            doca_priv_gpu_dev_verbs_update_dbr<qp_type>(qp, prod_index);

            // Use at least either GPU or Sys synchronization scope for the second DB ringing.
            constexpr enum doca_gpu_dev_verbs_sync_scope second_db_sync_scope =
                (sync_scope <= DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU)
                    ? sync_scope
                    : DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU;
            doca_gpu_dev_verbs_ring_db<second_db_sync_scope>(qp, prod_index, code_opt);
        }

        doca_gpu_dev_verbs_unlock<resource_sharing_mode>(&qp->sq_lock);
    }
}

/**
 * @brief Submit a work request to the NIC using the BlueFlame protocol.
 * This function requires a single thread. Users must pass a pointer to a WQE stored in shared
 * memory. Hopper or a newer generation is required to leaverage the BlueFlame protocol.
 *
 * @param qp - Queue Pair (QP)
 * @param prod_index - Producer index
 * @param smem_wqe - WQE to be submitted directly to the NIC. The buffer must be in shared memory.
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>
__device__ static __forceinline__ void doca_gpu_dev_verbs_submit_bf(
    struct doca_gpu_dev_verbs_qp *qp, uint64_t prod_index, struct doca_gpu_dev_verbs_wqe *smem_wqe,
    uint32_t code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT) {
#ifdef DOCA_GPUNETIO_VERBS_HAS_TMA_COPY
    doca_gpu_dev_verbs_lock<resource_sharing_mode>(&qp->sq_lock);
    unsigned long long int old_prod_index =
        doca_gpu_dev_verbs_atomic_max<unsigned long long int, resource_sharing_mode, true>(
            (unsigned long long int *)&qp->sq_wqe_pi, (unsigned long long int)prod_index);
    if (old_prod_index < prod_index) {
        doca_gpu_dev_verbs_ring_bf<sync_scope>(qp, smem_wqe);
        doca_priv_gpu_dev_verbs_update_dbr<DOCA_GPUNETIO_VERBS_QP_SQ>(qp, prod_index);
        constexpr enum doca_gpu_dev_verbs_sync_scope second_db_sync_scope =
            (sync_scope <= DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU) ? sync_scope
                                                               : DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU;
        doca_gpu_dev_verbs_ring_db<second_db_sync_scope>(qp, prod_index, code_opt);
    }
    doca_gpu_dev_verbs_unlock<resource_sharing_mode>(&qp->sq_lock);
#else
    doca_gpu_dev_verbs_submit_db<resource_sharing_mode, sync_scope, DOCA_GPUNETIO_VERBS_QP_SQ>(
        qp, prod_index, code_opt);
#endif
}

/**
 * @brief Submit all the WQEs up to the given producer index to the NIC using the BlueFlame
 * protocol. This function must be called by all threads in the warp. At least 8 threads are
 * required.
 *
 * @param qp - Queue Pair (QP)
 * @param prod_index - Producer index
 * @param wqe - WQE to be submitted directly to the NIC
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>
__device__ static __forceinline__ void doca_gpu_dev_verbs_submit_bf_warp(
    struct doca_gpu_dev_verbs_qp *qp, uint64_t prod_index, struct doca_gpu_dev_verbs_wqe *wqe,
    uint32_t code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT) {
    unsigned int lane_id = doca_gpu_dev_verbs_get_lane_id();
    unsigned long long int old_prod_index;
    if (lane_id == 0) {
        doca_gpu_dev_verbs_lock<resource_sharing_mode>(&qp->sq_lock);
        old_prod_index =
            doca_gpu_dev_verbs_atomic_max<unsigned long long int, resource_sharing_mode, true>(
                (unsigned long long int *)&qp->sq_wqe_pi, (unsigned long long int)prod_index);
    }
    __syncwarp();
    old_prod_index = __shfl_sync(0xFFFFFFFF, old_prod_index, 0);
    if (old_prod_index < prod_index) {
        doca_gpu_dev_verbs_ring_bf_warp(qp, wqe);
        __syncwarp();
        if (lane_id == 0) {
            doca_priv_gpu_dev_verbs_update_dbr<DOCA_GPUNETIO_VERBS_QP_SQ>(qp, prod_index);
            constexpr enum doca_gpu_dev_verbs_sync_scope second_db_sync_scope =
                (sync_scope <= DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU)
                    ? sync_scope
                    : DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU;
            doca_gpu_dev_verbs_ring_db<second_db_sync_scope>(qp, prod_index, code_opt);
        }
    }
    if (lane_id == 0) doca_gpu_dev_verbs_unlock<resource_sharing_mode>(&qp->sq_lock);
    __syncwarp();
}

/**
 * @brief Submit all the WQEs up to the given producer index to the NIC via the CPU proxy.
 *
 * @param qp - Queue Pair (QP)
 * @param prod_index - Producer index
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>
__device__ static __forceinline__ void doca_gpu_dev_verbs_submit_proxy(
    struct doca_gpu_dev_verbs_qp *qp, uint64_t prod_index,
    uint32_t code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT) {
    doca_gpu_dev_verbs_fence_release<sync_scope>();
    doca_gpu_dev_verbs_ring_proxy<resource_sharing_mode>(qp, prod_index);
    if (code_opt & DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_CPU_PROXY_UPDATE_PI) {
        doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode, true>(&qp->sq_wqe_pi,
                                                                             prod_index);
    }
}

template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
          enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
          enum doca_gpu_dev_verbs_qp_type qp_type = DOCA_GPUNETIO_VERBS_QP_SQ>
__device__ static __forceinline__ void doca_gpu_dev_verbs_submit(
    struct doca_gpu_dev_verbs_qp *qp, uint64_t prod_index,
    uint32_t code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT) {
    const enum doca_gpu_dev_verbs_nic_handler qp_nic_handler =
        (enum doca_gpu_dev_verbs_nic_handler)__ldg((int *)&qp->nic_handler);
    if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO) {
        if (qp_nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB)
            doca_gpu_dev_verbs_submit_db<resource_sharing_mode, sync_scope, qp_type>(qp, prod_index,
                                                                                     code_opt);
        else if (qp_nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_NO_DBR)
            doca_gpu_dev_verbs_submit_db_no_dbr<resource_sharing_mode, sync_scope, qp_type>(
                qp, prod_index, code_opt);
        else
            doca_gpu_dev_verbs_submit_proxy<resource_sharing_mode, sync_scope>(qp, prod_index, code_opt);
    } else if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB) {
        doca_gpu_dev_verbs_submit_db<resource_sharing_mode, sync_scope, qp_type>(qp, prod_index,
                                                                                 code_opt);
    } else if (qp_nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_NO_DBR) {
        doca_gpu_dev_verbs_submit_db_no_dbr<resource_sharing_mode, sync_scope, qp_type>(
            qp, prod_index, code_opt);
    } else {
        doca_gpu_dev_verbs_submit_proxy<resource_sharing_mode, sync_scope>(qp, prod_index, code_opt);
    }
}

/* *********** WQE PREPARATION *********** */

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_wqe_prepare_nop
// ════════════════════════════════════════════════════════════════════════════
// 填充一个 NOP（No-Operation）WQE。用于 size==0 时占位，不执行实际数据传输。
// NIC 收到后直接跳过并产生 CQE（如果设了 CQ_UPDATE）。
//
// WQE 64 字节中只使用第一个 16 字节的控制段 cseg：
//   cseg.opmod_idx_opcode = (wqe_idx << 8) | NOP_OPCODE
//     - 高 16 位：WQE 索引（NIC 用来匹配 CQE）
//     - 低  8 位：操作码 = NOP
//   cseg.qpn_ds = sq_num_shift8_be_1ds
//     - 高 24 位：QP 编号（大端序，左移 8 位预存）
//     - 低  8 位：WQE 包含的 data segment 数量 = 1
//   cseg.fm_ce_se = CQ_UPDATE（完成时写 CQE）
// ════════════════════════════════════════════════════════════════════════════
__device__ static __forceinline__ void doca_gpu_dev_verbs_wqe_prepare_nop(
    struct doca_gpu_dev_verbs_qp *qp, struct doca_gpu_dev_verbs_wqe *wqe_ptr,
    const uint16_t wqe_idx, enum doca_gpu_dev_verbs_wqe_ctrl_flags ctrl_flags) {
    struct doca_gpunetio_ib_mlx5_wqe_ctrl_seg cseg;

    cseg.opmod_idx_opcode =
        doca_gpu_dev_verbs_bswap32(((uint32_t)wqe_idx << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) |
                                   DOCA_GPUNETIO_IB_MLX5_OPCODE_NOP);
    cseg.qpn_ds = __ldg(&qp->sq_num_shift8_be_1ds);  // QPN | 1 data segment
    cseg.fm_ce_se = ctrl_flags;

    // 只写控制段（16 字节），其余 48 字节不用管（NOP 不需要地址/数据段）
    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg0), (uint64_t *)&(cseg));
}

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_wqe_prepare_write（单数据段版本）
// ════════════════════════════════════════════════════════════════════════════
// 填充一个 RDMA Write WQE（64 字节），包含 3 个 segment：
//
// WQE 64 字节布局（每段 16 字节）：
//   ┌─────────────────────────────────────────────────────────┐
//   │ dseg0 (字节 0-15)  : cseg  — 控制段                      │
//   │   opmod_idx_opcode = (wqe_idx << 8) | RDMA_WRITE        │
//   │   qpn_ds = QPN | 3（3 个 data segment）                  │
//   │   fm_ce_se = CQ_UPDATE                                   │
//   │   imm = immediate（RDMA Write with Immediate 时用，否则 0）│
//   ├─────────────────────────────────────────────────────────┤
//   │ dseg1 (字节 16-31) : rseg  — 远端地址段                    │
//   │   raddr = 对端 MR 内偏移（大端序 64 位）                   │
//   │   rkey  = 对端 MR 的 remote key（大端序 32 位）            │
//   ├─────────────────────────────────────────────────────────┤
//   │ dseg2 (字节 32-47) : dseg0 — 本地数据段                    │
//   │   byte_count = 传输字节数（大端序，最高位清零）              │
//   │   lkey = 本地 MR 的 local key（大端序 32 位）              │
//   │   addr = 本地 buffer 的虚拟地址（大端序 64 位）             │
//   ├─────────────────────────────────────────────────────────┤
//   │ dseg3 (字节 48-63) : (未使用)                              │
//   └─────────────────────────────────────────────────────────┘
//
// NIC 硬件处理：
//   1. 从 dseg0.addr（本地 GPU 显存）DMA 读 byte_count 字节
//   2. 通过 RDMA 网络传输到对端
//   3. 写入对端 rseg.raddr 处的内存（无需对端 CPU/GPU 参与）
//
// bswap32/bswap64：MLX5 WQE 格式要求大端序（网络字节序），
// GPU 是小端，需要字节翻转。
// ════════════════════════════════════════════════════════════════════════════

__device__ static __forceinline__ void doca_gpu_dev_verbs_wqe_prepare_write(
    struct doca_gpu_dev_verbs_qp *qp, struct doca_gpu_dev_verbs_wqe *wqe_ptr,
    const uint16_t wqe_idx, const uint32_t opcode,
    enum doca_gpu_dev_verbs_wqe_ctrl_flags ctrl_flags, const uint32_t immediate,
    const uint64_t raddr, const uint32_t rkey, const uint64_t laddr0, const uint32_t lkey0,
    const uint32_t bytes0) {
    struct doca_gpunetio_ib_mlx5_wqe_ctrl_seg cseg;
    struct doca_gpunetio_ib_mlx5_wqe_raddr_seg rseg;
    struct doca_gpunetio_ib_mlx5_wqe_data_seg dseg0;

    cseg.opmod_idx_opcode = doca_gpu_dev_verbs_bswap32(
        ((uint32_t)wqe_idx << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) | opcode);
    cseg.qpn_ds = __ldg(&qp->sq_num_shift8_be_3ds);
    cseg.fm_ce_se = ctrl_flags;
    cseg.imm = immediate;

    rseg.raddr = doca_gpu_dev_verbs_bswap64(raddr);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
    rseg.rkey = rkey;
#else
    rseg.rkey = doca_gpu_dev_verbs_bswap32(rkey);
#endif

    dseg0.byte_count =
        doca_gpu_dev_verbs_bswap32(bytes0 & uint32_t(DOCA_GPUNETIO_IB_MLX5_INLINE_SEG - 1));
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
    dseg0.lkey = lkey0;
#else
    dseg0.lkey = doca_gpu_dev_verbs_bswap32(lkey0);
#endif
    dseg0.addr = doca_gpu_dev_verbs_bswap64(laddr0);

    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg0), (uint64_t *)&(cseg));
    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg1), (uint64_t *)&(rseg));
    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg2), (uint64_t *)&(dseg0));
}

__device__ static __forceinline__ void doca_gpu_dev_verbs_wqe_prepare_write(
    struct doca_gpu_dev_verbs_qp *qp, struct doca_gpu_dev_verbs_wqe *wqe_ptr,
    const uint16_t wqe_idx, const uint32_t opcode,
    enum doca_gpu_dev_verbs_wqe_ctrl_flags ctrl_flags, const uint32_t immediate,
    const uint64_t raddr, const uint32_t rkey, const uint64_t laddr0, const uint32_t lkey0,
    const uint32_t bytes0, const uint64_t laddr1, const uint32_t lkey1, const uint32_t bytes1) {
    struct doca_gpunetio_ib_mlx5_wqe_ctrl_seg cseg;
    struct doca_gpunetio_ib_mlx5_wqe_raddr_seg rseg;
    struct doca_gpunetio_ib_mlx5_wqe_data_seg dseg0;
    struct doca_gpunetio_ib_mlx5_wqe_data_seg dseg1;

    cseg.opmod_idx_opcode = doca_gpu_dev_verbs_bswap32(
        ((uint32_t)wqe_idx << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) | opcode);
    cseg.qpn_ds = __ldg(&qp->sq_num_shift8_be_4ds);
    cseg.fm_ce_se = ctrl_flags;
    cseg.imm = immediate;

    rseg.raddr = doca_gpu_dev_verbs_bswap64(raddr);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
    rseg.rkey = rkey;
#else
    rseg.rkey = doca_gpu_dev_verbs_bswap32(rkey);
#endif

    dseg0.byte_count =
        doca_gpu_dev_verbs_bswap32(bytes0 & uint32_t(DOCA_GPUNETIO_IB_MLX5_INLINE_SEG - 1));
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
    dseg0.lkey = lkey0;
#else
    dseg0.lkey = doca_gpu_dev_verbs_bswap32(lkey0);
#endif
    dseg0.addr = doca_gpu_dev_verbs_bswap64(laddr0);

    dseg1.byte_count =
        doca_gpu_dev_verbs_bswap32(bytes1 & uint32_t(DOCA_GPUNETIO_IB_MLX5_INLINE_SEG - 1));
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
    dseg1.lkey = lkey1;
#else
    dseg1.lkey = doca_gpu_dev_verbs_bswap32(lkey1);
#endif
    dseg1.addr = doca_gpu_dev_verbs_bswap64(laddr1);

    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg0), (uint64_t *)&(cseg));
    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg1), (uint64_t *)&(rseg));
    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg2), (uint64_t *)&(dseg0));
    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg3), (uint64_t *)&(dseg1));
}

/**
 * @brief Prepare the header segment of an inline RDMA Write WQE.
 * The data segment is prepared separately.
 *
 * @param qp - Queue Pair (QP)
 * @param send_wr - Send Work Request to be prepared
 * @param wqe_idx - Index of the WQE to be prepared
 * @param out_wqes - Pointer to the WQE buffer to write the prepared WQE to
 */
__device__ static __forceinline__ void doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_header(
    struct doca_gpu_dev_verbs_qp *qp, struct doca_gpu_dev_verbs_wqe *wqe_ptr,
    const uint16_t wqe_idx, enum doca_gpu_dev_verbs_wqe_ctrl_flags ctrl_flags, const uint64_t raddr,
    const uint32_t rkey, const uint32_t bytes) {
    int ds;
    struct doca_gpunetio_ib_mlx5_wqe_ctrl_seg cseg;
    struct doca_gpunetio_ib_mlx5_wqe_raddr_seg rseg;

    if (bytes > sizeof(struct doca_gpunetio_ib_mlx5_wqe_data_seg) -
                    sizeof(struct doca_gpunetio_ib_mlx5_wqe_inl_data_seg))
        ds = DOCA_GPUNETIO_VERBS_WQE_SEG_CNT_RDMA_WRITE_INL_MAX;
    else
        ds = DOCA_GPUNETIO_VERBS_WQE_SEG_CNT_RDMA_WRITE_INL_MIN;

    assert(bytes <= DOCA_GPUNETIO_VERBS_MAX_INLINE_SIZE);

    cseg.opmod_idx_opcode =
        doca_gpu_dev_verbs_bswap32(((uint32_t)wqe_idx << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) |
                                   DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE);
    cseg.qpn_ds = doca_gpu_dev_verbs_bswap32(__ldg(&qp->sq_num_shift8) | ds);
    cseg.fm_ce_se = ctrl_flags;
    // cseg.imm = 0;

    rseg.raddr = doca_gpu_dev_verbs_bswap64(raddr);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
    rseg.rkey = rkey;
#else
    rseg.rkey = doca_gpu_dev_verbs_bswap32(rkey);
#endif

    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg0), (uint64_t *)&(cseg));
    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg1), (uint64_t *)&(rseg));
}

/**
 * @brief Prepare the data segment of an inline RDMA Write WQE.
 *
 * @param qp - Queue Pair (QP)
 * @param data - Data to be written
 * @param out_wqes - Pointer to the WQE buffer to write the prepared WQE to
 */
template <typename T>
__device__ static __forceinline__ void doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_data(
    struct doca_gpu_dev_verbs_qp *qp, struct doca_gpu_dev_verbs_wqe *wqe_ptr, T data) {
    struct doca_gpunetio_ib_mlx5_wqe_inl_data_seg *data_seg_ptr =
        (struct doca_gpunetio_ib_mlx5_wqe_inl_data_seg
             *)((uintptr_t)wqe_ptr + sizeof(struct doca_gpunetio_ib_mlx5_wqe_ctrl_seg) +
                sizeof(struct doca_gpunetio_ib_mlx5_wqe_raddr_seg));
    struct doca_gpunetio_ib_mlx5_wqe_inl_data_seg data_seg;
    uint32_t bytes = sizeof(T);

    data_seg.byte_count = doca_gpu_dev_verbs_bswap32(bytes | DOCA_GPUNETIO_IB_MLX5_INLINE_SEG);
    *(uint32_t *)data_seg_ptr = data_seg.byte_count;
    if (bytes <= sizeof(uint32_t)) {
        T *dst = (T *)((uintptr_t)data_seg_ptr + sizeof(data_seg));
        *dst = data;
    } else {
        uint32_t *dst32 = (uint32_t *)((uintptr_t)data_seg_ptr + sizeof(data_seg));
        dst32[0] = ((uint32_t *)&data)[0];
        dst32[1] = ((uint32_t *)&data)[1];
    }
}

/**
 * @brief Prepare a RDMA Write WQE with inline data
 *
 * @param qp - Queue Pair (QP)
 * @param send_wr - Send Work Request to be prepared
 * @param wqe_idx - Index of the WQE to be prepared
 * @param out_wqes - Pointer to the WQE buffer to write the prepared WQE to
 */
__device__ static __forceinline__ void doca_gpu_dev_verbs_wqe_prepare_write_inl(
    struct doca_gpu_dev_verbs_qp *qp, struct doca_gpu_dev_verbs_wqe *wqe_ptr,
    const uint16_t wqe_idx, enum doca_gpu_dev_verbs_wqe_ctrl_flags ctrl_flags, const uint64_t raddr,
    const uint32_t rkey, const uint64_t laddr, const uint32_t bytes) {
    struct doca_gpunetio_ib_mlx5_wqe_inl_data_seg data_seg;
    struct doca_gpunetio_ib_mlx5_wqe_inl_data_seg *data_seg_ptr =
        (struct doca_gpunetio_ib_mlx5_wqe_inl_data_seg
             *)((uintptr_t)wqe_ptr + sizeof(struct doca_gpunetio_ib_mlx5_wqe_ctrl_seg) +
                sizeof(struct doca_gpunetio_ib_mlx5_wqe_raddr_seg));

    doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_header(qp, wqe_ptr, wqe_idx, ctrl_flags, raddr,
                                                         rkey, bytes);

    data_seg.byte_count = doca_gpu_dev_verbs_bswap32(bytes | DOCA_GPUNETIO_IB_MLX5_INLINE_SEG);
    *(uint32_t *)data_seg_ptr = data_seg.byte_count;

    doca_gpu_dev_verbs_memcpy_data((void *)((uintptr_t)data_seg_ptr + sizeof(data_seg)),
                                   (void *)(uintptr_t)laddr, bytes);
}

__device__ static __forceinline__ void doca_gpu_dev_verbs_wqe_prepare_read(
    struct doca_gpu_dev_verbs_qp *qp, struct doca_gpu_dev_verbs_wqe *wqe_ptr,
    const uint16_t wqe_idx, enum doca_gpu_dev_verbs_wqe_ctrl_flags ctrl_flags, const uint64_t raddr,
    const uint32_t rkey, const uint64_t laddr0, const uint32_t lkey0, const uint32_t bytes0) {
    struct doca_gpunetio_ib_mlx5_wqe_ctrl_seg cseg;
    struct doca_gpunetio_ib_mlx5_wqe_raddr_seg rseg;
    struct doca_gpunetio_ib_mlx5_wqe_data_seg dseg0;

    cseg.opmod_idx_opcode =
        doca_gpu_dev_verbs_bswap32(((uint32_t)wqe_idx << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) |
                                   DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_READ);
    cseg.qpn_ds = __ldg(&qp->sq_num_shift8_be_3ds);
    cseg.fm_ce_se = ctrl_flags;

    rseg.raddr = doca_gpu_dev_verbs_bswap64(raddr);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
    rseg.rkey = rkey;
#else
    rseg.rkey = doca_gpu_dev_verbs_bswap32(rkey);
#endif

    dseg0.byte_count = doca_gpu_dev_verbs_bswap32(bytes0);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
    dseg0.lkey = lkey0;
#else
    dseg0.lkey = doca_gpu_dev_verbs_bswap32(lkey0);
#endif
    dseg0.addr = doca_gpu_dev_verbs_bswap64(laddr0);

    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg0), (uint64_t *)&(cseg));
    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg1), (uint64_t *)&(rseg));
    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg2), (uint64_t *)&(dseg0));
}

__device__ static __forceinline__ void doca_gpu_dev_verbs_wqe_prepare_read(
    struct doca_gpu_dev_verbs_qp *qp, struct doca_gpu_dev_verbs_wqe *wqe_ptr,
    const uint16_t wqe_idx, enum doca_gpu_dev_verbs_wqe_ctrl_flags ctrl_flags, const uint64_t raddr,
    const uint32_t rkey, const uint64_t laddr0, const uint32_t lkey0, const uint32_t bytes0,
    const uint64_t laddr1, const uint32_t lkey1, const uint32_t bytes1) {
    struct doca_gpunetio_ib_mlx5_wqe_ctrl_seg cseg;
    struct doca_gpunetio_ib_mlx5_wqe_raddr_seg rseg;
    struct doca_gpunetio_ib_mlx5_wqe_data_seg dseg0;
    struct doca_gpunetio_ib_mlx5_wqe_data_seg dseg1;

    cseg.opmod_idx_opcode =
        doca_gpu_dev_verbs_bswap32(((uint32_t)wqe_idx << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) |
                                   DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_READ);
    cseg.qpn_ds = doca_gpu_dev_verbs_bswap32(__ldg(&qp->sq_num_shift8) | 4);
    cseg.fm_ce_se = ctrl_flags;
    // cseg.imm = 0;

    rseg.raddr = doca_gpu_dev_verbs_bswap64(raddr);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
    rseg.rkey = rkey;
#else
    rseg.rkey = doca_gpu_dev_verbs_bswap32(rkey);
#endif

    dseg0.byte_count =
        doca_gpu_dev_verbs_bswap32(bytes0 & uint32_t(DOCA_GPUNETIO_IB_MLX5_INLINE_SEG - 1));
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
    dseg0.lkey = lkey0;
#else
    dseg0.lkey = doca_gpu_dev_verbs_bswap32(lkey0);
#endif
    dseg0.addr = doca_gpu_dev_verbs_bswap64(laddr0);

    dseg1.byte_count =
        doca_gpu_dev_verbs_bswap32(bytes1 & uint32_t(DOCA_GPUNETIO_IB_MLX5_INLINE_SEG - 1));
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
    dseg1.lkey = lkey1;
#else
    dseg1.lkey = doca_gpu_dev_verbs_bswap32(lkey1);
#endif
    dseg1.addr = doca_gpu_dev_verbs_bswap64(laddr1);

    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg0), (uint64_t *)&(cseg));
    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg1), (uint64_t *)&(rseg));
    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg2), (uint64_t *)&(dseg0));
    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg3), (uint64_t *)&(dseg1));
}

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_wqe_prepare_atomic
// ════════════════════════════════════════════════════════════════════════════
// 填充一个 Atomic WQE（64 字节），用于 RDMA 原子操作。
//
// 支持两种原子操作码（由 opcode 参数决定）：
//   ATOMIC_FA  (Fetch-And-Add)       — 对端 *raddr += compare_add，返回旧值
//   ATOMIC_CS  (Compare-And-Swap)    — 如果对端 *raddr == compare_add，
//                                       则 *raddr = swap_add，返回旧值
//
// WQE 64 字节布局（4 × 16 字节 segment）：
//   ┌─────────────────────────────────────────────────────────────────────┐
//   │ dseg0 (字节 0-15)  : cseg  — 控制段                                  │
//   │   opmod_idx_opcode = (wqe_idx << 8) | ATOMIC_FA/ATOMIC_CS           │
//   │   qpn_ds = QPN | 4（4 个 data segment = 64 字节 / 16）              │
//   │   fm_ce_se = CQ_UPDATE（完成时产生 CQE）                             │
//   ├─────────────────────────────────────────────────────────────────────┤
//   │ dseg1 (字节 16-31) : rseg  — 远端地址段                                │
//   │   raddr = 对端原子操作目标地址（如 signal_table[signalId] 的 VA）      │
//   │   rkey  = 对端 MR 的 remote key                                      │
//   │   ★ 注意：raddr 必须 8 字节对齐（IB Atomic 要求）                     │
//   ├─────────────────────────────────────────────────────────────────────┤
//   │ dseg2 (字节 32-47) : atseg — 原子操作段（Atomic Segment）              │
//   │   swap_add:                                                          │
//   │     ATOMIC_FA 模式：= compare_add（即加数）                          │
//   │     ATOMIC_CS 模式：= swap_add（替换值）                             │
//   │   compare:                                                           │
//   │     ATOMIC_FA 模式：= compare_add（NIC 忽略此字段，但代码统一赋值）   │
//   │     ATOMIC_CS 模式：= compare_add（期望的比较值）                    │
//   ├─────────────────────────────────────────────────────────────────────┤
//   │ dseg3 (字节 48-63) : dseg  — 本地数据段（接收返回值）                  │
//   │   byte_count = sizeof(uint64_t) = 8（原子操作固定 8 字节）            │
//   │   lkey  = 本地 MR 的 local key                                       │
//   │   addr  = 本地 buffer 地址（NIC 将旧值 DMA 写到这里 = sink buffer）   │
//   └─────────────────────────────────────────────────────────────────────┘
//
// NIC 硬件处理流程：
//   1. NIC 通过 RDMA 访问对端 raddr，执行原子操作（FAA 或 CAS）
//   2. 对端 NIC 在内存控制器级别原子执行（保证不被其他访问打断）
//   3. 将操作前的旧值返回，通过 DMA 写入本地 laddr（sink buffer）
//   4. 完成后在本地 CQ 产生 CQE
//
// 在 put_signal_counter 中的用途：
//   - signal FAA：对端 signal_table[signalId] += 1（通知对端数据已到达）
//   - counter FAA：本地 counter_table[counterId] += 1（记录本端操作完成数）
// ════════════════════════════════════════════════════════════════════════════
__device__ static __forceinline__ void doca_gpu_dev_verbs_wqe_prepare_atomic(
    struct doca_gpu_dev_verbs_qp *qp, struct doca_gpu_dev_verbs_wqe *wqe_ptr,
    const uint16_t wqe_idx, const uint32_t opcode,
    enum doca_gpu_dev_verbs_wqe_ctrl_flags ctrl_flags, const uint64_t raddr, const uint32_t rkey,
    const uint64_t laddr, const uint32_t lkey, const uint32_t bytes, const uint64_t compare_add,
    const uint64_t swap_add) {
    struct doca_gpunetio_ib_mlx5_wqe_ctrl_seg cseg;
    struct doca_gpunetio_ib_mlx5_wqe_raddr_seg rseg;
    struct doca_gpunetio_ib_mlx5_wqe_atomic_seg atseg;
    struct doca_gpunetio_ib_mlx5_wqe_data_seg dseg;

    // 控制段：(wqe_idx << 8) | opcode，大端序
    cseg.opmod_idx_opcode = doca_gpu_dev_verbs_bswap32(
        ((uint32_t)wqe_idx << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) | opcode);
    cseg.qpn_ds = __ldg(&qp->sq_num_shift8_be_4ds);  // QPN | 4 segments
    cseg.fm_ce_se = ctrl_flags;  // CQ_UPDATE = 完成时产生 CQE

    // 远端地址段：原子操作的目标地址
    rseg.raddr = doca_gpu_dev_verbs_bswap64(raddr);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
    rseg.rkey = rkey;
#else
    rseg.rkey = doca_gpu_dev_verbs_bswap32(rkey);
#endif

    // 原子操作段：根据操作类型设置 swap_add 字段
    //   ATOMIC_FA：swap_add = compare_add（即加数），NIC 执行 *raddr += swap_add
    //   ATOMIC_CS：swap_add = swap_add（替换值），NIC 执行 if(*raddr==compare) *raddr=swap_add
    atseg.swap_add = doca_gpu_dev_verbs_bswap64(
        opcode == DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA ? compare_add : swap_add);
    atseg.compare = doca_gpu_dev_verbs_bswap64(compare_add);  // 比较值（FAA 忽略）

    // 本地数据段：NIC 将旧值写到这个地址（sink buffer）
    dseg.byte_count = doca_gpu_dev_verbs_bswap32(bytes);  // 固定 8 字节
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
    dseg.lkey = lkey;
#else
    dseg.lkey = doca_gpu_dev_verbs_bswap32(lkey);
#endif
    dseg.addr = doca_gpu_dev_verbs_bswap64(laddr);

    // 通过 4 次 128-bit store 将 4 个 segment 写入 WQE 缓冲区
    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg0), (uint64_t *)&(cseg));
    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg1), (uint64_t *)&(rseg));
    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg2), (uint64_t *)&(atseg));
    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg3), (uint64_t *)&(dseg));
}

/**
 * @brief Prepare an Extended Atomic WQE
 *
 * @param qp - Queue Pair (QP)
 * @param send_wr - Send Work Request to be prepared
 * @param wqe_idx - Index of the WQE to be prepared
 * @param out_wqes - Pointer to the WQE buffer to write the prepared WQE to
 */
template <enum doca_gpu_dev_verbs_atomic_ext_bytes bytes = DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_4>
__device__ static __forceinline__ void doca_gpu_dev_verbs_wqe_prepare_atomic_ext(
    struct doca_gpu_dev_verbs_qp *qp, struct doca_gpu_dev_verbs_wqe *wqe_ptr0,
    struct doca_gpu_dev_verbs_wqe *wqe_ptr1, const uint16_t wqe_idx, const uint32_t opcode,
    enum doca_gpu_dev_verbs_wqe_ctrl_flags ctrl_flags, const uint64_t raddr, const uint32_t rkey,
    const uint64_t laddr, const uint32_t lkey, const uint64_t add_data,
    const uint64_t field_boundary, const uint64_t swap_data, const uint64_t compare_data,
    const uint64_t swap_mask, const uint64_t compare_mask) {
    struct doca_gpunetio_ib_mlx5_wqe_ctrl_seg cseg;
    struct doca_gpunetio_ib_mlx5_wqe_raddr_seg rseg;
    struct doca_gpunetio_ib_mlx5_wqe_atomic_seg aseg_1;
    struct doca_gpunetio_ib_mlx5_wqe_atomic_seg aseg_2 = {0};
    struct doca_gpunetio_ib_mlx5_wqe_data_seg dseg;

    rseg.raddr = doca_gpu_dev_verbs_bswap64(raddr);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
    rseg.rkey = rkey;
#else
    rseg.rkey = doca_gpu_dev_verbs_bswap32(rkey);
#endif

    cseg = {
        0,
    };

    if (opcode == DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA) {
        if (bytes == DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_4) {
            cseg.opmod_idx_opcode =
                doca_gpu_dev_verbs_bswap32(DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA |
                                           (wqe_idx << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) |
                                           DOCA_GPUNETIO_4_BYTE_ATOMIC_EXT_OPMOD);
            cseg.qpn_ds = __ldg(&qp->sq_num_shift8_be_4ds);
            cseg.fm_ce_se = ctrl_flags;

            doca_gpu_dev_verbs_atomic_32_masked_fa_seg_t *atomic_32_masked_fa_seg =
                (doca_gpu_dev_verbs_atomic_32_masked_fa_seg_t *)&aseg_1;
            atomic_32_masked_fa_seg->add_data = doca_gpu_dev_verbs_bswap32((uint32_t)add_data);
            atomic_32_masked_fa_seg->field_boundary = field_boundary;
        } else {
            cseg.opmod_idx_opcode =
                doca_gpu_dev_verbs_bswap32(DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA |
                                           (wqe_idx << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) |
                                           DOCA_GPUNETIO_8_BYTE_ATOMIC_EXT_OPMOD);
            cseg.qpn_ds = __ldg(&qp->sq_num_shift8_be_4ds);
            cseg.fm_ce_se = ctrl_flags;

            doca_gpu_dev_verbs_atomic_64_masked_fa_seg_t *atomic_64_masked_fa_seg =
                (doca_gpu_dev_verbs_atomic_64_masked_fa_seg_t *)&aseg_1;
            atomic_64_masked_fa_seg->add_data = doca_gpu_dev_verbs_bswap64((uint64_t)add_data);
            atomic_64_masked_fa_seg->field_boundary = field_boundary;
        }
    }

    if (opcode == DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS) {
        if (bytes == DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_4) {
            cseg.opmod_idx_opcode =
                doca_gpu_dev_verbs_bswap32(DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS |
                                           (wqe_idx << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) |
                                           DOCA_GPUNETIO_4_BYTE_ATOMIC_EXT_OPMOD);
            cseg.qpn_ds = __ldg(&qp->sq_num_shift8_be_4ds);
            cseg.fm_ce_se = ctrl_flags;

            doca_gpu_dev_verbs_atomic_32_masked_cs_seg_t *atomic_32_masked_cs_seg =
                (doca_gpu_dev_verbs_atomic_32_masked_cs_seg_t *)&aseg_1;
            atomic_32_masked_cs_seg->swap_data = doca_gpu_dev_verbs_bswap32((uint32_t)swap_data);
            atomic_32_masked_cs_seg->compare_data = compare_data;
            atomic_32_masked_cs_seg->swap_mask = doca_gpu_dev_verbs_bswap32((uint32_t)swap_mask);
            atomic_32_masked_cs_seg->compare_mask = compare_mask;
        } else {
            cseg.opmod_idx_opcode =
                doca_gpu_dev_verbs_bswap32(DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS |
                                           (wqe_idx << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) |
                                           DOCA_GPUNETIO_8_BYTE_ATOMIC_EXT_OPMOD);
            cseg.qpn_ds = __ldg(&qp->sq_num_shift8_be_5ds);
            cseg.fm_ce_se = ctrl_flags;

            doca_gpu_dev_verbs_atomic_64_masked_cs_seg_t *atomic_64_masked_cs_data_seg =
                (doca_gpu_dev_verbs_atomic_64_masked_cs_seg_t *)&aseg_1;
            atomic_64_masked_cs_data_seg->swap = doca_gpu_dev_verbs_bswap64((uint64_t)swap_data);
            atomic_64_masked_cs_data_seg->compare = compare_data;

            doca_gpu_dev_verbs_atomic_64_masked_cs_seg_t *atomic_64_masked_cs_mask_seg =
                (doca_gpu_dev_verbs_atomic_64_masked_cs_seg_t *)&aseg_2;
            atomic_64_masked_cs_mask_seg->swap = doca_gpu_dev_verbs_bswap64((uint64_t)swap_mask);
            atomic_64_masked_cs_mask_seg->compare = compare_mask;
        }
    }

    dseg.byte_count = doca_gpu_dev_verbs_bswap32((uint32_t)bytes);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
    dseg.lkey = lkey;
#else
    dseg.lkey = doca_gpu_dev_verbs_bswap32(lkey);
#endif
    dseg.addr = doca_gpu_dev_verbs_bswap64(laddr);

    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr0->dseg0), (uint64_t *)&(cseg));
    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr0->dseg1), (uint64_t *)&(rseg));
    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr0->dseg2), (uint64_t *)&(aseg_1));

    if (opcode == DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS &&
        bytes == DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_8) {
        doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr0->dseg3), (uint64_t *)&(aseg_2));
        doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr1->dseg0), (uint64_t *)&(dseg));
    } else {
        doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr0->dseg3), (uint64_t *)&(dseg));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_wqe_prepare_wait
// ════════════════════════════════════════════════════════════════════════════
// 填充一个 WAIT WQE，这是 MLX5 硬件特有的 WQE 类型（opcode = 0x0f）。
//
// ★ WAIT 语义：
//   NIC 在处理到这个 WQE 时，暂停当前 QP 的 SQ 流水线，
//   等待**另一个 QP** 的 CQ 中出现 max_index 对应的 CQE 后才继续。
//   这是硬件级的跨 QP 同步机制，无需 GPU/CPU 参与。
//
// ★ 参数含义：
//   max_index — 要等待的 WQE 索引（即被监控 QP 上第几个 WQE 完成）
//     例：main QP 上最后一个 WQE 是 signal FAA，其索引 = wqe_idx
//     → WAIT 的 max_index = wqe_idx → 等到 signal FAA 的 CQE 出现
//
//   qpn_cqn — 要监控的 CQ 编号（被等待 QP 的 SQ 关联 CQ 的 cq_num）
//     通过 qp->cq_sq.cq_num 获取（init 阶段创建 QP 时 NIC 固件分配）
//     NIC 硬件内部维护每个 CQ 的 consumer index，
//     当 CQ 的 consumer index >= max_index 时，WAIT 条件满足
//
// WQE 布局（只使用前 32 字节 = 2 个 segment）：
//   ┌──────────────────────────────────────────────────────────┐
//   │ dseg0 (字节 0-15) : cseg  — 控制段                        │
//   │   opmod_idx_opcode = (wqe_idx << 8) | WAIT (0x0f)        │
//   │   qpn_ds = QPN | SEG_CNT_WAIT（segment 数量）            │
//   │   fm_ce_se = CQ_UPDATE（WAIT 自己完成时也产生 CQE）       │
//   ├──────────────────────────────────────────────────────────┤
//   │ dseg1 (字节 16-31) : wseg — Wait 段                       │
//   │   resv[2]   = 保留字段                                    │
//   │   max_index = 等待的 WQE 完成索引（大端序）               │
//   │   qpn_cqn   = 监控的 CQ 编号（大端序）                   │
//   ├──────────────────────────────────────────────────────────┤
//   │ dseg2-dseg3 (字节 32-63) : 未使用                         │
//   └──────────────────────────────────────────────────────────┘
//
// 在 put_signal_counter 中的用途：
//   companion QP 上的 WAIT WQE 监控 main QP 的 CQ。
//   当 main QP 的 signal FAA 完成（CQE 产生）后，
//   WAIT 条件满足 → companion QP 继续执行下一个 Atomic FAA（counter++）。
//   这保证了 counter 只在数据+signal 都完成后才递增。
// ════════════════════════════════════════════════════════════════════════════
__device__ static __forceinline__ void doca_gpu_dev_verbs_wqe_prepare_wait(
    struct doca_gpu_dev_verbs_qp *qp, struct doca_gpu_dev_verbs_wqe *wqe_ptr, uint16_t wqe_idx,
    enum doca_gpu_dev_verbs_wqe_ctrl_flags ctrl_flags, const uint32_t max_index,
    const uint32_t qpn_cqn) {
    struct doca_gpunetio_ib_mlx5_wqe_ctrl_seg cseg;
    struct doca_gpunetio_ib_mlx5_wqe_wait_seg wseg;

    // 控制段：opcode = WAIT (0x0f)
    cseg.opmod_idx_opcode =
        doca_gpu_dev_verbs_bswap32(((uint32_t)wqe_idx << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) |
                                   DOCA_GPUNETIO_IB_MLX5_OPCODE_WAIT);
    // qpn_ds：QPN | segment 数量（WAIT 只有 cseg+wseg = 2 个 segment）
    cseg.qpn_ds = doca_gpu_dev_verbs_bswap32(__ldg(&qp->sq_num_shift8) |
                                             DOCA_GPUNETIO_VERBS_WQE_SEG_CNT_WAIT);
    cseg.fm_ce_se = ctrl_flags;  // CQ_UPDATE = WAIT 自身完成时也写 CQE
    // cseg.imm = 0;  // WAIT 不使用 immediate

    // Wait 段：指定等待条件
    wseg.max_index = doca_gpu_dev_verbs_bswap32(max_index);  // 等待的 WQE 完成索引
    wseg.qpn_cqn = doca_gpu_dev_verbs_bswap32(qpn_cqn);     // 监控的 CQ 编号

    // 只写 2 个 segment（32 字节），后 32 字节不用管
    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg0), (uint64_t *)&(cseg));
    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg1), (uint64_t *)&(wseg));
}

/**
 * @brief Prepare a Dump WQE
 *
 * @param qp - Queue Pair (QP)
 * @param wqe_ptr - Pointer to the WQE buffer to write the prepared WQE to
 * @param wqe_idx - Index of the WQE to be prepared
 * @param ctrl_flags - Control flags for the WQE
 * @param laddr - Local address for local dump
 * @param lkey - Local address mkey for local dump
 * @param bytes - Local address bytes to dump
 */
__device__ static inline void doca_gpu_dev_verbs_wqe_prepare_dump(
    struct doca_gpu_dev_verbs_qp *qp, struct doca_gpu_dev_verbs_wqe *wqe_ptr,
    const uint16_t wqe_idx, enum doca_gpu_dev_verbs_wqe_ctrl_flags ctrl_flags, const uint64_t laddr,
    const uint32_t lkey, const uint32_t bytes) {
    struct doca_gpunetio_ib_mlx5_wqe_ctrl_seg cseg;
    struct doca_gpunetio_ib_mlx5_wqe_data_seg dseg;

    cseg.opmod_idx_opcode =
        doca_gpu_dev_verbs_bswap32(((uint32_t)wqe_idx << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) |
                                   DOCA_GPUNETIO_IB_MLX5_OPCODE_DUMP);
    cseg.qpn_ds = __ldg(&qp->sq_num_shift8_be_2ds);
    cseg.fm_ce_se = ctrl_flags;

    dseg.byte_count = doca_gpu_dev_verbs_bswap32(bytes);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
    dseg.lkey = lkey;
#else
    dseg.lkey = doca_gpu_dev_verbs_bswap32(lkey);
#endif
    dseg.addr = doca_gpu_dev_verbs_bswap64(laddr);

    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg0), (uint64_t *)&(cseg));
    doca_gpu_dev_verbs_store_wqe_seg((uint64_t *)&(wqe_ptr->dseg1), (uint64_t *)&(dseg));
}
#endif /* DOCA_GPUNETIO_DEV_VERBS_QP_H */

/** @} */
