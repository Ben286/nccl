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
 * @file doca_gpunetio_dev_verbs_counter.cuh
 * @brief GDAKI CUDA device functions for One-sided Shared QP ops
 *
 * @{
 */

#ifndef DOCA_GPUNETIO_DEV_VERBS_COUNTER_CUH
#define DOCA_GPUNETIO_DEV_VERBS_COUNTER_CUH

#include "doca_gpunetio_dev_verbs_qp.cuh"
#include "doca_gpunetio_dev_verbs_cq.cuh"

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_submit_db_multi_qps_no_dbr
// ════════════════════════════════════════════════════════════════════════════
// GPU 直接敲 Doorbell 的简化版本（不更新 DBR，不加锁）。
//
// ★ 为什么不需要 DBR？
//   DBR（Doorbell Record）是 NIC 恢复路径的备用机制。正常情况下 NIC 通过
//   BlueFlame MMIO 写收到 Doorbell 通知；如果 NIC 忙或丢失了 MMIO 通知，
//   会去检查 DBR 来"补看"是否有新 WQE。在 NO_DBR 模式下假设 NIC 不会丢失
//   MMIO 通知，省去了写 DBR 的开销和序列化锁。
//
// ★ 为什么不需要 lock？
//   DB 模式需要 lock 是因为要保证 "atomic_max → ring_db → update_dbr → ring_db"
//   这个多步骤操作的原子性。NO_DBR 模式只做 "atomic_max → ring_db"，
//   而 atomic_max + ring_db 即使多线程交叉也是安全的（因为 atomic_max 已保证
//   sq_wqe_pi 单调递增，ring_db 只是通知 NIC 去看 SQ）。
//
// 流程（每条 QP）：
//   1. atomic_max(sq_wqe_pi, prod_index) — 推进全局 producer index
//      如果自己的 prod_index > 当前值，说明自己有新 WQE 需要通知 NIC
//      如果 ≤ 当前值，说明其他线程已经敲过更大的 DB，不需要重复敲
//   2. prepare_db — 构造 64-bit DB 值（QPN + WQE index 大端序）
//   3. fence_release + store_relaxed_mmio — 写 BlueFlame MMIO 地址
//      fence_release 确保 WQE 数据先于 DB 对 NIC 可见
//      store_relaxed_mmio 使用 PTX st.mmio 直达 PCIe BAR
//
// 参数：
//   qps          - QP 指针数组
//   prod_indices - 每条 QP 的 producer index
//   code_opt     - 可含 SKIP_DB_RINGING（完全跳过敲 DB）
//                  或 ASYNC_STORE_RELEASE（使用异步 store release 指令）
// ════════════════════════════════════════════════════════════════════════════
template <unsigned int num_qps,
          enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>
__device__ static __forceinline__ void doca_gpu_dev_verbs_submit_db_multi_qps_no_dbr(
    struct doca_gpu_dev_verbs_qp **qps, uint64_t *prod_indices,
    uint32_t code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT) {
    if (!(code_opt & DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_DB_RINGING)) {
        DOCA_GPUNETIO_VERBS_ASSERT(num_qps >= 2);
        uint64_t old_prod_indices[num_qps];
        __be64 db_vals[num_qps];

#pragma unroll 2
        for (unsigned int i = 0; i < num_qps; i++) {
            // atomic_max：推进 sq_wqe_pi 到 prod_indices[i]，返回旧值
            //   sq_wqe_pi 是 "NIC 应该处理到哪个 WQE" 的全局指示器
            //   atomic_max 保证只会往前推不会回退（多线程安全）
            old_prod_indices[i] =
                doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode, true>(
                    &qps[i]->sq_wqe_pi, prod_indices[i]);
            if (old_prod_indices[i] < prod_indices[i]) {
                // 自己确实推进了 sq_wqe_pi → 需要通知 NIC
                // __ldg：从只读 cache 加载 sq_db（BlueFlame MMIO 地址）
                __be64 *db_ptr = (__be64 *)__ldg((uintptr_t *)&qps[i]->sq_db);
                // prepare_db：构造 DB 值 = { qpn_ds, opmod_idx_opcode(prod_index << 8) }
                db_vals[i] = doca_gpu_dev_verbs_prepare_db(qps[i], prod_indices[i]);

#ifdef DOCA_GPUNETIO_VERBS_HAS_ASYNC_STORE_RELEASE
                if (code_opt & DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_ASYNC_STORE_RELEASE) {
                    // Hopper+ 异步 store release：合并 fence+store 为单条指令
                    doca_gpu_dev_verbs_async_store_release((uint64_t *)db_ptr,
                                                           (uint64_t)db_vals[i]);
                } else
#endif
                {
                    // 标准路径：fence_release + MMIO store
                    doca_gpu_dev_verbs_fence_release<sync_scope>();
#ifdef DOCA_GPUNETIO_VERBS_HAS_STORE_RELAXED_MMIO
                    {
                        // st.mmio.relaxed.sys.global — 写 NIC BlueFlame BAR 地址
                        doca_gpu_dev_verbs_store_relaxed_mmio((uint64_t *)db_ptr,
                                                              (uint64_t)db_vals[i]);
                    }
#else
                    {
                        // 回退方案：用 cuda::atomic_ref<sys> 的 relaxed store
                        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> db_ptr_aref(
                            *((uint64_t *)db_ptr));
                        db_ptr_aref.store(db_vals[i], cuda::memory_order_relaxed);
                    }
#endif
                }
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_submit_db_multi_qps
// ════════════════════════════════════════════════════════════════════════════
// GPU 直接敲 Doorbell 的完整版本（含锁 + DBR 更新 + 双敲 DB）。
// 这是 put_signal_counter 在默认 GPU_SM_DB 模式下的实际执行路径。
//
// ★ 为什么需要 lock + unlock？
//   多个 GPU 线程可能同时要敲同一个 QP 的 Doorbell。如果不加锁：
//   - 线程 A 敲了 DB(prod=10)
//   - 线程 B 敲了 DB(prod=12) 
//   - 线程 A 写 DBR(prod=10) ← 这会把 DBR 倒退！
//   lock 保证 "atomic_max → ring_db → update_dbr → ring_db" 的原子性。
//
// ★ 为什么敲两次 DB？（Double Doorbell 策略）
//   MLX5 NIC 有一个已知问题：如果 NIC 正在处理旧 WQE 时收到 DB，
//   可能在极端时序下丢失这次通知。为了可靠性：
//   1. 第一次 ring_db："尽早"通知 NIC 有新 WQE（Early ringing）
//   2. update_dbr：更新 Doorbell Record（NIC 恢复路径时查看的备份）
//   3. 第二次 ring_db：确保 NIC 看到最新的 prod_index
//   即使第一次 DB 丢失，第二次也会兜底。
//
// ★ second_db_sync_scope 的含义：
//   第二次 DB 的 fence_release 使用较弱的同步范围（至多 GPU 级别）。
//   因为第一次 DB 已经做了完整的 sys 级 fence_release，WQE 数据已经
//   对 NIC 可见了；第二次只是补发通知，不需要重新做 sys 级 fence。
//
// 整体流程（每条 QP）：
//   ┌─ lock ──────────────────────────────────────────────────┐
//   │ 1. atomic_max(sq_wqe_pi, prod_index)                    │
//   │ 2. if 推进了：                                           │
//   │    ├─ prepare_db → db_val                               │
//   │    └─ fence_release + store_mmio(db_ptr, db_val) ← 第一次│
//   └─────────────────────────────────────────────────────────┘
//   ┌─ 仍在 lock 内 ─────────────────────────────────────────┐
//   │ 3. if 推进了：                                           │
//   │    ├─ update_dbr(prod_index) ← 写 NIC 恢复备份         │
//   │    └─ fence_release + store_mmio(db_ptr, db_val) ← 第二次│
//   │ 4. unlock                                               │
//   └─────────────────────────────────────────────────────────┘
// ════════════════════════════════════════════════════════════════════════════
template <unsigned int num_qps,
          enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>
__device__ static __forceinline__ void doca_gpu_dev_verbs_submit_db_multi_qps(
    struct doca_gpu_dev_verbs_qp **qps, uint64_t *prod_indices,
    uint32_t code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT) {
    if (!(code_opt & DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_DB_RINGING)) {
        DOCA_GPUNETIO_VERBS_ASSERT(num_qps >= 2);
        uint64_t old_prod_indices[num_qps];
        __be64 db_vals[num_qps];

        // ── 第一轮：加锁 + 推进 sq_wqe_pi + 第一次敲 DB（Early Ringing）──
#pragma unroll 2
        for (unsigned int i = 0; i < num_qps; i++) {
            // 加锁：序列化对同一 QP Doorbell 的访问
            doca_gpu_dev_verbs_lock<resource_sharing_mode>(&qps[i]->sq_lock);
            // atomic_max：推进 sq_wqe_pi 到自己的 prod_index，返回旧值
            old_prod_indices[i] =
                doca_gpu_dev_verbs_atomic_max<uint64_t, resource_sharing_mode, true>(
                    &qps[i]->sq_wqe_pi, prod_indices[i]);
            if (old_prod_indices[i] < prod_indices[i]) {
                // 自己推进了 sq_wqe_pi → 第一次 Early Ringing
                __be64 *db_ptr = (__be64 *)__ldg((uintptr_t *)&qps[i]->sq_db);
                db_vals[i] = doca_gpu_dev_verbs_prepare_db(qps[i], prod_indices[i]);

#ifdef DOCA_GPUNETIO_VERBS_HAS_ASYNC_STORE_RELEASE
                if (code_opt & DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_ASYNC_STORE_RELEASE) {
                    doca_gpu_dev_verbs_async_store_release((uint64_t *)db_ptr,
                                                           (uint64_t)db_vals[i]);
                } else
#endif
                {
                    doca_gpu_dev_verbs_fence_release<sync_scope>();
#ifdef DOCA_GPUNETIO_VERBS_HAS_STORE_RELAXED_MMIO
                    {
                        doca_gpu_dev_verbs_store_relaxed_mmio((uint64_t *)db_ptr,
                                                              (uint64_t)db_vals[i]);
                    }
#else
                    {
                        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> db_ptr_aref(
                            *((uint64_t *)db_ptr));
                        db_ptr_aref.store(db_vals[i], cuda::memory_order_relaxed);
                    }
#endif
                }
            }
        }

        // ── 第二轮：更新 DBR + 第二次敲 DB（Recovery Path Safeguard）──
        // 第二次 DB 用较弱的同步范围：WQE 数据已在第一次 fence 后对 NIC 可见
        constexpr enum doca_gpu_dev_verbs_sync_scope second_db_sync_scope =
            (sync_scope <= DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU) ? sync_scope
                                                               : DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU;

#pragma unroll 2
        for (unsigned int i = 0; i < num_qps; i++) {
            if (old_prod_indices[i] < prod_indices[i]) {
                // update_dbr：将 prod_index 写入 NIC 的 Doorbell Record 内存
                // 如果 NIC 触发恢复路径（missed doorbell），会从 DBR 读取最新 prod_index
                doca_priv_gpu_dev_verbs_update_dbr(qps[i], prod_indices[i]);
                // 第二次敲 DB：兜底确保 NIC 收到通知
                __be64 *db_ptr = (__be64 *)__ldg((uintptr_t *)&qps[i]->sq_db);
#ifdef DOCA_GPUNETIO_VERBS_HAS_ASYNC_STORE_RELEASE
                if (code_opt & DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_ASYNC_STORE_RELEASE) {
                    doca_gpu_dev_verbs_async_store_release((uint64_t *)db_ptr,
                                                           (uint64_t)db_vals[i]);
                } else
#endif
                {
                    doca_gpu_dev_verbs_fence_release<second_db_sync_scope>();
#ifdef DOCA_GPUNETIO_VERBS_HAS_STORE_RELAXED_MMIO
                    {
                        doca_gpu_dev_verbs_store_relaxed_mmio((uint64_t *)db_ptr,
                                                              (uint64_t)db_vals[i]);
                    }
#else
                    {
                        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> db_ptr_aref(
                            *((uint64_t *)db_ptr));
                        db_ptr_aref.store(db_vals[i], cuda::memory_order_relaxed);
                    }
#endif
                }
            }
            // 解锁：允许其他线程敲同一个 QP 的 Doorbell
            doca_gpu_dev_verbs_unlock<resource_sharing_mode>(&qps[i]->sq_lock);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_submit_proxy_multi_qps
// ════════════════════════════════════════════════════════════════════════════
// CPU 代理模式下的多 QP 提交。
//
// 工作方式：
//   GPU 线程不直接敲 NIC 的 MMIO Doorbell，而是：
//   1. fence_release：确保 WQE 数据对 CPU proxy 线程可见
//   2. ring_proxy：将 prod_index 写到 sq_db 指向的共享内存地址
//      （此地址在 proxy 模式下被重定义为 CPU↔GPU 共享的 prod_index 通信区）
//   CPU proxy 线程持续轮询该地址，发现 prod_index 更新后，
//   由 CPU 端调用 ibv_post_send / 直接写 Doorbell 通知 NIC。
//
// ring_proxy 内部逻辑：
//   - EXCLUSIVE 模式：直接 store prod_idx
//   - GPU 模式：fetch_max(prod_idx) — 多线程写时取最大值，不丢失
// ════════════════════════════════════════════════════════════════════════════
template <unsigned int num_qps,
          enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>
__device__ static __forceinline__ void doca_gpu_dev_verbs_submit_proxy_multi_qps(
    struct doca_gpu_dev_verbs_qp **qps, uint64_t *prod_indices) {
    DOCA_GPUNETIO_VERBS_ASSERT(num_qps >= 2);
    // fence_release：确保所有 WQE 写入（store_wqe_seg）对系统可见
    // 必须在 ring_proxy 之前，否则 CPU proxy 可能读到未完成的 WQE
    doca_gpu_dev_verbs_fence_release<sync_scope>();

#pragma unroll 2
    for (unsigned int i = 0; i < num_qps; i++) {
        // 将 prod_index 写到 proxy 共享地址，通知 CPU proxy "有新 WQE 到此索引"
        doca_gpu_dev_verbs_ring_proxy<resource_sharing_mode>(qps[i], prod_indices[i]);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_submit_multi_qps
// ════════════════════════════════════════════════════════════════════════════
// 同时向多条 QP 提交工作请求（敲 Doorbell 通知 NIC）。
// 这是 put_signal_counter 第三阶段的入口函数，负责将已填充好的 WQE 告知 NIC。
//
// ★ 三种 NIC Handler 模式：
//
//   1. GPU_SM_DB（GPU 直接敲 Doorbell）= 默认高性能模式
//      GPU 线程直接写 NIC 的 BlueFlame MMIO 寄存器 + Doorbell Record（DBR）。
//      → 调用 submit_db_multi_qps：lock → atomic_max(sq_wqe_pi) → ring DB → update DBR → ring DB → unlock
//      优点：最低延迟（GPU→NIC 直通，无 CPU 中转）
//      缺点：需要序列化（lock/unlock）避免多线程写同一个 MMIO
//
//   2. GPU_SM_NO_DBR（GPU 敲 DB 但不写 DBR）= 简化模式
//      与 DB 模式类似但跳过 DBR 更新，依赖 NIC 固件不会触发恢复路径。
//      → 调用 submit_db_multi_qps_no_dbr：无 lock → atomic_max → ring DB
//      适用：UAR（User Access Region）注册成功且不会触发 NIC 恢复的场景
//
//   3. CPU_PROXY（CPU 代理模式）= 兼容模式
//      GPU 线程不直接敲 Doorbell，只将 prod_index 写到共享内存。
//      CPU 上有一个 proxy 线程轮询该地址，发现更新后由 CPU 代敲 Doorbell。
//      → 调用 submit_proxy_multi_qps：fence_release → ring_proxy
//      适用：NIC 不支持 GPU 直接 MMIO 写的场景（旧硬件或特殊配置）
//
// 模式选择：
//   nic_handler 模板参数 = AUTO 时，运行时从 QP 结构体读取实际模式
//   nic_handler 模板参数 = 具体值时，编译期确定（零运行时开销）
//
// 参数：
//   qps          - QP 指针数组（如 [main_qp, companion_qp]）
//   prod_indices - 每条 QP 的 producer index（= 最后一个 WQE 索引 + 1）
//   code_opt     - 运行时优化标志（如 SKIP_DB_RINGING 跳过敲 DB）
// ════════════════════════════════════════════════════════════════════════════
template <unsigned int num_qps,
          enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_sync_scope sync_scope = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
          enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static __forceinline__ void doca_gpu_dev_verbs_submit_multi_qps(
    struct doca_gpu_dev_verbs_qp **qps, uint64_t *prod_indices,
    uint32_t code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT) {
    DOCA_GPUNETIO_VERBS_ASSERT(num_qps >= 2);
    if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO) {
        // 运行时从第一个 QP 读取 NIC handler 模式（通过只读 cache __ldg）
        const enum doca_gpu_dev_verbs_nic_handler qp_nic_handler =
            (enum doca_gpu_dev_verbs_nic_handler)__ldg((int *)&qps[0]->nic_handler);
        if (qp_nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB)
            doca_gpu_dev_verbs_submit_db_multi_qps<num_qps, resource_sharing_mode, sync_scope>(
                qps, prod_indices, code_opt);
        else if (qp_nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_NO_DBR)
            doca_gpu_dev_verbs_submit_db_multi_qps_no_dbr<num_qps, resource_sharing_mode,
                                                          sync_scope>(qps, prod_indices, code_opt);
        else
            doca_gpu_dev_verbs_submit_proxy_multi_qps<num_qps, resource_sharing_mode, sync_scope>(
                qps, prod_indices);
    } else if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB) {
        doca_gpu_dev_verbs_submit_db_multi_qps<num_qps, resource_sharing_mode, sync_scope>(
            qps, prod_indices, code_opt);
    } else if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_NO_DBR) {
        doca_gpu_dev_verbs_submit_db_multi_qps_no_dbr<num_qps, resource_sharing_mode, sync_scope>(
            qps, prod_indices, code_opt);
    } else {
        doca_gpu_dev_verbs_submit_proxy_multi_qps<num_qps, resource_sharing_mode, sync_scope>(
            qps, prod_indices);
    }
}

template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static __forceinline__ void doca_gpu_dev_verbs_put_counter(
    struct doca_gpu_dev_verbs_qp *qp, struct doca_gpu_dev_verbs_addr raddr,
    struct doca_gpu_dev_verbs_addr laddr, size_t size, struct doca_gpu_dev_verbs_qp *companion_qp,
    struct doca_gpu_dev_verbs_addr counter_raddr, struct doca_gpu_dev_verbs_addr counter_laddr,
    uint64_t counter_val, uint32_t code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT) {
    constexpr unsigned int num_qps = 2;
    struct doca_gpu_dev_verbs_wqe *wqe_ptr;
    uint64_t base_wqe_idx;
    uint64_t wqe_idx;
    size_t remaining_size = size;
    size_t size_;
    uint64_t num_chunks =
        doca_gpu_dev_verbs_div_ceil_aligned_pow2(size, DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE_SHIFT);
    num_chunks = num_chunks > 1 ? num_chunks : 1;

    // DOCA_GPUNETIO_VERBS_ASSERT(out_ticket != NULL);
    DOCA_GPUNETIO_VERBS_ASSERT(qp != NULL);
    // DOCA_GPUNETIO_VERBS_ASSERT(qp->mem_type == DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU);

    base_wqe_idx =
        doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(qp, num_chunks, code_opt);
#pragma unroll 1
    for (uint64_t i = 0; i < num_chunks; i++) {
        wqe_idx = base_wqe_idx + i;
        size_ = remaining_size > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
                    ? DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
                    : remaining_size;
        wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

        [[likely]] if (size_ > 0) {
            doca_gpu_dev_verbs_wqe_prepare_write(
                qp, wqe_ptr, wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0,
                raddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE), raddr.key,
                laddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE), laddr.key, size_);
        } else {
            doca_gpu_dev_verbs_wqe_prepare_nop(qp, wqe_ptr, wqe_idx,
                                               DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE);
        }
        remaining_size -= size_;
    }

    doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(qp, base_wqe_idx, wqe_idx);

    uint64_t companion_base_wqe_idx =
        doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(companion_qp, 2, code_opt);
    uint64_t companion_wqe_idx = companion_base_wqe_idx;

    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
    doca_gpu_dev_verbs_wqe_prepare_wait(companion_qp, wqe_ptr, companion_wqe_idx,
                                        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, wqe_idx,
                                        qp->cq_sq.cq_num);

    ++companion_wqe_idx;
    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
    doca_gpu_dev_verbs_wqe_prepare_atomic(
        companion_qp, wqe_ptr, companion_wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, counter_raddr.addr, counter_raddr.key,
        counter_laddr.addr, counter_laddr.key, sizeof(uint64_t), counter_val, 0);
    doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(companion_qp, companion_base_wqe_idx,
                                                              companion_wqe_idx);

    doca_gpu_dev_verbs_qp *qps[num_qps] = {qp, companion_qp};
    uint64_t prod_indices[num_qps] = {wqe_idx + 1, companion_wqe_idx + 1};

    // mark_wqes_ready has already called fence.release with sufficiently strong scope. No need to
    // call it again in submit.
    constexpr enum doca_gpu_dev_verbs_sync_scope submit_sync_scope =
        (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU)
            ? DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD
            : DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU;
    doca_gpu_dev_verbs_submit_multi_qps<num_qps, resource_sharing_mode, submit_sync_scope,
                                        nic_handler>(qps, prod_indices, code_opt);
}

template <typename T,
          enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static __forceinline__ void doca_gpu_dev_verbs_p_counter(
    struct doca_gpu_dev_verbs_qp *qp, struct doca_gpu_dev_verbs_addr raddr, T value,
    struct doca_gpu_dev_verbs_qp *companion_qp, struct doca_gpu_dev_verbs_addr counter_raddr,
    struct doca_gpu_dev_verbs_addr counter_laddr, uint64_t counter_val,
    uint32_t code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT) {
    constexpr unsigned int num_qps = 2;
    uint64_t wqe_idx;
    struct doca_gpu_dev_verbs_wqe *wqe_ptr;

    // DOCA_GPUNETIO_VERBS_ASSERT(out_ticket != NULL);
    DOCA_GPUNETIO_VERBS_ASSERT(qp != NULL);
    // DOCA_GPUNETIO_VERBS_ASSERT(qp->mem_type == DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU);

    wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(qp, 1, code_opt);
    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

    doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_header(qp, wqe_ptr, wqe_idx,
                                                         DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
                                                         raddr.addr, raddr.key, sizeof(T));
    doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_data<T>(qp, wqe_ptr, value);
    doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(qp, wqe_idx, wqe_idx);

    uint64_t companion_base_wqe_idx =
        doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(companion_qp, 2, code_opt);
    uint64_t companion_wqe_idx = companion_base_wqe_idx;

    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
    doca_gpu_dev_verbs_wqe_prepare_wait(companion_qp, wqe_ptr, companion_wqe_idx,
                                        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, wqe_idx,
                                        qp->cq_sq.cq_num);

    ++companion_wqe_idx;
    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
    doca_gpu_dev_verbs_wqe_prepare_atomic(
        companion_qp, wqe_ptr, companion_wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, counter_raddr.addr, counter_raddr.key,
        counter_laddr.addr, counter_laddr.key, sizeof(uint64_t), counter_val, 0);
    doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(companion_qp, companion_base_wqe_idx,
                                                              companion_wqe_idx);

    doca_gpu_dev_verbs_qp *qps[num_qps] = {qp, companion_qp};
    uint64_t prod_indices[num_qps] = {wqe_idx + 1, companion_wqe_idx + 1};

    // mark_wqes_ready has already called fence.release with sufficiently strong scope. No need to
    // call it again in submit.
    constexpr enum doca_gpu_dev_verbs_sync_scope submit_sync_scope =
        (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU)
            ? DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD
            : DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU;
    doca_gpu_dev_verbs_submit_multi_qps<num_qps, resource_sharing_mode, submit_sync_scope,
                                        nic_handler>(qps, prod_indices, code_opt);
}

template <enum doca_gpu_dev_verbs_signal_op sig_op,
          enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_put_signal_counter
// ════════════════════════════════════════════════════════════════════════════
// 功能：在 main QP 上发起 RDMA Write（数据传输）+ Atomic FAA（远端 signal 通知），
//       在 companion QP 上发起 WAIT（等 main QP 完成）+ Atomic FAA（本地 counter 计数）。
//
// ★ CQE 产生分析：
//   IB 规范：只有 signaled WQE（ctrl_flags 含 CQ_UPDATE）完成后才产生本地 CQE。
//   本函数中所有 WQE 都设置了 CQ_UPDATE，所以每个 WQE 都会产生本地 Send CQ CQE。
//
//   各 WQE 类型的 CQE 行为：
//     WQE 类型              本地 Send CQ    对端 Recv CQ
//     ──────────────────    ────────────    ────────────
//     RDMA Write (数据)     ✓ (signaled)    ✗（单边操作，对端无感知）
//     Atomic FAA (signal)   ✓ (signaled)    ✗（原子操作不产生对端 CQE）
//     NOP (占位)            ✓ (signaled)    ✗（空操作）
//     WAIT (companion)      ✓ (signaled)    N/A（纯本地操作）
//     Atomic FAA (counter)  ✓ (signaled)    ✗（原子操作）
//
//   ★ 关键：对端完全不通过 CQE 感知数据到达！
//     对端通过 GPU 线程轮询 signal_table 的值变化来检测数据到达。
//     只有 RDMA Write with Immediate 才会在对端 Recv CQ 产生 CQE，
//     而这里用的是普通 RDMA Write（opcode=RDMA_WRITE, immediate=0）。
//
//   ★ 为什么所有 WQE 都必须设 CQ_UPDATE？
//     wait_until_slot_available 通过 poll_cq_at 检测旧 WQE 是否完成来回收
//     SQ 环形缓冲区槽位。此机制要求 WQE 索引与 CQE 索引一一对应，
//     即每个 WQE 都必须产生 CQE。如果某个 WQE 不是 signaled，
//     poll_cq_at 将永远等不到对应的 CQE，导致死等。
//
// 涉及两条 QP 的 WQE 布局：
//
//   Main QP (qp):
//   ┌──────────────────┐
//   │ WQE[0..N-1]      │  RDMA_WRITE × N（数据分片，每片 ≤ MAX_TRANSFER_SIZE）
//   │ WQE[N]           │  ATOMIC_FA（对端 signal += sig_val）
//   └──────────────────┘
//   → mark_wqes_ready  → NIC 按序执行：先写完数据，再原子加 signal
//
//   Companion QP (companion_qp):
//   ┌──────────────────┐
//   │ WQE[0]           │  WAIT（等待 main QP 的 WQE[N] 在 main QP 的 Send CQ 中产生 CQE）
//   │ WQE[1]           │  ATOMIC_FA（本地 counter += counter_val）
//   └──────────────────┘
//   → mark_wqes_ready  → NIC 执行：等 main QP 全部完成后，才增加本地 counter
//
//   最后同时敲两条 QP 的 Doorbell，通知 NIC 开始处理。
//
// 参数说明：
//   qp             - main QP：承载数据传输和 signal 操作
//   raddr/laddr    - 数据传输的远端/本地地址（含 MR key）
//   size           - 数据总字节数
//   sig_raddr/sig_laddr - signal 的远端/本地地址（Atomic FAA 的目标/返回值地址）
//   sig_val        - signal 增量（通常为 1）
//   companion_qp   - companion QP：独立 QP，承载 WAIT + counter 操作
//   counter_raddr/counter_laddr - counter 的远端/本地地址（FAA 写自己的 counter 表）
//   counter_val    - counter 增量（通常为 1）
// ════════════════════════════════════════════════════════════════════════════
__device__ static __forceinline__ void doca_gpu_dev_verbs_put_signal_counter(
    struct doca_gpu_dev_verbs_qp *qp, struct doca_gpu_dev_verbs_addr raddr,
    struct doca_gpu_dev_verbs_addr laddr, size_t size, struct doca_gpu_dev_verbs_addr sig_raddr,
    struct doca_gpu_dev_verbs_addr sig_laddr, uint64_t sig_val,
    struct doca_gpu_dev_verbs_qp *companion_qp, struct doca_gpu_dev_verbs_addr counter_raddr,
    struct doca_gpu_dev_verbs_addr counter_laddr, uint64_t counter_val,
    uint32_t code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT) {
    constexpr unsigned int num_qps = 2;  // main QP + companion QP
    struct doca_gpu_dev_verbs_wqe *wqe_ptr;
    uint64_t base_wqe_idx;
    uint64_t wqe_idx;
    size_t remaining_size = size;
    size_t size_;
    // 计算数据分片数：size / MAX_TRANSFER_SIZE 向上取整
    // MLX5 单个 WQE 最大传输 MAX_TRANSFER_SIZE 字节（通常 2GB），超过需拆分
    uint64_t num_chunks =
        doca_gpu_dev_verbs_div_ceil_aligned_pow2(size, DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE_SHIFT);
    num_chunks = num_chunks > 1 ? num_chunks : 1;  // 至少 1 片

    DOCA_GPUNETIO_VERBS_ASSERT(qp != NULL);

    // ════════════════════════════════════════════════════════════════════════
    // 第一阶段：Main QP — 数据传输 + Signal
    // ════════════════════════════════════════════════════════════════════════

    // 步骤 1：在 main QP 的 SQ 中原子预留 (num_chunks + 1) 个 WQE 槽位
    //   num_chunks 个用于 RDMA Write 数据分片
    //   1 个用于 Atomic FAA signal
    //   原子操作保证多线程并发时不会抢到同一个槽位
    base_wqe_idx =
        doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(qp, num_chunks + 1, code_opt);

    // 步骤 2：填充 RDMA Write WQE（数据分片）
    //   每个 WQE 对应一个 ≤ MAX_TRANSFER_SIZE 的数据块
    //   同一 QP 上 NIC 保证按 WQE 索引顺序执行
#pragma unroll 1  // 禁止循环展开（分片数运行时才知道）
    for (uint64_t i = 0; i < num_chunks; i++) {
        wqe_idx = base_wqe_idx + i;
        size_ = remaining_size > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
                    ? DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
                    : remaining_size;
        // 获取环形 WQE 缓冲区中第 wqe_idx 个 WQE 的指针（自动取模）
        wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

        // [[likely]] 是 C++20 属性，提示编译器此分支在运行时"很可能"成立。
        // 编译器会将 likely 分支的指令布局优化为直通（fall-through）路径，
        // 减少分支预测失败的性能损失。这里 size_ > 0 是常态，size_==0 只在
        // 最后一片可能出现（数据刚好是 MAX_TRANSFER_SIZE 整数倍时的边界情况）。
        [[likely]] if (size_ > 0) {
            // 填充 64 字节 WQE：
            //   cseg: 控制段 — opcode=RDMA_WRITE, CQ_UPDATE(完成时产生 CQE)
            //   rseg: 远端地址段 — raddr(对端 buffer VA) + rkey(对端 MR 密钥)
            //   dseg: 本地数据段 — laddr(本地 buffer VA) + lkey(本地 MR 密钥) + byte_count
            // NIC 执行：DMA 读本地 laddr → 网络传输 → DMA 写对端 raddr
            doca_gpu_dev_verbs_wqe_prepare_write(
                qp, wqe_ptr, wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0,
                raddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE), raddr.key,
                laddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE), laddr.key, size_);
        } else {
            // size_==0 时填 NOP（空操作），占位但不传数据
            doca_gpu_dev_verbs_wqe_prepare_nop(qp, wqe_ptr, wqe_idx,
                                               DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE);
        }
        remaining_size -= size_;
    }

    // 步骤 3：填充 Atomic FAA WQE（Signal 通知对端）
    //   紧跟在所有 RDMA Write 之后，同一 QP 保证顺序执行
    //   ★ 关键保证：NIC 在 Atomic FAA 执行前，所有前序 RDMA Write 已完成
    //     这是 IB 规范要求的同一 QP 内 WQE 顺序完成语义（PCIe ordering + NIC 内部流水线）
    //   操作：对端 signal_table[signalId] += sig_val
    //   Atomic FAA 会返回旧值到 sig_laddr（sink buffer），但我们不关心旧值
    ++wqe_idx;
    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);
    doca_gpu_dev_verbs_wqe_prepare_atomic(
        qp, wqe_ptr, wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, sig_raddr.addr, sig_raddr.key, sig_laddr.addr,
        sig_laddr.key, sizeof(uint64_t), sig_val, 0);

    // 步骤 4：标记 main QP 的 WQE[base..wqe_idx] 为"可消费"
    //   实现：CAS 原子推进 qp->sq_ready_index = wqe_idx + 1
    //   保证多线程间 WQE 按预留顺序依次 ready（不会跳过前面线程的 WQE）
    //   内含 fence.release 确保 WQE 数据对 NIC 可见
    doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(qp, base_wqe_idx, wqe_idx);

    // ════════════════════════════════════════════════════════════════════════
    // 第二阶段：Companion QP — WAIT + Counter
    // ════════════════════════════════════════════════════════════════════════
    // 为什么需要 companion QP？
    //   counter 是写给自己的（本地完成计数），语义是"main QP 全部完成后才 +1"
    //   如果放在 main QP 上，counter FAA 会和 signal FAA 竞争 WQE 槽位
    //   更重要的是：WAIT WQE 只能等待**另一个 QP** 的 CQ，不能等自己
    //   所以用独立的 companion QP 来实现"等 main QP 完成 → 再做 counter"

    // 步骤 5：在 companion QP 上预留 2 个 WQE 槽位（WAIT + Atomic FAA）
    uint64_t companion_base_wqe_idx =
        doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(companion_qp, 2, code_opt);
    uint64_t companion_wqe_idx = companion_base_wqe_idx;

    // 步骤 6：填充 WAIT WQE（等待 main QP 完成）
    //   WAIT 是 MLX5 硬件特有的 WQE 类型（opcode = 0x0f）
    //   语义：NIC 暂停处理 companion QP 的后续 WQE，直到 main QP 的 Send CQ
    //         中出现 wqe_idx 对应的 CQE（即 signal FAA 已完成并产生了 signaled CQE）
    //   ★ 注意：WAIT 监控的是 main QP 的本地 Send CQ（不是对端 Recv CQ）
    //     因为所有 WQE 都设了 CQ_UPDATE，signal FAA 完成时会在 main QP 的
    //     Send CQ 产生 CQE → WAIT 检测到 → 放行 companion QP 后续 WQE
    //   参数：
    //     max_index = wqe_idx  — 等待 main QP 的第几个 WQE 完成
    //     qpn_cqn = qp->cq_sq.cq_num — 监控 main QP 的 Send CQ 编号
    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
    doca_gpu_dev_verbs_wqe_prepare_wait(companion_qp, wqe_ptr, companion_wqe_idx,
                                        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, wqe_idx,
                                        qp->cq_sq.cq_num);

    // 步骤 7：填充 Atomic FAA WQE（本地 Counter 自增）
    //   WAIT 放行后才执行，保证此时 main QP 的所有操作（数据+signal）都已完成
    //   操作：自己的 counter_table[counterId] += counter_val（通常 +1）
    //   counter_raddr.key = 自己的 rkey（通过 RDMA Atomic 写自己，绕过 PCIe ordering 问题）
    //   counter_laddr = sink buffer（FAA 返回的旧值丢弃）
    ++companion_wqe_idx;
    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
    doca_gpu_dev_verbs_wqe_prepare_atomic(
        companion_qp, wqe_ptr, companion_wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, counter_raddr.addr, counter_raddr.key,
        counter_laddr.addr, counter_laddr.key, sizeof(uint64_t), counter_val, 0);
    // 步骤 8：标记 companion QP 的 WQE 为"可消费"
    doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(companion_qp, companion_base_wqe_idx,
                                                              companion_wqe_idx);

    // ════════════════════════════════════════════════════════════════════════
    // 第三阶段：同时敲两条 QP 的 Doorbell
    // ════════════════════════════════════════════════════════════════════════
    // prod_indices[i] = 该 QP 最新的 WQE 生产者索引（已填充完毕的 WQE 数量）
    // Doorbell 通知 NIC："从上次到 prod_index 的 WQE 都准备好了，可以处理"
    doca_gpu_dev_verbs_qp *qps[num_qps] = {qp, companion_qp};
    uint64_t prod_indices[num_qps] = {wqe_idx + 1, companion_wqe_idx + 1};

    // mark_wqes_ready 已经做了 fence.release，保证 WQE 数据在敲 Doorbell 前对 NIC 可见
    // 所以 submit 阶段不需要再做 GPU 级别的 fence，用 THREAD scope 即可
    constexpr enum doca_gpu_dev_verbs_sync_scope submit_sync_scope =
        (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU)
            ? DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD
            : DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU;
    // 根据 NIC handler 模式选择敲 Doorbell 的方式：
    //   GPU_SM_DB：GPU 线程直接写 BlueFlame Doorbell（MMIO）+ 写 DBR（Doorbell Record）
    //   GPU_SM_NO_DBR：GPU 线程写 BlueFlame 但不写 DBR（依赖 UAR 注册成功）
    //   CPU_PROXY：GPU 线程只推进 prod_index，CPU proxy 线程轮询后代敲 Doorbell
    doca_gpu_dev_verbs_submit_multi_qps<num_qps, resource_sharing_mode, submit_sync_scope,
                                        nic_handler>(qps, prod_indices, code_opt);
}

template <enum doca_gpu_dev_verbs_signal_op sig_op,
          enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static __forceinline__ void doca_gpu_dev_verbs_signal_counter(
    struct doca_gpu_dev_verbs_qp *qp, struct doca_gpu_dev_verbs_addr sig_raddr,
    struct doca_gpu_dev_verbs_addr sig_laddr, uint64_t sig_val,
    struct doca_gpu_dev_verbs_qp *companion_qp, struct doca_gpu_dev_verbs_addr counter_raddr,
    struct doca_gpu_dev_verbs_addr counter_laddr, uint64_t counter_val,
    uint32_t code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT) {
    constexpr unsigned int num_qps = 2;
    uint64_t wqe_idx;
    struct doca_gpu_dev_verbs_wqe *wqe_ptr;

    // DOCA_GPUNETIO_VERBS_ASSERT(out_ticket != NULL);
    DOCA_GPUNETIO_VERBS_ASSERT(qp != NULL);
    // DOCA_GPUNETIO_VERBS_ASSERT(qp->mem_type == DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU);

    // Signal
    wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(qp, 1, code_opt);
    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);
    doca_gpu_dev_verbs_wqe_prepare_atomic(
        qp, wqe_ptr, wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, sig_raddr.addr, sig_raddr.key, sig_laddr.addr,
        sig_laddr.key, sizeof(uint64_t), sig_val, 0);

    doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(qp, wqe_idx, wqe_idx);

    // Counter
    uint64_t companion_base_wqe_idx =
        doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(companion_qp, 2, code_opt);
    uint64_t companion_wqe_idx = companion_base_wqe_idx;

    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
    doca_gpu_dev_verbs_wqe_prepare_wait(companion_qp, wqe_ptr, companion_wqe_idx,
                                        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, wqe_idx,
                                        qp->cq_sq.cq_num);

    ++companion_wqe_idx;
    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
    doca_gpu_dev_verbs_wqe_prepare_atomic(
        companion_qp, wqe_ptr, companion_wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, counter_raddr.addr, counter_raddr.key,
        counter_laddr.addr, counter_laddr.key, sizeof(uint64_t), counter_val, 0);
    doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(companion_qp, companion_base_wqe_idx,
                                                              companion_wqe_idx);

    doca_gpu_dev_verbs_qp *qps[num_qps] = {qp, companion_qp};
    uint64_t prod_indices[num_qps] = {wqe_idx + 1, companion_wqe_idx + 1};

    // mark_wqes_ready has already called fence.release with sufficiently strong scope. No need to
    // call it again in submit.
    constexpr enum doca_gpu_dev_verbs_sync_scope submit_sync_scope =
        (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU)
            ? DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD
            : DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU;
    doca_gpu_dev_verbs_submit_multi_qps<num_qps, resource_sharing_mode, submit_sync_scope,
                                        nic_handler>(qps, prod_indices, code_opt);
}

template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode =
              DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
          enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
__device__ static __forceinline__ void doca_gpu_dev_verbs_get_counter(
    struct doca_gpu_dev_verbs_qp *qp, struct doca_gpu_dev_verbs_addr raddr,
    struct doca_gpu_dev_verbs_addr laddr, size_t size, struct doca_gpu_dev_verbs_qp *companion_qp,
    struct doca_gpu_dev_verbs_addr counter_raddr, struct doca_gpu_dev_verbs_addr counter_laddr,
    uint64_t counter_val, uint32_t code_opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT) {
    constexpr unsigned int num_qps = 2;
    struct doca_gpu_dev_verbs_wqe *wqe_ptr;
    uint64_t base_wqe_idx;
    uint64_t wqe_idx;
    size_t remaining_size = size;
    size_t size_;
    uint64_t num_chunks =
        doca_gpu_dev_verbs_div_ceil_aligned_pow2(size, DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE_SHIFT);
    num_chunks = num_chunks > 1 ? num_chunks : 1;

    base_wqe_idx =
        doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(qp, num_chunks, code_opt);
#pragma unroll 1
    for (uint64_t i = 0; i < num_chunks; i++) {
        wqe_idx = base_wqe_idx + i;
        size_ = remaining_size > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
                    ? DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE
                    : remaining_size;
        wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

        [[likely]] if (size_ > 0) {
            doca_gpu_dev_verbs_wqe_prepare_read(
                qp, wqe_ptr, wqe_idx,
                DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
                raddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE), raddr.key,
                laddr.addr + (i * DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE), laddr.key, size_);
        } else {
            doca_gpu_dev_verbs_wqe_prepare_nop(qp, wqe_ptr, wqe_idx,
                                               DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE);
        }
        remaining_size -= size_;
    }

    doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(qp, base_wqe_idx, wqe_idx);

    uint64_t companion_base_wqe_idx =
        doca_gpu_dev_verbs_reserve_wq_slots<resource_sharing_mode>(companion_qp, 2, code_opt);
    uint64_t companion_wqe_idx = companion_base_wqe_idx;

    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
    doca_gpu_dev_verbs_wqe_prepare_wait(companion_qp, wqe_ptr, companion_wqe_idx,
                                        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, wqe_idx,
                                        qp->cq_sq.cq_num);

    ++companion_wqe_idx;
    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(companion_qp, companion_wqe_idx);
    doca_gpu_dev_verbs_wqe_prepare_atomic(
        companion_qp, wqe_ptr, companion_wqe_idx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, counter_raddr.addr, counter_raddr.key,
        counter_laddr.addr, counter_laddr.key, sizeof(uint64_t), counter_val, 0);
    doca_gpu_dev_verbs_mark_wqes_ready<resource_sharing_mode>(companion_qp, companion_base_wqe_idx,
                                                              companion_wqe_idx);

    doca_gpu_dev_verbs_qp *qps[num_qps] = {qp, companion_qp};
    uint64_t prod_indices[num_qps] = {wqe_idx + 1, companion_wqe_idx + 1};

    // mark_wqes_ready has already called fence.release with sufficiently strong scope. No need to
    // call it again in submit.
    constexpr enum doca_gpu_dev_verbs_sync_scope submit_sync_scope =
        (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU)
            ? DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD
            : DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU;
    doca_gpu_dev_verbs_submit_multi_qps<num_qps, resource_sharing_mode, submit_sync_scope,
                                        nic_handler>(qps, prod_indices, code_opt);
}

#endif /* DOCA_GPUNETIO_DEV_VERBS_COUNTER_CUH */
