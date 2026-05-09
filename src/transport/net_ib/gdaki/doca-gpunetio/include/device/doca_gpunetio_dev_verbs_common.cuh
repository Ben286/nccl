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
 * @file doca_gpunetio_dev_verbs_common.cuh
 * @brief GDAKI common device structs and functions
 *
 * @{
 */
#ifndef DOCA_GPUNETIO_DEV_VERBS_COMMON_H
#define DOCA_GPUNETIO_DEV_VERBS_COMMON_H

#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda/atomic>
#include <math.h>

#include "../common/doca_gpunetio_verbs_dev.h"

#if __CUDA_ARCH__ >= 1000
#define DOCA_GPUNETIO_VERBS_HAS_ASYNC_STORE_RELEASE 1
#endif

#if __CUDA_ARCH__ >= 900
#define DOCA_GPUNETIO_VERBS_HAS_TMA_COPY 1
#endif

#if CUDA_VERSION >= 12020
#define DOCA_GPUNETIO_VERBS_HAS_STORE_RELAXED_MMIO 1
#else
#warning "warning: doca_gpunetio should be used with a CUDA version >= 12020."
#endif

#if CUDA_VERSION >= 12080 && __CUDA_ARCH__ >= 900
#define DOCA_GPUNETIO_VERBS_HAS_FENCE_ACQUIRE_RELEASE_PTX 1
#endif

/**
 * @brief Queries the global timer
 *
 * @return The value of the global timer
 */
__device__ static __forceinline__ uint64_t doca_gpu_dev_verbs_query_globaltimer() {
    uint64_t ret;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ret)::"memory");
    return ret;
}

__device__ static __forceinline__ unsigned int doca_gpu_dev_verbs_get_lane_id() {
    unsigned int ret;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(ret));
    return ret;
}

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_bswap64
// ════════════════════════════════════════════════════════════════════════════
// 64 位字节序反转（小端 ↔ 大端）。
// MLX5 WQE 格式要求大端序（网络字节序），GPU 是小端，需字节翻转。
//
// 算法：将 64 位拆成两个 32 位，分别用 prmt 指令反转字节序，再交换高低位。
//   prmt.b32 dst, src, ign, 0x0123 → 将 src 的字节 [3,2,1,0] 重排为 [0,1,2,3]
//   即：字节序反转。mask=0x0123 表示输出字节取自输入的 byte[0], byte[1], byte[2], byte[3]
//
// 示例：0x0102030405060708 → 0x0807060504030201
// ════════════════════════════════════════════════════════════════════════════
__device__ static __forceinline__ uint64_t doca_gpu_dev_verbs_bswap64(uint64_t x) {
    uint64_t ret;
    asm volatile(
        "{\n\t"
        ".reg .b32 mask;\n\t"
        ".reg .b32 ign;\n\t"
        ".reg .b32 lo;\n\t"
        ".reg .b32 hi;\n\t"
        ".reg .b32 new_lo;\n\t"
        ".reg .b32 new_hi;\n\t"
        "mov.b32 mask, 0x0123;\n\t"
        "mov.b64 {lo,hi}, %1;\n\t"
        "prmt.b32 new_hi, lo, ign, mask;\n\t"
        "prmt.b32 new_lo, hi, ign, mask;\n\t"
        "mov.b64 %0, {new_lo,new_hi};\n\t"
        "}"
        : "=l"(ret)
        : "l"(x));
    return ret;
}

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_bswap32
// ════════════════════════════════════════════════════════════════════════════
// 32 位字节序反转。使用 PTX prmt.b32 指令实现零延迟字节重排。
// prmt（permute）指令可在单周期内完成 4 字节的任意重排。
// mask=0x0123：输出 byte[3]=input byte[0], byte[2]=input byte[1], etc.
// 示例：0xAABBCCDD → 0xDDCCBBAA
// ════════════════════════════════════════════════════════════════════════════
__device__ static __forceinline__ uint32_t doca_gpu_dev_verbs_bswap32(uint32_t x) {
    uint32_t ret;
    asm volatile(
        "{\n\t"
        ".reg .b32 mask;\n\t"
        ".reg .b32 ign;\n\t"
        "mov.b32 mask, 0x0123;\n\t"
        "prmt.b32 %0, %1, ign, mask;\n\t"
        "}"
        : "=r"(ret)
        : "r"(x));
    return ret;
}

__device__ static __forceinline__ uint16_t doca_gpu_dev_verbs_bswap16(uint16_t x) {
    uint16_t ret;
    asm volatile(
        "{\n\t"
        ".reg .b8 hi;\n\t"
        ".reg .b8 lo;\n\t"
        "mov.b16 {hi, lo}, %1;\n\t"
        "mov.b16 %0, {lo, hi};\n\t"
        "}"
        : "=h"(ret)
        : "h"(x));
    return ret;
}

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_store_relaxed_mmio
// ════════════════════════════════════════════════════════════════════════════
// 以放松内存序写入 64 位 MMIO 地址（如 Doorbell）。
//
// PTX 指令：st.mmio.relaxed.sys.global.b64 [ptr], val
//   .mmio     = 标记目标为 Memory-Mapped I/O（非普通显存）
//               NIC 的 Doorbell 寄存器通过 BAR 映射到 GPU 地址空间
//               .mmio 提示硬件这是设备寄存器，不能被缓存或合并
//   .relaxed  = 放松内存序（调用者自行通过 fence 保证 ordering）
//   .sys      = 系统作用域（对 PCIe 设备可见）
//
// 在 DOCA 中的应用：
//   - ring_db 向 NIC Doorbell 写入 64 位值（QPN + prod_index）
//     通知 NIC 有新 WQE 需要处理
//   仅 CUDA 12.2+ 支持（旧版本回退到 cuda::atomic_ref::store）
// ════════════════════════════════════════════════════════════════════════════
#ifdef DOCA_GPUNETIO_VERBS_HAS_STORE_RELAXED_MMIO
__device__ static __forceinline__ void doca_gpu_dev_verbs_store_relaxed_mmio(uint64_t *ptr,
                                                                             uint64_t val) {
    asm volatile("st.mmio.relaxed.sys.global.b64 [%0], %1;" : : "l"(ptr), "l"(val));
}
#endif

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_fence_acquire
// ════════════════════════════════════════════════════════════════════════════
// 获取栅栏（Acquire Fence）：确保此指令之后的所有内存读写
// 都能看到此指令之前其他线程/设备通过 release 发布的数据。
//
// 与 fence_release 配对使用，构成经典的 Release-Acquire 同步语义：
//   线程 A: 写数据 → fence.release → 写 flag
//   线程 B: 读 flag → fence.acquire → 读数据 ← 保证看到线程 A 写的数据
//
// 在 DOCA 中的应用场景：
//   - poll_cq_at 检测到 CQE 后调用 fence.acquire.sys
//     → 保证后续读 WQE 数据时能看到 NIC DMA 写入的最新内容
//   - lock 加锁成功后调用 fence.acquire
//     → 保证临界区内的读写看到前一个持有者的修改
//
// sync_scope 控制栅栏的可见范围：
//   THREAD：无操作（单线程不需要同步）
//   CTA：block 内线程间可见
//   GPU：device 内所有线程可见
//   SYS：GPU + CPU + NIC + PCIe 全系统可见
// ════════════════════════════════════════════════════════════════════════════
template <enum doca_gpu_dev_verbs_sync_scope sync_scope>
__device__ static __forceinline__ void doca_gpu_dev_verbs_fence_acquire() {
#ifdef DOCA_GPUNETIO_VERBS_HAS_FENCE_ACQUIRE_RELEASE_PTX
    if (sync_scope == DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA)
        asm volatile("fence.acquire.cta;");
    else if (sync_scope == DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU)
        asm volatile("fence.acquire.gpu;");
    else if (sync_scope == DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS)
        asm volatile("fence.acquire.sys;");
    else
        ;  // no-op
#else
    // fence.acquire is not available in PTX. Emulate that with st.release.
    uint32_t dummy;
    uint32_t val = 0;
    if (sync_scope == DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA)
        asm volatile("ld.acquire.cta.b32 %0, [%1];" : "=r"(val) : "l"(&dummy));
    else if (sync_scope == DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU)
        asm volatile("ld.acquire.gpu.b32 %0, [%1];" : "=r"(val) : "l"(&dummy));
    else if (sync_scope == DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS)
        asm volatile("ld.acquire.sys.b32 %0, [%1];" : "=r"(val) : "l"(&dummy));
    else if (sync_scope == DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD)
        ;  // no-op
#endif
}

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_fence_release
// ════════════════════════════════════════════════════════════════════════════
// 释放栅栏（Release Fence）：确保此指令之前的所有内存写入
// 在此指令执行后对其他线程/设备可见。
//
// 在 DOCA 中的应用场景：
//   - mark_wqes_ready 中 CAS 之前调用 fence.release
//     → 保证 WQE 数据写入对后续读取 ready_index 的线程可见
//   - submit_db 敲 Doorbell 前调用 fence.release
//     → 保证 NIC 通过 PCIe 读取 WQE 时能看到 GPU 写入的最新数据
//   - unlock 释放锁时内含 store.release
//     → 保证临界区内的修改对下一个 lock 持有者可见
//
// 在不支持 fence PTX 的旧架构上，用 st.release 写一个 dummy 变量来模拟。
// ════════════════════════════════════════════════════════════════════════════
template <enum doca_gpu_dev_verbs_sync_scope sync_scope>
__device__ static __forceinline__ void doca_gpu_dev_verbs_fence_release() {
#ifdef DOCA_GPUNETIO_VERBS_HAS_FENCE_ACQUIRE_RELEASE_PTX
    if (sync_scope == DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA)
        asm volatile("fence.release.cta;");
    else if (sync_scope == DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU)
        asm volatile("fence.release.gpu;");
    else if (sync_scope == DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS)
        asm volatile("fence.release.sys;");
    else if (sync_scope == DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD)
        ;  // no-op
#else
    // fence.release is not available in PTX. Emulate that with st.release.
    uint32_t dummy;
    const uint32_t val = 0;
    if (sync_scope == DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA)
        asm volatile("st.release.cta.u32 [%0], %1;" : : "l"(&dummy), "r"(val));
    else if (sync_scope == DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU)
        asm volatile("st.release.gpu.u32 [%0], %1;" : : "l"(&dummy), "r"(val));
    else if (sync_scope == DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS)
        asm volatile("st.release.sys.u32 [%0], %1;" : : "l"(&dummy), "r"(val));
    else if (sync_scope == DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD)
        ;  // no-op
#endif
}

#ifdef DOCA_GPUNETIO_VERBS_HAS_ASYNC_STORE_RELEASE
__device__ static __forceinline__ void doca_gpu_dev_verbs_async_store_release(uint32_t *ptr,
                                                                              uint32_t val) {
    asm volatile("st.async.release.sys.global.b32 [%0], %1;" : : "l"(ptr), "r"(val));
}

__device__ static __forceinline__ void doca_gpu_dev_verbs_async_store_release(uint64_t *ptr,
                                                                              uint64_t val) {
    asm volatile("st.async.mmio.release.sys.global.b64 [%0], %1;" : : "l"(ptr), "l"(val));
}
#endif

__device__ static __forceinline__ bool doca_gpu_dev_verbs_isaligned(void *ptr, size_t alignment) {
    bool status;
    status = (((uintptr_t)ptr & (alignment - 1)) == 0);
    return status;
}

/**
 * @brief Copy data from src to dst. The data must have natural alignment with it's size.
 *
 * @param dst - Destination pointer
 * @param src - Source pointer
 * @param bytes - Number of bytes to copy
 */
__device__ static __forceinline__ void doca_gpu_dev_verbs_memcpy_aligned_data(void *dst, void *src,
                                                                              size_t bytes) {
    size_t remaining_bytes = bytes;
    size_t copied_size;
    while (remaining_bytes > 0) {
        if (remaining_bytes >= sizeof(uint32_t)) {
            *(uint32_t *)dst = *(uint32_t *)src;
            copied_size = sizeof(uint32_t);
        } else if (remaining_bytes >= sizeof(uint16_t)) {
            *(uint16_t *)dst = *(uint16_t *)src;
            copied_size = sizeof(uint16_t);
        } else {
            *(uint8_t *)dst = *(uint8_t *)src;
            copied_size = sizeof(uint8_t);
        }
        remaining_bytes -= copied_size;
        dst = (void *)((uintptr_t)dst + copied_size);
        src = (void *)((uintptr_t)src + copied_size);
    }
}

/**
 * @brief Copy data from src to dst. The data may or may not have natural alignment with it's size.
 *
 * @param dst - Destination pointer
 * @param src - Source pointer
 * @param bytes - Number of bytes to copy
 */
__device__ static __forceinline__ void doca_gpu_dev_verbs_memcpy_data(void *dst, void *src,
                                                                      size_t bytes) {
    size_t remaining_bytes = bytes;
    size_t copied_size;
    while (remaining_bytes > 0) {
        if (doca_gpu_dev_verbs_isaligned(dst, sizeof(uint64_t)) &&
            doca_gpu_dev_verbs_isaligned(src, sizeof(uint64_t)) &&
            remaining_bytes >= sizeof(uint64_t)) {
            *(uint64_t *)dst = *(uint64_t *)src;
            copied_size = sizeof(uint64_t);
        } else if (doca_gpu_dev_verbs_isaligned(dst, sizeof(uint32_t)) &&
                   doca_gpu_dev_verbs_isaligned(src, sizeof(uint32_t)) &&
                   remaining_bytes >= sizeof(uint32_t)) {
            *(uint32_t *)dst = *(uint32_t *)src;
            copied_size = sizeof(uint32_t);
        } else if (doca_gpu_dev_verbs_isaligned(dst, sizeof(uint16_t)) &&
                   doca_gpu_dev_verbs_isaligned(src, sizeof(uint16_t)) &&
                   remaining_bytes >= sizeof(uint16_t)) {
            *(uint16_t *)dst = *(uint16_t *)src;
            copied_size = sizeof(uint16_t);
        } else {
            *(uint8_t *)dst = *(uint8_t *)src;
            copied_size = sizeof(uint8_t);
        }
        remaining_bytes -= copied_size;
        dst = (void *)((uintptr_t)dst + copied_size);
        src = (void *)((uintptr_t)src + copied_size);
    }
}

template <typename T>
__device__ static __forceinline__ void doca_gpu_dev_verbs_memcpy_inl_aligned_data(T *dst, T *src,
                                                                                  size_t bytes) {
    size_t remaining_bytes = bytes;
    const size_t copied_size = sizeof(T);
    while (remaining_bytes > 0) {
        remaining_bytes -= copied_size;
        dst = (void *)((uintptr_t)dst + copied_size);
        src = (void *)((uintptr_t)src + copied_size);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_atomic_max
// ════════════════════════════════════════════════════════════════════════════
// 原子取最大值：*ptr = max(*ptr, val)，返回旧值。
//
// 在 DOCA 中的应用场景：
//   - submit_db 中推进 sq_wqe_pi（生产者索引）：
//     atomic_max(&qp->sq_wqe_pi, prod_index)
//     多线程可能并发提交不同批次的 WQE，只有最大的 prod_index 才是正确的
//     如果旧值 < 新值，说明是自己推进的，需要敲 Doorbell
//   - poll_cq_at 中推进 cqe_ci（CQ consumer index）：
//     atomic_max(&cq->cqe_ci, cons_index + 1)
//     多线程并发 poll 不同 CQE 时，只取最大进度
//
// resource_sharing_mode 决定原子操作的硬件范围：
//   EXCLUSIVE：单线程独占，无需原子（直接读写）
//   CTA：block 内共享（cuda::thread_scope_block）
//   GPU：device 内共享（cuda::thread_scope_device）
// ════════════════════════════════════════════════════════════════════════════
template <typename T, enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode,
          bool need_fence_acquire = false>
__device__ static __forceinline__ T doca_gpu_dev_verbs_atomic_max(T *ptr, T val) {
    if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE) {
        T old_val = *ptr;
        *ptr = max(old_val, val);
        return old_val;
    } else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA) {
        cuda::atomic_ref<T, cuda::thread_scope_block> ptr_aref(*ptr);
        return ptr_aref.fetch_max(
            val, need_fence_acquire ? cuda::memory_order_acquire : cuda::memory_order_relaxed);
    } else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU) {
        cuda::atomic_ref<T, cuda::thread_scope_device> ptr_aref(*ptr);
        return ptr_aref.fetch_max(
            val, need_fence_acquire ? cuda::memory_order_acquire : cuda::memory_order_relaxed);
    }
    return 0;
}

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_atomic_add
// ════════════════════════════════════════════════════════════════════════════
// 原子加法：old = *ptr; *ptr += val; return old;
//
// 在 DOCA 中的核心应用：
//   - reserve_wq_slots 中预留 SQ 槽位：
//     atomic_add(&qp->sq_rsvd_index, count)
//     多个 GPU 线程并发调用时，每个线程通过 fetch_add 得到不重叠的索引范围
//     这是整个并发 WQE 提交机制的基石
//
// 使用 memory_order_relaxed：不提供 ordering 保证。
// 调用者通过后续的 fence_release（在 mark_wqes_ready 中）来保证
// WQE 数据写入对 NIC 的可见性。
// ════════════════════════════════════════════════════════════════════════════
template <typename T, enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode>
__device__ static __forceinline__ T doca_gpu_dev_verbs_atomic_add(T *ptr, T val) {
    if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE) {
        T old_val = *ptr;
        *ptr = old_val + val;
        return old_val;
    } else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA) {
        cuda::atomic_ref<T, cuda::thread_scope_block> ptr_aref(*ptr);
        return ptr_aref.fetch_add(val, cuda::memory_order_relaxed);
    } else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU) {
        cuda::atomic_ref<T, cuda::thread_scope_device> ptr_aref(*ptr);
        return ptr_aref.fetch_add(val, cuda::memory_order_relaxed);
    }
    return 0;
}

template <typename T, enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode>
__device__ static __forceinline__ T doca_gpu_dev_verbs_atomic_read(T *ptr) {
    if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE)
        return *ptr;
    else
        return READ_ONCE(*ptr);
}

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_lock / doca_gpu_dev_verbs_unlock
// ════════════════════════════════════════════════════════════════════════════
// GPU 端自旋锁实现，用于保护 submit_db 中的 Doorbell 敲击操作。
//
// 为什么 submit_db 需要锁？
//   多个线程可能并发调用 submit_db，但 Doorbell 敲击必须序列化：
//   - atomic_max 推进 sq_wqe_pi 后，如果发现自己是最新的，就敲 Doorbell
//   - 但 "推进 + 敲 Doorbell + 更新 DBR + 再敲一次" 这组操作必须原子执行
//   - 否则另一个线程可能在中间插入，导致 NIC 看到不一致的状态
//
// 加锁：CAS 自旋（while atomicCAS(lock, 0, 1) != 0），成功后 fence.acquire
// 解锁：store.release(lock, 0)
//   acquire/release 配对保证临界区内的内存操作不会被重排到锁外
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Lock a resource
 *
 * @param lock - Pointer to the lock
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode>
__device__ static __forceinline__ void doca_gpu_dev_verbs_lock(int *lock) {
    if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE) {
        *lock = 1;
    } else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA) {
        while (atomicCAS_block(lock, 0, 1) != 0) continue;
        doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA>();
    } else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU) {
        while (atomicCAS(lock, 0, 1) != 0) continue;
        doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
    }
}

/**
 * @brief Unlock a resource
 *
 * @param lock - Pointer to the lock
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode>
__device__ static __forceinline__ void doca_gpu_dev_verbs_unlock(int *lock) {
    if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE) {
        *lock = 0;
    } else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA) {
        cuda::atomic_ref<int, cuda::thread_scope_block> lock_aref(*lock);
        lock_aref.store(0, cuda::memory_order_release);
    } else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU) {
        cuda::atomic_ref<int, cuda::thread_scope_device> lock_aref(*lock);
        lock_aref.store(0, cuda::memory_order_release);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_load_relaxed_sys_global (多重载版本)
// ════════════════════════════════════════════════════════════════════════════
// 以系统全局作用域 + 放松内存序加载数据（8/32/64 位）。
//
// PTX 指令：ld.relaxed.sys.global.L1::no_allocate.bN [ptr]
//   .relaxed  = 不保证 ordering（纯读操作，不构成同步点）
//   .sys      = 系统作用域（对 CPU/NIC/GPU 全局可见的最新值）
//   .global   = 全局地址空间
//   .L1::no_allocate = 绕过 L1 cache（不污染 L1，适合 one-shot 读取）
//
// 在 DOCA 中的应用：
//   - priv_poll_cq_at 中读取 CQE 的 op_own 和 wqe_counter 字段
//     CQE 由 NIC 通过 PCIe DMA 写入 GPU 显存，使用 .sys 确保读到 NIC 写的最新值
//     使用 no_allocate 避免 CQE 数据污染 L1 cache（CQE 只读一次）
// ════════════════════════════════════════════════════════════════════════════
__device__ static __forceinline__ uint8_t doca_gpu_dev_verbs_load_relaxed_sys_global(uint8_t *ptr) {
    uint16_t ret;
    asm volatile("ld.relaxed.sys.global.L1::no_allocate.b8 %0, [%1];" : "=h"(ret) : "l"(ptr));
    return (uint8_t)ret;
}

__device__ static __forceinline__ uint32_t
doca_gpu_dev_verbs_load_relaxed_sys_global(uint32_t *ptr) {
    uint32_t ret;
    asm volatile("ld.relaxed.sys.global.L1::no_allocate.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ static __forceinline__ uint64_t
doca_gpu_dev_verbs_load_relaxed_sys_global(uint64_t *ptr) {
    uint64_t ret;
    asm volatile("ld.relaxed.sys.global.L1::no_allocate.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_load_relaxed
// ════════════════════════════════════════════════════════════════════════════
// 以指定的资源共享模式作用域 + 放松内存序加载 64 位数据。
//
// 与 load_relaxed_sys_global 的区别：
//   - sys_global 固定使用系统作用域（用于读 NIC DMA 写入的数据）
//   - 这个版本根据 resource_sharing_mode 选择作用域：
//     EXCLUSIVE：直接读取（无原子开销）
//     CTA：ld.relaxed.cta（block 内可见）
//     GPU：ld.relaxed.gpu（device 内可见）
//
// 在 DOCA 中的应用：
//   - priv_poll_cq_at 中读取 cq->cqe_ci（软件维护的 CQ consumer index）
//     cqe_ci 是 GPU 线程间共享的，用 .gpu 作用域即可（不涉及 NIC 写入）
// ════════════════════════════════════════════════════════════════════════════
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode>
__device__ static __forceinline__ uint64_t doca_gpu_dev_verbs_load_relaxed(uint64_t *ptr) {
    uint64_t ret = 0;
    if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE)
        ret = *ptr;
    else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA)
        asm volatile("ld.relaxed.cta.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU)
        asm volatile("ld.relaxed.gpu.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

// ════════════════════════════════════════════════════════════════════════════
// doca_gpu_dev_verbs_div_ceil_aligned_pow2
// ════════════════════════════════════════════════════════════════════════════
// 向上取整除法（分母为 2 的幂）：ceil(x / 2^shift)
//
// 算法：
//   y = 1 << shift  (如 shift=31 → y=2GB)
//   对齐部分：(x & ~(y-1)) >> shift → 整除部分的商
//   余数检测：!!(x & (y-1)) → 如果有余数则 +1
//
// 在 DOCA 中的应用：
//   put_signal_counter 中计算数据分片数：
//   num_chunks = div_ceil(size, MAX_TRANSFER_SIZE_SHIFT)
//   将超大传输拆分成多个 ≤ MAX_TRANSFER_SIZE 的 WQE
// ════════════════════════════════════════════════════════════════════════════

// 向上取整整数除法
/**
 * @brief Calculate the ceiling of x / y, where y is (2^denominator_shift) 
 *
 * @param x - Numerator
 * @param denominator_shift - Denominator shift (y = 2^denominator_shift)
 * @return The ceiling of x / y
 */
__device__ static __forceinline__ uint64_t
doca_gpu_dev_verbs_div_ceil_aligned_pow2(uint64_t x, unsigned int denominator_shift) {
    uint64_t y = 1ULL << denominator_shift;
    return ((x & ~(y - 1)) >> denominator_shift) + (!!(x & (y - 1)));
}

/**
 * @brief Calculate the ceiling of x / y, where y is (2^denominator_shift).
 * The result must fit in 32 bits. This is a faster implementation than gdaki_div_ceil_aligned_pow2.
 *
 * @param x - Numerator
 * @param denominator_shift - Denominator shift (y = 2^denominator_shift)
 * @return The ceiling of x / y
 */
__device__ static __forceinline__ uint32_t
doca_gpu_dev_verbs_div_ceil_aligned_pow2_32bits(uint64_t x, int denominator_shift) {
    return uint32_t(x >> denominator_shift) + !!__funnelshift_r(0, uint32_t(x), denominator_shift);
}

#endif /* DOCA_GPUNETIO_DEV_VERBS_COMMON_H */
