/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// ===== lsa_barrier__funcs.h：LSA Barrier 所有成员函数的内联实现 =====

#ifndef _NCCL_DEVICE_MEM_BARRIER__FUNCS_H_
#define _NCCL_DEVICE_MEM_BARRIER__FUNCS_H_
#include "lsa_barrier__types.h"
#include "comm__types.h"

#if NCCL_CHECK_CUDACC
// 析构函数：将当前 epoch 写回 resource buffer 以持久化
// 写回公式：state[(multimem ? 0 : 1)*nBarriers + index] = epoch
// 目的：下一次创建相同 index 的 session 时，从这里读取 epoch，保证轮次连续
//
// 为什么需要持久化 epoch？
//   barrier 通过比较 inbox 值与 epoch+nRanks（multimem）或 epoch+1（unicast）来判断完成。
//   如果每次都从 0 开始，经过 nRanks 次 barrier 后计数器回绕会发生误判。
//   持久化 epoch 让每次 barrier 使用不同的目标值，从根本上避免回绕问题。
//
// 注意：epoch 只由 coop 内 thread_rank()==0 的线程写入，避免重复写
// CTK 12.0 + sm_90 的 ptxas bug workaround：index==0 时用特殊写法规避
// 构造函数（lsa tag 版本）：使用 DevComm 中默认的 lsa barrier 资源和 lsa team
// 等价于调用通用构造函数：
//   team = ncclTeamLsa(comm)
//   handle = comm.lsaBarrier
//   mmHandle = comm.lsaMultimem
// 构造函数（通用版本）：
//   1. 初始化所有内部字段（coop/comm/team/handle/index/multimem/mmHandle）
//   2. 从 resource buffer 中读取 epoch（持久化状态，跨 session 保持连续）
//
// state 布局（uint32_t 数组）：
//   [0..nBarriers-1]          = multimem 模式的 epoch state（mode=multimem 时读 index 0）
//   [nBarriers..2*nBarriers-1] = unicast 模式的 epoch state（mode=unicast 时读 nBarriers+index）
// 读取公式：epoch = state[(multimem ? 0 : 1)*nBarriers + index]
//
// CTK < 12.6 的 workaround：ptxas 有 bug，强制 multimem=false 避免触发
template<typename Coop>
NCCL_DEVICE_INLINE ncclLsaBarrierSession<Coop>::ncclLsaBarrierSession(
    Coop coop, ncclDevComm const& comm, ncclTeam team,
    ncclLsaBarrierHandle handle, uint32_t index,
    bool multimem, ncclMultimemHandle mmHandle
  ):
  ncclLsaBarrierSession_internal<Coop>{
    coop, comm, team, handle, (int)index,
#if CUDART_VERSION >= 12060
    multimem,
#else // WAR for an issue with ptxas in CTK < 12.6
    /*multimem=*/false,  // CTK < 12.6 时强制关闭 multimem，规避 ptxas bug
#endif
    mmHandle, /*epoch=*/0
  } {
  // 读取持久化的 epoch：从本 rank 的 resource buffer 中加载对应的 epoch 值
  // 这确保跨多次 session 的 barrier 轮次连续不重叠
  uint32_t* state = (uint32_t*)ncclGetResourceBufferLocalPointer(comm, handle.bufHandle);
  this->epoch = state[(this->multimem ? 0 : 1)*this->handle.nBarriers + this->index];
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclLsaBarrierSession<Coop>::ncclLsaBarrierSession(
    Coop coop, ncclDevComm const& comm, ncclTeamTagLsa, uint32_t index, bool multimem
  ): ncclLsaBarrierSession(
    coop, comm, ncclTeamLsa(comm), comm.lsaBarrier, index, multimem, comm.lsaMultimem
  ) {
}
#endif

#if NCCL_CHECK_CUDACC
// arrive：通知 team 中所有其他 rank，本 rank 已到达 barrier
//
// multimem 模式（sm_90+ Hopper）：
//   1. coop.sync() 确保 coop 内所有线程在 arrive 之前完成写操作
//   2. 仅 thread_rank()==0 执行实际写操作（避免重复 arrive）
//   3. 使用 PTX 指令 "multimem.red.release.sys.add.u32" 向 MultiCast 地址原子加 1
//      - multimem.red：对 MultiCast 地址执行 reduce 写（广播 +1 到所有 rank）
//      - .release.sys：release 内存序，system scope
//      - .add.u32：无符号 32 位加法
//   4. relaxed 模式时改用 "multimem.red.relaxed.sys.add.u32"
//
// unicast 模式（通用）：
//   1. coop 内每个线程负责向 nRanks-1 个 peer 中的若干个写信号
//      （for i in thread_rank..nRanks-1 step coop.size()）
//   2. 跳过 self（peer = i + (team.rank <= i ? 1 : 0)）
//   3. 使用 cuda::atomic_ref::store(epoch+1, releaseOrder) 写入 ucInbox(peer, team.rank)
//      - 写入 peer 的 ucInbox 中属于本 rank 的槽位
//      - release 内存序保证 arrive 前的数据对 peer wait 可见
template<typename Coop>
NCCL_DEVICE_INLINE ncclLsaBarrierSession<Coop>::~ncclLsaBarrierSession() {
  uint32_t* state = (uint32_t*)ncclGetResourceBufferLocalPointer(this->comm, this->handle.bufHandle);
  if (this->coop.thread_rank() == 0) {
#if __CUDA_ARCH__ == 1200 && CUDART_VERSION < 13000
    // WAR for a compiler issue with CTK < 13.0
    // sm_90（arch 1200）上 ptxas 有 bug，需要对 index==0 特殊处理
    if (this->index == 0)
      state[(this->multimem ? 0 : 1)*this->handle.nBarriers] = this->epoch;
    else
#endif
    // 将更新后的 epoch 写回对应槽位
    state[(this->multimem ? 0 : 1)*this->handle.nBarriers + this->index] = this->epoch;
  }
  // 先同步 coop 内所有线程，确保 arrive 前的写已完成（隐式 release）
  // sync 确保 epoch 写入对后续 coop 所有线程可见（barrier 完整性）
  this->coop.sync();
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE void ncclLsaBarrierSession<Coop>::arrive(Coop, cuda::memory_order order) {
  this->coop.sync();
  if (this->multimem) {
  #if __CUDA_ARCH__ >= 900
    if (this->coop.thread_rank() == 0) {
      uint32_t* inbox = this->mcInbox(/*multimem=*/true);
      if (nccl::utility::releaseOrderOf(order) != cuda::memory_order_relaxed) {
        // release 语义：保证 arrive 之前的写对其他 rank 可见
        asm volatile("multimem.red.release.sys.add.u32 [%0],1;" :: "l"(inbox) : "memory");
      } else {
        // relaxed 语义：不保证顺序，性能更高
        asm volatile("multimem.red.relaxed.sys.add.u32 [%0],1;" :: "l"(inbox) : "memory");
      }
    }
  #endif
  } else {
    if (this->team.nRanks > 1) {
      cuda::atomic_thread_fence(nccl::utility::releaseOrderOf(order));
    }
    // unicast 模式：向所有 peer 的 inbox 写 epoch+1
    #pragma unroll 1
    for (int i = this->coop.thread_rank(); i < this->team.nRanks-1; i += this->coop.size()) {
      // 跳过 self：peer = i，但如果 i >= team.rank 时要 +1（索引偏移）
      int peer = i + (this->team.rank <= i ? 1 : 0);
      // 向 peer 的 ucInbox[team.rank] 写 epoch+1（表示本 rank 已到达）
      cuda::atomic_ref<uint32_t> inbox(*this->ucInbox(peer, this->team.rank));
      inbox.store(this->epoch+1, cuda::memory_order_relaxed);
    }
  }
}
#endif

#if NCCL_CHECK_CUDACC
// wait：等待 team 中所有其他 rank 都完成 arrive
//
// multimem 模式（sm_90+）：
//   1. 仅 thread_rank()==0 的线程执行等待
//   2. 轮询 mcInbox(false)（本 rank 的 unicast 视图）
//      等待计数器 >= epoch + nRanks（每个 rank arrive 一次 +1，共 nRanks 次）
//   3. 使用 rollingLessEq 风格的无符号比较：got - (epoch+nRanks) <= (uint32_t(-1)>>1)
//      含义：got 是否已达到 epoch+nRanks（支持 epoch 轻微超前的情况）
//   4. wait 完成后 epoch += nRanks（为下次 barrier 准备新的目标值）
//
// unicast 模式：
//   1. coop 内每个线程负责等待若干 peer 的信号
//   2. 轮询 ucInbox(myRank, peer) 等待来自 peer 的 epoch+1
//   3. wait 完成后 epoch += 1
//
// testAbort：每次轮询前检查 abortFlag，若 host 端请求终止则跳出循环
// acquireOrderOf(order)：从双向内存序中提取 acquire 部分
template<typename Coop>
template<bool EnableTimeout>
NCCL_DEVICE_INLINE ncclResult_t ncclLsaBarrierSession_internal<Coop>::waitInternal(Coop, cuda::memory_order order,
                                                                                   uint64_t timeoutCycles) {
  using nccl::utility::testAbort;
  uint32_t steps;
  uint64_t startCycle;
  ncclResult_t ret = ncclSuccess;
  if NCCL_IF_CONSTEXPR (EnableTimeout) {
    startCycle = clock64();
  } else {
    steps = 0;
  }
  if (this->multimem) {
  #if __CUDA_ARCH__ >= 900
    if (this->coop.thread_rank() == 0) {
      // 轮询本 rank 的 mcInbox（unicast 视图）等待所有 arrive 的累积
      cuda::atomic_ref<uint32_t> inbox(*this->mcInbox(/*multimem=*/false));
    // unicast 模式：coop 内每个线程等待若干 peer 的信号
      #pragma unroll 1
      while (true) {
        uint32_t got = inbox.load(nccl::utility::acquireOrderOf(order));
        // 无符号回绕比较：got - (epoch+nRanks) <= INT32_MAX 即为"已到达"
        if (got - (this->epoch + this->team.nRanks) <= uint32_t(-1)>>1) break;

        if NCCL_IF_CONSTEXPR (EnableTimeout) {
          if (clock64() - startCycle >= timeoutCycles) {
            ret = ncclTimeout;
            goto exit;
          }
        } else {
          if (testAbort(this->comm.abortFlag, steps)) break;
        }
      }
      this->epoch += this->team.nRanks;  // 为下一次 barrier 准备新 epoch
    }
  #endif
  } else {
    #pragma unroll 1
    for (int i = this->coop.thread_rank(); i < this->team.nRanks-1; i += this->coop.size()) {
      int peer = i + (this->team.rank <= i ? 1 : 0);
      // 轮询来自 peer 的信号槽
      cuda::atomic_ref<uint32_t> inbox(*this->ucInbox(this->team.rank, peer));
      #pragma unroll 1
      while (true) {
        uint32_t got = inbox.load(nccl::utility::acquireOrderOf(order));
        // 无符号比较：got - (epoch+1) <= INT32_MAX 即为 peer 已 arrive
        if (got - (this->epoch + 1) <= uint32_t(-1)>>1) break;

        if NCCL_IF_CONSTEXPR (EnableTimeout) {
          if (clock64() - startCycle >= timeoutCycles) {
            ret = ncclTimeout;
            goto exit;
          }
        } else {
          if (testAbort(this->comm.abortFlag, steps)) break;
        }
      }
    }
    this->epoch += 1;  // 为下一次 barrier 准备新 epoch
  }
  goto exit; // Silence a compiler warning.
exit:
  // 同步 coop 内所有线程，确保所有人都完成了 wait
  this->coop.sync();
  return ret;
}
#endif

#if NCCL_CHECK_CUDACC
// sync：arrive + wait 的组合，完成一次完整的双向 barrier
// 内存语义：order 同时应用于 arrive 的 release 和 wait 的 acquire
template<typename Coop>
NCCL_DEVICE_INLINE void ncclLsaBarrierSession<Coop>::wait(Coop coop, cuda::memory_order order) {
  (void)(this->template waitInternal</*EnableTimeout=*/false>(coop, order, 0ULL));
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclResult_t ncclLsaBarrierSession<Coop>::wait(
    Coop coop, cuda::memory_order order, uint64_t timeoutCycles) {
  this->coop.sync();
  return this->template waitInternal</*EnableTimeout=*/true>(coop, order, timeoutCycles);
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE void ncclLsaBarrierSession<Coop>::sync(Coop coop, cuda::memory_order order) {
  this->arrive(coop, order);
  this->wait(coop, order);
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclResult_t ncclLsaBarrierSession<Coop>::sync(
    Coop coop, cuda::memory_order order, uint64_t timeoutCycles) {
  this->arrive(coop, order);
  return this->wait(coop, order, timeoutCycles);
}
#endif

#endif // _NCCL_DEVICE_MEM_BARRIER__FUNCS_H_
