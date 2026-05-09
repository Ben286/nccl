/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// ===== ll_a2a__funcs.h：Low-Latency A2A 函数实现 =====
// 本文件实现 ncclLLA2ASession<Coop> 的所有成员函数。
//
// LL 协议核心原理：
//   发送：将数据拆成 uint32_t 对，以 uint4（128位）为单位写入 peer 的 slot，
//         格式为 {data_lo, epoch_tag, data_hi, epoch_tag}
//         使用 st.relaxed.sys.v4.u32（>=sm_70）或 st.volatile.v4.u32（<sm_70）
//   接收：循环 ld.v4.u32 读取本地 slot，检查 y==epoch && w==epoch，
//         两个 epoch_tag 均匹配时说明数据已完整写入（无撕裂写）。
//
// union 技巧：
//   union { T tmp; uint32_t u32[divUp(sizeof(T),8)][2]; };
//   tmp = data;
//   将类型 T 的值按位解释为 uint32_t 数组，u32[v][0] = data_lo，u32[v][1] = data_hi。
//   每组 (data_lo, data_hi) 对应一个 uint4 中的 x/z 字段，y/w 固定为 epoch。
//
// pitch 的作用：
//   当 sizeof(T) > 8 时，T 被分解为多个 uint4（v = 0..divUp(sizeof(T),8)-1），
//   每个 uint4 写到 buf + u*pitch + v 处（u=elt index，v=T 内部 chunk 索引）。
//   pitch = maxElts 确保不同 elt 的 chunk 不交叠。

#ifndef _NCCL_DEVICE_LL_A2A__FUNCS_H_
#define _NCCL_DEVICE_LL_A2A__FUNCS_H_
#include "ll_a2a__types.h"
#include "comm__types.h"
#include "../utility.h"

// -------------------------------------------------------------------------
// recv<T>：从本地 slot 接收数据（简单包装 recvUnrolled<1,1>）
// -------------------------------------------------------------------------
// 读取本地 resource buffer 的 slot（发送方已写入），
// spin-wait 直到 epoch_tag 匹配（即 y==epoch && w==epoch）
// -------------------------------------------------------------------------
// 析构函数
// -------------------------------------------------------------------------
// 将 epoch-2 写回 resource buffer 的持久化行，供下次构造时恢复。
// 只需 thread_rank==0 的线程写（单线程写避免竞争）。
// 最后 coop.sync() 确保持久化写完成后所有线程才退出。
//
// 为什么写 epoch-2 而非 epoch？
//   epoch 在 endEpoch 中每次 +1（或 +3 绕回）。
//   构造时会 +2 来"跳过"上一轮使用的两个可能残留的 epoch 值，
//   确保新 epoch 的 tag 不与 buffer 中残留的旧 tag 冲突。
// -------------------------------------------------------------------------
// 构造函数
// -------------------------------------------------------------------------
// 流程：
//   1. 调用基类 ncclLLA2ASession_internal<Coop> 构造，初始化 coop/comm/team/handle/block/pitch/multimem/mmHandle
//      epoch=0，slotsOffset=0（占位，后面覆盖）
//   2. 定位 epoch 持久化行：
//      line = resource_buffer_local_base + block * (1 + 2*nSlots)
//   3. 读取 epoch：epoch = line->x + 2
//      line->x 存储上次析构写回的 epoch - 2（初始为 0 → epoch = 2）
//   4. 计算 slotsOffset = calcSlotOffset()（根据 epoch & 1 选择双 buffer）
#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclLLA2ASession<Coop>::ncclLLA2ASession(
    Coop coop, ncclDevComm const& comm, ncclTeam team,
    ncclLLA2AHandle handle, uint32_t block, int maxElts,
    bool multimem, ncclMultimemHandle mmHandle
  ):
  ncclLLA2ASession_internal<Coop>{
    coop, comm, team, handle, (int)block, /*pitch=*/maxElts,
    multimem, mmHandle, /*epoch=*/0, /*slotsOffset=*/0
  } {
  // 定位本 block 的 epoch 持久化行（每个 block 占 1+2*nSlots 行，第 0 行是持久化行）
  uint4* line = (uint4*)ncclGetResourceBufferLocalPointer(comm, handle.bufHandle);
  line += block*(1 + 2*handle.nSlots);
  // epoch = line->x + 2：从持久化行恢复 epoch，+2 保证新 epoch 跳过残留值
  // 初次调用时 line->x = 0，epoch = 2（从 2 开始使用）
  this->epoch = line->x + 2;
  // 根据恢复的 epoch 计算对应的 slot buffer 偏移
  this->slotsOffset = this->calcSlotOffset();
}
#endif

// -------------------------------------------------------------------------
// send<T>：向 peer 的 slot 写入数据（LL 协议格式）
// -------------------------------------------------------------------------
// 参数：
//   peer — 目标 peer 在 team 内的 rank
//   elt  — slot 编号（0..nSlots-1）
//   data — 要发送的数据（类型 T）
//
// 写入格式（每个 uint4）：
//   {data_lo, epoch, data_hi, epoch}
//   y 和 w 字段固定为 epoch_tag，接收方用这两个字段检测数据有效性。
//   两个 epoch_tag 分别保护 x（data_lo）和 z（data_hi），
//   防止 uint4 写被分裂（torn write）时接收方误读半数据。
//
// PTX 指令选择：
//   __CUDA_ARCH__ >= 700（Volta+）：st.relaxed.sys.v4.u32
//     relaxed 内存序（无 fence），系统级可见（sys = 跨 GPU 可见）
//     比 volatile 更高效（允许编译器重排，但保证最终写入 sys 内存）
//   __CUDA_ARCH__ < 700（Pascal及以下）：st.volatile.v4.u32
//     volatile 防止编译器缓存，强制每次写到内存
#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclLLA2ASession<Coop>::~ncclLLA2ASession() {
  // 定位本 block 的 epoch 持久化行
  uint4* line = (uint4*)ncclGetResourceBufferLocalPointer(this->comm, this->handle.bufHandle);
  line += this->block*(1 + 2*this->handle.nSlots);
  // 只有 thread 0 写（其他线程无需重复写同一位置）
  if (this->coop.thread_rank() == 0) line->x = this->epoch - 2;
  // 等所有线程确认写完成后再退出（保证析构是 coop 同步点）
  this->coop.sync();  // 先确保所有当前 epoch 的 recv 已完成
}
#endif

// -------------------------------------------------------------------------
// bcast<T>：向所有 peer 广播数据
// -------------------------------------------------------------------------
// 两种实现路径：
//
// 路径 1：multimem 模式（multimem=true，sm_90+）
//   写一次 MultiCast 地址，NVLink 硬件自动将数据广播到所有参与 GPU 的对应地址。
//   MultiCast 基地址由 ncclGetResourceBufferMultimemPointer 获取（mmHandle）。
//   优点：只需一次 store 指令，延迟和带宽均优于 unicast 循环。
//
// 路径 2：unicast 模式（multimem=false）
//   循环对每个 peer 调用 ncclGetResourceBufferPeerPointer 并 store。
//   使用分段循环（外层 8 步展开 + 尾余段）最大化指令级并行：
//     - 外层循环：每次处理 8 个 peer，全部展开（#pragma unroll），
//       允许编译器重叠多个 store 的地址计算和 TLB 查询
//     - 尾余段：处理 team.nRanks % 8 个剩余 peer，静态展开 8 次但动态判断
//   peer 按 (r+0, r+1, ...) 顺序遍历（r 从 team.rank 开始循环，不避免自己）
//   注意：bcast 包括向自己发送（self-send），以保证 recv 时自己的 slot 也有数据
#if NCCL_CHECK_CUDACC
template<typename Coop>
template<typename T>
NCCL_DEVICE_INLINE void ncclLLA2ASession<Coop>::send(int peer, int elt, T data) {
  using nccl::utility::divUp;
  // union：将 T 按位解释为 uint32_t[divUp(sizeof(T),8)][2]
  //   u32[v][0] = v 号 chunk 的低 32 位（data_lo）
  //   u32[v][1] = v 号 chunk 的高 32 位（data_hi）
  union { T tmp; uint32_t u32[divUp(sizeof(T), 8)][2]; };
  tmp = data;
  // 获取 peer 的 slot buffer 基地址（unicast：通过 symmetric memory 直接写 peer 的内存）
  uint4* buf = (uint4*)ncclGetResourceBufferPeerPointer(this->comm, this->handle.bufHandle, this->team, peer);
  // 定位到具体的 slot：slotsOffset（当前 epoch buffer 偏移）+ elt（slot 编号）
  buf += this->slotsOffset + elt;
  // 对 T 的每个 8 字节 chunk（v=0..divUp(sizeof(T),8)-1）写一个 uint4
  #pragma unroll
  for (int u=0; u < divUp(sizeof(T), 8); u++) {
    #if __CUDA_ARCH__ >= 700
      // st.relaxed.sys.v4.u32 [addr], {data_lo, epoch, data_hi, epoch}
      // %0 = 地址（64位），%1 = data_lo，%2 = data_hi，%3 = epoch
      // 注意参数顺序：{x=data_lo, y=epoch, z=data_hi, w=epoch}
      asm volatile("st.relaxed.sys.v4.u32 [%0],{%1,%3,%2,%3};" ::
        "l"(buf + u*this->pitch),  // 第 u 个 chunk 的地址（间隔 pitch 行）
        "r"(u32[u][0]), "r"(u32[u][1]), "r"(this->epoch)
        : "memory"
      );
    #else  // __CUDA_ARCH__ < 700（Pascal 及以下）
      // < sm_70：使用 volatile（无 relaxed 支持）
      asm volatile("st.volatile.v4.u32 [%0],{%1,%3,%2,%3};" ::
        "l"(buf + u*this->pitch),
        "r"(u32[u][0]), "r"(u32[u][1]), "r"(this->epoch)
        : "memory"
      );
    #endif
  }
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
template<typename T>
NCCL_DEVICE_INLINE void ncclLLA2ASession<Coop>::bcast(int elt, T data) {
  using nccl::utility::divUp;
  if (this->multimem) {
    // ---- multimem 路径：一次 MultiCast store ----
    union { T tmp; uint32_t u32[divUp(sizeof(T),8)][2]; };
    tmp = data;
    // 获取 MultiCast 视图基地址（写此地址相当于广播给所有 GPU）
    uint4* bufmc = (uint4*)ncclGetResourceBufferMultimemPointer(this->comm, this->handle.bufHandle, this->mmHandle);
    bufmc += this->slotsOffset + elt;
    #pragma unroll
    for (int u=0; u < divUp(sizeof(T), 8); u++) {
      #if __CUDA_ARCH__ >= 700
        asm volatile("st.relaxed.sys.v4.u32 [%0],{%1,%3,%2,%3};" ::
          "l"(bufmc + this->pitch*u),
          "r"(u32[u][0]), "r"(u32[u][1]), "r"(this->epoch)
          : "memory"
        );
      #else
        asm volatile("st.volatile.v4.u32 [%0],{%1,%3,%2,%3};" ::
          "l"(bufmc + this->pitch*u),
          "r"(u32[u][0]), "r"(u32[u][1]), "r"(this->epoch)
          : "memory"
        );
      #endif
    }
  } else {
    // ---- unicast 路径：循环 send 给每个 peer ----
    union { T tmp; uint32_t u32[divUp(sizeof(T), 8)][2]; };
    tmp = data;
    int dr = 0;
    int r = this->team.rank;  // 从自己的 rank 开始（包含自己）
    // 外层循环：每次处理 8 个 peer（#pragma unroll 1 防止外层展开，内层 ur 展开）
    #pragma unroll 1  // 禁止循环体展开（spin-wait 循环不应展开）
    for (; dr+8 <= this->team.nRanks; dr += 8) {
    // 尾余段：处理 team.nRanks % 8 个剩余 peer（静态展开 8 次，动态判断 break）
      // 内层 8 次展开：同时发起 8 个不同 peer 的 store（提高 MLP：内存级并行）
      #pragma unroll
      for (int ur=0; ur < 8; ur++) {
        uint4* buf = (uint4*)ncclGetResourceBufferPeerPointer(this->comm, this->handle.bufHandle, this->team, r);
        buf += this->slotsOffset + elt;
        #pragma unroll
        for (int u=0; u < divUp(sizeof(T),8); u++) {
          #if __CUDA_ARCH__ >= 700
            asm volatile("st.relaxed.sys.v4.u32 [%0],{%1,%3,%2,%3};" ::
              "l"(buf + u*this->pitch),
              "r"(u32[u][0]), "r"(u32[u][1]), "r"(this->epoch)
              : "memory"
            );
          #else
            asm volatile("st.volatile.v4.u32 [%0],{%1,%3,%2,%3};" ::
              "l"(buf + u*this->pitch),
              "r"(u32[u][0]), "r"(u32[u][1]), "r"(this->epoch)
              : "memory"
            );
          #endif
        }
        r += 1;
        if (r == this->team.nRanks) r = 0;  // 循环回到 rank 0
      }
    }
    #pragma unroll
    for (int ur=0; ur < 8; ur++, dr++) {
      if (dr == this->team.nRanks) break;  // 已处理完所有 peer
      uint4* buf = (uint4*)ncclGetResourceBufferPeerPointer(this->comm, this->handle.bufHandle, this->team, r);
      buf += this->slotsOffset + elt;
      #pragma unroll
      for (int u=0; u < divUp(sizeof(T),8); u++) {
        #if __CUDA_ARCH__ >= 700
          asm volatile("st.relaxed.sys.v4.u32 [%0],{%1,%3,%2,%3};" ::
            "l"(buf + u*this->pitch),
            "r"(u32[u][0]), "r"(u32[u][1]), "r"(this->epoch)
            : "memory"
          );
        #else
          asm volatile("st.volatile.v4.u32 [%0],{%1,%3,%2,%3};" ::
              "l"(buf + u*this->pitch),
              "r"(u32[u][0]), "r"(u32[u][1]), "r"(this->epoch)
              : "memory"
            );
        #endif
      }
      r += 1;
      if (r == this->team.nRanks) r = 0;
    }
  }
}
#endif

// -------------------------------------------------------------------------
// recvUnrolled<MinEltCount, MaxEltCount, T>：批量接收并展开循环
// -------------------------------------------------------------------------
// 关键实现技术：
//
// 1. 双模板参数 MinEltCount + MaxEltCount：
//    MinEltCount：编译期已知的最小读取数量（静态展开，无分支）
//    MaxEltCount：数组大小（需静态分配 tmp[MaxEltCount][...]）
//    判断是否读取 slot u：u < MinEltCount || u < eltCount
//      u < MinEltCount → 编译期 true，无动态分支
//      u < eltCount    → 运行时判断
//    这样编译器可以消除 u < MinEltCount 段的分支，提高吞吐量。
//
// 2. spin-wait 循环（testAbort）：
//    while (!testAbort(abortFlag, steps))：每隔一定步数检查 abortFlag，
//    实现故障容忍（abort 时退出 spin-wait 并处理错误）。
//
// 3. PTX 指令选择（读取 slot）：
//    __CUDA_ARCH__ == 900（Hopper sm_90）：ld.acquire.sys.v4.u32
//      acquire 语义：读到后续的 load/store 均可见（保证 LL 数据的完整可见性）
//    __CUDA_ARCH__ >= 700（非 900）：ld.relaxed.sys.v4.u32
//      relaxed 即可（Hopper 上 acquire 是特殊需求，其他架构 relaxed 足够）
//    __CUDA_ARCH__ < 700：ld.volatile.v4.u32
//
// 4. epoch_tag 检查（okAll）：
//    tmp[u][v].y == epoch && tmp[u][v].w == epoch
//    两个 epoch_tag 均匹配：说明 x（data_lo）和 z（data_hi）都已被完整写入
//    任一不匹配：继续 spin-wait
//    __builtin_expect(okAll, true)：提示编译器"正常情况下数据已就绪"，
//    优化分支预测（减少 spin-wait 路径的代价）
//
// 5. 数据提取（读完后）：
//    u32[v][0] = tmp[u][v].x（data_lo）
//    u32[v][1] = tmp[u][v].z（data_hi）
//    通过 union 重新解释为 T 类型
#if NCCL_CHECK_CUDACC
template<typename Coop>
template<typename T>
NCCL_DEVICE_INLINE T ncclLLA2ASession<Coop>::recv(int elt) {
  T ret[1];
  // MinEltCount=1, MaxEltCount=1：只读 1 个 slot，无循环
  this->template recvUnrolled</*MinEltCount=*/1, /*MaxEltCount=*/1>(elt, 1, 0, ret);
  return ret[0];
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
template<int MinEltCount, int MaxEltCount, typename T>
NCCL_DEVICE_INLINE void ncclLLA2ASession<Coop>::recvUnrolled(int eltStart, int eltCount, int eltStride, T(&elts)[MaxEltCount]) {
  using nccl::utility::divUp;
  using nccl::utility::testAbort;
  // 获取本地（local）resource buffer 基地址（接收自己这侧的数据）
  uint4* buf = (uint4*)ncclGetResourceBufferLocalPointer(this->comm, this->handle.bufHandle);
  // 定位到起始 slot：slotsOffset（当前 epoch buffer 偏移）+ eltStart
  buf += this->slotsOffset + eltStart;
  uint32_t steps = 0;  // testAbort 计步器（每隔若干步检查 abortFlag）
  // tmp[u][v]：存储每个 slot(u) 每个 chunk(v) 读取到的 uint4 原始值
  uint4 tmp[MaxEltCount][divUp(sizeof(T), 8)];
  // spin-wait：循环读取，直到所有 epoch_tag 匹配（或 abortFlag 触发退出）
  #pragma unroll 1
  while (!testAbort(this->comm.abortFlag, steps)) {
            // Volta~Ampere（sm_70~sm_80）：relaxed sys，跨 GPU 可见即可
            // Hopper（sm_90）：acquire 语义，确保读取后数据对后续操作完全可见
    // 读取所有需要的 slot 的所有 chunk
    #pragma unroll
    for (int u=0; u < MaxEltCount; u++) {
      if (u < MinEltCount || u < eltCount) {  // 静态+动态控制是否读取
        #if __CUDA_ARCH__ >= 700
          #if __CUDA_ARCH__ == 900
            #pragma unroll
            for (int v=0; v < divUp(sizeof(T), 8); v++) {
              asm volatile("ld.acquire.sys.v4.u32 {%0,%1,%2,%3},[%4];"
                : "=r"(tmp[u][v].x), "=r"(tmp[u][v].y), "=r"(tmp[u][v].z), "=r"(tmp[u][v].w)
                : "l"(buf + u*eltStride + v*this->pitch)
                : "memory");
            }
          #else
            #pragma unroll
            for (int v=0; v < divUp(sizeof(T), 8); v++) {
              asm volatile("ld.relaxed.sys.v4.u32 {%0,%1,%2,%3},[%4];"
                : "=r"(tmp[u][v].x), "=r"(tmp[u][v].y), "=r"(tmp[u][v].z), "=r"(tmp[u][v].w)
                : "l"(buf + u*eltStride + v*this->pitch)
                : "memory");
            }
          #endif
        #else // __CUDA_ARCH__ >= 700
          #pragma unroll
          for (int v=0; v < divUp(sizeof(T), 8); v++) {
            asm volatile("ld.volatile.v4.u32 {%0,%1,%2,%3},[%4];"
              : "=r"(tmp[u][v].x), "=r"(tmp[u][v].y), "=r"(tmp[u][v].z), "=r"(tmp[u][v].w)
              : "l"(buf + u*eltStride + v*this->pitch)
              : "memory");
          }
        #endif
      }
    }
    // 检查所有 slot 的所有 chunk 的 epoch_tag 是否均匹配
    bool okAll = true;
  // 从 tmp 中提取实际数据（忽略 epoch_tag 字段 y/w）
    #pragma unroll
    for (int u=0; u < MaxEltCount; u++) {
      #pragma unroll
      for (int v=0; v < divUp(sizeof(T), 8); v++) {
        if (u < MinEltCount || u < eltCount) {
          // LL 协议：y（epoch_tag for x）和 w（epoch_tag for z）均须等于 epoch
          bool ok = tmp[u][v].y == this->epoch &&
                    tmp[u][v].w == this->epoch;
          okAll &= ok;
        }
      }
    }
    // 数据已就绪（正常情况）：退出 spin-wait
    if (__builtin_expect(okAll, true)) break;
  }

  #pragma unroll
  for (int u=0; u < MaxEltCount; u++) {
    if (MinEltCount <= u && u == eltCount) break;  // 超过实际 eltCount 则停止
    union { T val; uint32_t u32[divUp(sizeof(T), 8)][2]; };
    #pragma unroll
    for (int v=0; v < divUp(sizeof(T), 8); v++) {
      u32[v][0] = tmp[u][v].x;  // data_lo（uint4 的 x 字段）
      u32[v][1] = tmp[u][v].z;  // data_hi（uint4 的 z 字段，跳过 epoch_tag）
    }
    elts[u] = val;  // 通过 union 将 uint32_t 数组重新解释为 T
  }
}
#endif

// -------------------------------------------------------------------------
// endEpoch()：结束当前 epoch，推进到下一轮
// -------------------------------------------------------------------------
// 详细流程：
//
// Step 1: epoch 回绕检查
//   epoch 是 uint32_t，每次 +1 最终会接近最大值。
//   若 epoch >= -2u（即 epoch == UINT32_MAX-1 或 UINT32_MAX）：
//     需要清零当前 slot buffer，防止新 epoch 的值（从 2 开始）
//     与 buffer 中残留的旧数据（epoch = UINT32_MAX 等）冲突。
//
// Step 2: 清零 slot buffer（仅在 epoch >= -2u 时执行）
//   所有 coop 线程协作将 slotsOffset 处的 nSlots 个 uint4 清零。
//   #pragma unroll 4：部分展开，平衡寄存器压力和 MLP。
//   coop.sync()：清零完成后才能推进 epoch。
//
// Step 3: epoch 推进
//   epoch += (epoch == -1u) ? 3 : 1
//     正常情况：+1（UINT32_MAX-1 → UINT32_MAX-1+1 = UINT32_MAX-1... 不对）
//     实际：epoch >= -2u 时已清零，所以只需特殊处理 -1u（UINT32_MAX）：
//       epoch == UINT32_MAX（-1u）→ +3 → 2（绕回到 2，跳过 0 和 1）
//       epoch == UINT32_MAX-1（-2u）→ +1 → UINT32_MAX（-1u）
//       下一次 endEpoch 再 +3 → 2
//     目的：epoch 永远不经过 0 和 1（0 是 buffer 初始值，1 可能造成混淆）
//
// Step 4: 更新 slotsOffset
//   calcSlotOffset() 根据新的 epoch & 1 选择另一半 buffer，
//   send/recv 下一轮时使用新的 slotsOffset。
// -------------------------------------------------------------------------
// recvReduce<Unroll, Elt, EltToAcc, Reduce>：接收并规约多个 slot
// -------------------------------------------------------------------------
// 实现技巧（两段循环）：
//
//   第一段（整块 Unroll）：
//     for i in [0, eltCount-Unroll, step=Unroll]：
//       recvUnrolled<Min=Unroll>(eltStart+i*eltStride, Unroll, eltStride, got)
//       —— Min=Unroll：全部展开，无运行时分支
//       eltToAcc(got[0]) 作为初始累加量，got[1..Unroll-1] 继续 reduce
//       注意：i==0 时 acc = acc0（初始化），i>0 时 acc = reduce(acc, acc0)
//
//   第二段（尾余段）：
//     if (i < eltCount)：
//       recvUnrolled<Min=1>(eltStart+i*eltStride, eltCount-i, eltStride, got)
//       —— Min=1：只保证读 1 个，尾余段（0..Unroll-2）动态判断
//       同样将尾余段 reduce 进 acc
//
//   返回类型：Acc = decltype(eltToAcc(Elt{}))，通过 trailing return type 推导
#if NCCL_CHECK_CUDACC
template<typename Coop>
template<int Unroll, typename Elt, typename EltToAcc, typename Reduce>
NCCL_DEVICE_INLINE auto ncclLLA2ASession<Coop>::recvReduce(
    int eltStart, int eltCount, int eltStride, EltToAcc eltToAcc, Reduce reduce
  ) -> decltype(eltToAcc(nccl::utility::declval<Elt>())) {
  using Acc = decltype(eltToAcc(nccl::utility::declval<Elt>()));
  Acc acc;
  int i = 0;
  // 第一段：整块 Unroll 处理（静态展开 recvUnrolled<Unroll>）
  #pragma unroll 1
  for (; i+Unroll <= eltCount; i += Unroll) {
    Elt got[Unroll];
    // Min=Unroll：编译器知道恰好读 Unroll 个，全部静态展开
    this->template recvUnrolled</*Min=*/Unroll>(eltStart + i*eltStride, Unroll, eltStride, got);
    Acc acc0 = eltToAcc(got[0]);
    // i==0 时直接赋值（避免未初始化的 acc 参与 reduce）
    acc = i==0 ? acc0 : reduce(acc, acc0);
    // 展开 Unroll-1 次，动态判断 i+j < eltCount
    // 将同批次的 got[1..Unroll-1] 继续 reduce
    #pragma unroll
    for (int j=1; j < Unroll; j++) acc = reduce(acc, eltToAcc(got[j]));
  }
  // 第二段：尾余段处理（eltCount % Unroll 个 slot）
  if (i < eltCount) {
    Elt got[Unroll];
    // Min=1：只保证最少读 1 个，其余动态判断
    this->template recvUnrolled</*Min=*/1>(eltStart + i*eltStride, eltCount-i, eltStride, got);
    Acc acc0 = eltToAcc(got[0]);
    acc = i==0 ? acc0 : reduce(acc, acc0);
    #pragma unroll
    for (int j=1; j < Unroll-1; j++) {
      if (i+j < eltCount) acc = reduce(acc, eltToAcc(got[j]));
    }
  }
  return acc;
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE void ncclLLA2ASession<Coop>::endEpoch(Coop) {
  // Step 1: 检查是否接近 uint32_t 回绕（epoch >= -2u = UINT32_MAX-1）
  if (__builtin_expect(this->epoch >= -2u, false)) {
  // Step 3: 所有线程同步（保证清零完成 + 解除旧 epoch 的 recv 依赖）
    // Step 2: 清零 slot buffer（防止新 epoch 读到旧数据）
    this->coop.sync();
    uint4* buf = (uint4*)ncclGetResourceBufferLocalPointer(this->comm, this->handle.bufHandle);
    buf += this->slotsOffset;
    // 协作清零：每个线程负责部分 slot，#pragma unroll 4 提升 MLP
    #pragma unroll 4
    for (int i=this->coop.thread_rank(); i < this->handle.nSlots; i += this->coop.size()) {
      buf[i] = uint4{0, 0, 0, 0};  // 全零：epoch_tag 为 0，不匹配任何有效 epoch
    }
  }
  this->coop.sync();
  // epoch 推进：正常 +1；若 epoch==-1u（UINT32_MAX）则 +3 跳过 0/1 回到 2
  this->epoch += (this->epoch == -1u) ? 3 : 1;
  // Step 4: 更新 slotsOffset，切换到新 epoch 对应的另一半 buffer
  this->slotsOffset = this->calcSlotOffset();
}
#endif

#endif // _NCCL_DEVICE_LL_A2A__FUNCS_H_
