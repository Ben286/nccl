/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_CORE__TYPES_H_
#define _NCCL_DEVICE_CORE__TYPES_H_
#include "../core.h"
#if defined(NCCL_OS_WINDOWS)
/* Minimal types instead of nccl_device/gin/gin_device_host_common.h (GIN is Linux-only) */
#define NCCL_GIN_MAX_CONNECTIONS 4
typedef void *ncclGinWindow_t;
#else
#include "nccl_device/gin/gin_device_host_common.h"
#endif

// nccl.h has: typedef ncclWindow_vidmem* ncclWindow_t;
// ===== ncclWindow_vidmem：Symmetric Memory 窗口的底层实现结构 =====
// nccl.h 中定义：typedef ncclWindow_vidmem* ncclWindow_t;
//
// 背景：NCCL Symmetric Memory 要求每个 rank 在相同的 cudaMalloc 虚拟地址段
// 内映射所有 rank 的 buffer，以便用一个 base+偏移 就可以访问任意 rank 的内存。
// 具体布局：
//   rank 0 的 buffer 位于 lsaFlatBase
//   rank 1 的 buffer 位于 lsaFlatBase + stride4G * 4GB
//   rank i 的 buffer 位于 lsaFlatBase + i * stride4G * 4GB
// 这种布局通过修改 64 位指针的高 32 位（即 add4G）来实现寻址，
// 而低 32 位（buffer 内 offset）完全相同 → 称为 "symmetric"。
//
struct ncclWindow_vidmem {
  // winHost：对应的 host 端 window 对象指针，仅供 host 侧调试/管理使用
  void* winHost;

  // lsaFlatBase：LSA（Local Symmetric Accessible）flat 地址空间的起始指针。
  // 指向 lsa team 中 rank 0 的 buffer 起始字节。
  // lsa rank i 的 buffer 起始地址 = add4G(lsaFlatBase, i * stride4G)
  // 其中 add4G 只修改 64 位地址的高 32 位（代表 stride4G 个 4GB 跨度），
  // 低 32 位（buffer 内偏移）保持不变。
  char* lsaFlatBase; // pointer to first byte for rank 0 of lsa team

  // lsaRank：本 rank 在 lsa team 内的编号（0..lsaSize-1）。
  // 用于计算 ncclGetLocalPointer：local = add4G(lsaFlatBase, lsaRank * stride4G)
  int lsaRank;

  // worldRank：本 rank 在 world team 内的编号（0..nRanks-1）。
  // 用于 ncclGetPeerPointer(w, offset, worldPeer)：
  //   先计算 delta = worldPeer - worldRank，再转换为 lsa 偏移 lsaRank + delta
  int worldRank;

  // stride4G：各 rank buffer 之间的虚拟地址步长，单位是 4GB（即 2^32 字节）。
  // 物理意义：rank i 和 rank i+1 的 buffer 在虚拟地址空间上相差 stride4G * 4GB。
  // 通常 stride4G == 1（即相邻 4GB），但多路 NVLink 配置下可能更大。
  // 使用 add4G(ptr, delta4G) 修改 64 位指针的高 32 位（低 32 位 offset 不变）。
  uint32_t stride4G;

  // mcOffset4K：MultiCast（NVLink multicast）地址相对于 mcBasePtr 的偏移，
  // 单位是 4KB（即 4096 字节）。
  // 用于 ncclGetMultimemPointer：
  //   ptr = reinterpret_cast<char(*)[4096]>(mm.mcBasePtr) + mcOffset4K
  // MultiCast 写：向该地址写数据会原子地广播到 lsa team 中所有 GPU 的同一 buffer
  uint32_t mcOffset4K;

  // ginOffset4K：GIN（GPU Interconnect Notification）地址相对于 GIN 基址的偏移，
  // 单位是 4KB。GIN 是一种不经过 CPU 的 GPU→GPU 触发通知机制。
  uint32_t ginOffset4K;

  // ginWins：每个 GIN 连接对应一个 ncclGinWindow_t，描述通过该 GIN 连接可访问的
  // 远端 inbox 窗口（用于 put 数据到对端 rank 的 GIN inbox）。
  // 数组大小为 NCCL_GIN_MAX_CONNECTIONS（当前为 4）。
  ncclGinWindow_t ginWins[NCCL_GIN_MAX_CONNECTIONS];
};

// ===== ncclMultimemHandle：NVLink MultiCast 句柄 =====
// NVLink MultiCast（也称 multimem）是一种特殊的内存映射方式：
// 所有 lsa team GPU 都映射到同一个 MultiCast 虚拟地址。
// 向该地址发出的写请求会自动广播到所有 rank 的物理内存。
// 从该地址发出的读请求会聚合（reduce）所有 rank 的值后返回。
// 用于：AllReduce 中的 reduce-scatter 和 all-gather 加速。
struct ncclMultimemHandle {
  // mcBasePtr：NVLink MultiCast 虚拟地址基址（所有 lsa rank 共享同一基址）。
  // 实际访问时需要加上 mcOffset4K（来自 ncclWindow_vidmem）对齐到具体 window。
  void* mcBasePtr;
};

#endif
