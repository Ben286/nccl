/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/


// ===== comm__types.h：ncclDevComm — Device 端通信上下文结构 =====
// ncclDevComm 是所有 Device API 操作的根对象，包含：
//   - rank/团队信息（world/lsa rank 编号和大小）
//   - window 表（用于 ncclFindWindow 协作搜索）
//   - resource window（用户自定义 buffer 的 symmetric 窗口）
//   - MultiCast 句柄（lsaMultimem）
//   - Barrier 句柄（lsaBarrier/railGinBarrier + hybridLsaBarrier/hybridRailGinBarrier/worldGinBarrier）
//   - GIN（GPU Interconnect Notification）相关字段
//   - 故障容忍（abortFlag）
//
// 用法：
//   ncclDevComm const& comm = *devComm;  // 从 ncclDevComm_t* 解引用
//   ncclTeam lsa = ncclTeamLsa(comm);    // 获取 lsa team
//   void* ptr = ncclGetLocalPointer(window, offset);
#ifndef _NCCL_DEVICE_COMM__TYPES_H_
#define _NCCL_DEVICE_COMM__TYPES_H_
#include "../comm.h"
#include "core__types.h"
#include "ll_a2a__types.h"
#include "lsa_barrier__types.h"
#include "gin_barrier__types.h"

#if __cplusplus
struct ncclDevCommWindowTable {
  // Entry：一个 window 的地址范围描述
  struct Entry {
    uintptr_t base, size;
    ncclWindow_t window;  // 对应的 ncclWindow_t 指针（window==0 表示无效）
  } entries[32];  // 每个表节点最多 32 个 window，与并行搜索的 warp 宽度对应
  struct ncclDevCommWindowTable* next;  // 链表下一节点，nullptr 表示结尾
};
#endif
typedef struct ncclDevCommWindowTable* ncclDevCommWindowTable_t;

// ncclDevComm：Device 端通信上下文（所有 Device API 的根对象）
struct ncclDevComm {
  // [2.30.3 新增] DevComm 版本化元数据
  // Internal NCCL structure versioning metadata.  Do not modify.
  unsigned int magic;
  unsigned int version;

  int rank, nRanks;  // world rank 和 world 大小
  uint32_t nRanks_rcp32;  // nRanks 的预计算倒数，供 idivFast32 使用（避免除法指令）
  int lsaRank, lsaSize;
  uint32_t lsaSize_rcp32;

  ncclDevCommWindowTable_t windowTable;

  // ===== Resource Window =====
  // resourceWindow：用户通过 ncclDevResourceRequirements 申请的 resource buffer 的 window
  // ncclGetResourceBuffer* 系列函数通过此 window 寻址
  ncclWindow_t resourceWindow;
  ncclWindow_vidmem_t resourceWindow_inlined;

  // ===== MultiCast 句柄 =====
  // lsaMultimem：lsa team 的 NVLink MultiCast 句柄（mcBasePtr）
  // 当 lsaMultimem==true 时由 NCCL 分配，否则为空
  // 用于 ncclGetLsaMultimemPointer / ncclGetResourceBufferLsaMultimemPointer
  ncclMultimemHandle_t lsaMultimem;

  // ===== Barrier 句柄 =====
  // lsaBarrier：跨 lsa team 所有 rank 的 barrier 句柄
  // 通过 symmetric memory 实现，无需经过 CPU
  ncclLsaBarrierHandle_t lsaBarrier;

  // railGinBarrier：跨 rail 维度（跨节点）的 GIN barrier 句柄
  // 通过 GIN DMA 实现跨节点同步
  ncclGinBarrierHandle_t railGinBarrier;

  // ===== GIN（GPU Interconnect Notification）相关字段 =====
  // ginConnectionCount：当前 DevComm 中活跃的 GIN 连接数（0..NCCL_GIN_MAX_CONNECTIONS）
  uint8_t ginConnectionCount;

  // ginNetDeviceTypes[i]：第 i 个 GIN 连接的网络设备类型（PROXY 或 GDAKI）
  // PROXY：通过 CPU proxy 调度；GDAKI：通过 GPU DMA（GDAKI）直接执行
  uint8_t ginNetDeviceTypes[NCCL_GIN_MAX_CONNECTIONS];

  // ginHandles[i]：第 i 个 GIN 连接的句柄（由 GIN 后端分配）
  // proxy 类型为 ncclGinProxyHandle*；gdaki 类型为相应的 GDAKI 句柄
  void* ginHandles[NCCL_GIN_MAX_CONNECTIONS];
  int ginSignalCount;  // 用户分配的 GIN signal 数量
  int ginCounterCount;  // 用户分配的 GIN counter 数量
  // ginSignalShadows：GIN signal 的 shadow 副本（host 端写、device 端读）
  // 由于 GIN signal 可能存放在 GIN 的特殊内存中（device 端不可直接读），
  // shadow 副本提供 device 端可访问的镜像，供 spin-poll 等待
  uint64_t* ginSignalShadows;

  // ginContextCount：实际分配的 GIN context 数量
  // 每个 GIN context 对应一个独立的 GIN queue / DMA 请求通道
  uint32_t ginContextCount;

  // ginIsRailed：GIN 连接是否为 rail 模式
  // true：GIN 连接仅在 rail 维度（跨节点）之间建立
  // false：GIN 连接在 full 拓扑（所有 rank 两两）之间建立
  bool ginIsRailed; // Whether the GIN connections are railed

  // FT related
  // ===== 故障容忍（FT）字段 =====
  // abortFlag：指向 host 端的 abort 标志，device 端通过 testAbort 周期性轮询
  // 当 host 端设置 *abortFlag != 0 时，device kernel 应终止运行
  uint32_t* abortFlag;

  // [2.30.3 新增] hybrid barriers —— 用于对称 kernel 的跨层同步
  // hybridLsaBarrier：节点内 LSA barrier（hybrid 模式下使用）
  ncclLsaBarrierHandle_t hybridLsaBarrier;
  // hybridRailGinBarrier：跨节点 Rail GIN barrier（hybrid 模式下使用）
  ncclGinBarrierHandle_t hybridRailGinBarrier;

  // [2.30.3 新增] worldGinBarrier：跨所有 rank 的全局 GIN barrier
  ncclGinBarrierHandle_t worldGinBarrier;
};

#endif // _NCCL_DEVICE_COMM__TYPES_H_
