/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_GIN_HOST_H_
#define _NCCL_GIN_HOST_H_

#include "allocator.h"
#include "nccl.h"
#include "nccl_gin.h"
#include "nccl_device/gin/gin_device_host_common.h"
#include <thread>
#include <mutex>
#include <condition_variable>

// [2.30.3 新增] ncclGinStateDevComm：per-DevComm 的 GIN 状态
// 2.30.3 将 GIN context/handle 从全局 ncclGinState 拆分到 per-DevComm 结构
// 每个 DevComm 创建时分配自己的 GIN context，生命周期跟随 DevComm
struct ncclGinStateDevComm {
  int contextCount;
                        // 由 ncclGin->connect() 返回，包含 IB QP 句柄/DOCA QP 等连接状态
                        // NCCL_GIN_MAX_CONNECTIONS = 4，对应最大 4 个连接（跨节点最多 4 个 NIC）
  void* ginCtx[NCCL_GIN_MAX_CONNECTIONS];  // 每条 GIN 连接的 context
  ncclNetDeviceHandle_t* devHandles[NCCL_GIN_MAX_CONNECTIONS];
  struct ncclGinStateDevComm* next;  // 链表指针，用于遍历所有 DevComm 的 GIN 状态
                        // 可能值：NONE / FULL / RAIL
                        //   NONE：未建立连接
                        //   FULL：与所有 world rank 建立连接
                        //   RAIL：只与 rail 内 rank 建立连接（RailRing AllGather/ReduceScatter 使用）
};

// ============================================================================
// ncclGinState：GIN（GPU Initiated Network）的全局状态结构体
//
// GIN 概述：
//   传统 NCCL 流程： GPU kernel → workFifo → CPU proxy 线程 → IB
//   GIN 流程：GPU kernel 直接发起 RDMA ，有两种实现方式：
//
//   1. GIN_PROXY 后端：
//      GPU kernel 将 GFD（Go Forward Descriptor，128字节）写入共享队列
//      CPU proxy 线程轮询队列，通过 IB Verbs 将 GFD 转发给远端
//      优点：兼容性好，不需要特殊硬件
//      缺点：不能完全 bypass CPU（还是需要 CPU 中集）
//
//   2. GIN_GDAKI 后端：
//      GPU kernel 直接操作 DOCA GPUNetIO Verbs QP
//      完全 bypass CPU，真正的 GPU-side RDMA
//      需要特殊硬件：带 DOCA GPUNetIO 支持的 NIC
//
// ncclGinState 与 ncclComm 的关系：
//   每个 ncclComm 拥有一个 ncclGinState。
//   ncclGinState 生命周期随 comm 创建/销毁。
//   GIN 连接直到第一次使用时才建立（ncclGinConnectOnce）。
// ============================================================================
struct ncclGinState {
  // ---------------------------------------------------------------------------
  // 插件接口和全局实例
  // ---------------------------------------------------------------------------
  ncclGin_t* ncclGin;  // GIN 插件接口表指针（类似 ncclNet_t 与常规网络插件的关系）
                        // 包含：getProperties/listen/connect/createContext/progress/close 等函数指针
                        // 由 ncclLoadGinPlugin() 动态加载充入
  void* ginInstance;  // GIN 插件全局实例（每个进程内全局唯一）
                        // 由 ncclGin->createInstance() 创建，包含插件内部状态（如 IBQP 资源池）
  bool connected;  // 是否已完成 GIN 连接建立（ncclGinConnectOnce 返回后置为 true）
  // ---------------------------------------------------------------------------
  // GIN 类型与连接数量
  // ---------------------------------------------------------------------------
  ncclGinType_t ginType;  // 当前进程支持的 GIN 类型（GIN_NONE/GIN_PROXY/GIN_GDAKI）
                        // 注：这是已与其他 rank 协商后的全局类型，不仅仅是本地支持
  int ginCommCount;  // 已建立的 GIN 连接总数
  // ---------------------------------------------------------------------------
  // 连接句柄和上下文
  // ---------------------------------------------------------------------------
  void* ginComms[NCCL_GIN_MAX_CONNECTIONS];  // 每条 GIN 连接的连接层句柄
  // [2.30.3 新增] 每条连接的网络属性（如设备名、速度、端口号等）
  ncclNetProperties_t ginProps[NCCL_GIN_MAX_CONNECTIONS];
  // ---------------------------------------------------------------------------
  // GIN_PROXY 后端的进展线程
  // ---------------------------------------------------------------------------
                        // GPU 内可用的 device handle，封装了 QP/GFD 队列地址
                        // ncclNetDeviceHandle_t = 面向 device 的网络针对 handle，包含 context/signal/counter 就位
  int needsProxyProgress;  // Whether we need to progress GIN operations with the proxy
                        // =1 表示需要 CPU proxy 线程轮询 GFD 队列（GIN_PROXY 后端恰此标志）
                        // =0 表示 GPU 可纯硬件操作不需要 CPU 现驾（GIN_GDAKI）
  int ginProgress;         // GIN progress is enabled
                        // 状态机：
                        //   0  → 线程正在初始化，等待条件变量通知
                        //   1  → 正常运行，轮询所有 ginCtx[] 的 progress
                        //  -1  → 请求退出（ncclGinHostFinalize 设置）
                        //  -2  → 错误状态（线程自行设置）
  std::thread thread;
  std::mutex mutex;
  std::condition_variable cond;
  ncclResult_t asyncResult;
                        // 主线程通过 ncclGinQueryLastError 查询
  // [2.30.3 新增] GIN 插件版本号
  int ginVersion;

  // [2.30.3 新增] per-DevComm GIN 状态链表头
  struct ncclGinStateDevComm* devComms;
  // ---------------------------------------------------------------------------
  // 连接类型
  // ---------------------------------------------------------------------------
  ncclGinConnectionType_t ginConnectionType;  // 当前建立的连接类型
};

extern int64_t ncclParamGinType();

// Sets the local GIN type for comm. The GIN type that is set for comm is the
// GIN type supported by the call process itself, without taking into account
// (1) GIN support of other ranks, and (2) additional local constraints like
// cross-NIC
ncclResult_t setLocalGinType(struct ncclComm* comm);
// Get the GIN type from comm. ginType is set to the GIN type that can be used
// by the comm to communicate with other nodes.
ncclResult_t getGlobalGinType(struct ncclComm* comm, ncclGinType_t* ginType);
ncclResult_t getGlobalRailedGinType(struct ncclComm* comm, ncclGinType_t* ginType);

// FIXME change to ncclGinState instead of ncclComm, no need to pass comm
// [2.30.3 变更] ncclGinConnectOnce 签名简化（移除冗余参数）
ncclResult_t ncclGinConnectOnce(struct ncclComm* comm);
ncclResult_t ncclGinHostFinalize(struct ncclComm* comm);
// [2.30.3 新增] per-DevComm GIN 资源管理
ncclResult_t ncclGinDevCommSetup(struct ncclComm* comm, struct ncclDevCommRequirements const* reqs, struct ncclDevComm* devComm);
ncclResult_t ncclGinDevCommFree(struct ncclComm* comm, struct ncclDevComm const* devComm);
// [2.30.3 变更] ncclGinRegister 新增 multiSegment 参数
ncclResult_t ncclGinRegister(struct ncclComm* comm, void* address, size_t size,
                             void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS],
                             ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONNECTIONS], int winFlags,
                             bool multiSegment = false);
ncclResult_t ncclGinDeregister(struct ncclComm* comm, void* ginHostWins[NCCL_GIN_MAX_CONNECTIONS]);
ncclResult_t ncclGinQueryLastError(struct ncclGinState* ginState, bool* hasError);

#endif
