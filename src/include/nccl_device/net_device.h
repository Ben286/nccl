/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// ===== net_device.h：网络设备类型枚举和句柄 =====
// 本文件定义 NCCL 网络设备插件（network plugin）的 offload 类型和句柄结构。
//
// 背景：NCCL 支持多种网络硬件（InfiniBand、RoCE 等），
// 每种网络硬件可提供不同级别的"设备端卸载"（device-side offload）能力：
//   - HOST：无 device 卸载，所有网络操作在 CPU/proxy thread 完成
//   - UNPACK：网络数据在 device 端直接解包（减少 CPU-GPU 数据搬运）
//   - GIN_PROXY：通过 GIN 代理模式发送信号（CPU proxy 辅助）
//   - GIN_GDAKI：通过 GIN GDAKI 模式直接在 device 端触发 RDMA（无需 CPU 介入）
//
// 版本兼容性：
//   ncclNetDeviceHandle 结构随 NCCL 版本演进（v7 ~ v12），
//   每个版本新增字段时通过 typedef 保持向后兼容。
//   插件通过 net->getProperties() 返回 netDeviceVersion，
//   NCCL 内部校验该版本号是否与 NCCL_NET_DEVICE_UNPACK_VERSION 匹配。

#ifndef NCCL_NET_DEVICE_H_
#define NCCL_NET_DEVICE_H_

// NCCL_NET_DEVICE_INVALID_VERSION：版本号 0，表示网络设备插件不支持任何 device offload
// 若 getProperties() 返回此版本号，NCCL 认为插件无 device 端能力，退化为纯 host 模式
#define NCCL_NET_DEVICE_INVALID_VERSION      0x0

// NCCL_NET_MTU_SIZE：网络 MTU（最大传输单元）= 4096 字节
// 用于对齐和分段网络包（如 GIN DMA 操作的粒度）
#define NCCL_NET_MTU_SIZE                    4096

// Arbitrary version number - A given NCCL build will only be compatible with a single device networking plugin
// version. NCCL will check the supplied version number from net->getProperties() and compare to its internal version.
// NCCL_NET_DEVICE_UNPACK_VERSION：当前 NCCL 构建支持的 device offload 版本号
// 版本号 0x7 对应 ncclNetDeviceHandle_v7_t 结构定义。
// NCCL 会通过 net->getProperties().netDeviceVersion 校验插件版本是否匹配，
// 不匹配时禁用 device offload 功能（避免 ABI 不兼容导致崩溃）。
// 注意：单次 NCCL 构建只兼容一种版本（hardcoded），插件必须精确匹配。
#define NCCL_NET_DEVICE_UNPACK_VERSION 0x7

// -------------------------------------------------------------------------
// ncclNetDeviceType：网络设备 offload 类型枚举
// -------------------------------------------------------------------------
typedef enum {
  // NCCL_NET_DEVICE_HOST：无 device 端卸载
  //   所有网络收发操作由 CPU proxy thread 完成，GPU 仅触发网络请求
  //   对应传统 NCCL 的通信路径（无 device API 参与网络操作）
  NCCL_NET_DEVICE_HOST=0,

  // NCCL_NET_DEVICE_UNPACK：device 端数据解包卸载
  //   网络数据由 NIC 直接 DMA 到 GPU 内存，GPU kernel 负责解包（unpack）
  //   适用于支持 GPU Direct RDMA 的 InfiniBand/RoCE 网卡
  //   版本要求：netDeviceVersion == NCCL_NET_DEVICE_UNPACK_VERSION（0x7）
  NCCL_NET_DEVICE_UNPACK=1,

  // NCCL_NET_DEVICE_GIN_PROXY：GIN 代理模式
  //   GPU 通过 GIN（GPU Interconnect Notification）机制触发信号，
  //   由 CPU proxy thread 监听 GIN 信号并转发给网络（间接模式）
  //   适用于网卡不支持 GDAKI 但支持 GIN 信号的场景
  NCCL_NET_DEVICE_GIN_PROXY=2,

  // NCCL_NET_DEVICE_GIN_GDAKI：GIN + GDAKI 模式（GPU Direct Async KI）
  //   GPU kernel 直接通过 GDAKI 触发 NIC 的 RDMA 操作，无需 CPU 介入
  //   "GDAKI" = GPU Direct Async Kernel Initiated：GPU kernel 异步发起 RDMA
  //   适用于最高性能场景（NVIDIA ConnectX-7+ 等支持 GDAKI 的网卡）
  //   此模式下 GIN signal 直接映射到 NIC 的发送队列触发器
  NCCL_NET_DEVICE_GIN_GDAKI=3,
} ncclNetDeviceType;

// -------------------------------------------------------------------------
// ncclNetDeviceHandle_v7_t：网络设备句柄（version 7）
// -------------------------------------------------------------------------
// 插件通过此结构向 NCCL 传递 device offload 所需的配置信息。
// NCCL 在建立连接时检查此结构并配置相应的 device 端资源。
typedef struct {
  // netDeviceType：offload 类型（见 ncclNetDeviceType 枚举）
  // NCCL 根据此字段选择 device 端处理路径
  ncclNetDeviceType netDeviceType; // Network offload type

  // netDeviceVersion：插件报告的版本号，NCCL 与 NCCL_NET_DEVICE_UNPACK_VERSION 比较
  // 不匹配时 NCCL 禁用该 offload 类型并报告 warning
  int netDeviceVersion;            // Version number for network offload

  // handle：指向插件自定义的 device 端句柄数据（plugin-specific，NCCL 不解析）
  // 内容由插件定义，可以是 RDMA QP 信息、GIN 配置等
  // 该内存通常需要 device 可访问（如 pinned memory 或 device memory）
  void* handle;

  // size：handle 指向数据的字节大小
  // NCCL 用于拷贝/传递 handle 数据时的边界检查
  size_t size;

  // needsProxyProgress：是否需要 CPU proxy thread 持续轮询
  //   1：需要（如 GIN_PROXY 模式，CPU 需要监听 GIN 信号并转发）
  //   0：不需要（如 GIN_GDAKI 模式，GPU 直接触发网络，无需 CPU 干预）
  int needsProxyProgress;
} ncclNetDeviceHandle_v7_t;

// -------------------------------------------------------------------------
// 版本别名（向后兼容）
// -------------------------------------------------------------------------
// v7 到 v12 保持相同的结构定义，通过 typedef 链实现向前声明兼容。
// 未来新版本若需要新增字段，只需在新版本结构体中添加，
// 旧版本插件仍然可以与旧版本 NCCL 匹配（版本号校验确保精确匹配）。
typedef ncclNetDeviceHandle_v7_t ncclNetDeviceHandle_v8_t;  // v8 = v7（结构不变）
typedef ncclNetDeviceHandle_v8_t ncclNetDeviceHandle_v9_t;  // v9 = v8
typedef ncclNetDeviceHandle_v9_t ncclNetDeviceHandle_v10_t;  // v10 = v9
typedef ncclNetDeviceHandle_v10_t ncclNetDeviceHandle_v11_t;  // v11 = v10
typedef ncclNetDeviceHandle_v11_t ncclNetDeviceHandle_v12_t;  // v12 = v11（当前最新）
// 当前版本的 canonical typedef（代码中使用 ncclNetDeviceHandle_t 引用最新结构）
typedef ncclNetDeviceHandle_v12_t ncclNetDeviceHandle_t;

#endif
