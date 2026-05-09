/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// ===== core.h：NCCL Device API 核心头文件总览 =====
// 本文件定义了 Device API 的所有核心类型和函数接口：
//
// 1. ncclTeam：逻辑 rank 组抽象，描述一组 GPU rank 的编号方式
//    - ncclTeamWorld：全部 rank（stride=1）
//    - ncclTeamLsa：本节点 lsa（Local Symmetric Accessible）组
//    - ncclTeamRail：rail 维度组（= outer factor of lsa team）
//
// 2. ncclWindow_t（= ncclWindow_vidmem*）：Symmetric Memory 窗口
//    - 每个 window 对应一段对齐的跨 GPU 共享内存区域
//    - ncclGetLocalPointer / ncclGetLsaPointer / ncclGetPeerPointer：
//      通过 add4G（修改高 32 位）寻址不同 rank 的 buffer
//    - ncclGetMultimemPointer：NVLink MultiCast 地址（写→广播/读→reduce）
//
// 3. ncclDevCommRequirements：创建 DevComm 时的静态声明
//    - 声明所需的 GIN context/signal/counter 数量
//    - 声明所需的 barrier/lsaBarrier/railGinBarrier/LLA2A 数量
//
// 4. ncclDevResourceHandle：用户自定义 resource buffer 的不透明句柄
//    每个 handle 对应 128 字节对齐的 resource buffer slot
//
// 5. Window API（Device side, __device__ only）：
//    ncclGetLocalPointer / ncclGetLsaPointer / ncclGetPeerPointer /
//    ncclGetMultimemPointer / ncclGetLsaMultimemPointer
//    以及对应的 ncclGetResourceBuffer*** 便捷函数

#ifndef _NCCL_DEVICE_CORE_H_
#define _NCCL_DEVICE_CORE_H_
#include <nccl.h>
#include "coop.h"
#include "utility.h"

// ncclDevComm：DevComm 对象（Device 端通信上下文），详见 impl/comm__types.h
struct ncclDevComm;
typedef struct ncclDevComm ncclDevComm_t;

// ncclTeam：逻辑 rank 组（见下方 ncclTeam 结构体定义）
struct ncclTeam;
typedef struct ncclTeam ncclTeam_t;

// typedef struct ncclWindow_vidmem* ncclWindow_t; // in nccl.h
typedef struct ncclWindow_vidmem ncclWindow_vidmem_t;

// ncclMultimemHandle：NVLink MultiCast 句柄（见 impl/core__types.h）
// ncclWindow_t 定义在 nccl.h：typedef struct ncclWindow_vidmem* ncclWindow_t
struct ncclMultimemHandle;
typedef struct ncclMultimemHandle ncclMultimemHandle_t;

// ncclDevResourceHandle：用户 resource buffer 的不透明句柄（32 位整数）
// 每个 handle 对应 resource window 中偏移 handle*128 字节处的 128B 槽位
typedef uint32_t ncclDevResourceHandle;
typedef ncclDevResourceHandle ncclDevResourceHandle_t;

// ncclGinSignal_t / ncclGinCounter_t：GIN 信号和计数器的类型
// GIN（GPU Interconnect Notification）用于 GPU→GPU 直接通知，不经过 CPU
typedef uint32_t ncclGinSignal_t;
typedef uint32_t ncclGinCounter_t;

typedef struct alignas(uint64_t) {
  char opaque[16];
} ncclGinRequest_t;

// 各类 barrier 和 LLA2A 句柄的前向声明（详见对应 barrier.h / lsa_barrier.h 等）
struct ncclLsaBarrierHandle;
typedef struct ncclLsaBarrierHandle ncclLsaBarrierHandle_t;

struct ncclGinBarrierHandle;
typedef struct ncclGinBarrierHandle ncclGinBarrierHandle_t;

struct ncclLLA2AHandle;
typedef struct ncclLLA2AHandle ncclLLA2AHandle_t;

// ===== ncclTeam：逻辑 rank 组抽象 =====
// ncclTeam 描述一组逻辑上相邻的 GPU rank 集合，用于寻址和跨 GPU 操作。
// 每个 ncclTeam 实质上定义了一个等差数列：全局 rank 的子集 {rank + k*stride | 0<=k<nRanks}
//
// 字段说明：
//   nRanks：该组中的 rank 总数
//   rank：本节点（本 rank）在该组中的编号（0..nRanks-1）
//   stride：该组的全局 rank 步长，即该组中相邻 rank k 和 k+1 
//           对应的全局 rank 相差 stride
//
// 举例：8 个全局 rank，分为 2 个 LSA team（每组 4 个）：
//   ncclTeamWorld： {nRanks=8, rank=myRank, stride=1}
//   ncclTeamLsa：  {nRanks=4, rank=myRank%4, stride=1}
//   ncclTeamRail： {nRanks=2, rank=myRank/4, stride=4}  (即 lsa 的 outer factor)
struct ncclTeam {
  // nRanks：该组内的 rank 总数
  // rank：本 rank 在该组内的编号（0..nRanks-1）
  // stride：该组内相邻 rank 对应的全局 rank 步长（world team 中 stride=1，rail team 中 stride=lsaSize）
  int nRanks, rank, stride;
};

#if __cplusplus
template<typename T> struct ncclSymPtr;
#endif

#if __cplusplus
struct ncclTeamTagWorld {};
struct ncclTeamTagLsa {};
struct ncclTeamTagRail {};
#endif

struct ncclDevCommRequirements;
typedef struct ncclDevCommRequirements ncclDevCommRequirements_t;

struct ncclDevResourceRequirements;
typedef struct ncclDevResourceRequirements ncclDevResourceRequirements_t;

struct ncclTeamRequirements;
typedef struct ncclTeamRequirements ncclTeamRequirements_t;

struct ncclCommProperties;
typedef struct ncclCommProperties ncclCommProperties_t;

// ===== ncclGinConnectionType_t：GIN 连接拓扑类型 =====
// NCCL_GIN_CONNECTION_NONE：不使用 GIN
// NCCL_GIN_CONNECTION_FULL：所有 rank 两两之间都有 GIN 连接（全连拓扑）
// NCCL_GIN_CONNECTION_RAIL：只有 rail 维度的 GIN 连接（跨节点）
typedef enum {
  NCCL_GIN_CONNECTION_NONE,
  NCCL_GIN_CONNECTION_FULL,
  NCCL_GIN_CONNECTION_RAIL,
} ncclGinConnectionType_t;

// ===== ncclDevCommRequirements：创建 DevComm 时的静态声明结构 =====
// 用户在创建 DevComm 前填写此结构，向 NCCL 说明自己需要哪些资源。
// NCCL 会根据这些要求预分配对应资源并返回一个配置好的 DevComm 对象。
// 使用方法：
//   ncclDevCommRequirements_t req = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
//   req.ginSignalCount = 4;  // 提出所需资源
//   ncclDevCommCreate(comm, &req, &devComm);
struct ncclDevCommRequirements {
  /* attributes that users should never touch. */
  // size：结构体大小，用于 ABI 版本检查
  // magic：魔数，防止错误传入非 NCCL 结构体
  // version： NCCL 版本号检查
  size_t size;
  unsigned int magic;
  unsigned int version;

  /* attributes that users are able to customize. */

  // resourceRequirementsList：单链表，每个节点对应一个用户 resource buffer 窗口需求
  // 每个 ncclDevResourceRequirements 包含 bufferSize/bufferAlign/ginSignal/ginCounter 需求量
  ncclDevResourceRequirements_t* resourceRequirementsList;

  // teamRequirementsList：单链表，每个节点声明一个需要 MultiCast 支持的 team
  ncclTeamRequirements_t* teamRequirementsList;

  // lsaMultimem：是否为 lsa team 开启 MultiCast 支持（NVLink multicast）
  // 开启后可以通过 ncclGetLsaMultimemPointer 获取 lsa 的 MultiCast 地址
  bool lsaMultimem; // Enable multimem on lsa team

  // barrierCount：需要的 CTA 级 barrier 数量（对应 barrier.h 中的 ncclBarrierHandle）
  // 这种 barrier 用于 CTA 内部的同步（不涉及趪 CUDA CTA）
  int barrierCount;

  // lsaBarrierCount：需要的 lsa 级 barrier 数量（对应 lsa_barrier.h）
  // lsa barrier 可跨 lsa team 内所有 rank 提供同步，通过 symmetric memory 实现
  int lsaBarrierCount;

  // railGinBarrierCount：需要的 rail GIN barrier 数量（对应 gin_barrier.h）
  // rail GIN barrier 通过 GIN 实现跨节点（rail 组）的同步
  int railGinBarrierCount;

  // lsaLLA2ABlockCount / lsaLLA2ASlotCount：Low-Latency All-to-All 资源配置
  // lsaLLA2ABlockCount：每个 工作组占用的 LLA2A block 数
  // lsaLLA2ASlotCount：每个 LLA2A block 中的 slot 数（同时飞行的 A2A 数）
  int lsaLLA2ABlockCount, lsaLLA2ASlotCount;

  // ginForceEnable：即使硬件不支持 GIN 也强制开启（测试用）
  bool ginForceEnable;

  // ginContextCount：提示 NCCL 预分配的 GIN context 数（实际 DevComm 中的数量可能不同）
  // 每个 GIN context 对应一个 GIN queue/DMA 请求通道
  int ginContextCount; // This is a hint, the actual context count in the devcomm may not match.

  // ginSignalCount：需要的 GIN signal 数量，保证从 id=0 开始连续分配
  // GIN signal 用于远端 rank 通过 GIN DMA 触发本 rank 的内存中的一个 slot
  int ginSignalCount; // Guaranteed to start at id=0

  // ginCounterCount：需要的 GIN counter 数量，保证从 id=0 开始连续分配
  // GIN counter 用于计数已完成的 GIN put 次数，device 端可 spin-poll
  int ginCounterCount; // Guaranteed to start at id=0

  // ginConnectionType： GIN 连接拓扑类型（NONE/FULL/RAIL）
  ncclGinConnectionType_t ginConnectionType;

  // ginExclusiveContexts：是否为每个 GIN context 分配独占的 DMA 通道
  // true：每个 context 不共享，降低串行延迟；
  // false：多个 context 可能共用相同 DMA 通道（节省资源）
  bool ginExclusiveContexts;

  // ginQueueDepth：每个 GIN context 的队列深度（同时飞行的 put 数）
  // 设置为 0 表示使用默认深度
  int ginQueueDepth;
  int ginTrafficClass;

  int worldGinBarrierCount;
};

#define NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER {                               \
    sizeof(ncclDevCommRequirements_t),           /* size */                    \
    NCCL_API_MAGIC,                              /* magic */                   \
    NCCL_VERSION_CODE,                           /* version */                 \
    nullptr,                                     /* resourceRequirementsList*/ \
    nullptr,                                     /* teamRequirementsList */    \
    false,                                       /* lsaMultimem */             \
    0,                                           /* barrierCount */            \
    0,                                           /* lsaBarrierCount */         \
    0,                                           /* railGinBarrierCount */     \
    0,                                           /* lsaLLA2ABlockCount */      \
    0,                                           /* lsaLLA2ASlotCount */       \
    false,                                       /* ginForceEnable */          \
    4,                                           /* ginContextCount */         \
    0,                                           /* ginSignalCount */          \
    0,                                           /* ginCounterCount */         \
    NCCL_GIN_CONNECTION_NONE,                    /* ginConnectionType */       \
    false,                                       /* ginExclusiveContexts */    \
    0,                                           /* ginQueueDepth */           \
    NCCL_CONFIG_UNDEF_INT,                       /* ginTrafficClass */         \
    0,                                           /* worldGinBarrierCount */    \
}

// ===== ncclDevResourceRequirements：单个 Resource Buffer 需求描述 =====
// 作为 resourceRequirementsList 链表中的节点，每个节点描述一个独立的
// symmetric resource buffer 需求。NCCL 会分配对应的 buffer 并写入 out* 字段。
struct ncclDevResourceRequirements {
  // next：链表下一节点，nullptr 表示链表结尾
  ncclDevResourceRequirements_t* next;

  // bufferSize / bufferAlign：申请的 buffer 大小和对齐要求（单位：字节）
  // NCCL 会在 resource window 中分配一块满足对齐要求的连续内存
  size_t bufferSize, bufferAlign;

  // outBufferHandle：分配成功后， NCCL 向该指针写入分配结果（句柄）
  // 如果为 nullptr，表示用户不需要返回句柄
  ncclDevResourceHandle_t* outBufferHandle; // If non-null, target assigned during ncclDevCommCreate.

  // ginSignalCount / ginCounterCount：该 buffer 需要的额外 GIN signal / counter 数量
  // 这些 signal/counter 将与 resourceRequirementsList 中说明的整体句柄相关联
  int ginSignalCount;
  int ginCounterCount;

  // outGinSignalStart / outGinCounterStart：NCCL 将分配到的起始 signal/counter
  // 地址写入这两个指针，用户可用它们寻址具体的 signal/counter 数组
  ncclGinSignal_t* outGinSignalStart;
  ncclGinCounter_t* outGinCounterStart;
};

// ===== ncclTeamRequirements：单个 Team MultiCast 需求描述 =====
// 作为 teamRequirementsList 链表中的节点，
// 声明对某个 team 需要 NVLink MultiCast 支持。
struct ncclTeamRequirements {
  // next：链表下一节点
  ncclTeamRequirements_t* next;

  // team：该需求对应的 team（由 ncclTeamXxx 函数获取）
  ncclTeam_t team;

  // multimem：是否需要为此 team 开启 NVLink MultiCast
  bool multimem;

  // outMultimemHandle：如果非 nullptr，NCCL 在 ncclDevCommCreate 时将分配到的
  // MultiCast 句柄写入此指针，用户后续可用它调用 ncclGetMultimemPointer
  ncclMultimemHandle_t* outMultimemHandle; // If non-null, target assigned during ncclDevCommCreate.
};

#define NCCL_COMM_PROPERTIES_INITIALIZER {                           \
  sizeof(ncclCommProperties_t),                  /* size */          \
  NCCL_API_MAGIC,                                /* magic */         \
  NCCL_VERSION_CODE,                             /* version */       \
}

// ===== ncclGinType_t：GIN 硬件实现类型 =====
// NCCL_GIN_TYPE_NONE：无 GIN(不支持或未开启）
// NCCL_GIN_TYPE_PROXY：通过 CPU proxy 调度的 GIN（较高延迟）
//                   值=2，与 NCCL_NET_DEVICE_GIN_PROXY 兼容
// NCCL_GIN_TYPE_GDAKI：通过 GPU DMA（GDAKI）直接执行的 GIN（低延迟）
//                    值=3，与 NCCL_NET_DEVICE_GIN_GDAKI 兼容
typedef enum {
  NCCL_GIN_TYPE_NONE = 0,
  NCCL_GIN_TYPE_PROXY = 2, // intentially not 1. Must match NCCL_NET_DEVICE_GIN_PROXY for backward compatibility
  NCCL_GIN_TYPE_GDAKI = 3, // intentially not 2. Must match NCCL_NET_DEVICE_GIN_GDAKI for backward compatibility
} ncclGinType_t;

// ===== ncclCommProperties：查询 NCCL Communicator 属性的只读结构 =====
// 用户通过 ncclCommQueryProperties() 获取本结构，了解当前通信子的硬件支持情况。
struct ncclCommProperties {
  /* internal use only */
  // size/magic/version：内部 ABI 检查字段，用户不应修改
  size_t size;
  unsigned int magic;
  unsigned int version;

  /* attributes for users. */
  // rank / nRanks：本 rank 编号和全局 rank 总数
  int rank;
  int nRanks;

  // cudaDev / nvmlDev：本 rank 对应的 CUDA 设备号和 NVML 设备号
  int cudaDev;
  int nvmlDev;

  // deviceApiSupport：是否支持 Device API（即是否支持创建 DevComm）
  // 主要取决于硬件（需要 NVLink Symmetric Memory 支持）
  bool deviceApiSupport;

  // multimemSupport：是否支持 NVLink MultiCast
  bool multimemSupport;

  // ginType：本节点 GIN 硬件类型（NONE/PROXY/GDAKI）
  ncclGinType_t ginType;

  // nLsaTeams：本节点属于的 lsa team 总数
  // 即全局 nRanks / lsaSize
  int nLsaTeams;

  // hostRmaSupport：是否支持 Host 端 RMA（Remote Memory Access）
  bool hostRmaSupport;

  // railedGinType：rail 维度的 GIN 类型（跨节点 GIN）
  ncclGinType_t railedGinType;
};

NCCL_EXTERN_C __host__ ncclResult_t ncclCommQueryProperties(ncclComm_t comm, ncclCommProperties_t* props);
NCCL_EXTERN_C __host__ ncclResult_t ncclDevCommCreate(ncclComm_t comm, ncclDevCommRequirements_t const* reqs, ncclDevComm_t* outDevComm);
NCCL_EXTERN_C __host__ ncclResult_t ncclDevCommDestroy(ncclComm_t comm, ncclDevComm_t const* devComm);

// Host 端获取窗口指针的辅助函数：
// ncclGetLsaMultimemDevicePointer：获取 lsa MultiCast 地址（host 上计算用）
// ncclGetMultimemDevicePointer：获取指定 multimem 句柄的 MultiCast 地址
// ncclGetLsaDevicePointer：获取指定 lsa rank 的 buffer 地址
// ncclGetPeerDevicePointer：获取指定 peer rank 的 buffer 地址
NCCL_EXTERN_C __host__ ncclResult_t ncclGetLsaMultimemDevicePointer(ncclWindow_t window, size_t offset, void** outPtr);
NCCL_EXTERN_C __host__ ncclResult_t ncclGetMultimemDevicePointer(ncclWindow_t window, size_t offset, ncclMultimemHandle_t multimem, void** outPtr);
NCCL_EXTERN_C __host__ ncclResult_t ncclGetLsaDevicePointer(ncclWindow_t window, size_t offset, int lsaRank, void** outPtr);
NCCL_EXTERN_C __host__ ncclResult_t ncclGetPeerDevicePointer(ncclWindow_t window, size_t offset, int peer, void** outPtr);

////////////////////////////////////////////////////////////////////////////////
// Team API:
// ncclTeamRankToLsa(comm, tm, rank)：将 tm team 中的 rank 转换为 lsa rank
// ncclTeamRankToWorld(comm, tm, rank)：将 tm team 中的 rank 转换为全局 world rank
// ===== Team API：获取各种逻辑 rank 组 =====
// ncclTeamWorld：返回全局 team（包含所有 nRanks 个 rank，stride=1）
// ncclTeamLsa：返回 lsa team（本节点内部共享 NVLink Symmetric Memory 的一组 rank）
// ncclTeamRail：返回 rail team = ncclTeamOuterFactor(lsa, lsaSize)
//   即跨节点的同 rank（每节点取同一 lsa rank），stride=lsaSize
#if __cplusplus
NCCL_IR_EXTERN_C NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamWorld(ncclDevComm const&);
#endif
#ifndef __clang_llvm_bitcode_lib__
NCCL_EXTERN_C __host__ ncclTeam_t ncclTeamWorld(ncclComm_t);
#endif

#if __cplusplus
NCCL_IR_EXTERN_C NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamLsa(ncclDevComm const&);
#endif
#ifndef __clang_llvm_bitcode_lib__
NCCL_EXTERN_C __host__ ncclTeam_t ncclTeamLsa(ncclComm_t);
#endif

// ncclTeamRankIsMember(a, b, bPeer)：判断 b 中的 rank bPeer 是否也属于 a team
NCCL_EXTERN_C NCCL_HOST_DEVICE_INLINE bool ncclTeamRankIsMember(ncclTeam_t a, ncclTeam_t b, int bPeer);
// ncclTeamRankToTeam(a, b, bPeer)：将 b 中的 rank bPeer 转换为 a 中的 rank
NCCL_EXTERN_C NCCL_HOST_DEVICE_INLINE int ncclTeamRankToTeam(ncclTeam_t a, ncclTeam_t b, int bPeer);

#if __cplusplus
NCCL_IR_EXTERN_C NCCL_HOST_DEVICE_INLINE int ncclTeamRankToWorld(ncclDevComm const&, ncclTeam, int rank);
#endif
#ifndef __clang_llvm_bitcode_lib__
NCCL_EXTERN_C __host__ int ncclTeamRankToWorld(ncclComm_t, ncclTeam_t, int rank);
#endif

// ncclTeamRail：等价于 ncclTeamOuterFactor(lsaTeam, lsaSize)
// 返回跨节点维度的 rail team：每个节点取同一个 lsa rank，stride=lsaSize
#if __cplusplus
NCCL_IR_EXTERN_C NCCL_HOST_DEVICE_INLINE int ncclTeamRankToLsa(ncclDevComm const&, ncclTeam, int rank);
#endif
#ifndef __clang_llvm_bitcode_lib__
NCCL_EXTERN_C __host__ int ncclTeamRankToLsa(ncclComm_t, ncclTeam_t, int rank);
#endif

// ncclTeamInnerFactor(parent, innerSize)：取 parent 的内层子团（前 innerSize 个 rank）
// ncclTeamOuterFactor(parent, innerSize)：取 parent 的外层子团（每隔 innerSize 取一）
NCCL_EXTERN_C NCCL_HOST_DEVICE_INLINE ncclTeam_t ncclTeamInnerFactor(ncclTeam_t parent, int innerSize);
NCCL_EXTERN_C NCCL_HOST_DEVICE_INLINE ncclTeam_t ncclTeamOuterFactor(ncclTeam_t parent, int innerSize);

// Interpret each team as a set of ranks. This function assumes that `subset`
// is a subset of `parent`. Thus the number of ranks in the set difference of
// `parent` minus `subset` is `super.nRanks - subset.nRanks`. Given `index` this
// function returns the index'th element of `parent` minus `subset`.
// ===== ncclTeamRankInDifference：集合差中的第 index 个元素 =====
// 概念：将 parent team 寊为一个 rank 集合，寊 subset 也是其中一个子集。
// 该函数返回集合差（parent - subset）中第 index 个元素（rank 编号）。
// 算法要点：在排序的 parent rank 列表中，跳过 subset 中有的 rank，
// 按顺序取第 index 个剩下的。
// 用途：在均衡分配中，给除了 lsa 子集之外的其他 rank 分配任务。
NCCL_EXTERN_C NCCL_HOST_DEVICE_INLINE int ncclTeamRankInDifference(ncclTeam_t parent, ncclTeam_t subset, int index);

// Equivalent to ncclTeamOuterFactor of lsa team.
#if __cplusplus
NCCL_IR_EXTERN_C NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamRail(ncclDevComm const&);
#endif
#ifndef __clang_llvm_bitcode_lib__
NCCL_EXTERN_C __host__ ncclTeam_t ncclTeamRail(ncclComm_t);
#endif

// Get offset of resource buffer within `comm.resourceWindow`.
// ncclGetResourceBufferOffset：由 handle 计算在 resourceWindow 中的字节偏移
// 公式：offset = h * 128（每个 handle 对应 128 字节对齐的 slot）
NCCL_EXTERN_C NCCL_HOST_DEVICE_INLINE size_t ncclGetResourceBufferOffset(ncclDevResourceHandle_t);

// ===== Window API：Device 端寻址函数（仅在 __device__ 代码中可用）=====
//
// 所有函数内部用 add4G（修改 64 位指针的高 32 位）寻址不同 rank 的 buffer。
// 全部指针从 texture cache（__ldg）读取，避免占用 L1 d-cache。
//
// ncclGetLocalPointer(w, offset)：当前 rank 的 buffer 地址
//   = add4G(lsaFlatBase, lsaRank * stride4G) + offset
//
// ncclGetLsaPointer(w, offset, peer)：指定 lsa rank 的 buffer 地址
//   peer 是 0..lsaSize-1 范围内的 lsa rank 编号
//   = add4G(lsaFlatBase, peer * stride4G) + offset
//
// ncclGetPeerPointer(w, offset, peer)：指定全局 world rank 的 buffer 地址
//   peer 是全局 world rank 编号
//   内部计算：i = lsaRank + (peer - worldRank)
//
// ncclGetPeerPointer(w, offset, tm, peer)：指定 team tm 中 rank peer 的 buffer 地址
//   内部计算：i = lsaRank + (peer - tm.rank)*tm.stride
//
// ncclGetMultimemPointer(w, offset, mmHandle)：NVLink MultiCast 地址
//   ptr = mm.mcBasePtr + mcOffset4K * 4096 + offset
//   对应该 window 在 lsa MultiCast 空间中的位置。
//   写：广播到所有 lsa rank 的同一 offset；读：reduce 所有 lsa rank 的对应内容。
//
// ncclGetLsaMultimemPointer(w, offset, comm)：使用 comm.lsaMultimem 句柄的 MultiCast 地址
#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE ncclSymPtr<char> ncclGetResourceBuffer(ncclDevComm const&, ncclDevResourceHandle);
#endif

////////////////////////////////////////////////////////////////////////////////
// Window API:

#if NCCL_CHECK_CUDACC
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetLocalPointer(ncclWindow_t w, size_t offset);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetLsaPointer(ncclWindow_t w, size_t offset, int peer);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetPeerPointer(ncclWindow_t w, size_t offset, int peer);
NCCL_DEVICE_INLINE void* ncclGetPeerPointer(ncclWindow_t w, size_t offset, ncclTeam tm, int peer);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetMultimemPointer(ncclWindow_t w, size_t offset, ncclMultimemHandle mmHandle);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetLsaMultimemPointer(ncclWindow_t w, size_t offset, ncclDevComm const&);
#endif

#if NCCL_CHECK_CUDACC
// Convenience for combining ncclGet***Pointer() with resource handle.
// ===== ncclGetResourceBuffer*** 便捷函数 =====
// 将 ncclGetResourceBufferOffset + ncclGet***Pointer 合并为一步，
// 直接从 DevComm 的 resourceWindow 内置对象中根据 handle 寻址。
// 内置对象（comm.resourceWindow_inlined）不需要额外齐内存，读取更快。
//
// ncclGetResourceBufferLocalPointer(comm, h)：本 rank 的 h 句柄 slot
// ncclGetResourceBufferLsaPointer(comm, h, peer)： lsa rank peer 的 slot
// ncclGetResourceBufferPeerPointer(comm, h, team, peer)： team 中 rank peer 的 slot
// ncclGetResourceBufferMultimemPointer(comm, h, mm)：MultiCast 的 slot
// ncclGetResourceBufferLsaMultimemPointer(comm, h)：使用 lsaMultimem 的 slot
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetResourceBufferLocalPointer(ncclDevComm const&, ncclDevResourceHandle);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetResourceBufferLsaPointer(ncclDevComm const&, ncclDevResourceHandle, int peer);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetResourceBufferPeerPointer(ncclDevComm const&, ncclDevResourceHandle, ncclTeam, int peer);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetResourceBufferMultimemPointer(ncclDevComm const&, ncclDevResourceHandle, ncclMultimemHandle);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetResourceBufferLsaMultimemPointer(ncclDevComm const&, ncclDevResourceHandle);
#endif

#endif // _NCCL_DEVICE_CORE_H_
