/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_SYM_KERNELS_H_
#define NCCL_SYM_KERNELS_H_
#include "nccl.h"
#include "nccl_device.h"
#include "nccl_common.h"
#include "device.h"
#if !defined(NCCL_OS_WINDOWS)
#include "../device/symmetric/gin_scratch.h"
#else
#include "nccl_device/gin_win_stub.h"
#endif

////////////////////////////////////////////////////////////////////////////////
// ncclSymk[Foo]: Kernels built on the device API

//
// ============================================================================
// Symmetric Kernel 系统概述
// ============================================================================
// "Symmetric Kernel" 指通过 ncclDevCommCreate() 创建 Device-side comm，
// 在 GPU kernel 内直接调用 NCCL 通信原语（put/waitSignal 等）的一批 kernel。
//
// 与传统 NCCL kernel 的差异：
//   传统：host 上的 ncclAllReduce() 调用 → CPU 处理，最终发射 kernel
//   Symmetric：GPU kernel 本身调用 nccl 原语，不需要 CPU 中间层
//
// SymkKernelId 命名规则（为了理解这些 kernel）：
//   AllReduce_AGxLL_R： AllGather(算法=LL协议)居然后做 Reduce(居然R）
//   AllReduce_RSxLD_AGxST：RS(居然Load) + AG(居然Store) = 两阶段 AllReduce
//   尾缀 MC = MultiCast（使用 NVLS 多播内存），比普通 LD/ST 更高效
//   RailRing = 多节点坏速 Ring（每条 rail 内环形）
//   LsaXxx = LSA（Local Symmetric Area）维度的操作（节点内）
//   A2A = All-to-All布局的操作
//
// SymkDevComm： Kernel 登入后看到的最小上下文结构体，包含：
//   - devComm：普通 ncclDevComm（rank/nRanks/channels等）
//   - lsaLLA2A：节点内 LL A2A（Low Latency All-to-All）句柄
//   - ginOutbox： GIN 出站缓冲运管句柄（流水线信用管理）
//   - ginInboxRail： Rail 方向的 A2A 入站缓冲句柄
//   - ginSyncHandle： GIN 同步句柄（信号/计数器）
//
// SymkState： Host 侧保存的 Symmetric kernel 全局状态。
//   kcomm：和 kernel 共享的设备状态，在 ncclSymkInitOnce 时初始化
//
// SymkDevWork： 每次 symmetric collective 的工作元素（任务描述符）
//   inputWin/outputWin： 对称内存窗口，指向跨节点对称内存
//   sChannelId/nChannels：该工作使用的 channel 范围
////////////////////////////////////////////////////////////////////////////////
#define NCCL_SYM_KERNEL_CELL_SIZE 1024 // no less than 16 bytes minimal cell size

constexpr int ncclSymkMaxBlocks = 64;
constexpr int ncclSymkMaxThreads = 512;
constexpr int ncclSymkLLMaxEltSize = 8;

constexpr __host__ __device__ int ncclSymkLLMaxSlots(int eltSize = ncclSymkLLMaxEltSize) {
  return ncclSymkMaxThreads*ncclSymkLLMaxEltSize/eltSize;
}

// ============================================================================
// ncclSymkKernelId：Symmetric kernel 的种类标识符
// 命名規则：{集体类型}_{算法简称}
//   AG = AllGather,  RS = ReduceScatter
//   LL = Low Latency 协议， ST = Store， LD = Load
//   MC = MultiCast（使用 NVSwitch NVLS 多播）
//   R  = Reduce
//   RailRing = 每个节点内先内部融合，节点间再环形传输
//   LsaSTMC = LSA（节点内）维度用 Store MultiCast
//   RailA2A = Rail方向的 All-to-All
// ============================================================================
enum ncclSymkKernelId {
  ncclSymkKernelId_AllReduce_AGxLL_R,
  ncclSymkKernelId_AllReduce_AGxLLMC_R,
  ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST,
  ncclSymkKernelId_AllReduce_RSxLD_AGxST,
  ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC,

  ncclSymkKernelId_AllGather_LL,
  ncclSymkKernelId_AllGather_LLMC,
  ncclSymkKernelId_AllGather_TmaST,
  ncclSymkKernelId_AllGather_ST,
  ncclSymkKernelId_AllGather_TmaSTMC,
  ncclSymkKernelId_AllGather_STMC,
  ncclSymkKernelId_AllGather_RailRing_LsaSTMC,

  ncclSymkKernelId_ReduceScatter_LL,
  ncclSymkKernelId_ReduceScatter_TmaLD,
  ncclSymkKernelId_ReduceScatter_LD,
  ncclSymkKernelId_ReduceScatter_LDMC,
  ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD,
  ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC,

  ncclSymkKernelId_Count
};

// ============================================================================
// ncclSymkDevComm： Symmetric Kernel 看到的设备上下文，包含所有 kernel 需要的句柄
// 通过 ncclSymkDevWorkArgs 嵌入 kernel launch args 传入 kernel
// ============================================================================
struct ncclSymkDevComm {
  struct ncclDevComm devComm;
  struct ncclLLA2AHandle lsaLLA2A;
  struct ncclGinOutboxHandle ginOutbox;
  ncclGinCounter_t ginCounterPerBlock;
  struct ncclGinInboxA2AHandle ginInboxRail;
  struct ncclGinSyncHandle ginSyncHandle;
};

// ============================================================================
// ncclSymkState：Host 侧为每个 comm 维护的 Symmetric kernel 全局状态
// 存在于 comm->symkState，由 ncclSymkInitOnce() 初始化
//   initialized：是否已调用 ncclSymkInitOnce()
//   hasLsaMultimem：节点内是否支持 NVLS 多播（LSA 维度）
//   kcomm：和 kernel 共享的设备状态，工作时嵌入 kernel 参数
// ============================================================================
struct ncclSymkState {
  bool initialized;
  bool hasLsaMultimem;
  int maxGinInboxBlocks;
  struct ncclSymkDevComm kcomm;
};

struct ncclSymkChannelWorkRange {
  uint16_t workHi; // inclusive index of my ending work
  uint16_t fracHi; // 16-bit fraction in (0.0, 1.0] indicating where my part ends
};

// 16 bytes aligned
// ============================================================================
// ncclSymkDevWork：每次 Symmetric collective 调用的工作元素
//   16 字节对齐，嘉入 ncclSymkDevWorkArgs 后连续存储
//   inputWin/outputWin：指向对称内存窗口（跨节点 VMM 映射）
//   inputOff/outputOff：窗口内的偶同（包含 CBD分割偏移）
//   nElts：本次工作处理的元素数
// ============================================================================
struct alignas(16) ncclSymkDevWork {
  uint64_t redOpArg; // must be collectively uniform
  size_t nElts;
  struct ncclWindow_vidmem* inputWin, *outputWin;
  size_t inputOff, outputOff; // these = origUserOffset + cbdPartOffset
  int rootRank;
  uint64_t sChannelId:16, nChannels:16, padding:32;
};

struct alignas(16) ncclSymkDevWorkArgs {
  struct ncclSymkDevComm kcomm;
  int nMaxChannels;
  int maxDynamicSmem;
  // starting of channelWorkRange will be aligned to 16 bytes
  // channelWorkRange[nChannels];
  // ncclSymDevWork[nWorks];
  // aux functions
  __host__ static constexpr size_t calcArgsSize(int nChannels, int nWorks) {
    return alignUp(sizeof(struct ncclSymkDevWorkArgs), 16) + alignUp(nChannels * sizeof(struct ncclSymkChannelWorkRange), 16) + nWorks * sizeof(struct ncclSymkDevWork);
  }
  __host__ __device__ struct ncclSymkChannelWorkRange* getWorkRange() const {
    return (struct ncclSymkChannelWorkRange*)((uint8_t*)this + alignUp(sizeof(struct ncclSymkDevWorkArgs), 16));
  }
  __host__ __device__ struct ncclSymkDevWork* getWorks(int nChannels) const {
    return (struct ncclSymkDevWork*)((uint8_t*)this->getWorkRange() + alignUp(nChannels * sizeof(struct ncclSymkChannelWorkRange), 16));
  }
};

union ncclSymkDevWorkArgs4K {
  struct ncclSymkDevWorkArgs args;
  char buf4K[4096];
};

typedef enum {
  ncclSymSendNonregRecvNonreg = 0,
  ncclSymSendNonregRecvReg = 1,
  ncclSymSendRegRecvNonreg = 2,
  ncclSymSendRegRecvReg = 3,
  ncclNumSymRegTypes = 4
} ncclSymRegType_t;

// We assume ncclComm contains a field: `ncclSymkState symkState`
ncclResult_t ncclSymkInitOnce(struct ncclComm* comm);
ncclResult_t ncclSymkFinalize(struct ncclComm* comm);

bool ncclSymkAvailable(struct ncclComm* comm, ncclFunc_t coll, int/*ncclDevRedOp_t*/ red,
                       ncclDataType_t ty, size_t nElts);
ncclResult_t ncclSymkPickKernel(struct ncclComm* comm, ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                                size_t nEltsTotal, size_t nEltsMax, int nWorks, ncclSymRegType_t winRegType,
                                float* estTimeUs, ncclSymkKernelId* kernelId, int* nBlocks, int* nWarps, bool* forced);

ncclResult_t ncclSymkMakeDevWork(struct ncclComm* comm, struct ncclTaskColl* task, struct ncclSymkDevWork* outDevWork);

// Generated by src/device/symmetric/generate.py
extern int const ncclSymkKernelCount;
extern void* ncclSymkKernelList[/*ncclSymkKernelCount*/];
extern int ncclSymkKernelRequirements[/*ncclSymkKernelCount*/];
extern int ncclSymkKernelMaxDynamicSmem[/*ncclSymkKernelCount*/]; // initialized by ncclInitKernelsForDevice()
int ncclSymkGetKernelIndex(ncclSymkKernelId kernelId, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty);
const char* ncclSymkKernelIdToString(int kernelId);
ncclResult_t ncclGetSymRegType(struct ncclDevrWindow* sendWin, struct ncclDevrWindow* recvWin, ncclSymRegType_t* winRegType);

int ncclSymkLLKernelMask();
int ncclSymkDynamicSmemKernelMask();

constexpr int ncclSymkAllGather_RailRing_ChunkSize = 1<<20;
#endif
