/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
#ifndef GIN_PROXY_DEFS_H
#define GIN_PROXY_DEFS_H

#include <stdint.h>
#include <stddef.h>

#define NCCL_GIN_PROXY_VERSION 100
#define NCCL_GIN_PROXY_GFD_VERSION 1

// ---------------------------------------------------------------------------
// ncclGinProxyGfdQwordIdx_t：GFD 中各 qword 的位置索引
// ---------------------------------------------------------------------------
// 一个完整的 GFD 布局（2.30.3: 16 qwords = 128 字节）：
//   [qword 0] header      - version + 数据大小（2.30.3: op 已移到 qword[7] headerExt）
//   [qword 1] srcOff/inlineLow/vaSignalOff  - 源偏移 或 inline低位 或 VA信号偏移
//   [qword 2] srcHandle/inlineHigh/vaSignalHandle - 源句柄 或 inline高位 或 VA信号句柄
//   [qword 3] dstOff      - 目标地址偏移
//   [qword 4] dstHandle   - 目标窗口句柄
//   [qword 5] completion  - counterId + signalId + signalValLow
//   [qword 6] signalVal   - signalVal高位
//   [qword 7] headerExt   - op 类型（uint16_t，2.30.3 从 header 移到此处）
//   [qword 8..15]         - 保留扩展（flag 统一设为 1，CPU 通过最后一个 qword 的 flag 判断 GFD 完整性）
// ---------------------------------------------------------------------------
// ncclGinProxyOp_t：GFD（Go Forward Descriptor）的操作类型 bitmask
// ---------------------------------------------------------------------------
// 每个 GFD 的 headerExt qword（qword[7]）中有 16 bit 用于编码操作类型（op 字段）
// 下面的 bit 可以两两 OR 组合，形成" PUT + 信号 + 计数器"这样的复合操作
typedef enum {
  ncclGinProxyOpPut = 1 << 0,
  ncclGinProxyOpBaseMask = 1 << 0,
  ncclGinProxyOpWithInline = 1 << 1,
                                       // 适用于 putValue（最多 8 字节内嵌小值）
  ncclGinProxyOpWithCounter = 1 << 2,  // 完成后递增本地计数器 counter[counterId]<>++
  ncclGinProxyOpWithSignalInc = 1 << 3,  // 完成后递增远端信号 signal[signalId]++
  ncclGinProxyOpWithSignalAdd = 1 << 4,  // 完成后远端信号 signal[signalId] += signalVal
  ncclGinProxyOpVASignal = 1 << 5, // VA signals do not include put.
  ncclGinProxyOpGet = 1 << 6,
  ncclGinProxyOpFlush = 1 << 7,
                                       // 注意：VA signal + PUT 必须拆为两个 GFD，第一个只做 PUT，第二个只做 VA signal
} ncclGinProxyOp_t;

static_assert(sizeof(void *) == sizeof(uint64_t) && sizeof(size_t) == sizeof(uint64_t),
              "The proxy code is built on the assumption that the pointer size is 64 bits and at "
              "most 57 bits are used for the actual pointer.");

// ---------------------------------------------------------------------------
// ncclGinProxyQword_t：GFD 中单个 qword（64bit）的 union 表示
// ---------------------------------------------------------------------------
// GFD 由 16 个 qword 组成（共 128 字节）。
// 每个 qword 根据其在 GFD 中的位置（参见 ncclGinProxyGfdQwordIdx_t）有不同含义。
// 使用 union 是为了方便以不同字段解释同一 64bit 内存位置（公用 uint64_t raw）。
// 注意：这里的每个 struct 成员不是 GFD 的"第几个 qword"，
//       而是"如果某个 qword 要按此类型解释，应使用此 struct"
//       具体哪个 qword 用哪个 struct，由 ncclGinProxyGfdQwordIdx_t 枚举决定。
//
// 重要约束：
//   所有字段的 bit[0] 都是一个 "flag" 位，CPU 通过此位判断该 qword 是否已由 GPU 写入。
//   flag=1 表示已写，flag=0 表示未写（CPU 完成后复位为 0，以便下一轮检测）。
//
//   所有 offset/handle 字段占用 bit[63:1]（63 bit），因为 bit[0] 被 flag 占用。
//   GPU 写入时将实际值存入 bit[63:1]，CPU 读取时通过位域名直接访问（编译器自动移位）。
typedef union {
  uint64_t raw;  // 原始 64bit 访问（将整个 qword 读/写）
  struct {
    uint64_t v : 1;  // flag 位：=1 表示有效。用于 CPU 侧实现无锁 GFD 检测
    uint64_t resv : 63;
  } __attribute__((packed)) flag;
  // [qword 0] header：GFD 头部，包含协议版本 + 数据大小
  // op 字段在 2.30.3 中已从 header 移到 qword[7] headerExt
  struct {
    uint64_t flag : 1;  // =1 表示此 qword 已写入
    uint64_t version : 4;  // GFD 协议版本号（NCCL_GIN_PROXY_GFD_VERSION）
    uint64_t resv : 2;
    uint64_t size : 57;  // 本次 RDMA 操作的字节数
  } __attribute__((packed)) header;
  // [qword 1] srcOff：源地址偏移（当 op 无 WithInline 且无 VASignal 时使用）
  // 对应 IB Verbs 中 sge.addr = base_vas[rank] + srcOff
  struct {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag : 1;
    uint64_t srcOff : 63;  // 源窗口内的字节偏移
  } __attribute__((packed)) srcOff;
  // [qword 2] srcHandle：源窗口句柄（当 op 无 WithInline 且无 VASignal 时使用）
  // CPU proxy 通过此 handle 查找 MR，获取 IB Verbs lkey
  struct {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag : 1;
    uint64_t srcHandle : 63;
  } __attribute__((packed)) srcHandle;
  // [qword 1] vaSignalOff：VA 信号偏移（当 op 为 VASignal 时使用，与 srcOff 共用 qword[1]）
  // VA signal 模式：不含数据 PUT，仅将小值写入远端指定虚拟地址
  struct {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag : 1;
    uint64_t vaSignalOff : 63;  // 信号在远端窗口中的字节偏移
  } __attribute__((packed)) vaSignalOff;
  // [qword 2] vaSignalHandle：VA 信号窗口句柄（当 op 为 VASignal 时使用，与 srcHandle 共用 qword[2]）
  struct {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag : 1;
    uint64_t vaSignalHandle : 63;  // 信号窗口 handle（远端窗口句柄，rkey 来源）
  } __attribute__((packed)) vaSignalHandle;
  // [qword 1] inlineLow：inline 小值低位（当 op 含 WithInline 时使用，与 srcOff 共用 qword[1]）
  // 最大支持 96 bit / 12 字节内嵌小值（通过 qword[1] 和 qword[2] 拼接）：
  //   qword[1].inlineLow.inlineValLow   = bits [0:31]
  //   qword[1].inlineLow.inlineValLow2  = bits [32:47]
  //   qword[2].inlineHigh.inlineValHigh = bits [48:63]
  struct {
    uint8_t flag : 1;
    uint8_t resv : 7;
    uint32_t inlineValLow;   // 小值的 bits [0:31]
    uint16_t inlineValLow2;  // 小值的 bits [32:47]
  } __attribute__((packed)) inlineLow;
  // [qword 2] inlineHigh：inline 小值高位（当 op 含 WithInline 时使用，与 srcHandle 共用 qword[2]）
  // inline supports a max of 96 bit / 12 byte values
  struct {
    uint8_t flag : 1;
    uint8_t resv : 7;
    uint16_t inlineValHigh;  // 小值的 bits [48:63]
    uint8_t resv1;
    uint32_t resv2;
  } __attribute__((packed)) inlineHigh;
  struct {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag : 1;
    uint64_t dstOff : 63;
  } __attribute__((packed)) dstOff;
  struct {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag : 1;
    uint64_t dstHandle : 63;
  } __attribute__((packed)) dstHandle;
  struct {
    uint8_t flag : 1;
    // We need to keep the size of counterId and signalId in sync with the
    // NCCL_GIN_COUNTER_POOL_SIZE / NCCL_GIN_SIGNAL_POOL_SIZE upper limits
    // in gin_host.cc.
    // must be non-zero if WITH_COUNTER is set
    uint32_t counterId : 23;  // 计数器 ID，最大支持 2^23 个计数器
    // must be non-zero if WITH_SIGNAL_INC, WITH_SIGNAL_ADD, or WITH_SIGNAL_SET is set
    uint32_t signalId : 24;
    uint16_t signalValLow;  // 信号加数的低 16 bit
  } __attribute__((packed)) completion;
  struct {
    uint8_t flag : 1;
    uint8_t resv : 7;
    uint16_t signalValLow2;  // [16:31] bit
    uint32_t signalValHigh;  // [32:63] bit
  } __attribute__((packed)) signalVal;
  struct {
    uint8_t flag : 1;
    uint8_t resv : 7;
    uint16_t op;
    uint8_t resv2;
    uint32_t resv3;
  } __attribute__((packed)) headerExt;
} ncclGinProxyQword_t;
static_assert(sizeof(ncclGinProxyQword_t) == sizeof(uint64_t),
              "sizeof(ncclGinProxyQword_t) != sizeof(uint64_t)");
static_assert(NCCL_GIN_PROXY_GFD_VERSION < (1 << 4),
              "NCCL_GIN_PROXY_GFD_VERSION must be less than 2^4");

typedef enum {
  ncclGinProxyGfdHeader = 0,
  ncclGinProxyGfdInlineLow = 1,
  ncclGinProxyGfdInlineHigh = 2,
  ncclGinProxyGfdSrcOff = 1, // re-uses the inline word
  ncclGinProxyGfdSrcHandle = 2, // re-uses the inline word
  ncclGinProxyGfdVASignalOff = 1, // re-uses the inline word, VA signals with PUT must be split into two GFDs
  ncclGinProxyGfdVASignalHandle = 2, // re-uses the inline word, VA signals with PUT must be split into two GFDs
  ncclGinProxyGfdDstOff = 3,
  ncclGinProxyGfdDstHandle = 4,
  ncclGinProxyGfdCompletion = 5,
  ncclGinProxyGfdSignalVal = 6,
  ncclGinProxyGfdHeaderExt = 7,
  ncclGinProxyGfdQwords = 16,
} ncclGinProxyGfdQwordIdx_t;

// GFD 结构：128 字节对齐，由 16 个 ncclGinProxyQword_t 组成
// 为什么必须 128 字节对齐：
//   1. 与数据 cache line 对齐，避免 false sharing
//   2. GPU 使用 sizeof(GFD)/sizeof(uint4) 次 uint4 stwt（write-through）写入，需要对齐
//   3. CPU proxy 可以用 cache line 读取完成 GFD 解析
typedef struct __attribute__((packed)) {
  ncclGinProxyQword_t qword[ncclGinProxyGfdQwords];
} ncclGinProxyGfd_t;
static_assert(sizeof(ncclGinProxyGfd_t) == 128,
              "sizeof(ncclGinProxyGfd_t) != 128 - Backwards compat requires ncclGinProxyGfd to be 128 bytes!");

// ---------------------------------------------------------------------------
// ncclGinProxyGpuCtx_t：GIN_PROXY 后端在 GPU 上保存的上下文
// ---------------------------------------------------------------------------
// 每个 GIN context（对应一个 ncclGin 实例）对应一个 ncclGinProxyGpuCtx_t。
// GPU kernel 通过这个结构访问队列、生产者索引、消费者索引、信号和计数器。
typedef struct {
  int nranks;  // rail 内的 rank 数，即 GFD 队列的数量（每个 peer 一个队列）
  uint32_t queueSize;  // 单个队列容量（必须是 2 的幂次，用于取模运算）
                       // 队列满时，GPU 必须等待 CPU 消费才能继续写入
  ncclGinProxyGfd_t *queues;  // 指向每个 peer 的 GFD 环形队列
                       // 总大小 = nranks * queueSize * 128 字节
                       // queues[pe * queueSize ... (pe+1) * queueSize - 1] = 指向第 pe 个 peer 的队列
  uint32_t *pis;  // Producer Indices：GPU 维护的生产者索引（每个 peer 一个）
  // The consumer indices will reside in CPU or GPU memory depending on the availability of GDR
                       // GPU 每次 postGfd 时先对应 pis[pe] ++ 申请一个槽位
                       // pis[pe] 在 GPU GPU 可以并发递增（cuda::atomic fetch_add）
  uint32_t *cis;  // Consumer Indices：CPU proxy 维护的消费者索引（每个 peer 一个）

                       // 可以在 CPU 内存或如果小架支持 GDR（GPU Direct RDMA）就在 GPU 内存
                       // GDR 下 CPU 更新 cis 可以被 GPU 通过 NVLink 直接读取
  uint64_t *counters;  // 计数器数组：本节点这个 context 的全部计数器
                       // GPU 通过 signals 地址池访问（readCounter/resetCounter）
                       // CPU proxy 读取 GFD completion.counterId 并递增对应计数器
  uint64_t *signals;  // 信号数组：远端对 indexed 信号的读取和写入
                       // GPU 通过 readSignal/waitSignal 读取
                       // CPU proxy 读取 GFD completion.signalId 并操作远端内存中的信号
} ncclGinProxyGpuCtx_t;

#endif
