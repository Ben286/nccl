/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_SYMMETRIC_SCHED_H_
#define NCCL_SYMMETRIC_SCHED_H_

#include "device.h"
#include "nccl.h"
#include "scheduler.h"
#include <cuda_fp16.h>
#if defined(__CUDA_FP8_TYPES_EXIST__)
#include <cuda_fp8.h>
#endif

extern int64_t ncclParamSingleProcMemRegEnable();

NCCL_PARAM(SymNoWinEnable, "SYM_NOWIN_ENABLE", 0);

// symkRedOp: 将 host 侧的归约操作映射为 symk 内部的设备归约操作
// ncclAvg 在 symk 中实现为 Sum + PostDiv（先求和再乘以 1/nRanks）
// 其他操作直接使用原有的 device op
ncclDevRedOp_t symkRedOp(ncclRedOp_t redOp, ncclDevRedOp_t devRedOp) {
  if (redOp == ncclAvg) {
    return ncclDevSumPostDiv;
  }
  return devRedOp;
}

// convertCollTaskToSymmetricTask: 将传统 collective task 转换为 symmetric task
// 主要工作：调整归约操作的 scalarArg 以适配 symk 的 PostDiv 实现
// 对 ncclAvg:
//   - 通常的 kernel 在 device 端做 Sum 然后除以 nRanks
//   - symk 的 PostDiv 实际上是 *乘以* scalarArg，所以需要设置 scalarArg = 1.0/nRanks
//   - FP16/BF16: 累加器用 float，scalarArg 用 float 精度的 1/nRanks
//   - FP8: 累加器用 half，scalarArg 用 half 精度的 1/nRanks
//   - LDMC kernel 例外：累加器类型 = 数据类型，不需要重新打包标量
void convertCollTaskToSymmetricTask(struct ncclComm* comm,struct ncclTaskColl* task) {
  task->opDev.op = symkRedOp(task->opHost, task->opDev.op);
  if (task->opDev.op == ncclDevSumPostDiv) {
    // LDMC uses the same accumulator type as data type. Do not re-pack the scalar.
    if (task->devFuncId == (uint32_t)ncclSymkKernelId_ReduceScatter_LDMC) {
      return;
    }
    union {
      __half f16; float f32; uint64_t u64;
      void *ptr;
    };
    u64 = 0;
    switch (task->datatype) {
      // 16-bit floats use float accumulator
      case ncclFloat16:
#if defined(__CUDA_BF16_TYPES_EXIST__)
      case ncclBfloat16:
#endif
        f32 = float(1.0/comm->nRanks);  // ncclDevSumPostDiv actually multiplies by the scalar, not divides.
        task->opDev.scalarArg = u64;
        return;
#if defined(__CUDA_FP8_TYPES_EXIST__)
      case ncclFloat8e4m3:
      case ncclFloat8e5m2:
        f16 = __float2half(float(1.0/comm->nRanks));
        task->opDev.scalarArg = u64;
        return;
#endif
      default:
        break;
      }
  }
}

// ════════════════════════════════════════════════════════════════════════════════
// ncclMakeSymmetricTaskList — Symmetric Kernel 任务分流 & 内核选择
// ════════════════════════════════════════════════════════════════════════════════
// 调用时机：ncclPrepareTasks() 在处理传统 collective 之前调用本函数
// 输入：task — 所有待处理 collective 任务组成的单链表
// 输出：symTaskQueue — 适合 symk 的任务入此队列，后续由 ncclSymmetricTaskScheduler 处理
//       remainTasksHead — 不适合 symk 的任务返回此链表，继续走传统 algo/proto 路径
//
// 整体流程分 3 个阶段：
//   阶段 1（L78-106）: 遍历所有任务，按 ncclSymkAvailable() 判断是否可用 symk
//     可用的按 (func, op, type, regType) 四维 key 分桶存入 tasksSymByFnOpTy[]
//     不可用的追加到 remainTasks 链表
//   阶段 2（L112-140）: 对每个桶内的任务组，尝试打包多个 work 到一次 kernel launch
//     调用 ncclSymkPickKernel() 选择最优 kernel（考虑数据量、SM 占用、TMA 支持等）
//   阶段 3（L141-213）: 对选定的 kernel 执行 fallback 检查（LL 内核限制、sysmem 段限制）
//     通过检查的任务转换为 symmetric task 入队；未通过的回退到 remain 链表
//
// 关键概念：
//   Window（对称内存窗口）: 用户通过 ncclMemAlloc/ncclCommRegister 注册的内存区域
//     每个 window 在所有 rank 上有对应的 vidmem 映射，symk 可直接跨 GPU 读写
//   RegType: send/recv 缓冲区是否在对称窗口内，共 4 种组合
//     ncclSymSendRegRecvReg = 3（最理想，所有 symk 都可用）
//     其他组合只能用 LL 内核（通过 bounce buffer 传输）
//   LL 内核: 低延迟内核，用 128B 原子操作 + flag 实现同步
//     不需要 TMA/LDST 对齐要求，但吞吐有限
// ════════════════════════════════════════════════════════════════════════════════
ncclResult_t ncclMakeSymmetricTaskList(struct ncclComm* comm, struct ncclTaskColl* task, struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* symTaskQueue, struct ncclTaskColl** remainTasksHead) {
  ncclResult_t ret = ncclSuccess;
  int fnOpTySymCount = 0;
  // tasksSymByFnOpTy: 按 (func, redOp, datatype, symRegType) 四维组合索引的分桶数组
  // 每个桶是一个单链表头，用 task->next 串联同一桶的所有任务
  // 数组大小 = ncclNumFuncs * ncclNumDevRedOps * ncclNumTypes * ncclNumSymRegTypes
  struct ncclTaskColl* tasksSymByFnOpTy[ncclNumFuncs * ncclNumDevRedOps * ncclNumTypes * ncclNumSymRegTypes];
  // fnOpTySymIndices: 记录哪些桶被使用了，避免遍历整个稀疏数组
  int fnOpTySymIndices[ncclNumFuncs * ncclNumDevRedOps * ncclNumTypes * ncclNumSymRegTypes];
  struct ncclKernelPlanner* planner = &comm->planner;
  struct ncclTaskColl* remainTasksTail = nullptr;
  bool foundSymm = false;

  memset(tasksSymByFnOpTy, 0, sizeof(tasksSymByFnOpTy));
  *remainTasksHead = nullptr;
  if (task) {
    // 首次使用 symmetric 功能时初始化 Device Runtime（DevR）
    // 包括：对称内存窗口排序数组、vidmem 映射等
    NCCLCHECK(ncclDevrInitOnce(comm));
  }
  // ── 阶段 1: 任务分类 ──────────────────────────────────────────────────────
  // 遍历所有 collective 任务，将可用 symk 的分桶到 tasksSymByFnOpTy[]，
  // 不可用的追加到 remainTasks 链表
  while (task != nullptr) {
    int index;
    struct ncclTaskColl* next = task->next;
    // symkRedOp: 将 ncclAvg 映射为 ncclDevSumPostDiv（symk 内部实现 Avg = Sum + PostDiv）
    ncclDevRedOp_t symkOp = symkRedOp(task->opHost, task->opDev.op);
    // ncclSymkAvailable 检查三个条件：
    //   (1) comm->isAllDirectNvlink: 所有 GPU 间必须通过 All-Direct NVLink 连接
    //   (2) ncclSymkImplemented(func, op, ty): 该 (集体操作, 归约, 数据类型) 组合有 symk 实现
    //   (3) ncclSymkMask != 0: 该数据量范围有适用的 kernel
    bool symAvailable = ncclSymkAvailable(comm, task->func, symkOp, task->datatype, task->count);

    if (symAvailable) {
      // 在 DevR 排序窗口数组中二分查找 sendbuff/recvbuff 所属的对称内存窗口
      // 如果用户缓冲区在 ncclMemAlloc 分配的区域内，findWindow 返回非空
      NCCLCHECK(ncclDevrFindWindow(comm, task->sendbuff, &task->sendWin));
      NCCLCHECK(ncclDevrFindWindow(comm, task->recvbuff, &task->recvWin));
      // 根据 send/recv 窗口是否带 NCCL_WIN_COLL_SYMMETRIC 标志，确定注册类型：
      //   0: SendNonreg+RecvNonreg  1: SendNonreg+RecvReg
      //   2: SendReg+RecvNonreg     3: SendReg+RecvReg（最优，所有 kernel 可用）
      NCCLCHECK(ncclGetSymRegType(task->sendWin, task->recvWin, &task->winRegType));

      // 计算四维哈希索引 = func * DevRedOps * Types * RegTypes + op * Types * RegTypes + ty * RegTypes + regType
      // 这样相同 (func, op, type, regType) 的任务会落入同一个桶
      index = (((int)task->func * ncclNumDevRedOps + symkOp) * ncclNumTypes + (int)task->datatype) * ncclNumSymRegTypes + (int)task->winRegType;
      // 如果是这个桶的第一个任务，记录索引到 fnOpTySymIndices[] 以便后续快速遍历
      if (tasksSymByFnOpTy[index] == nullptr) fnOpTySymIndices[fnOpTySymCount++] = index;
      // 头插法将任务插入对应桶的链表
      task->next = tasksSymByFnOpTy[index];
      tasksSymByFnOpTy[index] = task;
      // 从传统 collective 计数中移除（该任务已被 symk 接管）
      planner->nTasksColl--;
      foundSymm = true;
    } else {
      // 不可用 symk 的任务追加到 remain 链表尾部
      if (*remainTasksHead) {
        remainTasksTail->next = task;
        remainTasksTail = task;
      } else {
        *remainTasksHead = remainTasksTail = task;
      }
    }
    task = next;
  }
  if (remainTasksTail) remainTasksTail->next = nullptr;
  // 如果没有任何任务适合 symk，直接返回（所有任务都在 remain 链表中）
  if (!foundSymm) goto exit;

  // 断言：kernel args 空间至少能容纳 1 个 work 单元
  // calcArgsSize(MAXCHANNELS, 1) = sizeof(ncclSymkDevWorkArgs) + MAXCHANNELS*sizeof(ChannelWorkRange) + 1*sizeof(ncclSymkDevWork)
  assert(comm->workArgsBytes >= ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 1));

  // ── 阶段 2+3: 内核选择 + Fallback 检查 + 任务入队 ─────────────────────────
  // 遍历每个非空桶，同一桶内的任务共享 (func, op, type, regType)，
  // 尝试将尽可能多的 work 打包到一次 kernel launch 中
  for (int cursor = 0; cursor < fnOpTySymCount; cursor++) {
    struct ncclTaskColl* task = tasksSymByFnOpTy[fnOpTySymIndices[cursor]];
    // 外层循环：同一桶可能有太多任务无法一次 launch，所以分批处理
    while (task != NULL) {
      ncclSymkKernelId kernelId = ncclSymkKernelId_Count; // 初始化为无效 ID
      int nChannels = MAXCHANNELS;
      int nWarps = 0;
      int nWorks = 0;              // 本批次打包的 work 数量
      float estTimeUs = 1.e18;     // 估计执行时间（微秒），初始化为极大值
      size_t countTotal = 0, countMax = 0;  // 总元素数 和 单个 work 的最大元素数
      struct ncclTaskColl* headTask = task;  // 记录本批次的头任务
      // cellCount = 1024 / sizeof(datatype)，即 NCCL_SYM_KERNEL_CELL_SIZE 对应的元素个数
      // symk 以 1024 字节为最小工作粒度，所有 count 都向上对齐到 cellCount
      size_t cellCount = NCCL_SYM_KERNEL_CELL_SIZE / ncclTypeSize(headTask->datatype);
      bool forced = false; // 是否由用户通过 NCCL_SYM_KERNEL 环境变量强制指定了内核
      ncclDevRedOp_t symkOp = symkRedOp(task->opHost, task->opDev.op);
      // ── 内层循环：贪心打包 works ──
      // 尽可能多地将同桶任务打包为一个 batch，直到：
      //   (1) kernel args 空间不够容纳 nWorks+1 个 work，或
      //   (2) 到达链表末尾
      // 设计原则：更高 kernel ID 倾向于处理更大数据量
      while (task != nullptr) {
        size_t count;
        nWorks++;
        // 将 count 向上对齐到 cellCount（1024B 粒度）
        count = alignUp(task->count, cellCount);
        countTotal += count;
        if (count > countMax) countMax = count;
        // calcArgsSize 计算 nWorks+1 个 work 时的 args 总大小
        // 如果再加一个 work 就超出 workArgsBytes 限制，或已到链表末尾，
        // 则标记当前 task 为 isSymLast（本批次最后一个），终止打包
        if (ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, nWorks + 1) > comm->workArgsBytes || task->next == nullptr) {
          task->isSymLast = 1;
          break;
        }
        task = task->next;
      }
      // ── ncclSymkPickKernel: 从可用内核中选择最优 ──
      // 输入：func, op, datatype, countTotal, countMax, nWorks, winRegType
      // 输出：kernelId, nChannels(SM blocks), nWarps(固定=16), estTimeUs, forced
      // 选择过程：
      //   1) 根据 (func, op, ty, count) 获取初始 kernel mask
      //   2) 根据 nWorks>1 排除 LL 内核（LL 不支持分组）
      //   3) 根据 winRegType 进一步过滤：
      //      - AllReduce: 非双注册只能用 LL
      //      - AllGather: 非双注册/RecvReg 只能用 LL；多节点排除 GIN 内核
      //      - ReduceScatter: 非双注册/SendReg 只能用 LL
      //   4) 对剩余 kernel 逐个调用 queryModel 获取 (estTime, nBlocks)
      //      用 time * (1 + 0.025 * nBlocks) 做 SM 惩罚后取最优
      NCCLCHECK(ncclSymkPickKernel(comm, headTask->func, symkOp, headTask->datatype,
                                   countTotal, countMax, nWorks, headTask->winRegType,
                                   &estTimeUs, &kernelId, &nChannels, &nWarps, &forced));
      task = headTask; // 重置 task 指针到本批次头部，后续遍历用
      // 判断选出的 kernel 是否为 LL 类型（低延迟协议）
      // ncclSymkLLKernelMask() 返回所有 LL 内核的位掩码：
      //   AllReduce_AGxLL_R, AllReduce_AGxLLMC_R, AllGather_LL, AllGather_LLMC, ReduceScatter_LL
      bool isLLKernel = (1 << kernelId) & ncclSymkLLKernelMask();
      // 单进程管理多 GPU 且未启用 SingleProcMemReg 时，LL 内核对非注册内存有限制
      bool isOneThreadMultiGpus = comm->intraRanks > 1 && !ncclParamSingleProcMemRegEnable();
      bool needFallback = false;

      // ── 阶段 3: LL 内核的 Fallback 决策逻辑 ──
      // 这是 symk 调度中最复杂的决策点，决定是否回退到传统 kernel
      //
      // Fallback logic for symmetric LL kernels:
      // - If both src and dst are registered, we don't fall back if a symmetric kernel is available.
      // - Otherwise, we have to fall back to a legacy kernel if running the selected symmetric LL kernel is
      //   not possible (if the buffers are not registered and we manage multiple GPUs).
      // - If the user forced a symmetric kernel via NCCL_SYM_KERNEL or requested preference for using
      //   symmetric kernels even without symmetric buffers via NCCL_SYM_NOWIN_ENABLE, we respect that.
      // - Otherwise, we query the legacy cost model and if it selects a non-LL proto, we pick that.
      //
      // 判断层次（优先级从高到低）：
      //   ① SendReg+RecvReg 或 ALGO_UNDEF → 不回退（最理想情况，所有 kernel 可用）
      //   ② 非 LL 内核 → 不需要此处回退（非 LL 要求双注册，前面 mask 已过滤）
      //   ③ LL 内核 + 单进程多 GPU + 双非注册 → 必须回退（无法安全访问跨 GPU 内存）
      //   ④ LL 内核 + 用户强制 → 不回退（尊重用户选择）
      //   ⑤ LL 内核 + 双非注册 + 未启用 SYM_NOWIN_ENABLE → 回退（默认保守策略）
      //   ⑥ 查询传统 tuner，如果传统路径选择的不是 LL 协议 → 回退（传统路径可能更优）
      if (headTask->winRegType == ncclSymSendRegRecvReg || headTask->algorithm == NCCL_ALGO_UNDEF) {
        needFallback = false; // ① 双注册 或 首次路由（UNDEF）→ 放行
      } else if (isLLKernel) {
        // ③ 单进程管理多 GPU + 双非注册 → 必须回退
        // 原因：CUDA IPC 限制使得非注册缓冲区无法在 LL 模式下安全跨 GPU 访问
        needFallback = isOneThreadMultiGpus && headTask->winRegType == ncclSymSendNonregRecvNonreg;
        if (!needFallback && !forced) {
          // ⑤ 非强制 + 双非注册 + 未启用 NCCL_SYM_NOWIN_ENABLE → 回退
          needFallback = !ncclParamSymNoWinEnable() && headTask->winRegType == ncclSymSendNonregRecvNonreg;
          if (!needFallback) {
            // ⑥ 最后一道检查：查询传统 tuner（ncclGetAlgoInfo）
            // 如果传统路径选择了非 LL 协议（如 Simple/LL128），说明数据量较大，
            // 传统大消息 kernel 可能比 symk LL 更优，回退
            int collNetSupport = 0;
            int nvlsSupport = comm->nvlsSupport && (ncclNvlsSupported(task->opDev.op, headTask->datatype) || headTask->func == ncclFuncAllGather);
            NCCLCHECK(ncclGetCollNetSupport(comm, headTask, &collNetSupport));
            NOWARN(ncclGetAlgoInfo(comm, headTask, collNetSupport, nvlsSupport, 1), NCCL_COLL);
            // 如果传统 tuner 选了 LL 以外的协议，说明这个消息大小不适合 LL
            needFallback = (headTask->protocol != NCCL_PROTO_LL);
          }
        }
      }

      // ── 额外 Fallback: sysmem 段检查 ──
      // 即使缓冲区在注册窗口内，如果该窗口的虚拟地址范围包含 sysmem（系统内存）段，
      // symk 也无法正确工作（symk 要求所有段都在 vidmem 中），必须回退
      // 注意：window==NULL 时函数返回 false，所以非注册情况也被正确处理
      // Override needFallback when buffers are registered but VAs contain sysmem segments.
      // The below functions return false when the window is NULL, so this covers non-reg cases as well.
      if (!needFallback) {
        bool hasSysmemSegment = ncclDevrWindowHasSysmemSegment(headTask->sendWin) || ncclDevrWindowHasSysmemSegment(headTask->recvWin);
        needFallback = hasSysmemSegment;
      }

      // ── Fallback 执行：将本批次任务退回 remain 链表 ──
      if (kernelId == ncclSymkKernelId_Count || needFallback) {
        // kernelId == Count 表示 PickKernel 没找到任何可用 kernel（mask 为空）
        // needFallback 表示虽然选出了 kernel 但不满足运行条件
        // 两种情况都需要回退到传统 kernel 路径
        // cannot find appropriate symmetric kernel for the tasks
        // fallback to legacy kernels
        while (task != nullptr) {
          struct ncclTaskColl* next = task->next;
          int isSymLast = task->isSymLast;
          // 将任务追加回 remain 链表尾部
          if (*remainTasksHead) {
            remainTasksTail->next = task;
            remainTasksTail = task;
          } else {
            *remainTasksHead = remainTasksTail = task;
          }
          // 恢复传统 collective 计数（阶段 1 中减少了）
          planner->nTasksColl++;
          task = next;
          if (isSymLast) break; // isSymLast 标记了本批次的边界
        }
        continue; // 继续处理同一桶的下一批任务
      }

      // ── LL 内核特殊初始化 ──
      // 非注册内存上的 LL 内核需要额外的 bounce buffer 等资源
      // ncclSymkInitOnce 只执行一次，分配 LL 所需的同步标志和临时缓冲区
      // initialize symmetric objects for LL kernels
      if (isLLKernel && headTask->winRegType == ncclSymSendNonregRecvNonreg) {
        NCCLCHECK(ncclSymkInitOnce(comm));
      }

      // ── 最终入队：将本批次所有任务标记为选定 kernel 并入 symTaskQueue ──
      // set all symmetric tasks to the same kernel
      while (task != nullptr) {
        struct ncclTaskColl* next = task->next;
        int isSymLast = task->isSymLast;
        // 将 PickKernel 的结果写入每个 task
        task->devFuncId = (uint32_t)kernelId;   // 选定的 symmetric kernel ID
        task->nMaxChannels = nChannels;          // 该 kernel 使用的 SM block 数
        task->nWarps = nWarps;                   // 固定为 16 warps/block = 512 threads
        // convertCollTaskToSymmetricTask: 转换归约操作
        //   - ncclAvg → ncclDevSumPostDiv + 设置 scalarArg = 1.0/nRanks
        //   - FP16/BF16 用 float 精度的 1/nRanks，FP8 用 half 精度
        //   - LDMC kernel 例外：累加器与数据类型相同，不需要重打包标量
        convertCollTaskToSymmetricTask(comm, task);
        // 入队到 planner 的 symTask 队列
        // 后续由 ncclSymmetricTaskScheduler（在 ncclLaunchPrepare 中调用）处理
        ncclIntruQueueEnqueue(&planner->collSymTaskQueue, task);
        task = next;
        if (isSymLast) break; // 本批次结束，外层 while 继续处理同桶的下一批
      }
    }
  }

exit:
  return ret;
}

ncclResult_t ncclSymmetricTaskScheduler(struct ncclComm* comm, struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* symTaskQueue, struct ncclKernelPlan* plan) {
  struct ncclTaskColl* headTask = ncclIntruQueueHead(symTaskQueue);
  int devFuncId = headTask->devFuncId;
  struct ncclTaskColl* task = NULL;
  ssize_t totalCount = 0;  // aligned bytes
  ssize_t logCount = 0;
  ssize_t remainCell = 0;
  ssize_t cellPerChannel = 0;
  int workCount = 0, workIndex = 0;
  size_t cellCount = NCCL_SYM_KERNEL_CELL_SIZE / ncclTypeSize(headTask->datatype); // minimal cell size
  ncclResult_t ret = ncclSuccess;
  int curChannel = 0;
  int curChannelWork = 0;
  int nMaxChannels = headTask->nMaxChannels;
  struct ncclSymkDevWork* workBufPtr = NULL;
  struct ncclSymkChannelWorkRange* workRangePtr = NULL;
  const char* funcName = ncclFuncToString(headTask->func);
  const char* kernelName = ncclSymkKernelIdToString(headTask->devFuncId);
  struct ncclSymkDevWorkArgs* argsBuf = NULL;

  plan->isSymColl = true;
  plan->threadPerBlock = headTask->nWarps * WARP_SIZE;
  plan->hasProxyOps = false;
  ncclSymkKernelId kernelId = (ncclSymkKernelId)headTask->devFuncId;
  int kernelIndex = ncclSymkGetKernelIndex(kernelId, headTask->opDev.op, headTask->datatype);
  plan->kernelFn = ncclSymkKernelList[kernelIndex];
  int maxDynamicSmem = ncclSymkKernelMaxDynamicSmem[kernelIndex];
  plan->kernelDynSmem = (1 & ncclSymkDynamicSmemKernelMask()>>(int)kernelId) ? maxDynamicSmem : 0;
  task = headTask;
  while (task != nullptr && task->devFuncId == devFuncId) {
    workCount++;
    totalCount += alignUp(task->count, cellCount);
    logCount += task->count;
    if (task->isSymLast == 1) break;
    task = task->next;
  }

  plan->kernelArgsSize = ncclSymkDevWorkArgs::calcArgsSize(nMaxChannels, workCount);
  argsBuf = (struct ncclSymkDevWorkArgs*)calloc(1, plan->kernelArgsSize);

  remainCell = cellPerChannel = DIVUP(DIVUP(totalCount, nMaxChannels), cellCount);
  workRangePtr = argsBuf->getWorkRange();
  workBufPtr = argsBuf->getWorks(nMaxChannels);
  argsBuf->nMaxChannels = nMaxChannels;
  argsBuf->maxDynamicSmem = maxDynamicSmem;

  while (!ncclIntruQueueEmpty(symTaskQueue)) {
    struct ncclSymkDevWork devWork = {};
    size_t cellLeft = 0, taskCell = 0;
    uint8_t isSymLast = 0;

    if (ncclIntruQueueHead(symTaskQueue)->devFuncId != devFuncId) break; // scheduling is done

    task = ncclIntruQueueDequeue(symTaskQueue);
    isSymLast = task->isSymLast;

    NCCLCHECKGOTO(ncclSymkMakeDevWork(comm, task, &devWork), ret, fail);

    cellLeft = taskCell = DIVUP(task->count, cellCount);
    for (;curChannel < nMaxChannels;) {
      workRangePtr[curChannel].workHi = workIndex;
      if (curChannelWork == 0) {
        if (devWork.nChannels == 0) {
          devWork.sChannelId = curChannel;
          devWork.nChannels = 1;
        } else if (cellLeft <= remainCell) {
          // the last segment of the task
          assert(devWork.nChannels > 0);
          // if the remaining cell is less than 1024 bytes, we can fuse the last channel
          if ((remainCell - cellLeft) * NCCL_SYM_KERNEL_CELL_SIZE <= (1 << 10) || ncclIntruQueueEmpty(symTaskQueue)) devWork.nChannels++;
        } else {
          // middle segment of the task
          devWork.nChannels++;
        }
      } else {
        assert(cellLeft == taskCell);
        if (taskCell <= remainCell) {
          // the first segment of the task is fully scheduled onto the channel
          devWork.sChannelId = curChannel;
          devWork.nChannels = 1;
        }
      }
      if (cellLeft < remainCell) {
        workRangePtr[curChannel].fracHi = uint16_t(0x10000UL - 1);
        remainCell -= cellLeft;
        curChannelWork++;
        break;
      } else if (cellLeft == remainCell) {
        workRangePtr[curChannel].fracHi = uint16_t(0x10000UL - 1);
        remainCell = cellPerChannel;
        curChannel++;
        curChannelWork = 0;
        break;
      } else {
        // cellLeft > remainCell; the task is partially scheduled onto the channel
        cellLeft -= remainCell;
        workRangePtr[curChannel].fracHi = uint16_t(DIVUP(0x10000L * (taskCell - cellLeft), taskCell) - 1);
        remainCell = cellPerChannel;
        curChannel++;
        curChannelWork = 0;
      }
    }
    memcpy(workBufPtr + workIndex, &devWork, sizeof(struct ncclSymkDevWork));
    workIndex++;

    // Profiler
    plan->groupApiEventHandle = task->groupApiEventHandle;

    ncclMemoryPoolFree<struct ncclTaskColl>(&comm->memPool_ncclTaskColl, task);
    if (isSymLast == 1) break;
    if (curChannel == nMaxChannels) {
      WARN("ncclSymmetricTaskScheduler ran out of channel space (nMaxChannels=%d, workCount=%d, workIndex=%d)",
           nMaxChannels, workCount, workIndex);
      goto fail;
    }
  }
  if (remainCell < cellPerChannel) curChannel++;

  memcpy(&argsBuf->kcomm, &comm->symkState.kcomm, sizeof(comm->symkState.kcomm));
  plan->workBytes = totalCount * ncclTypeSize(headTask->datatype);
  plan->channelMask = uint64_t(-1) >> (64 - curChannel);
  plan->kernelSymArgs = (void*)argsBuf;
  plan->workStorageType = ncclDevWorkStorageTypeArgs;

  if (comm->rank == 0) {
    INFO(NCCL_TUNING, "%s [Symmetric]: %ld Bytes -> Kernel %s nchannels %d nthreads %d nWorks %d", funcName,
         logCount * ncclTypeSize(headTask->datatype), kernelName, curChannel, plan->threadPerBlock, workCount);
  }

exit:
  return ret;
fail:
  goto exit;
}
#endif // NCCL_SYMMETRIC_SCHED_H_

