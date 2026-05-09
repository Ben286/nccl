/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// ===== core__funcs.h：Team API 和 Window API 的内联实现 =====
// 本文件包含所有 ncclTeam*/ncclGet***Pointer 函数的 __host__/__device__ inline 实现。
// 所有寻址函数的关键技术：
//   - add4G(base, delta)：修改 64 位指针高 32 位，实现 Symmetric Memory 寻址
//   - loadConst：通过 __ldg（texture cache）读取常量内容，避免占用 L1 d-cache
//   - reinterpret_cast<char(*)[N]>(ptr) + k：计算 ptr + k*N 字节偏移

#ifndef _NCCL_DEVICE_CORE__FUNCS_H_
#define _NCCL_DEVICE_CORE__FUNCS_H_
#include "core__types.h"
#include "comm__types.h"
#include "ptr__types.h"

#if __cplusplus
// ncclTeamWorld：返回全局 team
// 全局 team 包含所有 nRanks 个 rank，本 rank 编号为 comm.rank，stride=1
NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamWorld(ncclDevComm const &comm) {
  ncclTeam ans;
  ans.nRanks = comm.nRanks;  // 全局 rank 总数
  ans.rank = comm.rank;  // 本 rank 在全局中的编号
  ans.stride = 1;  // lsa team 内连续编号
  return ans;
}
#endif

#if __cplusplus
// ncclTeamLsa：返回 lsa team
// lsa（Local Symmetric Accessible）team 是本节点内部开启 NVLink Symmetric Memory 的一组 rank
// 本 rank 编号为 comm.lsaRank，stride=1（节点内部按顺序编号）
NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamLsa(ncclDevComm const &comm) {
  ncclTeam ans;
  ans.nRanks = comm.lsaSize;  // lsa team 的 rank 数（= 节点内 GPU 数）
  ans.rank = comm.lsaRank;  // 本 rank 在 lsa team 中的编号
  ans.stride = 1;
  return ans;
}
#endif

#if __cplusplus
// ncclTeamRail：返回 rail team（跨节点维度）
// rail team = ncclTeamOuterFactor(lsa, lsaSize)：每个节点取相同 lsa rank 的一组 rank
// 例：8 个全局 rank，lsaSize=4：rail team 有 2 个 rank，stride=4
//   全局 rank 0,4 属于同一 rail team（lsaRank=0）；rank 1,5 属于另一个
// 关键计算：用 idivFast32 替代局 % / 除法操作，避免 GPU 上高延迟除法指令
NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamRail(ncclDevComm const& comm) {
  ncclTeam ans;
  // idivFast32(n, d, rcp)：n/d 的快速整数除法（用预计算倒数 rcp 替代除法指令）
  // comm.lsaSize_rcp32 是 comm 初始化时预计算的 lsaSize 倒数
  ans.nRanks = nccl::utility::idivFast32(comm.nRanks, comm.lsaSize, comm.lsaSize_rcp32);  // 整个 rail 维度有多少 rank
  ans.rank = nccl::utility::idivFast32(comm.rank, comm.lsaSize, comm.lsaSize_rcp32);  // 本 rank 在 rail team 中的编号 = rank / lsaSize
  ans.stride = comm.lsaSize;  // stride = lsaSize（相邻 rail rank 的全局 rank 相差 lsaSize）
  return ans;
}
#endif

// ncclTeamRankIsMember(a, b, brank)：判断 b team 中的 rank brank 是否在 a team 中
// 算法：
//   1. 将 b team 中的 brank 转换为全局 world 偏移：wrank = (brank - b.rank)*b.stride
//   2. 尝试映射到 a team：adelta = wrank/a.stride，amod = wrank%a.stride
//   3. 如果 amod==0 且 0<=arank<a.nRanks，则属于 a team
NCCL_HOST_DEVICE_INLINE bool ncclTeamRankIsMember(ncclTeam_t a, ncclTeam_t b, int brank) {
  // 将 b 中的 brank 转换为相对 a.rank 的全局偏移
  // 将 b 中的 brank 转换为相对 a.rank 的全局偏移
  int wrank = (brank - b.rank)*b.stride;
  // a 中的相对步数和余数
  uint32_t adelta = wrank/a.stride;  // 不检查 amod（调用者保证 amod==0）
  uint32_t amod = wrank%a.stride;  // 必须为 0 才能属于 a team
  int arank = a.rank + adelta;  // a team 中的 rank 编号
  return 0 <= arank && arank < a.nRanks && amod == 0;
}

// ncclTeamRankToTeam(a, b, brank)：将 b team 中的 brank 转换为 a team 中的 rank 编号
// 前提：brank 必须是 a team 的成员（调用方负责确保）
// 公式：arank = a.rank + (brank - b.rank)*b.stride / a.stride
NCCL_HOST_DEVICE_INLINE int ncclTeamRankToTeam(ncclTeam_t a, ncclTeam_t b, int brank) {
  int wrank = (brank - b.rank)*b.stride;
  uint32_t adelta = wrank/a.stride;
  //uint32_t amod = wrank%a.stride;
  int arank = a.rank + adelta;
  return arank;
}

#if __cplusplus
// ncclTeamRankToWorld(comm, tm, rank)：将 tm team 中的 rank 转换为全局 world rank
// 公式：worldRank = comm.rank + (rank - tm.rank)*tm.stride
// 解释：comm.rank 是本 rank 的全局编号；
//   (rank - tm.rank) 是在 tm team 中的相对差；
//   乘以 stride 得到全局差；将全局差加到 comm.rank 得到全局 rank
NCCL_HOST_DEVICE_INLINE int ncclTeamRankToWorld(ncclDevComm const& comm, ncclTeam tm, int rank) {
  return comm.rank + (rank - tm.rank)*tm.stride;
}
#endif

#if __cplusplus
// ncclTeamRankToLsa(comm, tm, rank)：将 tm team 中的 rank 转换为 lsa rank
// 公式：lsaRank_result = comm.lsaRank + (rank - tm.rank)*tm.stride
// 解释：与 ncclTeamRankToWorld 相同逻辑，但以 comm.lsaRank 为基准
NCCL_HOST_DEVICE_INLINE int ncclTeamRankToLsa(ncclDevComm const& comm, ncclTeam tm, int rank) {
  return comm.lsaRank + (rank - tm.rank)*tm.stride;
}
#endif

// ncclTeamInnerFactor(parent, innerSize)：取 parent team 的内层子团
// 内层：将 parent 分层，内层包含前 innerSize 个 rank
// 结果的 rank = parent.rank % innerSize，stride = parent.stride（继承）
// 例：parent={nRanks=8, rank=5, stride=1}, innerSize=4
//   inner={nRanks=4, rank=1, stride=1}（rank 5 在内层的编号=5%4=1）
NCCL_HOST_DEVICE_INLINE ncclTeam_t ncclTeamInnerFactor(ncclTeam_t parent, int innerSize) {
  ncclTeam_t ans;
  ans.nRanks = innerSize;  // 内层共 innerSize 个 rank
  ans.rank = parent.rank%innerSize;  // 内层编号 = parent.rank 模 innerSize
  ans.stride = parent.stride;  // stride 不变（继承 parent）
  return ans;
}

// ncclTeamOuterFactor(parent, innerSize)：取 parent team 的外层子团
// 外层：将 parent 分层，外层包含每隔 innerSize 取一个 rank
// 结果的 rank = parent.rank / innerSize，stride = parent.stride * innerSize
// 例：parent={nRanks=8, rank=5, stride=1}, innerSize=4
//   outer={nRanks=2, rank=1, stride=4}（rank 5 在外层的编号=5/4=1）
NCCL_HOST_DEVICE_INLINE ncclTeam_t ncclTeamOuterFactor(ncclTeam_t parent, int innerSize) {
  ncclTeam_t ans;
  ans.nRanks = parent.nRanks/innerSize;  // 外层共 nRanks/innerSize 个 rank
  ans.rank = parent.rank/innerSize;  // 外层编号 = parent.rank / innerSize
  ans.stride = parent.stride*innerSize;  // 外层 stride 放大 innerSize 倍
  return ans;
}

// ncclTeamRankInDifference(parent, subset, index)：返回集合差（parent - subset）中第 index 个元素
// 前提：subset 是 parent 的子集，两者 rank 按步长（stride）排序。
// 算法说明：
//   stride = subset.stride / parent.stride：subset rank 在 parent 中的步长（subset 每隔多少 parent rank）
//   below：index=0 之前（寻 parent.rank 之前）在差集中有多少元素
//   分三段处理：
//     - index < below：在 subset 第一个元素之前，直接返回 index
//     - 中间段：跨过 subset 元素间隙中的差集元素
//     - 最后段：subset 最后一个元素之后
NCCL_HOST_DEVICE_INLINE int ncclTeamRankInDifference(ncclTeam_t parent, ncclTeam_t subset, int index) {
  // stride：subset 中相邻 rank 在 parent 中的间距
  int stride = subset.stride/parent.stride;
  // below：本 rank 在 subset 第一个元素之前，在差集中有多少 parent rank
  int below = parent.rank - subset.rank*stride;
  if (stride < 0) {
    // stride < 0 表示 subset 在 parent 中是逆序的（如 rail 跨节点）
    stride = -stride;
    below -= (subset.nRanks-1)*stride;  // 适配逆序子集
  }
  if (index < below) {
    // 在 subset 第一个元素之前的差集元素
    return index;
  } else if (index-below < (subset.nRanks-1)*(stride-1)) {
    // 中间段：在 subset 相邻元素之间的差集元素
    return below + 1 + ((index-below)/(stride-1))*stride + (index-below)%(stride-1);
  } else {
    // 最后段：subset 最后一个元素之后的差集元素
    return below + 1 + (subset.nRanks-1)*stride + (index - below - (subset.nRanks-1)*(stride-1));
  }
}

#if NCCL_CHECK_CUDACC
// ncclGetLocalPointer(w, offset)：获取本 rank 对应的 buffer 地址
// 寻址公式：base_i = add4G(lsaFlatBase, lsaRank * stride4G)
//   - lsaFlatBase：rank 0 的 buffer 起始地址（通过 __ldg 从 texture cache 加载）
//   - stride4G * lsaRank：修改 64 位指针高 32 位（= lsaRank 个 4GB 单元）
//   - + offset：buffer 内偏移（低 32 位，不变）
// 使用 loadConst(__ldg)：防止编译器将 window 字段缓入 L1 d-cache
NCCL_DEVICE_INLINE void* ncclGetLocalPointer(ncclWindow_t w, size_t offset) {
  char* base = nccl::utility::loadConst(&w->lsaFlatBase);  // __ldg 加载，不占 L1
  uint32_t stride4G = nccl::utility::loadConst(&w->stride4G);  // __ldg 加载 stride
  int i = nccl::utility::loadConst(&w->lsaRank);  // __ldg 加载 lsaRank
  // add4G(base, i*stride4G)：将 base 的高 32 位加上 i*stride4G，低 32 位不变
  return (void*)(nccl::utility::add4G(base, i*stride4G) + offset);
}
#endif

#if NCCL_CHECK_CUDACC
// ncclGetLsaPointer(w, offset, peer)：获取 lsa team 中 rank peer 的 buffer 地址
// peer 是 0..lsaSize-1 范围内的 lsa rank 编号
// 寻址公式： add4G(lsaFlatBase, peer * stride4G) + offset
NCCL_DEVICE_INLINE void* ncclGetLsaPointer(ncclWindow_t w, size_t offset, int peer) {
  char* base = nccl::utility::loadConst(&w->lsaFlatBase);
  uint32_t stride4G = nccl::utility::loadConst(&w->stride4G);
  int i = peer;  // 直接用 lsa rank 编号寻址
  return (void*)(nccl::utility::add4G(base, i*stride4G) + offset);
}
#endif

#if NCCL_CHECK_CUDACC
// ncclGetPeerPointer(w, offset, peer)：获取全局 world rank peer 的 buffer 地址
// peer 是全局 world rank 编号（可能跨节点）
// 寻址公式：i = lsaRank + (peer - worldRank)
//   解释：worldRank 和 lsaRank 属于本 rank；
//   (peer - worldRank) 是全局差，加到 lsaRank 得到对应的 lsa 偏移
//   前提：全局 rank peer 必须在同一个 lsa team 内
//   （跨节点情形需要确保跨 lsa 访问的内存已映射）
NCCL_DEVICE_INLINE void* ncclGetPeerPointer(ncclWindow_t w, size_t offset, int peer) {
  char* base = nccl::utility::loadConst(&w->lsaFlatBase);
  uint32_t stride4G = nccl::utility::loadConst(&w->stride4G);
  int worldRank = nccl::utility::loadConst(&w->worldRank);  // 本 rank 的全局编号
  int lsaRank = nccl::utility::loadConst(&w->lsaRank);  // 本 rank 的 lsa 编号
  int i = lsaRank + (peer - worldRank);  // 对应的 lsa 偏移
  return (void*)(nccl::utility::add4G(base, i*stride4G) + offset);
}
#endif

#if NCCL_CHECK_CUDACC
// ncclGetPeerPointer(w, offset, tm, peer)：获取 team tm 中 rank peer 的 buffer 地址
// 寻址公式：i = lsaRank + (peer - tm.rank)*tm.stride
//   这里将 team 内的 rank 差乘以 stride 转换为全局差，再对应到 lsa 偏移
NCCL_DEVICE_INLINE void* ncclGetPeerPointer(ncclWindow_t w, size_t offset, ncclTeam tm, int peer) {
  char* base = nccl::utility::loadConst(&w->lsaFlatBase);
  uint32_t stride4G = nccl::utility::loadConst(&w->stride4G);
  int lsaRank = nccl::utility::loadConst(&w->lsaRank);
  // (peer - tm.rank)*tm.stride：将 team 内 rank 差转换为全局 rank 差，再对应到 lsa 偏移
  int i = lsaRank + (peer - tm.rank)*tm.stride;
  return (void*)(nccl::utility::add4G(base, i*stride4G) + offset);
}
#endif

#if NCCL_CHECK_CUDACC
// ncclGetMultimemPointer(w, offset, mm)：获取 NVLink MultiCast 地址
// 寻址公式：ptr = mm.mcBasePtr + mcOffset4K * 4096 + offset
//   - mm.mcBasePtr：MultiCast 虚拟地址基址
//   - mcOffset4K * 4096：window 在 MultiCast 空间中的偏移（4096 = 4KB）
//   - + offset：buffer 内偏移
// reinterpret_cast<char(*)[4096]>(ptr) + k：计算 ptr + k*4096 字节偏移的技巧
//   即将指针重解释为指向 4096 字节数组的指针，然后索引加 k
NCCL_DEVICE_INLINE void* ncclGetMultimemPointer(ncclWindow_t w, size_t offset, ncclMultimemHandle mm) {
  void* ptr = mm.mcBasePtr;  // MultiCast 基址
  // 将 ptr 当作指向 char[4096] 数组的指针，索引十就是加 mcOffset4K * 4096 字节
  ptr = reinterpret_cast<char(*)[4096]>(ptr) + nccl::utility::loadConst(&w->mcOffset4K);
  return (void*)((char*)ptr + offset);
}
#endif

#if NCCL_CHECK_CUDACC
// ncclGetLsaMultimemPointer(w, offset, comm)：使用 DevComm 的 lsaMultimem 句柄获取 MultiCast 地址
// 是 ncclGetMultimemPointer 的便捷包装：直接使用 comm.lsaMultimem
NCCL_DEVICE_INLINE void* ncclGetLsaMultimemPointer(ncclWindow_t w, size_t offset, ncclDevComm const& comm) {
  return ncclGetMultimemPointer(w, offset, comm.lsaMultimem);
}
#endif

#if NCCL_CHECK_CUDACC
// ncclFindWindow(coop, comm, ptr)：在 DevComm 的 windowTable 中查找包含地址 ptr 的 window
// 功能：给定一个内存地址 ptr，找到包含该地址的 ncclWindow_t
//
// 协作搜索机制（第一个寻址中的精华）：
//   windowTable 有多个表项 Entry，每项存储（base, size, window）三元组。
//   直接用单线程轮询所有 entry 太慢，ncclFindWindow 用 coop 内所有线程并行搜索：
//   1. 每个线程属于 coalesced coop（寻址调用时的有效线程子集）
//   2. 每个线程以步长 coalesced.size() 遍历多个 entry（相当于 warp 级并行 for 循环）
//   3. 找到后设置 found=true，将 index 上报
//   4. __ballot_sync 收集所有线程的 found 标志，生成 mask
//   5. 如果 mask != 0，选最低位的控制线程（第一个找到的线程）
//   6. __shfl_sync 广播该线程的 index，返回对应 window
//   7. 如果本表全没找到，继续查询下一个链表节点（t = t->next）
//
// 注意：
//   - 使用 loadConst(__ldg) 读取 entry，通过 texture cache 加速
//   - uptr - e.base < e.size 是无符号比较，告知编译器 uptr 在范围内
template<typename Coop>
NCCL_DEVICE_INLINE ncclWindow_t ncclFindWindow(Coop coop, ncclDevComm const& comm, void const *ptr) {
  using nccl::utility::loadConst;
  // coalesced：获取 coop 中实际活跃的线程子集（加速多播和兼容性）
  auto coalesced = ncclCoopCoalesced(coop);
  ncclDevCommWindowTable* t = comm.windowTable;
  while (true) {
    bool found = false;
    // index：本线程负责搜索的 entry 起始索引
    int index = coalesced.thread_rank();
    #pragma unroll 1
    while (index < 32) {  // windowTable 每个节点最多 32 个 entry
      uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
      // 通过 __ldg 加载 entry（避免占用 L1 d-cache）
      ncclDevCommWindowTable::Entry e = loadConst(&t->entries[index]);
      if ((e.base != 0) && (e.size != 0) && (e.window != 0)) {
        // uptr - e.base < e.size：无符号范围检查（uptr 在 [e.base, e.base+e.size) 内）
        if (uptr - uintptr_t(e.base) < uintptr_t(e.size)) {
          found = true;
          break;
        }
      }
      index += coalesced.size();  // 跳过 coalesced.size() 个 entry，让所有线程繆密覆盖
    }
    // __ballot_sync：收集所有线程的 found 标志生成 mask
    uint32_t mask = __ballot_sync(ncclCoopGetLaneMask(coalesced), found);
    if (mask != 0) {
      // __popc(mask-1)：找到 mask 中最低位=1 的位置（即第一个找到的线程）
      int source = __popc(mask-1);
      // __shfl_sync：将找到 window 的线程的 index 广播给所有线程
      index = __shfl_sync(ncclCoopGetLaneMask(coalesced), index, source);
      return loadConst(&t->entries[index].window);
    }
    // 本表未找到，继续遍历链表下一节点
    t = loadConst(&t->next);
  }
}
#endif

// ncclGetResourceBufferOffset(h)：由 resource handle 计算字节偏移
// 公式：offset = h * 128
// 每个 resource handle 对应 128 字节对齐的 slot，使用 <<7 优化为移位
NCCL_HOST_DEVICE_INLINE size_t ncclGetResourceBufferOffset(ncclDevResourceHandle_t h) {
  return ((size_t)h)*128;  // h * 128 = h << 7
}

#if NCCL_CHECK_CUDACC
// ncclGetResourceBufferLocalPointer(comm, h)：本 rank 的 resource buffer slot 地址
// 寻址公式：
//   local = add4G(lsaFlatBase, lsaRank * stride4G)  -- 本 rank 的 buffer 基址
//   ptr = reinterpret_cast<char(*)[128]>(local) + h  -- 加上 handle 个 128B 偏移
// 使用内置 resourceWindow_inlined （對内存内置），无需际外读取
NCCL_DEVICE_INLINE void* ncclGetResourceBufferLocalPointer(ncclDevComm const& comm, ncclDevResourceHandle h) {
  void* lsaFlatBase = comm.resourceWindow_inlined.lsaFlatBase;
  uint32_t stride4G = comm.resourceWindow_inlined.stride4G;
  void* local = nccl::utility::add4G(lsaFlatBase, comm.lsaRank*stride4G);  // 本 rank buffer 基址
  return (void*)(reinterpret_cast<char(*)[128]>(local) + h);  // 加上 h * 128 字节
}
#endif

#if NCCL_CHECK_CUDACC
// ncclGetResourceBufferLsaPointer(comm, h, peer)： lsa rank peer 的 resource buffer slot
// peer 是 0..lsaSize-1 范围内的 lsa rank 编号
NCCL_DEVICE_INLINE void* ncclGetResourceBufferLsaPointer(ncclDevComm const& comm, ncclDevResourceHandle h, int peer) {
  int r = peer;  // 直接用 lsa rank 编号
  void* lsaFlatBase = comm.resourceWindow_inlined.lsaFlatBase;
  uint32_t stride4G = comm.resourceWindow_inlined.stride4G;
  void* local = nccl::utility::add4G(lsaFlatBase, r*stride4G);  // lsa rank r 的 buffer 基址
  return (void*)(reinterpret_cast<char(*)[128]>(local) + h);
}
#endif

#if NCCL_CHECK_CUDACC
// ncclGetResourceBufferPeerPointer(comm, h, team, peer)： team 中 rank peer 的 resource buffer slot
// 寻址公式：r = lsaRank + (peer - team.rank)*team.stride
NCCL_DEVICE_INLINE void* ncclGetResourceBufferPeerPointer(ncclDevComm const& comm, ncclDevResourceHandle h, ncclTeam team, int peer) {
  // 将 team 中 rank peer 转换为 lsa rank 偏移
  int r = comm.lsaRank + (peer - team.rank)*team.stride;
  void* lsaFlatBase = comm.resourceWindow_inlined.lsaFlatBase;
  uint32_t stride4G = comm.resourceWindow_inlined.stride4G;
  void* local = nccl::utility::add4G(lsaFlatBase, r*stride4G);
  return (void*)(reinterpret_cast<char(*)[128]>(local) + h);
}
#endif

#if NCCL_CHECK_CUDACC
// ncclGetResourceBufferMultimemPointer(comm, h, mm)：MultiCast 路径的 resource buffer slot
// 寻址公式：ptr = mm.mcBasePtr + mcOffset4K * 4096 + h * 128
//   先定位到 MultiCast window（mcOffset4K * 4096），再定位到 slot（h * 128）
NCCL_DEVICE_INLINE void* ncclGetResourceBufferMultimemPointer(ncclDevComm const& comm, ncclDevResourceHandle h, ncclMultimemHandle mm) {
  void* ptr = mm.mcBasePtr;
  // 先加上 window 的 MultiCast 偏移（mcOffset4K 个 4096B）
  ptr = reinterpret_cast<char(*)[4096]>(ptr) + comm.resourceWindow_inlined.mcOffset4K;
  // 再加上 handle 的 slot 偏移（h 个 128B）
  ptr = reinterpret_cast<char(*)[128]>(ptr) + h;
  return ptr;
}
#endif

#if NCCL_CHECK_CUDACC
// ncclGetResourceBufferLsaMultimemPointer(comm, h)：使用 lsaMultimem 的 resource buffer slot
// 是 ncclGetResourceBufferMultimemPointer 的便捷包装
NCCL_DEVICE_INLINE void* ncclGetResourceBufferLsaMultimemPointer(ncclDevComm const& comm, ncclDevResourceHandle h) {
  return ncclGetResourceBufferMultimemPointer(comm, h, comm.lsaMultimem);
}
#endif

#if NCCL_CHECK_CUDACC
// ncclGetResourceBuffer(comm, h)：返回封装成 ncclSymPtr<char> 的 resource buffer 地址
// ncclSymPtr 封装了 Symmetric Memory 寻址，包含 window 和 offset 两个分量
NCCL_DEVICE_INLINE ncclSymPtr<char> ncclGetResourceBuffer(ncclDevComm const& comm, ncclDevResourceHandle h) {
  // ncclSymPtr<char>(window, offset)：绑定 window 和 offset，后续可用 .localPtr()/.lsaPtr() 等获取具体地址
  return ncclSymPtr<char>(comm.resourceWindow, size_t(h)*128);
}
#endif

#endif
