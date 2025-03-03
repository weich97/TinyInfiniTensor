#pragma once
#include "core/runtime.h"
#include "core/tensor.h"
#ifdef BUILD_TEST
#include "gtest/gtest.h"
#endif
#include <cstddef>
#include <map>
#include <unordered_set>

namespace infini {
  class Allocator
  {
  private:
    Runtime runtime;

    size_t used;

    size_t peak;

    size_t alignment;

    // pointer to the memory actually allocated
    void *ptr;

    // =================================== 作业 ===================================
    // TODO：可能需要设计一个数据结构来存储free block，以便于管理和合并
    // HINT: 可以使用一个 map 来存储 free block，key 为 block 的起始/结尾地址，value 为 block 的大小
    // =================================== 作业 ===================================

    struct freeBlockInfo {
      size_t addr;
      size_t blockSize;
    };  

    struct cmpFreeBlockInfo {
      bool operator()(const freeBlockInfo &a, const freeBlockInfo &b) const {
          return (a.blockSize != b.blockSize) ? (a.blockSize < b.blockSize)
                                              : (a.addr < b.addr);
      }
    };    

    // free balanced tree, maintains all free memory blocks
    std::set<freeBlockInfo, cmpFreeBlockInfo> freeBlocks;

    // key: head address offset of the free memory block
    // value: blockSize of the block
    std::unordered_map<size_t, size_t> headAddrToBlockSize;

    // key: tail address offset of the free memory block
    // value: blockSize of the block
    std::unordered_map<size_t, size_t> tailAddrToBlockSize;    

  public:
    Allocator(Runtime runtime);

    virtual ~Allocator();

    // function: simulate memory allocation
    // arguments：
    //     size: size of memory block to be allocated
    // return: head address offset of the allocated memory block
    size_t alloc(size_t size);

    // function: simulate memory free
    // arguments:
    //     addr: head address offset of memory block to be free
    //     size: size of memory block to be freed
    void free(size_t addr, size_t size);

    // function: perform actual memory allocation
    // return: pointer to the head address of the allocated memory
    void *getPtr();

    void info();

  private:
    // function: memory alignment, rouned up
    // return: size of the aligned memory block
    size_t getAlignedSize(size_t size);
  };
}
