#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        auto it = this->freeBlocks.lower_bound(freeBlockInfo{(size_t)0, size});

        size_t retAddr = this->peak;
        if (it != this->freeBlocks.end()) {
            // found an alvailable free memory block for allocation
            size_t blockSize = it->blockSize;
            retAddr = it->addr;
            size_t tailAddr = retAddr + size;
            // update the map of head and tail address offset of memory blocks
            this->headAddrToBlockSize.erase(retAddr);
            this->tailAddrToBlockSize.erase(tailAddr);
            // memory block splitting
            if (blockSize > tailAddr - retAddr) {
                freeBlockInfo newBlock = {tailAddr,
                                          blockSize - (tailAddr - retAddr)};
                this->headAddrToBlockSize[tailAddr] = newBlock.blockSize;
                this->tailAddrToBlockSize[retAddr + blockSize] = newBlock.blockSize;
                this->freeBlocks.insert(newBlock);
            }
            // update the free balanced tree
            this->freeBlocks.erase(it);
            this->used += tailAddr - retAddr;
        } else {
            // the allocated memory space is not sufficient for reallocation, it
            // needs to be extended
            auto blockTailWithPeak = this->tailAddrToBlockSize.find(this->peak);
            if (blockTailWithPeak != this->tailAddrToBlockSize.end()) {
                // there is a free block located at the end of the currently
                // allocated memory, where this free block has its tail address as
                // 'peak'
                retAddr = this->peak - blockTailWithPeak->second;
                IT_ASSERT(blockTailWithPeak->second < size);
                this->peak += (size - blockTailWithPeak->second);
                // updata freeBlocks, headAddrToBlockSize and tailAddrToBlockSize
                freeBlockInfo endBlock = {retAddr, blockTailWithPeak->second};
                this->freeBlocks.erase(endBlock);
                this->headAddrToBlockSize.erase(endBlock.addr);
                this->tailAddrToBlockSize.erase(endBlock.addr + endBlock.blockSize);
            } else {
                this->peak = this->peak + size;
            }
            this->used += size;
        }
    
        return retAddr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        auto tailAddr = addr + size;
        freeBlockInfo block = {addr, tailAddr - addr};
        this->headAddrToBlockSize[addr] = block.blockSize;
        this->tailAddrToBlockSize[tailAddr] = block.blockSize;
        auto preFreeBlockIter = this->tailAddrToBlockSize.find(addr);
        auto subFreeBlockIter = this->headAddrToBlockSize.find(tailAddr);
        if (preFreeBlockIter != this->tailAddrToBlockSize.end()) {
            // the head address of the memory block to be freed matches the end of a
            // free block, merge them together
            size_t preBlockSize = preFreeBlockIter->second;
            this->headAddrToBlockSize.erase(block.addr);
            this->headAddrToBlockSize[block.addr - preBlockSize] += block.blockSize;
            this->tailAddrToBlockSize.erase(block.addr);
            this->tailAddrToBlockSize[tailAddr] += preBlockSize;
            block.addr -= preBlockSize;
            block.blockSize += preBlockSize;
            // delete the preceding adjacent free block
            this->freeBlocks.erase(freeBlockInfo{block.addr, preBlockSize});
        }
        if (subFreeBlockIter != this->headAddrToBlockSize.end()) {
            // the tail address of the memory block to be freed matches the start of
            // a free block, merge them together
            auto subBlockSize = subFreeBlockIter->second;
            this->headAddrToBlockSize.erase(tailAddr);
            this->headAddrToBlockSize[block.addr] += subBlockSize;
            this->tailAddrToBlockSize.erase(tailAddr);
            this->tailAddrToBlockSize[tailAddr + subBlockSize] += block.blockSize;
            tailAddr += subBlockSize;
            block.blockSize += subBlockSize;
            // delete the succeeding adjacent memory block
            this->freeBlocks.erase(
                freeBlockInfo{tailAddr - subBlockSize, subBlockSize});
        }
        this->freeBlocks.insert(block);
        this->used -= size;    
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
