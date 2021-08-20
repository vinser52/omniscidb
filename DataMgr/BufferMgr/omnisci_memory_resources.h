#pragma once

#include <list>
#include <memory>
#include <unordered_map>
#include "Logger/Logger.h"
#include <memory_resource>
#include "Shared/measure.h"
#include "DataMgr/BufferMgr/BufferMgr.h"
#include "HeteroMem/MemResourceProvider.h"

namespace Buffer_Namespace {
struct MemSeg{
    int start_page;
    size_t num_pages;
    MemStatus mem_status;
    int slab_num;

    MemSeg(): start_page(-1), num_pages(0), mem_status(FREE), slab_num(-1) {}
    MemSeg(int start_page_, size_t num_page_, int slab_num_) : start_page(start_page_), num_pages(num_page_),
                                                                mem_status(FREE), slab_num(slab_num_) {}
};

using SegList = std::list<MemSeg>;

class omnisci_memory_resource : public std::pmr::memory_resource {
    private:
        void* do_allocate(std::size_t bytes, std::size_t alignment) override;
        void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override;
        bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override;
        SegList::iterator findFreeSeg(size_t bytes);
        SegList::iterator findFreeSegInSlab(size_t slab_num, size_t num_pages_requested);
        void addSlab(size_t slab_size);
        void removeSegment(SegList::iterator& seg_it);
        void clearAll();
        void reinit();
    public:
        explicit omnisci_memory_resource(std::pmr::memory_resource* memory_resource, 
                                                            MemType memory_type_,
                                                            size_t page_size_, 
                                                            size_t max_buffer_pool_size, 
                                                            size_t min_slab_size, 
                                                            size_t max_slab_size,
                                                            AbstractBufferMgr* parent_mgr_);
        ~omnisci_memory_resource();
        size_t getPageSize() const;
        size_t getMaxSize() const;
        const std::vector<SegList>& getSlabSegments() const;
        bool isAllocationCapped() const;
        size_t getAllocated() const;
        MemType getMemType() const;
        void clearSlabs();
    private:
        const size_t page_size;
        MemType memory_type;
        std::pmr::memory_resource* gen_mem;
        AbstractBufferMgr* parent_mgr;

        size_t num_pages_allocated;
        bool allocations_capped;

        std::vector<int8_t*> slabs; 
        std::vector<SegList> slab_segments;
        std::unordered_map<int8_t*, SegList::iterator> map_segments;

        size_t max_buffer_pool_num_pages; 
        size_t min_num_pages_per_slab;
        size_t max_num_pages_per_slab;
        size_t current_max_slab_page_size;

        std::mutex mem_mutex;
}; // omnisci_memory_resource
} // namespace Buffer_Namespace


