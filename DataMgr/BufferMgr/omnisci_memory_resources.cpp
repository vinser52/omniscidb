
#include "DataMgr/BufferMgr/omnisci_memory_resources.h"

namespace Buffer_Namespace{
omnisci_memory_resource::omnisci_memory_resource(std::pmr::memory_resource* memory_resource, 
                                                            MemType memory_type_,
                                                            size_t page_size_, 
                                                            size_t max_buffer_pool_size, 
                                                            size_t min_slab_size, 
                                                            size_t max_slab_size,
                                                            AbstractBufferMgr* parent_mgr_) 
    : page_size(page_size_)
    , memory_type(memory_type_)
    , gen_mem(memory_resource)
    , parent_mgr(parent_mgr_)
    , num_pages_allocated(0)
    , allocations_capped(false)
    , max_buffer_pool_num_pages(max_buffer_pool_size / page_size)
    , min_num_pages_per_slab(min_slab_size / page_size)
    , max_num_pages_per_slab(max_slab_size / page_size)
    , current_max_slab_page_size(max_num_pages_per_slab)
    {
        CHECK(max_buffer_pool_size > 0);
        CHECK(page_size > 0);
        CHECK(min_slab_size > 0);
        CHECK(max_slab_size > 0);
        CHECK(min_slab_size <= max_slab_size);
        CHECK(min_slab_size % page_size == 0);
        CHECK(max_slab_size % page_size == 0);
    }


void omnisci_memory_resource::addSlab(size_t bytes){
    try {
        int8_t* tmp =  reinterpret_cast<int8_t*>(gen_mem->allocate(bytes, page_size));
        slabs.push_back(tmp);
    } catch (std::bad_alloc&) {
        //throw std::bad_alloc();   
        throw FailedToCreateSlab(bytes);
    }
    slab_segments.emplace_back(SegList());
    slab_segments[slab_segments.size() - 1].push_back(MemSeg(0, bytes / page_size, slabs.size() - 1));
    num_pages_allocated += bytes / page_size;
}

void* omnisci_memory_resource::do_allocate(std::size_t bytes, std::size_t alignment){
    std::lock_guard<std::mutex> lock(mem_mutex);
    SegList::iterator seg_it = findFreeSeg(bytes);
    int8_t* new_mem_seg = slabs[seg_it->slab_num] + seg_it->start_page * page_size;
    map_segments[new_mem_seg] = seg_it;
    return new_mem_seg;
}
SegList::iterator omnisci_memory_resource::findFreeSegInSlab(size_t slab_num, size_t num_pages_requested){
    for (auto seg_it = slab_segments[slab_num].begin(); seg_it != slab_segments[slab_num].end();++seg_it) {
        if (seg_it->mem_status == FREE && seg_it->num_pages >= num_pages_requested) {
            size_t excess_pages = seg_it->num_pages - num_pages_requested;
            seg_it->num_pages = num_pages_requested;
            seg_it->mem_status = USED;
            if (excess_pages > 0) {
                MemSeg free_seg(seg_it->start_page + num_pages_requested, excess_pages, slab_num);
                auto temp_it = seg_it;  
                temp_it++;
                slab_segments[slab_num].insert(temp_it, free_seg);
            }
            return seg_it;
        } // if
    } // for
    return slab_segments[slab_num].end();
} // findFreeSegInSlab

SegList::iterator omnisci_memory_resource::findFreeSeg(size_t bytes){
    size_t num_pages_requested = (bytes + page_size - 1) / page_size;
    
    if (num_pages_requested > max_num_pages_per_slab) {
        throw TooBigForSlab(bytes);
    }

    size_t num_slabs = slab_segments.size();

    for (size_t slab_it = 0; slab_it != num_slabs; ++slab_it) {
        auto seg_it = findFreeSegInSlab(slab_it, num_pages_requested);
        if (seg_it != slab_segments[slab_it].end()) {
            return seg_it;
        }
    }

    while (!allocations_capped && num_pages_allocated < max_buffer_pool_num_pages) {
        try{
            size_t pagesLeft = max_buffer_pool_num_pages - num_pages_allocated;
            if (pagesLeft < current_max_slab_page_size) {
                current_max_slab_page_size = pagesLeft;
            }

            if(num_pages_requested <= current_max_slab_page_size){
                auto alloc_ms = measure<>::execution([&]() { addSlab(current_max_slab_page_size * page_size); });
                LOG(INFO) << "ALLOCATION slab of " << current_max_slab_page_size << " pages ("
                << current_max_slab_page_size * page_size << "B) created in "
                << alloc_ms << " ms " << parent_mgr->getStringMgrType() << ":" << parent_mgr->getDeviceId();
            }
            else break;

            return findFreeSegInSlab(num_slabs, num_pages_requested);

        }catch(std::runtime_error& error){
            LOG(INFO) << "ALLOCATION Attempted slab of " << current_max_slab_page_size
            << " pages (" << current_max_slab_page_size * page_size << "B) failed "
            << parent_mgr->getStringMgrType() << ":" << parent_mgr->getDeviceId();
            
            if (num_pages_requested > current_max_slab_page_size / 2 && current_max_slab_page_size != num_pages_requested){
                current_max_slab_page_size = num_pages_requested;
            }
            else{
                current_max_slab_page_size /= 2;
                if (current_max_slab_page_size < min_num_pages_per_slab){
                    allocations_capped = true;
                    LOG(INFO) << "ALLOCATION Capped " << current_max_slab_page_size
                    << " Minimum size = " << min_num_pages_per_slab << " "
                    << parent_mgr->getStringMgrType() << ":" << parent_mgr->getDeviceId();
                }
            }
        } // try,catch
    } // while
    if (num_pages_allocated == 0 && allocations_capped) {
        throw FailedToCreateFirstSlab(bytes);
    }
    return slab_segments[0].end();
} // findFreeSeg

void omnisci_memory_resource::do_deallocate(void* p, std::size_t bytes, std::size_t alignment){
    int8_t* dealloc_ptr =  reinterpret_cast<int8_t*>(p);
    std::lock_guard<std::mutex> lock(mem_mutex);
    auto map_seg_it = map_segments.find(dealloc_ptr);
    SegList::iterator seg_it = map_seg_it->second;
    map_segments.erase(map_seg_it);

    removeSegment(seg_it);
} // do_deallocate

void omnisci_memory_resource::removeSegment(SegList::iterator& seg_it){
    int slab_num = seg_it->slab_num;
    if (seg_it != slab_segments[slab_num].begin()){
        auto prev_it = std::prev(seg_it);
        if (prev_it->mem_status == FREE){
            seg_it->start_page = prev_it->start_page;
            seg_it->num_pages += prev_it->num_pages;
            slab_segments[slab_num].erase(prev_it);
        }
    }

    auto next_it = std::next(seg_it);
    if ( next_it != slab_segments[slab_num].end() ){
        if (next_it->mem_status == FREE){
            seg_it->num_pages += next_it->num_pages;
            slab_segments[slab_num].erase(next_it);
        }
    }

    seg_it->mem_status = FREE;
} // removeSegment

void omnisci_memory_resource::clearSlabs(){
    clearAll();
    reinit();
}

void omnisci_memory_resource::reinit() {
    num_pages_allocated = 0;
    current_max_slab_page_size =
    max_num_pages_per_slab;  
    allocations_capped = false;
}

void omnisci_memory_resource::clearAll(){
    std::lock_guard<std::mutex> lock(mem_mutex);
    map_segments.clear();
    int size = slabs.size();
    for (int i = 0; i < size; ++i){
        auto seg_it = slab_segments[i].end();
        --seg_it;
        size_t bytes = (seg_it->start_page + seg_it->num_pages) * page_size;
        gen_mem->deallocate(slabs[i], bytes, page_size);
    }
    slab_segments.clear();
    slabs.clear();
} // clearAll

omnisci_memory_resource::~omnisci_memory_resource(){
    clearAll();
}

bool omnisci_memory_resource::do_is_equal(const std::pmr::memory_resource& other) const noexcept {
    const omnisci_memory_resource* other_ptr = dynamic_cast<const omnisci_memory_resource*>(&other);
    return (other_ptr != nullptr && this == other_ptr);
}
size_t omnisci_memory_resource::getPageSize() const{
    return page_size;
}

size_t omnisci_memory_resource::getMaxSize() const{
    return page_size * max_buffer_pool_num_pages;
}

const std::vector<SegList>& omnisci_memory_resource::getSlabSegments() const{
    return slab_segments;
}

bool omnisci_memory_resource::isAllocationCapped() const{
    return allocations_capped;
}

MemType omnisci_memory_resource::getMemType() const {
    return memory_type;
}

size_t omnisci_memory_resource::getAllocated() const{
    return num_pages_allocated * page_size;
}
} // Buffer_Namespace