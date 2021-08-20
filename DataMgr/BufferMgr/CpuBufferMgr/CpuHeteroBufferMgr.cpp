/*
 * Copyright 2020 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "DataMgr/BufferMgr/CpuBufferMgr/CpuHeteroBufferMgr.h"

namespace Buffer_Namespace {

#ifdef HAVE_DCPMM
static MemRequirements get_mem_characteristics(BufferProperty bufProp) {
  switch (bufProp) {
    case HIGH_BDWTH:
      return MemRequirements::HIGH_BDWTH;
    case LOW_LATENCY:
      return MemRequirements::LOW_LATENCY;
    case CAPACITY:
    default:
      return MemRequirements::CAPACITY;
  }
  return MemRequirements::CAPACITY;
}
#endif /* HAVE_DCPMM */

CpuHeteroBufferMgr::CpuHeteroBufferMgr(const int device_id,
                                       const size_t max_dram_buffer_pool_size,
                                       CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                       const size_t min_slab_size, 
                                       const size_t max_slab_size,
                                       const std::string& pmm_path,
                                       const size_t pmem_size,
                                       const size_t page_size,
                                       AbstractBufferMgr* parent_mgr)
    : HeteroBufferMgr(device_id, cuda_mgr, min_slab_size,
                  max_slab_size, page_size, parent_mgr),  
                  max_dram_buffer_pool_size_(max_dram_buffer_pool_size),
                  mem_resource_provider_(pmm_path, pmem_size) {

}

CpuHeteroBufferMgr::CpuHeteroBufferMgr(const int device_id,
                                       const size_t max_buffer_pool_size,
                                       CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                       const size_t min_slab_size, 
                                       const size_t max_slab_size,
                                       const size_t page_size,
                                       AbstractBufferMgr* parent_mgr)
    : HeteroBufferMgr(device_id, cuda_mgr, min_slab_size,
                  max_slab_size, page_size, parent_mgr), 
                  max_dram_buffer_pool_size_(max_buffer_pool_size) {
}

size_t CpuHeteroBufferMgr::getMaxPoolForLayer(std::pmr::memory_resource* mem_resource){
  switch (mem_resource_provider_.getMemoryType(mem_resource))
  {
  case DRAM:
  {
    if(max_dram_buffer_pool_size_ == 0) {
      size_t total_system_memory = mem_resource_provider_.getAvailableMemorySize(mem_resource);
      VLOG(1) << "Detected " << (float)total_system_memory / (1024 * 1024)
            << "M of total system memory(DRAM).";
      max_dram_buffer_pool_size_ = total_system_memory * 0.8;
    }
      LOG(INFO) << "Max memory pool on DRAM size for CPU is " << (float)max_dram_buffer_pool_size_ / (1024 * 1024)
            << "MB";
    return max_dram_buffer_pool_size_;
  }
    break;
  case PMEM:
  {
      size_t max_pmem_buffer_pool_size_ = mem_resource_provider_.getAvailableMemorySize(mem_resource);
      VLOG(1) << "Detected " << (float)max_pmem_buffer_pool_size_ / (1024 * 1024)
            << "M of total available memory.";
      LOG(INFO) << "Max memory pool size for CPU is " << (float)max_pmem_buffer_pool_size_ / (1024 * 1024)
            << "MB";
    return max_pmem_buffer_pool_size_;
  }
    break;
  default:
    LOG(FATAL) << "HBM is not implemented";
    break;
  }
  return 0;
}

CpuHeteroBufferMgr::~CpuHeteroBufferMgr() {
  clear();
  
  for( auto it = gen_mem_resource.begin(); it != gen_mem_resource.end(); ++it){
    delete it->second;
  }
  gen_mem_resource.clear();
}

size_t CpuHeteroBufferMgr::getMaxSize(){
  UNREACHABLE();
  return 0;
}

void CpuHeteroBufferMgr::releaseSlabs(){
  for(auto it = gen_mem_resource.begin(); it != gen_mem_resource.end(); ++it){
      Buffer_Namespace::omnisci_memory_resource* omnisci_memory_resource = 
         dynamic_cast<Buffer_Namespace::omnisci_memory_resource*>(it->second);
      omnisci_memory_resource->clearSlabs();
    }
}

std::pmr::memory_resource* CpuHeteroBufferMgr::getMemoryResource(MemRequirements property, size_t page_size){
  std::pmr::memory_resource* tmp_mem_resource = mem_resource_provider_.get(property);
  auto find_mem_res = gen_mem_resource.find(tmp_mem_resource);
  if(find_mem_res == gen_mem_resource.end()){
    size_t BufferPoolSize = getMaxPoolForLayer(tmp_mem_resource);
    size_t minSlabSize = std::min(min_slab_size_, BufferPoolSize);
    minSlabSize = (minSlabSize / page_size_) * page_size_;
    size_t maxSlabSize = std::min(max_slab_size_, BufferPoolSize);
    maxSlabSize = (maxSlabSize / page_size_) * page_size_;
    LOG(INFO) << "Min CPU Slab Size is " << (float)minSlabSize / (1024 * 1024) << "MB";
    LOG(INFO) << "Max CPU Slab Size is " << (float)maxSlabSize / (1024 * 1024) << "MB";
    omnisci_memory_resource* new_omnisci_mem_res = new omnisci_memory_resource(
                                                                tmp_mem_resource, 
                                                                mem_resource_provider_.getMemoryType(tmp_mem_resource),
                                                                page_size, 
                                                                BufferPoolSize, 
                                                                minSlabSize, 
                                                                maxSlabSize,
                                                                this);
    gen_mem_resource.insert({tmp_mem_resource, new_omnisci_mem_res});
    return new_omnisci_mem_res;
  } else {
    return find_mem_res->second;
  }
}

AbstractBuffer* CpuHeteroBufferMgr::constructBuffer(
#ifdef HAVE_DCPMM
                                                    BufferProperty bufProp,
#endif /* HAVE_DCPMM */
                                                    const size_t chunk_page_size,
                                                    const size_t initial_size) {
  return new buffer_type(device_id_, cuda_mgr_, chunk_page_size, initial_size, getMemoryResource(
#ifdef HAVE_DCPMM
  get_mem_characteristics(bufProp),
#else
  MemRequirements::CAPACITY,
#endif /* HAVE_DCPMM */
  chunk_page_size));
  
}
void CpuHeteroBufferMgr::destroyBuffer(AbstractBuffer* buffer) {
  buffer_type* casted_buffer = dynamic_cast<buffer_type*>(buffer);
  if (casted_buffer == 0) {
    LOG(FATAL) << "Wrong buffer type - expects base class pointer to buffer_type type.";
  }

  delete casted_buffer;
}
} // namespace Buffer_Namespace