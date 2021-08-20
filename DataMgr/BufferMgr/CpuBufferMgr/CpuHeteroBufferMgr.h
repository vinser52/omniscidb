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

#pragma once

#include "DataMgr/BufferMgr/HeteroBufferMgr.h"
#include "DataMgr/BufferMgr/CpuBufferMgr/CpuHeteroBuffer.h"

#include "HeteroMem/MemResourceProvider.h"
#include "DataMgr/BufferMgr/omnisci_memory_resources.h"

namespace CudaMgr_Namespace {
class CudaMgr;
}

namespace Buffer_Namespace {

class CpuHeteroBufferMgr : public HeteroBufferMgr {
public:
  CpuHeteroBufferMgr(const int device_id,
                     const size_t max_dram_buffer_pool_size,
                     CudaMgr_Namespace::CudaMgr* cuda_mgr,
                     const size_t min_slab_size, 
                     const size_t max_slab_size,
                     const std::string& pmm_path,
                     const size_t pmem_size,
                     const size_t page_size = 512,
                     AbstractBufferMgr* parent_mgr = nullptr);

  CpuHeteroBufferMgr(const int device_id,
                     const size_t max_buffer_pool_size,
                     CudaMgr_Namespace::CudaMgr* cuda_mgr,
                     const size_t min_slab_size, 
                     const size_t max_slab_size,
                     const size_t page_size = 512,
                     AbstractBufferMgr* parent_mgr = nullptr);

  ~CpuHeteroBufferMgr() override;

  inline MgrType getMgrType() override { return CPU_MGR; }
  inline std::string getStringMgrType() override { return ToString(CPU_MGR); }

  using map_type = std::unordered_map<std::pmr::memory_resource*, omnisci_memory_resource*>;
  using memory_layer_iterator = typename map_type::const_iterator;
  using memory_layer_info_type = omnisci_memory_resource;
  class memory_info_iterator {
    private:
      memory_layer_iterator map_it_;
    public:
    	memory_info_iterator(memory_layer_iterator map_it){
        map_it_ = map_it;
      }
      memory_info_iterator& operator++(){
        ++map_it_;
        return *this;
      }
      memory_info_iterator operator++(int){
        memory_info_iterator prev_it = *this;
        ++map_it_;
        return prev_it;
      }
      const memory_layer_info_type& operator*(){
        return *(map_it_->second);
      }
      const memory_layer_info_type* operator->(){
        return map_it_->second;
      }
      bool operator!=(const memory_info_iterator& other){
        return map_it_!=other.map_it_;
      }
  };
  memory_info_iterator memory_layers_begin() const {
    return memory_info_iterator(gen_mem_resource.begin());
  }
  memory_info_iterator memory_layers_end() const {
    return memory_info_iterator(gen_mem_resource.end());
  }
  size_t getMaxSize() override;
protected:
  void releaseSlabs() override;
  using buffer_type = CpuHeteroBuffer;
  AbstractBuffer* constructBuffer(
#ifdef HAVE_DCPMM
                                  BufferProperty bufProp,
#endif /* HAVE_DCPMM */
                                  const size_t chunk_page_size,
                                  const size_t initial_size) override;
  void destroyBuffer(AbstractBuffer* buffer) override;
  size_t getMaxPoolForLayer(std::pmr::memory_resource* mem_resource);
private:
  std::pmr::memory_resource* getMemoryResource(MemRequirements property, size_t page_size);
  size_t max_dram_buffer_pool_size_;
  MemoryResourceProvider mem_resource_provider_;
  map_type gen_mem_resource;
};
} // namespace Buffer_Namespace
