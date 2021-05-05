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

CpuHeteroBufferMgr::CpuHeteroBufferMgr(const int device_id,
                                       const size_t max_buffer_size,
                                       const std::string& pmm_path,
                                       CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                       const size_t page_size,
                                       AbstractBufferMgr* parent_mgr)
    : HeteroBufferMgr(device_id, max_buffer_size, cuda_mgr, page_size, parent_mgr), mem_resource_provider_(pmm_path) {
}

CpuHeteroBufferMgr::CpuHeteroBufferMgr(const int device_id,
                                       const size_t max_buffer_size,
                                       CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                       const size_t page_size,
                                       AbstractBufferMgr* parent_mgr)
    : HeteroBufferMgr(device_id, max_buffer_size, cuda_mgr, page_size, parent_mgr) {
}

CpuHeteroBufferMgr::~CpuHeteroBufferMgr() {
  clear();
}

AbstractBuffer* CpuHeteroBufferMgr::constructBuffer(const size_t chunk_page_size,
                                                    const size_t initial_size) {
  return new buffer_type(device_id_, cuda_mgr_, chunk_page_size, initial_size, mem_resource_provider_.get(MemRequirements::CAPACITY));
}
void CpuHeteroBufferMgr::destroyBuffer(AbstractBuffer* buffer) {
  buffer_type* casted_buffer = dynamic_cast<buffer_type*>(buffer);
  if (casted_buffer == 0) {
    LOG(FATAL) << "Wrong buffer type - expects base class pointer to buffer_type type.";
  }

  delete casted_buffer;
}
} // namespace Buffer_Namespace