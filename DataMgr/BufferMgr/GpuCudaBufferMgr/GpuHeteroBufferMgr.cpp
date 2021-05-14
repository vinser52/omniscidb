/*
 * Copyright 2021 MapD Technologies, Inc.
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

#include "DataMgr/BufferMgr/GpuCudaBufferMgr/GpuCudaHeteroBuffer.h"
#include "DataMgr/BufferMgr/GpuCudaBufferMgr/GpuHeteroBufferMgr.h"

namespace Buffer_Namespace {
GpuHeteroBufferMgr::GpuHeteroBufferMgr(const int device_id,
                                       const size_t max_buffer_size,
                                       CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                       const size_t page_size,
                                       AbstractBufferMgr* parent_mgr)
    : HeteroBufferMgr(device_id, max_buffer_size, cuda_mgr, page_size, parent_mgr) {
}

GpuHeteroBufferMgr::~GpuHeteroBufferMgr() {
  clear();
}

void GpuHeteroBufferMgr::clearSlabs() {
  base_type::clearSlabs();
  bufferFactory_.releaseMemory();
}

AbstractBuffer* GpuHeteroBufferMgr::constructBuffer(
#ifdef HAVE_DCPMM
                                                    BufferProperty bufProp,
#endif /* HAVE_DCPMM */
                                                    const size_t chunk_page_size,
                                                    const size_t initial_size) {
  return bufferFactory_.construct(device_id_, cuda_mgr_, chunk_page_size, initial_size);
}

void GpuHeteroBufferMgr::destroyBuffer(AbstractBuffer* buffer) {
  bufferFactory_.destroy(buffer);
}
} // namespace Buffer_Namespace