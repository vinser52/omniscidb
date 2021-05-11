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
#ifndef __CUDA_ARCH__
#include "DataMgr/BufferMgr/GpuCudaBufferMgr/GpuCudaHeteroBuffer.h"

#include <thrust/device_vector.h>
#include <thrust/system_error.h>
#include <thrust/mr/allocator.h>
#include <thrust/mr/pool_options.h>
#include <thrust/mr/memory_resource.h>

namespace Buffer_Namespace {
template<typename T>
using allocator_type = thrust::mr::allocator<T, GpuCudaBufferFactory::disjoint_mr_type>;

template<typename T>
using device_vector_type = thrust::device_vector<T, allocator_type<T>>;

using GpuCudaHeteroBuffer = HeteroBuffer<device_vector_type, GpuCudaBufferFactory::disjoint_mr_type, GPU_LEVEL>;

using buffer_type = GpuCudaHeteroBuffer;

GpuCudaBufferFactory::GpuCudaBufferFactory() {
  thrust::mr::pool_options pool_opt = disjoint_mr_type::get_default_options();
#if 0
  pool_opt.largest_block_size = 2ull << 23;
  pool_opt.max_bytes_per_chunk = 2ull*1024*1024*1024;
#endif
  mr = new disjoint_mr_type(pool_opt);
}

GpuCudaBufferFactory::~GpuCudaBufferFactory() {
  delete mr;
}

void GpuCudaBufferFactory::releaseMemory() {
  mr->release();
}

Data_Namespace::AbstractBuffer* GpuCudaBufferFactory::construct(const int device_id,
                                                                CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                                                const size_t chunk_page_size,
                                                                const size_t initial_size) {
  return new buffer_type(device_id, cuda_mgr, chunk_page_size, initial_size, mr);
}

void GpuCudaBufferFactory::destroy(Data_Namespace::AbstractBuffer* buffer) {
  buffer_type* casted_buffer = dynamic_cast<buffer_type*>(buffer);
  if (casted_buffer == 0) {
    LOG(FATAL) << "Wrong buffer type - expects base class pointer to buffer_type type.";
  }

  delete casted_buffer;
}
}
#endif // __CUDA_ARCH__