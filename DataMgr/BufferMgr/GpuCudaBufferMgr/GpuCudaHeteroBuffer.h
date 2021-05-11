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

#pragma once
#include "DataMgr/AbstractBuffer.h"
#include "DataMgr/BufferMgr/HeteroBuffer.h"

#ifdef HAVE_CUDA
#include <thrust/system/cuda/memory_resource.h>
#include <thrust/mr/disjoint_sync_pool.h>
#include <thrust/mr/new.h>
#endif // HAVE_CUDA
namespace CudaMgr_Namespace {
class CudaMgr;
}

namespace Buffer_Namespace {
#ifdef HAVE_CUDA
class GpuCudaBufferFactory {
public:
  using memory_resource_type = thrust::system::cuda::memory_resource;
  using disjoint_mr_type = thrust::mr::disjoint_synchronized_pool_resource<memory_resource_type, thrust::mr::new_delete_resource>;

  GpuCudaBufferFactory();
  ~GpuCudaBufferFactory();
  GpuCudaBufferFactory(const GpuCudaBufferFactory&) = delete;

  void releaseMemory();
  
  Data_Namespace::AbstractBuffer* construct(const int device_id,
                                            CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                            const size_t chunk_page_size,
                                            const size_t initial_size);
  void destroy(Data_Namespace::AbstractBuffer* buffer);
private:
  disjoint_mr_type* mr;
};
#else
class GpuCudaBufferFactory {
public:
  inline void releaseMemory() {
    UNREACHABLE();
  }
  
  inline Data_Namespace::AbstractBuffer* construct(const int device_id,
                                                   CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                                   const size_t chunk_page_size,
                                                   const size_t initial_size) {
    UNREACHABLE();
    return nullptr;
  }
  inline void destroy(Data_Namespace::AbstractBuffer* buffer) {
    UNREACHABLE();
  }
};
#endif

} // namespace Buffer_Namespace