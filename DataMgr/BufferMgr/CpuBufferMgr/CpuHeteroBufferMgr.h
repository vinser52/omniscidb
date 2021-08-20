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


namespace CudaMgr_Namespace {
class CudaMgr;
}

namespace Buffer_Namespace {

class CpuHeteroBufferMgr : public HeteroBufferMgr {
public:
  CpuHeteroBufferMgr(const int device_id,
                     const size_t max_buffer_size,
                     const std::string& pmm_path,
                     CudaMgr_Namespace::CudaMgr* cuda_mgr,
                     const size_t page_size = 512,
                     AbstractBufferMgr* parent_mgr = nullptr);

  CpuHeteroBufferMgr(const int device_id,
                     const size_t max_buffer_size,
                     CudaMgr_Namespace::CudaMgr* cuda_mgr,
                     const size_t page_size = 512,
                     AbstractBufferMgr* parent_mgr = nullptr);

  ~CpuHeteroBufferMgr() override;

  inline MgrType getMgrType() override { return CPU_MGR; }
  inline std::string getStringMgrType() override { return ToString(CPU_MGR); }

protected:
  using buffer_type = CpuHeteroBuffer;
  AbstractBuffer* constructBuffer(const size_t chunk_page_size,
                                  const size_t initial_size) override;
  void destroyBuffer(AbstractBuffer* buffer) override;
private:
  MemoryResourceProvider mem_resource_provider_;
};
} // namespace Buffer_Namespace
