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

#include "DataMgr/AbstractBufferMgr.h"

#include "HeteroMem/MemResourceProvider.h"

#include <memory>
#include <map>
#include <unordered_map>
#include <mutex>
#include <atomic>

namespace CudaMgr_Namespace {
class CudaMgr;
}
using namespace Data_Namespace;

namespace Buffer_Namespace {
class HeteroBufferMgr : public AbstractBufferMgr {
public:
  HeteroBufferMgr(const int device_id,
                  CudaMgr_Namespace::CudaMgr* cuda_mgr,
                  const size_t min_slab_size, 
                  const size_t max_slab_size,
                  const size_t page_size = 512,
                  AbstractBufferMgr* parent_mgr = nullptr);

  ~HeteroBufferMgr() override;

#ifdef HAVE_DCPMM_STORE
  AbstractBuffer* createBuffer(BufferProperty bufProp,
                               const ChunkKey& key,
                               const size_t maxRows,
                               const int sqlTypeSize,
                               const size_t pageSize = 0) override;
#endif /* HAVE_DCPMM_STORE */

  /// Creates a chunk with the specified key and page size.
  AbstractBuffer* createBuffer(
#ifdef HAVE_DCPMM
                               BufferProperty bufProp,
#endif /* HAVE_DCPMM */
                               const ChunkKey& key,
                               const size_t page_size = 0,
                               const size_t initial_size = 0) override;

  void deleteBuffer(const ChunkKey& key,
                    const bool purge = true) override;  // purge param only used in FileMgr

  void deleteBuffersWithPrefix(const ChunkKey& keyPrefix,
                               const bool purge = true) override;

  AbstractBuffer* getBuffer(
#ifdef HAVE_DCPMM
                            BufferProperty bufProp,
#endif /* HAVE_DCPMM */
                            const ChunkKey& key, const size_t numBytes = 0) override;

  void fetchBuffer(const ChunkKey& key,
                   AbstractBuffer* destBuffer,
                   const size_t numBytes = 0) override;
  
  AbstractBuffer* putBuffer(const ChunkKey& key,
                            AbstractBuffer* srcBuffer,
                            const size_t numBytes = 0) override;

  void getChunkMetadataVecForKeyPrefix(ChunkMetadataVector& chunkMetadataVec,
                                       const ChunkKey& keyPrefix) override;

  bool isBufferOnDevice(const ChunkKey& key) override;

#ifdef HAVE_DCPMM_STORE
  bool isBufferInPersistentMemory(const ChunkKey& key) override { return false; }
#endif /* HAVE_DCPMM_STORE */

  std::string printSlabs() override { return "Not Implemented"; }
  virtual void clearSlabs();
  size_t getInUseSize() override;
  size_t getAllocated() override;
  bool isAllocationCapped() override;

  void checkpoint() override;
  void checkpoint(const int db_id, const int tb_id) override;
  void removeTableRelatedDS(const int db_id, const int table_id) override;

  // Buffer API
  AbstractBuffer* alloc(const size_t numBytes = 0) override;
  void free(AbstractBuffer* buffer) override;
  size_t getNumChunks();

  size_t getPageSize() const {
    return page_size_;
  }

protected:
  using global_mutex_type = std::mutex;
  using chunk_index_mutex_type = std::mutex;
  using chunk_index_type= std::map<ChunkKey, AbstractBuffer*>;
  using reverse_index_type = std::unordered_map<AbstractBuffer*, typename chunk_index_type::iterator>;
  using chunk_index_iterator = typename chunk_index_type::iterator;

  virtual AbstractBuffer* constructBuffer(
#ifdef HAVE_DCPMM
                                          BufferProperty bufProp,
#endif /* HAVE_DCPMM */
                                          const size_t chunk_page_size,
                                          const size_t initial_size) = 0;
  virtual void destroyBuffer(AbstractBuffer* buffer) = 0;

  void clearSlabsUnlocked();

  bool removeUnpinnedBuffers(chunk_index_iterator first, chunk_index_iterator last);

  CudaMgr_Namespace::CudaMgr* cuda_mgr_;
  
  void clear();

private:
  void checkpoint(chunk_index_iterator first, chunk_index_iterator last);
  
  AbstractBufferMgr* parent_mgr_;

  global_mutex_type global_mutex_;
  chunk_index_mutex_type chunk_index_mutex_;
  chunk_index_type chunk_index_;
  reverse_index_type reverse_index_;

  std::atomic<int> max_buffer_id_;
  size_t buffer_epoch_;

protected:
  virtual void releaseSlabs() = 0;
  size_t min_slab_size_;
  size_t max_slab_size_;
  const size_t page_size_;
};
} // namespace Buffer_Namespace