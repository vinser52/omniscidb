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

#include <fstream>
#include <sstream>
#include <string>

#include "DataMgr/BufferMgr/HeteroBufferMgr.h"
#include "DataMgr/ForeignStorage/ForeignStorageException.h"

using namespace Data_Namespace;

namespace Buffer_Namespace {

HeteroBufferMgr::HeteroBufferMgr(const int device_id,
                                 CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                 const size_t min_slab_size, 
                                 const size_t max_slab_size,
                                 const size_t page_size,
                                 AbstractBufferMgr* parent_mgr)
    : AbstractBufferMgr(device_id), cuda_mgr_(cuda_mgr),  parent_mgr_(parent_mgr), max_buffer_id_(0), buffer_epoch_(0),  
      min_slab_size_(min_slab_size), max_slab_size_(max_slab_size), page_size_(page_size)
     {
}

HeteroBufferMgr::~HeteroBufferMgr() {
  // clear() should be called from the derived class destructor
}

#ifdef HAVE_DCPMM_STORE
AbstractBuffer* HeteroBufferMgr::createBuffer(BufferProperty bufProp,
                                              const ChunkKey& key,
                                              const size_t maxRows,
                                              const int sqlTypeSize,
                                              const size_t pageSize) {
  AbstractBuffer *buffer = createBuffer(bufProp, key, pageSize, maxRows * sqlTypeSize);
  buffer->setMaxRows(maxRows);
  return buffer;
}
#endif /* HAVE_DCPMM_STORE */

/// Throws a runtime_error if the Chunk already exists
AbstractBuffer* HeteroBufferMgr::createBuffer(
#ifdef HAVE_DCPMM
                                              BufferProperty bufProp,
#endif /* HAVE_DCPMM */
                                              const ChunkKey& chunk_key,
                                              const size_t chunk_page_size,
                                              const size_t initial_size) {
  size_t actual_chunk_page_size = chunk_page_size;
  if (actual_chunk_page_size == 0) {
    actual_chunk_page_size = page_size_;
  }
  std::lock_guard<chunk_index_mutex_type> index_lock(chunk_index_mutex_);
  CHECK(chunk_index_.find(chunk_key) == chunk_index_.end());

  AbstractBuffer* buffer = nullptr;
  try {
    buffer = constructBuffer(
#ifdef HAVE_DCPMM
                             bufProp,
#endif /* HAVE_DCPMM */
                             chunk_page_size, initial_size);
  } catch (const std::bad_alloc& e) {
    this->clearSlabsUnlocked();
    buffer = constructBuffer(
#ifdef HAVE_DCPMM
                             bufProp,
#endif /* HAVE_DCPMM */
                             chunk_page_size, initial_size);
  }

  CHECK(buffer != nullptr);
  auto res = chunk_index_.emplace(chunk_key, buffer);
  CHECK(res.second);
  reverse_index_.emplace(buffer, res.first);
  CHECK(chunk_index_.size() == reverse_index_.size());

  return buffer;
}

void HeteroBufferMgr::deleteBuffer(const ChunkKey& key, const bool) {
  std::unique_lock<chunk_index_mutex_type> index_lock(chunk_index_mutex_);
  auto buffer_it = chunk_index_.find(key);
  CHECK(buffer_it != chunk_index_.end());
  auto buff = buffer_it->second;
  chunk_index_.erase(buffer_it);
  auto res = reverse_index_.erase(buff);
  CHECK(res == size_t(1));
  index_lock.unlock();

  destroyBuffer(buff);
}

void HeteroBufferMgr::deleteBuffersWithPrefix(const ChunkKey& keyPrefix, const bool) {
  std::unique_lock<chunk_index_mutex_type> index_lock(chunk_index_mutex_);
  auto first = chunk_index_.lower_bound(keyPrefix);
  ChunkKey endPrefix(keyPrefix);
  ++(endPrefix.back());
  auto last = chunk_index_.lower_bound(endPrefix);
  removeUnpinnedBuffers(first, last);
}

AbstractBuffer* HeteroBufferMgr::getBuffer(
#ifdef HAVE_DCPMM
                                           BufferProperty bufProp,
#endif /* HAVE_DCPMM */
                                           const ChunkKey& key, const size_t numBytes) {
  std::lock_guard<global_mutex_type> lock(global_mutex_);  // granular lock
  std::unique_lock<chunk_index_mutex_type> index_lock(chunk_index_mutex_);
  auto chunk_it = chunk_index_.find(key);
  bool found_buffer = chunk_it != chunk_index_.end();

  if (found_buffer) {
    AbstractBuffer* buffer = chunk_it->second;
    buffer->pin();
    index_lock.unlock();
    
    // TODO: update last touched
    buffer_epoch_++;

    if (buffer->size() < numBytes) {
      parent_mgr_->fetchBuffer(key, buffer, numBytes);
    }
    return buffer;
  } else {
    index_lock.unlock();
    AbstractBuffer* buffer = createBuffer(
#ifdef HAVE_DCPMM
                                          bufProp,
#endif /* HAVE_DCPMM */
                                          key, page_size_, numBytes);
    try {
      parent_mgr_->fetchBuffer(
          key, buffer, numBytes);
    } catch (const foreign_storage::ForeignStorageException& error) {
      deleteBuffer(key);  // buffer failed to load, ensure it is cleaned up
      LOG(WARNING) << "Get chunk - Could not load chunk key: " << show_chunk(key)
                   << " from foreign storage. Error was " << error.what();
      throw error;
    } catch (std::exception& error) {
      LOG(FATAL) << "Get chunk - Could not find chunk key: " << show_chunk(key)
                 << " in buffer pool or parent buffer pools. Error was " << error.what();
    }
    return buffer;
  }
}

void HeteroBufferMgr::fetchBuffer(const ChunkKey& key,
                                     AbstractBuffer* destBuffer,
                                     const size_t numBytes) {
  AbstractBuffer* buffer = getBuffer(
#ifdef HAVE_DCPMM
                                     BufferProperty::CAPACITY,
#endif /* HAVE_DCPMM */
                                     key, numBytes);
  size_t chunk_size = numBytes == 0 ? buffer->size() : numBytes;

  destBuffer->reserve(chunk_size);
  if (buffer->isUpdated()) {
    buffer->read(destBuffer->getMemoryPtr(),
                 chunk_size,
                 0,
                 destBuffer->getType(),
                 destBuffer->getDeviceId());
  } else {
    buffer->read(destBuffer->getMemoryPtr() + destBuffer->size(),
                 chunk_size - destBuffer->size(),
                 destBuffer->size(),
                 destBuffer->getType(),
                 destBuffer->getDeviceId());
  }
  destBuffer->setSize(chunk_size);
  destBuffer->syncEncoder(buffer);
  buffer->unPin();
}
  
AbstractBuffer* HeteroBufferMgr::putBuffer(const ChunkKey& key,
                                              AbstractBuffer* srcBuffer,
                                              const size_t numBytes) {
  std::unique_lock<chunk_index_mutex_type> index_lock(chunk_index_mutex_);
  auto buffer_it = chunk_index_.find(key);
  bool found_buffer = buffer_it != chunk_index_.end();
  index_lock.unlock();
  AbstractBuffer* buffer;
  if (!found_buffer) {
    buffer = createBuffer(
#ifdef HAVE_DCPMM
                          BufferProperty::CAPACITY,
#endif /* HAVE_DCPMM */
                          key, page_size_);
  } else {
    buffer = buffer_it->second;
  }

  size_t old_buffer_size = buffer->size();
  size_t new_buffer_size = numBytes == 0 ? srcBuffer->size() : numBytes;
  CHECK(!buffer->isDirty());

  if (srcBuffer->isUpdated()) {
    //@todo use dirty flags to only flush pages of chunk that need to
    // be flushed
    buffer->write((int8_t*)srcBuffer->getMemoryPtr(),
                  new_buffer_size,
                  0,
                  srcBuffer->getType(),
                  srcBuffer->getDeviceId());
  } else if (srcBuffer->isAppended()) {
    CHECK(old_buffer_size < new_buffer_size);
    buffer->append((int8_t*)srcBuffer->getMemoryPtr() + old_buffer_size,
                   new_buffer_size - old_buffer_size,
                   srcBuffer->getType(),
                   srcBuffer->getDeviceId());
  }
  srcBuffer->clearDirtyBits();
  buffer->syncEncoder(srcBuffer);
  return buffer;
}

void HeteroBufferMgr::getChunkMetadataVecForKeyPrefix(ChunkMetadataVector&,
                                                         const ChunkKey&) {
  LOG(FATAL) << "getChunkMetadataVecForPrefix not supported for BufferMgr.";
}

bool HeteroBufferMgr::isBufferOnDevice(const ChunkKey& key) {
  std::lock_guard<chunk_index_mutex_type> index_lock(chunk_index_mutex_);
  if (chunk_index_.find(key) == chunk_index_.end()) {
    return false;
  } else {
    return true;
  }
}

void HeteroBufferMgr::clearSlabs() {
  std::lock_guard<chunk_index_mutex_type> index_lock(chunk_index_mutex_);
  clearSlabsUnlocked();
}

void HeteroBufferMgr::clearSlabsUnlocked() {
  bool pinned_exists = removeUnpinnedBuffers(chunk_index_.begin(), chunk_index_.end());
  if (!pinned_exists) {
    releaseSlabs();
  }
}

bool HeteroBufferMgr::removeUnpinnedBuffers(chunk_index_iterator first, chunk_index_iterator last) {
  bool pinned_exists = false;
  auto chunk_it = first;
  while(chunk_it != last) {
    auto buffer = chunk_it->second;
    if(buffer->getPinCount() < 1) {
      destroyBuffer(buffer);
      chunk_index_.erase(chunk_it++);
      auto ret = reverse_index_.erase(buffer);
      CHECK(ret == size_t(1));
      continue;
    } else {
      pinned_exists = true;
    }
    ++chunk_it;
  }
  return pinned_exists;
}


size_t HeteroBufferMgr::getInUseSize() {
  size_t in_use = 0;
  std::lock_guard<std::mutex> chunk_index_lock(chunk_index_mutex_);
  
  for (auto& index_it : chunk_index_) {
    in_use += index_it.second->reservedSize();
  }

  return in_use;
}

size_t HeteroBufferMgr::getAllocated() {
  // TODO: check it is correct
  return getInUseSize();
}

bool HeteroBufferMgr::isAllocationCapped() {
  // TODO: check it is correct
  return false;
}

void HeteroBufferMgr::checkpoint() {
  std::lock_guard<global_mutex_type> lock(global_mutex_);  // granular lock
  std::lock_guard<chunk_index_mutex_type> chunk_index_lock(chunk_index_mutex_); 

  checkpoint(chunk_index_.begin(), chunk_index_.end());
}

void HeteroBufferMgr::checkpoint(const int db_id, const int tb_id) {
  ChunkKey key_prefix;
  key_prefix.push_back(db_id);
  key_prefix.push_back(tb_id);

  std::lock_guard<global_mutex_type> lock(global_mutex_);  // granular lock
  std::lock_guard<std::mutex> chunk_index_lock(chunk_index_mutex_);
  auto first = chunk_index_.lower_bound(key_prefix);
  ++(key_prefix.back());
  auto last = chunk_index_.lower_bound(key_prefix);

  checkpoint(first, last);
}

void HeteroBufferMgr::checkpoint(chunk_index_iterator first, chunk_index_iterator last) {
  for (; first != last; ++first) {
    const ChunkKey &chunk_key = first->first;
    AbstractBuffer* buffer = first->second;
    // checks that buffer is actual chunk (not just buffer) and is dirty
    if (chunk_key[0] != -1 && buffer->isDirty()) {
      parent_mgr_->putBuffer(chunk_key, buffer);
      buffer->clearDirtyBits();
    }
  }
}

void HeteroBufferMgr::removeTableRelatedDS(const int db_id, const int table_id) {
  UNREACHABLE();
}

/// client is responsible for deleting memory allocated for b->mem_
AbstractBuffer* HeteroBufferMgr::alloc(const size_t num_bytes) {
  ChunkKey chunk_key = {-1, max_buffer_id_++};
  std::lock_guard<std::mutex> lock(global_mutex_);
  // TODO: Should we require CAPACITY or LOW_LATENCY or HIGH_BDWTH
  return createBuffer(
#ifdef HAVE_DCPMM
                      BufferProperty::CAPACITY,
#endif /* HAVE_DCPMM */
                      chunk_key, page_size_, num_bytes);
}

void HeteroBufferMgr::free(AbstractBuffer* buffer) {
  std::unique_lock<chunk_index_mutex_type> index_lock(chunk_index_mutex_);
  auto res = reverse_index_.find(buffer);
  CHECK(res != reverse_index_.end());
  chunk_index_.erase(res->second);
  reverse_index_.erase(res);
  index_lock.unlock();
  
  destroyBuffer(buffer);
}

size_t HeteroBufferMgr::getNumChunks() {
  std::lock_guard<chunk_index_mutex_type> index_lock(chunk_index_mutex_);
  return chunk_index_.size();
}

void HeteroBufferMgr::clear() {
  std::lock_guard<chunk_index_mutex_type> index_lock(chunk_index_mutex_);
  for(auto& chunk : chunk_index_) {
    auto buffer = chunk.second;
    destroyBuffer(buffer);
  }
  CHECK(chunk_index_.size() == reverse_index_.size());
  chunk_index_.clear();
  reverse_index_.clear();
  buffer_epoch_ = 0;
}
} // namespace Buffer_Namespace
