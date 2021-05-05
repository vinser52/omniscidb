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

#include "CudaMgr/CudaMgr.h"
#include "Logger/Logger.h"
#include "DataMgr/AbstractBuffer.h"

#include <mutex>
#include <vector>
#include <memory_resource>

namespace CudaMgr_Namespace {
class CudaMgr;
}

using namespace Data_Namespace;

namespace Buffer_Namespace {
template<typename T>
inline T* get_raw(T* p) {
  return p;
}

template<typename Pointer, typename RawPtr = decltype(Pointer().get())>
inline RawPtr get_raw(const Pointer& p){
  return p.get();
}

template<template<typename> typename Storage, typename MemResource, MemoryLevel memLevel = CPU_LEVEL>
class HeteroBuffer : public AbstractBuffer {
public:
  using vector_type = Storage<int8_t>;
  using allocator_type = typename vector_type::allocator_type;
  using memory_resource_type = MemResource;

  HeteroBuffer(const int device_id,
               CudaMgr_Namespace::CudaMgr* cuda_mgr,
               const size_t page_size,
               const size_t num_bytes,
               memory_resource_type* mem_resource);

  HeteroBuffer(const int device_id,
               CudaMgr_Namespace::CudaMgr* cuda_mgr,
               const size_t page_size,
               const size_t num_bytes);

  /// Destructor
  ~HeteroBuffer() override {}

  HeteroBuffer(const HeteroBuffer&) = delete;
  HeteroBuffer& operator=(const HeteroBuffer&) = delete;

  /**
   * @brief Reads (copies) data from the buffer to the destination (dst) memory location.
   * Reads (copies) nbytes of data from the buffer, beginning at the specified byte
   * offset, into the destination (dst) memory location.
   *
   * @param dst       The destination address to where the buffer's data is being copied.
   * @param offset    The byte offset into the buffer from where reading (copying) begins.
   * @param nbytes    The number of bytes being read (copied) into the destination (dst).
   */
  void read(int8_t* const dst,
            const size_t num_bytes,
            const size_t offset = 0,
            const MemoryLevel dst_buffer_type = CPU_LEVEL,
            const int device_id = -1) override;

  /**
   * @brief Writes (copies) data from src into the buffer.
   * Writes (copies) nbytes of data into the buffer at the specified byte offset, from
   * the source (src) memory location.
   *
   * @param src        The source address from where data is being copied to the buffer.
   * @param num_bytes  The number of bytes being written (copied) into the buffer.
   * @param offset     The byte offset into the buffer to where writing begins.
   */
  void write(int8_t* src,
             const size_t num_bytes,
             const size_t offset = 0,
             const MemoryLevel src_buffer_type = CPU_LEVEL,
             const int device_id = -1) override;

  void reserve(const size_t num_bytes) override;

  void append(int8_t* src,
              const size_t num_bytes,
              const MemoryLevel src_buffer_type = CPU_LEVEL,
              const int deviceId = -1) override;

  /**
   * @brief Returns a raw, constant (read-only) pointer to the underlying buffer.
   * @return A constant memory pointer for read-only access.
   */
  inline int8_t* getMemoryPtr() override { return get_raw(buffer_.data()); };

  /// Returns the number of pages in the buffer.
  inline size_t pageCount() const override { return num_pages_; }

  /// Returns the size in bytes of each page in the buffer.
  inline size_t pageSize() const override { return page_size_; }

  /// Returns the total number of bytes allocated for the buffer.
  inline size_t reservedSize() const override { return buffer_.size(); }

  inline MemoryLevel getType() const override { return memLevel; }

  inline int pin() override {
    std::lock_guard<pin_mutex_type> pin_lock(pin_mutex_);
    return (++pin_count_);
  }

  inline int unPin() override {
    std::lock_guard<pin_mutex_type> pin_lock(pin_mutex_);
    return (--pin_count_);
  }

  inline int getPinCount() override {
    std::lock_guard<pin_mutex_type> pin_lock(pin_mutex_);
    return (pin_count_);
  }
private:
  void readData(int8_t* const dst,
                const size_t num_bytes,
                const size_t offset = 0,
                const MemoryLevel dst_buffer_type = CPU_LEVEL,
                const int dst_device_id = -1);

  void writeData(int8_t* const src,
                 const size_t num_bytes,
                 const size_t offset = 0,
                 const MemoryLevel src_buffer_type = CPU_LEVEL,
                 const int src_device_id = -1);


  using pin_mutex_type = std::mutex;

  size_t page_size_;  /// the size of each page in the buffer
  size_t num_pages_;
  int epoch_;  /// indicates when the buffer was last flushed
  
  // TODO: Should we use the same allocator as for the buffer_?
  std::vector<bool> page_dirty_flags_;
  int pin_count_;
  pin_mutex_type pin_mutex_;

  vector_type buffer_;

  CudaMgr_Namespace::CudaMgr* cuda_mgr_;
};

template<template<typename> typename Storage, typename MemResource, MemoryLevel memLevel>
HeteroBuffer<Storage, MemResource, memLevel>::HeteroBuffer(const int device_id,
                                 CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                 const size_t page_size,
                                 const size_t num_bytes,
                                 memory_resource_type* mem_resource)
                : AbstractBuffer(device_id)
                , page_size_(page_size)
                , num_pages_(0)
                , pin_count_(0)
                , buffer_(mem_resource)
                , cuda_mgr_(cuda_mgr) {
  pin();
  if (num_bytes > 0) {
    reserve(num_bytes);
  }
}

template<template<typename> typename Storage, typename MemResource, MemoryLevel memLevel>
HeteroBuffer<Storage, MemResource, memLevel>::HeteroBuffer(const int device_id,
                                 CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                 const size_t page_size,
                                 const size_t num_bytes)
                : AbstractBuffer(device_id)
                , page_size_(page_size)
                , num_pages_(0)
                , pin_count_(0)
                , cuda_mgr_(cuda_mgr) {
  pin();
  if (num_bytes > 0) {
    reserve(num_bytes);
  }
}

// TODO: should we accept AbstractBuffer instead of int8_t*
template<template<typename> typename Storage, typename MemResource, MemoryLevel memLevel>
void HeteroBuffer<Storage, MemResource, memLevel>::read(int8_t* const dst,
                           const size_t num_bytes,
                           const size_t offset,
                           const MemoryLevel dst_buffer_type,
                           const int dst_device_id) {
  if (num_bytes == 0) {
    return;
  }
  CHECK(dst && getMemoryPtr());
#ifdef BUFFER_MUTEX
  boost::shared_lock<boost::shared_mutex> read_lock(read_write_mutex_);
#endif

  if (num_bytes + offset > this->size()) {
    LOG(FATAL) << "Buffer: Out of bounds read error";
  }
  readData(dst, num_bytes, offset, dst_buffer_type, dst_device_id);
}

template<template<typename> typename Storage, typename MemResource, MemoryLevel memLevel>
void HeteroBuffer<Storage, MemResource, memLevel>::write(int8_t* src,
                            const size_t num_bytes,
                            const size_t offset,
                            const MemoryLevel src_buffer_type,
                            const int src_device_id) {
  CHECK_GT(num_bytes, size_t(0));
#ifdef BUFFER_MUTEX
  boost::unique_lock<boost::shared_mutex> write_lock(read_write_mutex_);
#endif
  if (num_bytes + offset > reservedSize()) {
    reserve(num_bytes + offset);
  }

  // write source contents to buffer
  writeData(src, num_bytes, offset, src_buffer_type, src_device_id);

  // update dirty flags for buffer and each affected page
  setDirty();
  if (offset < size_) {
    setUpdated();
  }
  if (offset + num_bytes > size_) {
    setAppended();
    size_ = offset + num_bytes;
  }

  size_t first_dirty_page = offset / page_size_;
  size_t last_dirty_page = (offset + num_bytes - 1) / page_size_;
  for (size_t i = first_dirty_page; i <= last_dirty_page; ++i) {
    page_dirty_flags_[i] = true;
  }
}

template<template<typename> typename Storage, typename MemResource, MemoryLevel memLevel>
void HeteroBuffer<Storage, MemResource, memLevel>::reserve(const size_t num_bytes) {
#ifdef BUFFER_MUTEX
  boost::unique_lock<boost::shared_mutex> write_lock(read_write_mutex_);
#endif
  size_t num_pages = (num_bytes + page_size_ - 1) / page_size_;
  if (num_pages > num_pages_) {
    buffer_.resize(page_size_ * num_pages);
    page_dirty_flags_.resize(num_pages);
    num_pages_ = num_pages;
  }
}

template<template<typename> typename Storage, typename MemResource, MemoryLevel memLevel>
void HeteroBuffer<Storage, MemResource, memLevel>::append(int8_t* src,
                             const size_t num_bytes,
                             const MemoryLevel src_buffer_type,
                             const int src_device_id) {
#ifdef BUFFER_MUTEX
  boost::shared_lock<boost::shared_mutex> read_lock(
      read_write_mutex_);  // keep another thread from getting a write lock
  boost::unique_lock<boost::shared_mutex> append_lock(
      append_mutex_);  // keep another thread from getting an append lock
#endif

  setAppended();

  if (num_bytes + size_ > reservedSize()) {
    reserve(num_bytes + size_);
  }

  writeData(src, num_bytes, size_, src_buffer_type, src_device_id);
  size_ += num_bytes;
  // Do we worry about dirty flags here or does append avoid them
}

template<template<typename> typename Storage, typename MemResource, MemoryLevel memLevel>
void HeteroBuffer<Storage, MemResource, memLevel>::readData(int8_t* const dst,
                               const size_t num_bytes,
                               const size_t offset,
                               const MemoryLevel dst_buffer_type,
                               const int dst_device_id) {
  int8_t* src = getMemoryPtr() + offset;
  if (dst_buffer_type == CPU_LEVEL) {
    memcpy(dst, src, num_bytes);
  } else if (dst_buffer_type == GPU_LEVEL) {
    CHECK_GE(dst_device_id, 0);
    cuda_mgr_->copyHostToDevice(dst, src, num_bytes, dst_device_id);
  } else {
    LOG(FATAL) << "Unsupported buffer type";
  }
}

template<template<typename> typename Storage, typename MemResource, MemoryLevel memLevel>
void HeteroBuffer<Storage, MemResource, memLevel>::writeData(int8_t* const src,
                                const size_t num_bytes,
                                const size_t offset,
                                const MemoryLevel src_buffer_type,
                                const int src_device_id) {
  CHECK(num_bytes + offset <= reservedSize());
  int8_t* dst = getMemoryPtr() + offset;
  if (src_buffer_type == CPU_LEVEL) {
    memcpy(dst, src, num_bytes);
  } else if (src_buffer_type == GPU_LEVEL) {
    // std::cout << "Writing to CPU from source GPU" << std::endl;
    CHECK_GE(src_device_id, 0);
    cuda_mgr_->copyDeviceToHost(dst, src, num_bytes, src_device_id);
  } else {
    LOG(FATAL) << "Unsupported buffer type";
  }
}

}  // namespace Buffer_Namespace
