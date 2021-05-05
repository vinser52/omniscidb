// SPDX-License-Identifier: BSD-2-Clause
/* Copyright (C) 2020 Intel Corporation. */

#include "HeteroMem/MemResourceProvider.h"

#include "Logger/Logger.h"

// MemKind Allocator
#include "HeteroMem/MemResources/memory_resources.h"

#include <memory>

#include "memkind.h"

namespace Buffer_Namespace {
using pmem_memory_resource_type = libmemkind::pmem::memory_resource;
using static_kind_memory_resource_type = libmemkind::static_kind::memory_resource;

MemoryResourceProvider::MemoryResourceProvider() :
    dram_mem_resource_(new static_kind_memory_resource_type(libmemkind::kinds::REGULAR)),
    mem_resources_(3, dram_mem_resource_.get())
{
  if (memkind_check_available(MEMKIND_DAX_KMEM_ALL) == MEMKIND_SUCCESS) {
    pmem_mem_resource_ = std::make_unique<static_kind_memory_resource_type>(libmemkind::kinds::DAX_KMEM_ALL);
    mem_resources_[CAPACITY] = pmem_mem_resource_.get();
    CHECK(mem_resources_[CAPACITY]);
    LOG(INFO) << "KMEM DAX nodes are detected - will use it as a capacity pool";
  }
}

MemoryResourceProvider::MemoryResourceProvider(const std::string& pmmPath) : 
    dram_mem_resource_(new static_kind_memory_resource_type(libmemkind::kinds::REGULAR)),
    mem_resources_(3, dram_mem_resource_.get())
{
    initPmm(pmmPath);
}

std::pmr::memory_resource* MemoryResourceProvider::get(const MemRequirements& req) const
{
  CHECK(req < mem_resources_.size());
  CHECK(mem_resources_[req]);
  return mem_resources_[req];
}

void MemoryResourceProvider::initPmm(const std::string& pmmPath)
{
  std::ifstream pmem_dirs_file(pmmPath);
  if (pmem_dirs_file.is_open()) {
    std::string line;
    while (!pmem_dirs_file.eof()) {
      std::getline(pmem_dirs_file, line);
      if (!line.empty()) {
        std::stringstream ss;
        std::string path;
        size_t size;

        ss << line;
        ss >> path;
        ss >> size;

        // TODO: need to support multiple directories.
        if(pmem_mem_resource_.get()) {
          LOG(FATAL) << "For now OmniSciDB does not support more than one directory for PMM volatile";
          return;
        }
        
        if (isDAXPath(path)) {
          LOG(INFO) << path << " is on DAX-enabled file system.";
        } else {
          LOG(WARNING) << path << " is not on DAX-enabled file system.";
        }

        pmem_mem_resource_ = std::make_unique<pmem_memory_resource_type>(path, size * 1024 * 1024 * 1024);
        mem_resources_[CAPACITY] = pmem_mem_resource_.get();
        CHECK(mem_resources_[CAPACITY]);
      }
    }
    pmem_dirs_file.close();
  }
  else{
    LOG(FATAL) << "Unable to open file " << pmmPath;
    return;
  }

}

bool MemoryResourceProvider::isDAXPath(const std::string& path) {
  int status = memkind_check_dax_path(path.c_str());
    if (status) {
        return false;
    }

    return true;
}

} // namespace Buffer_Namespace
