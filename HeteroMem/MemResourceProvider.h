// SPDX-License-Identifier: BSD-2-Clause
/* Copyright (C) 2020 Intel Corporation. */

#pragma once

#include <memory_resource>
#include <vector>
#include <string>


namespace Buffer_Namespace {

enum MemRequirements {
  CAPACITY,
  HIGH_BDWTH,
  LOW_LATENCY
};

class MemoryResourceProvider {
public:
  MemoryResourceProvider();

  MemoryResourceProvider(const std::string& pmmPath);

  std::pmr::memory_resource* get(const MemRequirements& req) const;
private:
  void initPmm(const std::string& pmmPath);

  bool isDAXPath(const std::string& path);
  
  using mem_resources_storage_type = std::vector<std::pmr::memory_resource*>;
 
  std::unique_ptr<std::pmr::memory_resource> dram_mem_resource_;
  std::unique_ptr<std::pmr::memory_resource> pmem_mem_resource_;

  mem_resources_storage_type mem_resources_;
};

} // namespace Buffer_Namespace
