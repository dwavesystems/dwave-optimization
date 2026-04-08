// Copyright 2026 D-Wave Systems Inc.
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#pragma once

#include "dwave-optimization/cp/core/domain_listener.hpp"

namespace dwave::optimization::cp {
class TestListener : public DomainListener {
 public:
    void bind(ssize_t i) override {}
    void change(ssize_t i) override {}
    void change_max(ssize_t i) override {}
    void change_min(ssize_t i) override {}
    void change_array_size(ssize_t i) override {}
};
}  // namespace dwave::optimization::cp
