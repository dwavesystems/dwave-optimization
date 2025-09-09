// Copyright 2024 D-Wave
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

#include "dwave-optimization/nodes/constants.hpp"

#include <algorithm>

namespace dwave::optimization {

ValuesInfo calculate_values_info(std::span<const double> buffer) {
    if (buffer.empty()) {
        return ValuesInfo(0.0, 0.0, true);
    }

    return ValuesInfo(std::ranges::min(buffer), std::ranges::max(buffer),
                      std::ranges::all_of(buffer, is_integer));
}

ConstantNode::ConstantNode(const double* data_ptr, std::initializer_list<ssize_t> shape)
        : ArrayOutputMixin(shape),
          buffer_ptr_(data_ptr),
          values_info_(calculate_values_info(std::span<const double>(buffer_ptr_, this->size()))) {}

ConstantNode::ConstantNode(const double* data_ptr, const std::span<const ssize_t> shape)
        : ArrayOutputMixin(shape),
          buffer_ptr_(data_ptr),
          values_info_(calculate_values_info(std::span<const double>(buffer_ptr_, this->size()))) {}

ConstantNode::ConstantNode(std::unique_ptr<DataSource> data_source, const double* data_ptr,
                           const std::span<const ssize_t> shape)
        : ArrayOutputMixin(shape),
          buffer_ptr_(data_ptr),
          values_info_(calculate_values_info(std::span<const double>(buffer_ptr_, this->size()))),
          data_source_(std::move(data_source)) {}

ConstantNode::ConstantNode(OwningDataSource&& data_source, std::initializer_list<ssize_t> shape)
        : ArrayOutputMixin(shape),
          buffer_ptr_(data_source.get()),
          values_info_(calculate_values_info(std::span<const double>(buffer_ptr_, this->size()))),
          data_source_(std::make_unique<OwningDataSource>(std::move(data_source))) {}

ConstantNode::ConstantNode(OwningDataSource&& data_source, const std::span<const ssize_t> shape)
        : ArrayOutputMixin(shape),
          buffer_ptr_(data_source.get()),
          values_info_(calculate_values_info(std::span<const double>(buffer_ptr_, this->size()))),
          data_source_(std::make_unique<OwningDataSource>(std::move(data_source))) {}

bool ConstantNode::integral() const { return this->values_info_.integral; }

double ConstantNode::min() const { return this->values_info_.min; }

double ConstantNode::max() const { return this->values_info_.max; }

void ConstantNode::update(State& state, int index) const {
    throw std::logic_error("update() called on a constant");
}

}  // namespace dwave::optimization
