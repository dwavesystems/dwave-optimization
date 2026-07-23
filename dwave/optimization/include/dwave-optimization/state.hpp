// Copyright 2023 D-Wave Systems Inc.
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

#include <cassert>
#include <memory>
#include <vector>

namespace dwave::optimization {

// Generic base class for encoding the state of the model. In general, nodes
// will subclass it to encode whatever information they need. The only universal
// data is the `mark` which used to track nodes that are seen in updates.
struct NodeStateData {
    virtual ~NodeStateData() = default;

    virtual std::unique_ptr<NodeStateData> copy() const {
        assert(typeid(*this) == typeid(NodeStateData) && "subclasses should overload copy()");
        return std::make_unique<NodeStateData>(*this);
    }

    bool mark = false;
};

// Foward declaration
class DecisionNode;

class State {
    friend class Graph;

 public:
    template <typename... Args>
    State(Args&&... args) : node_data_(std::forward<Args...>(args)...) {}

    template <typename... Args>
    void emplace_back(Args&&... args) {
        node_data_.emplace_back(std::forward<Args...>(args)...);
    }

    template <typename index_type>
    auto& operator[](index_type index) {
        return node_data_[index];
    }

    template <typename index_type>
    auto& operator[](index_type index) const {
        return node_data_[index];
    }

    template <typename size_type>
    void resize(size_type size) {
        node_data_.resize(size);
    }

    size_t size() const { return node_data_.size(); }

 private:
    std::vector<std::unique_ptr<NodeStateData>> node_data_;
    std::vector<const DecisionNode*> mutated_nodes_;
};

}  // namespace dwave::optimization
