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

#include "dwave-optimization/nodes/manipulation.hpp"

#include "_state.hpp"

namespace dwave::optimization {

ReshapeNode::ReshapeNode(ArrayNode* node_ptr, std::span<const ssize_t> shape)
        : ArrayOutputMixin(shape), array_ptr_(node_ptr) {
    // Don't (yet) support non-contiguous predecessors.
    // In some cases with non-contiguous predecessors we need to make a copy.
    // See https://github.com/dwavesystems/dwave-optimization/issues/200
    // There are also cases where we want reshape non-contiguous nodes.
    if (!array_ptr_->contiguous()) {
        throw std::invalid_argument("cannot reshape a non-contiguous array");
    }

    // Don't (yet) support dynamic predecessors.
    // We could support reshaping "down", e.g. (-1, 2) -> (-1,).
    // But we cannot support reshaping "up", e.g. (-1,) -> (-1, 2).
    // This is because in that case we would need the predecessor to grow/shrink
    // by a multiple of two each time.
    if (array_ptr_->dynamic()) {
        throw std::invalid_argument("cannot reshape a dynamic array");
    }

    // NumPy let's you use -1 in exactly one axis which is then inferred from
    // the others. We could support that in the future, including the dynamic
    // case.
    if (this->dynamic()) {
        throw std::invalid_argument("cannot reshape to a dynamic array");
    }

    if (this->size() != array_ptr_->size()) {
        // Use the same error message as NumPy
        throw std::invalid_argument("cannot reshape array of size " +
                                    std::to_string(array_ptr_->size()) + " into shape " +
                                    shape_to_string(this->shape()));
    }

    this->add_predecessor(node_ptr);
}

ReshapeNode::ReshapeNode(ArrayNode* node_ptr, std::vector<ssize_t>&& shape)
        : ReshapeNode(node_ptr, std::span(shape)) {}

double const* ReshapeNode::buff(const State& state) const { return array_ptr_->buff(state); }

void ReshapeNode::commit(State& state) const {}  // stateless node

std::span<const Update> ReshapeNode::diff(const State& state) const {
    return array_ptr_->diff(state);
}

void ReshapeNode::revert(State& state) const {}  // stateless node

class SizeNodeData : public ScalarNodeStateData {
 public:
    explicit SizeNodeData(std::integral auto value) : ScalarNodeStateData(value) {}
    void set(std::integral auto value) { ScalarNodeStateData::set(value); }
};

SizeNode::SizeNode(ArrayNode* node_ptr) : array_ptr_(node_ptr) { this->add_predecessor(node_ptr); }

double const* SizeNode::buff(const State& state) const {
    return data_ptr<SizeNodeData>(state)->buff();
}

void SizeNode::commit(State& state) const { return data_ptr<SizeNodeData>(state)->commit(); }

std::span<const Update> SizeNode::diff(const State& state) const {
    return data_ptr<SizeNodeData>(state)->diff();
}

void SizeNode::initialize_state(State& state) const {
    int index = this->topological_index();
    assert(index >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > index && "unexpected state length");
    assert(state[index] == nullptr && "already initialized state");

    state[index] = std::make_unique<SizeNodeData>(array_ptr_->size(state));
}

double SizeNode::max() const {
    // exactly the size of fixed-length predecessors
    if (!array_ptr_->dynamic()) return array_ptr_->size();

    // Ask the predecessor for its size, though in some cases it doesn't know
    // so fall back on max of ssize_t
    return array_ptr_->sizeinfo().max.value_or(std::numeric_limits<ssize_t>::max());
}

double SizeNode::min() const {
    // exactly the size of fixed-length predecessors
    if (!array_ptr_->dynamic()) return array_ptr_->size();

    // Ask the predecessor for its size, though in some cases it doesn't know
    // so fall back on 0
    return array_ptr_->sizeinfo().min.value_or(0);
}

void SizeNode::propagate(State& state) const {
    return data_ptr<SizeNodeData>(state)->set(array_ptr_->size(state));
}

void SizeNode::revert(State& state) const { return data_ptr<SizeNodeData>(state)->revert(); }

}  // namespace dwave::optimization
