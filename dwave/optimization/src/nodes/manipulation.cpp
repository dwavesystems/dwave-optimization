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

std::vector<ssize_t> make_concatenate_shape(std::vector<ArrayNode*> array_ptrs, ssize_t axis) {
    // One or more arrays must be given
    if (array_ptrs.size() < 1) {
        throw std::invalid_argument("need at least one array to concatenate");
    }

    for (auto it = std::next(array_ptrs.begin()), stop = array_ptrs.end(); it != stop; ++it) {

        // Arrays must have the same number of dimensions
        if ((*std::prev(it))->ndim() != (*it)->ndim()) {
            throw std::invalid_argument(
                    "all the input arrays must have the same number of dimensions," +
                    std::string(" but the array at index ") +
                    std::to_string(std::distance(array_ptrs.begin(), std::prev(it))) +
                    " has " + std::to_string((*std::prev(it))->ndim()) +
                    " dimension(s) and the array at index " +
                    std::to_string(std::distance(array_ptrs.begin(), it)) +
                    " has " +
                    std::to_string((*it)->ndim()) +
                    " dimension(s)");
        }

        // Array shapes must be the same except for on the concatenation axis
        for (ssize_t i = 0, stop = (*it)->ndim(); i < stop; ++i) {
            if (i != axis) {
                if ( (*std::prev(it))->shape()[i] != (*it)->shape()[i] ) {
                    throw std::invalid_argument(
                            "all the input array dimensions for the concatenation" +
                            std::string(" axis must match exactly, but along dimension ") +
                            std::to_string(i) + ", the array at index " +
                            std::to_string(std::distance(array_ptrs.begin(), std::prev(it))) +
                            " has size " +
                            std::to_string((*std::prev(it))->shape()[i]) +
                            " and the array at index " +
                            std::to_string(std::distance(array_ptrs.begin(), it)) +
                            " has size " +
                            std::to_string((*it)->shape()[i]));
                }
            }
        }
    }

    // Axis must be in range 0..ndim-1
    // We can do this check on the first input array since
    // we at this point know they all have the same shape
    if (!(0 <= axis && axis < array_ptrs.front()->ndim())) {
        throw std::invalid_argument(
                "axis " +
                std::to_string(axis) +
                std::string(" is out of bounds for array of dimension ") +
                std::to_string(array_ptrs.front()->ndim()));
    }


    // The shape of the input arrays, which will be the
    // same except for possibly on the concatenation axis
    std::span<const ssize_t> shape0 = array_ptrs.front()->shape();
    std::vector<ssize_t> shape(shape0.begin(), shape0.end());

    // On the concatenation axis we sum the axis dimension sizes
    for (auto it = std::next(array_ptrs.begin()), stop = array_ptrs.end(); it != stop; ++it) {
       shape[axis] = shape[axis] + (*it)->shape()[axis];
    }

    return shape;
}

ConcatenateNode::ConcatenateNode(std::vector<ArrayNode*> array_ptrs, const ssize_t axis)
        : ArrayOutputMixin(make_concatenate_shape(array_ptrs, axis)), axis_(axis), array_ptrs_(array_ptrs) {

    for (auto it = array_ptrs.begin(), stop = array_ptrs.end(); it != stop; it++) {
        this->add_predecessor((*it));
    }
}

double const* ConcatenateNode::buff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->buff();
}

std::span<const Update> ConcatenateNode::diff(const State& state) const {
   return data_ptr<ArrayNodeStateData>(state)->diff();
}

void ConcatenateNode::initialize_state(State& state) const {
    int index = topological_index();
    assert(index >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > index && "unexpected state length");
    assert(state[index] == nullptr && "already initialized state");

    std::vector<double> values;
    values.reserve(size());

    // Prefix is defined over dimensions 0..axis-1
    std::vector<std::vector<ssize_t>> prefix_dims(axis_);
    for (ssize_t dim = 0, stop = prefix_dims.size(); dim < stop; ++dim) {
        for (ssize_t i = 0, stop = shape()[dim]; i < stop; ++i) {
            prefix_dims[dim].push_back(i);
        }
    }

    // Suffix is defined over dimensions axis+1..ndim
    std::vector<std::vector<ssize_t>> suffix_dims(ndim() - axis_ - 1);
    for (ssize_t dim = 0, stop = suffix_dims.size(); dim < stop; ++dim) {
        for (ssize_t i = 0, stop = shape()[axis_ + 1 + dim]; i < stop; ++i) {
            suffix_dims[dim].push_back(i);
        }
    }

    auto prefix_prod = cartesian_product(prefix_dims);
    auto suffix_prod = cartesian_product(suffix_dims);

    std::vector<ssize_t> indices;
    for (auto prefix : prefix_prod) {
        for (ssize_t arr_i = 0, stop = array_ptrs_.size(); arr_i < stop; ++arr_i) {
            for (ssize_t arr_axis_i = 0, stop = array_ptrs_[arr_i]->shape()[axis_]; arr_axis_i < stop; ++arr_axis_i) {
                for (auto suffix : suffix_prod) {
                    indices.insert(indices.begin(), prefix.begin(), prefix.end());
                    indices.insert(indices.begin() + prefix_dims.size(), arr_axis_i);
                    indices.insert(indices.begin() + prefix_dims.size() + 1, suffix.begin(), suffix.end());

                    ssize_t idx = ravel_multi_index(array_ptrs_[arr_i]->strides(), indices);
                    values.emplace_back(array_ptrs_[arr_i]->buff(state)[idx]);
                    indices.clear();
                }
            }
        }
    }

    state[index] = std::make_unique<ArrayNodeStateData>(std::move(values));
}

void ConcatenateNode::commit(State& state) const {
    data_ptr<ArrayNodeStateData>(state)->commit();
}

void ConcatenateNode::revert(State& state) const {
    data_ptr<ArrayNodeStateData>(state)->revert();
}

void ConcatenateNode::propagate(State& state) const {

}

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
