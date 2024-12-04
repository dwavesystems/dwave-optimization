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

#include "dwave-optimization/nodes/lambda.hpp"

#include "_state.hpp"
#include "dwave-optimization/array.hpp"
#include "dwave-optimization/nodes/constants.hpp"
#include "dwave-optimization/nodes/mathematical.hpp"
#include "dwave-optimization/state.hpp"
#include "dwave-optimization/utils.hpp"

namespace dwave::optimization {

void InputNode::initialize_state(State& state, std::span<const double> data) const {
    int index = this->topological_index();
    assert(index >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > index && "unexpected state length");
    assert(state[index] == nullptr && "already initialized state");

    if (static_cast<ssize_t>(data.size()) != this->size()) {
        throw std::invalid_argument("data size does not match size of InputNode");
    }

    std::vector<double> copy(data.begin(), data.end());

    state[index] = std::make_unique<ArrayNodeStateData>(std::move(copy));
}

double const* InputNode::buff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->buff();
}

std::span<const Update> InputNode::diff(const State& state) const noexcept {
    return data_ptr<ArrayNodeStateData>(state)->diff();
}

void InputNode::commit(State& state) const noexcept {
    data_ptr<ArrayNodeStateData>(state)->commit();
}

void InputNode::revert(State& state) const noexcept {
    data_ptr<ArrayNodeStateData>(state)->revert();
}

void InputNode::assign(State& state, std::span<const double> new_values) const {
    if (static_cast<ssize_t>(new_values.size()) != this->size()) {
        throw std::invalid_argument("size of new values must match");
    }

    double min_val = std::numeric_limits<double>::infinity();
    double max_val = -std::numeric_limits<double>::infinity();

    static double dummy = 0;
    bool all_is_integral = true;
    for (const double& v : new_values) {
        min_val = std::min(min_val, v);
        max_val = std::max(min_val, v);
        all_is_integral &= (std::modf(v, &dummy) == 0.0);
    }

    if (min_val < min()) {
        throw std::invalid_argument("new data contains a value smaller than the min");
    }
    if (max_val > max()) {
        throw std::invalid_argument("new data contains a value smaller than the min");
    }
    if (integral() && !all_is_integral) {
        throw std::invalid_argument("new data contains a non-integral value");
    }

    data_ptr<ArrayNodeStateData>(state)->assign(new_values);
}

void InputNode::assign(State& state, const std::vector<double>& new_values) const {
    this->assign(state, std::span(new_values));
}

class NaryReduceNodeData : public ArrayNodeStateData {
 public:
    explicit NaryReduceNodeData(std::vector<double>&& values,
                                std::vector<Array::const_iterator>&& iterators, State&& state)
            : ArrayNodeStateData(std::move(values)),
              iterators(std::move(iterators)),
              register_(std::move(state)) {}

    // used to avoid reallocating memory for predecessor iterators every propagation
    std::vector<Array::const_iterator> iterators;

    State register_;
};

Graph validate_expression(Graph&& expression, const std::vector<InputNode*> inputs,
                          const ArrayNode* output) {
    if (!expression.topologically_sorted()) {
        throw std::invalid_argument("Expression must be topologically sorted");
    }

    if (expression.num_decisions()) {
        // At least one decision, so the first node must be a decision
        throw std::invalid_argument(
                R"({"message": "Expression should not have any decision variables", "node_ptr": )" +
                std::to_string((uintptr_t)(void*)expression.nodes()[0].get()) + "}");
    }

    for (const auto& node_ptr : expression.nodes()) {
        const ArrayNode* array_node = dynamic_cast<const ArrayNode*>(node_ptr.get());
        if (!array_node) {
            throw std::invalid_argument(
                    R"({"message": "Expression should contain only array nodes", "node_ptr": )" +
                    std::to_string((uintptr_t)(void*)node_ptr.get()) + "}");
        }

        if (!is_variant<InputNode, ConstantNode, MaximumNode, NegativeNode, AddNode, SubtractNode,
                        MultiplyNode>(array_node)) {
            throw std::invalid_argument(
                    R"({"message": "Expression contains unsupported node", "node_ptr": )" +
                    std::to_string((uintptr_t)(void*)node_ptr.get()) + "}");
        }

        if (array_node->ndim() != 0) {
            throw std::invalid_argument(
                    R"({"message": "Expression should only contain scalars", "node_ptr": )" +
                    std::to_string((uintptr_t)(void*)node_ptr.get()) + "}");
        }
    }

    return expression;
}

auto get_operands_shape(const std::vector<InputNode*>& inputs,
                        const std::vector<double>& initial_values,
                        const std::vector<ArrayNode*>& operands) {
    if (operands.size() == 0) {
        throw std::invalid_argument("Must have at least one operand");
    }

    if (operands.size() + 1 != inputs.size()) {
        throw std::invalid_argument("Expression must have one more InputNode than operands");
    }

    if (operands.size() + 1 != initial_values.size()) {
        throw std::invalid_argument("Must have same number of initial values as operands");
    }

    std::vector<const Array*> array_ops;
    for (const ArrayNode* op : operands) {
        array_ops.push_back(op);
    }

    if (!array_shape_equal(array_ops)) {
        throw std::invalid_argument("All operands must have the same shape");
    }

    return operands[0]->shape();
}

NaryReduceNode::NaryReduceNode(Graph&& expression, const std::vector<InputNode*>& inputs,
                               const ArrayNode* output, const std::vector<double>& initial_values,
                               const std::vector<ArrayNode*>& operands)
        : ArrayOutputMixin(get_operands_shape(inputs, initial_values, operands)),
          expression_(validate_expression(std::move(expression), inputs, output)),
          inputs_(inputs),
          output_(output),
          operands_(operands),
          initial_values_(initial_values) {
    for (const auto& op : operands_) {
        add_predecessor(op);
    }
}

double const* NaryReduceNode::buff(const State& state) const {
    return data_ptr<NaryReduceNodeData>(state)->buffer.data();
};

std::span<const Update> NaryReduceNode::diff(const State& state) const {
    return data_ptr<NaryReduceNodeData>(state)->diff();
}

ssize_t NaryReduceNode::size(const State& state) const { return operands_[0]->size(state); }

std::span<const ssize_t> NaryReduceNode::shape(const State& state) const {
    return operands_[0]->shape(state);
}

ssize_t NaryReduceNode::size_diff(const State& state) const {
    return data_ptr<NaryReduceNodeData>(state)->size_diff();
}

SizeInfo NaryReduceNode::sizeinfo() const { return operands_[0]->sizeinfo(); }

bool NaryReduceNode::integral() const { return false; }

double NaryReduceNode::min() const { return -std::numeric_limits<double>::infinity(); }

double NaryReduceNode::max() const { return std::numeric_limits<double>::infinity(); }

void NaryReduceNode::commit(State& state) const { data_ptr<NaryReduceNodeData>(state)->commit(); }

double NaryReduceNode::evaluate_expression(State& register_) const {
    // First propagate all the nodes
    for (const auto& node_ptr : expression_.nodes()) {
        node_ptr->propagate(register_);
    }
    // Then commit to clear the diffs
    for (const auto& node_ptr : expression_.nodes()) {
        node_ptr->commit(register_);
    }
    return output_->view(register_)[0];
}

void NaryReduceNode::initialize_state(State& state) const {
    int node_idx = topological_index();
    assert(node_idx >= 0 && "must be topologically sorted");
    assert(state[node_idx] == nullptr && "already initialized state");

    ssize_t start_size = this->size(state);
    ssize_t num_args = operands_.size();
    std::vector<double> values;
    State reg;
    reg = expression_.empty_state();

    std::vector<Array::const_iterator> iterators;
    for (const ArrayNode* array_ptr : operands_) {
        iterators.push_back(array_ptr->begin(state));
    }

    // Get the initial output of the expression
    for (ssize_t inp_index = 0; inp_index < num_args + 1; inp_index++) {
        inputs_[inp_index]->initialize_state(reg, std::span(initial_values_).subspan(inp_index, 1));
    }

    // Finish the initialization after the input states have been set
    expression_.initialize_state(reg);

    double val = evaluate_expression(reg);

    // Compute the expression for each subsequent index
    for (ssize_t index = 0; index < start_size; ++index) {
        for (ssize_t arg_index = 0; arg_index < num_args; ++arg_index) {
            double input_val = *iterators[arg_index];
            inputs_[arg_index]->assign(reg, std::span<double>(&input_val, 1));
            iterators[arg_index]++;
        }
        // Final input comes from the previous expression
        inputs_[num_args]->assign(reg, std::span(&val, 1));
        val = evaluate_expression(reg);
        values.push_back(val);
    }

    state[node_idx] = std::make_unique<NaryReduceNodeData>(std::move(values), std::move(iterators),
                                                           std::move(reg));
}

void NaryReduceNode::propagate(State& state) const {
    NaryReduceNodeData* data = data_ptr<NaryReduceNodeData>(state);
    ssize_t new_size = this->size(state);
    ssize_t num_args = operands_.size();

    data->iterators.clear();
    for (const ArrayNode* array_ptr : operands_) {
        data->iterators.push_back(array_ptr->begin(state));
    }

    // Set inputs to the initial values
    for (ssize_t inp_index = 0; inp_index < num_args + 1; inp_index++) {
        inputs_[inp_index]->assign(data->register_,
                                   std::span(initial_values_).subspan(inp_index, 1));
    }
    double val = evaluate_expression(data->register_);

    for (ssize_t index = 0; index < new_size; ++index) {
        for (ssize_t arg_index = 0; arg_index < num_args; ++arg_index) {
            double arg_val = *data->iterators[arg_index];
            inputs_[arg_index]->assign(data->register_, std::span(&arg_val, 1));
            data->iterators[arg_index]++;
        }
        // Final input comes from the previous expression
        inputs_[num_args]->assign(data->register_, std::span(&val, 1));
        val = evaluate_expression(data->register_);
        data->set(index, val);
    }

    if (data->diff().size()) Node::propagate(state);
}

void NaryReduceNode::revert(State& state) const { data_ptr<NaryReduceNodeData>(state)->revert(); }

}  // namespace dwave::optimization
