// Copyright 2024 D-Wave Inc.
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
#include "dwave-optimization/nodes/inputs.hpp"
#include "dwave-optimization/state.hpp"
#include "dwave-optimization/utils.hpp"

namespace dwave::optimization {

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

// Returns whether the node pointer can be cast to one of the types in the supplied variant
template <class V, std::size_t type_index = 0>
bool variant_supports(const ArrayNode* node_ptr) {
    if constexpr (type_index < std::variant_size_v<V>) {
        if (dynamic_cast<std::variant_alternative_t<type_index, V>>(node_ptr)) {
            return true;
        }
        return variant_supports<V, type_index + 1>(node_ptr);
    }
    return false;
}

void validate_expression(const Graph& expression) {
    if (!expression.topologically_sorted()) {
        throw std::invalid_argument("Expression must be topologically sorted");
    }

    if (!expression.objective()) {
        throw std::invalid_argument(
                R"({"message": "expression must have output (objective) set"})");
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

        if (!variant_supports<NaryReduceSupportedNodes>(array_node)) {
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
}

Graph _validate_expression(Graph&& expression) {
    validate_expression(expression);
    return expression;
}

auto get_operands_shape(const Graph& expression, std::span<ArrayNode* const> operands) {
    if (operands.size() == 0) {
        throw std::invalid_argument("Must have at least one operand");
    }

    if (static_cast<ssize_t>(operands.size()) + 1 != expression.num_inputs()) {
        throw std::invalid_argument("Expression must have one more inputs than operands");
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

void validate_naryreduce_arguments(const Graph& expression,
                                   const std::vector<ArrayNode*> operands) {
    auto output = expression.objective();

    auto operand_inputs = expression.inputs().subspan(1);

    assert(operand_inputs.size() == operands.size());

    for (ssize_t op_idx = 0; op_idx < static_cast<ssize_t>(operands.size()); op_idx++) {
        if (operands[op_idx]->min() < operand_inputs[op_idx]->min()) {
            throw std::invalid_argument(
                    R"({"message": "operand with index )" + std::to_string(op_idx) +
                    R"( has minimum smaller than corresponding input in expression"})");
        } else if (operands[op_idx]->max() > operand_inputs[op_idx]->max()) {
            throw std::invalid_argument(
                    R"({"message": "operand with index )" + std::to_string(op_idx) +
                    R"( has maximum larger than corresponding input in expression"})");
        } else if (operand_inputs[op_idx]->integral() && !operands[op_idx]->integral()) {
            throw std::invalid_argument(
                    R"({"message": "operand with index )" + std::to_string(op_idx) +
                    R"( is non-integral, but corresponding input is integral"})");
        }
    }

    auto reduction_input = expression.inputs()[0];

    if (reduction_input->integral() && !output->integral()) {
        throw std::invalid_argument(
                R"({"message": "if expression output can be non-integral, last input must not be integral"})");
        ;
    } else if (output->min() < reduction_input->min()) {
        throw std::invalid_argument(
                R"({"message": "expression output must not have a lower min than the last input"})");
    } else if (output->max() > reduction_input->max()) {
        throw std::invalid_argument(
                R"({"message": "expression output must not have a higher max than the last input"})");
    }
}

Graph _validate_naryreduce_arguments(Graph&& expression, const std::vector<ArrayNode*> operands) {
    validate_naryreduce_arguments(expression, operands);
    return expression;
}

NaryReduceNode::NaryReduceNode(Graph&& expression, const std::vector<ArrayNode*>& operands,
                               double initial)
        : ArrayOutputMixin(get_operands_shape(expression, operands)),
          initial(initial),
          expression_(_validate_naryreduce_arguments(_validate_expression(std::move(expression)),
                                                     operands)),
          operands_(operands),
          output_(expression_.objective()) {
    for (const auto& op : operands_) {
        add_predecessor(op);
    }
}

double const* NaryReduceNode::buff(const State& state) const {
    return data_ptr<NaryReduceNodeData>(state)->buffer.data();
};

void NaryReduceNode::commit(State& state) const { data_ptr<NaryReduceNodeData>(state)->commit(); }

std::span<const Update> NaryReduceNode::diff(const State& state) const {
    return data_ptr<NaryReduceNodeData>(state)->diff();
}

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

    // Initialize the inputs
    for (const auto inp : expression_.inputs()) {
        double val = inp->min();
        inp->initialize_state(reg, std::span(&val, 1));
    }

    // Finish the initialization after the input states have been set
    expression_.initialize_state(reg);

    double val = initial;

    // Compute the expression for each subsequent index
    for (ssize_t index = 0; index < start_size; ++index) {
        // First input comes from the previous expression
        val = std::clamp(val, reduction_input()->min(), reduction_input()->max());
        reduction_input()->assign(reg, std::span(&val, 1));

        for (ssize_t arg_index = 0; arg_index < num_args; ++arg_index) {
            double input_val = *iterators[arg_index];
            operand_inputs()[arg_index]->assign(reg, std::span<double>(&input_val, 1));
            iterators[arg_index]++;
        }
        val = evaluate_expression(reg);
        values.push_back(val);
    }

    state[node_idx] = std::make_unique<NaryReduceNodeData>(std::move(values), std::move(iterators),
                                                           std::move(reg));
}

bool NaryReduceNode::integral() const { return output_->integral(); }

double NaryReduceNode::max() const { return output_->max(); }

double NaryReduceNode::min() const { return output_->min(); }

std::span<const InputNode* const> NaryReduceNode::operand_inputs() const {
    return expression_.inputs().subspan(1);
}

void NaryReduceNode::propagate(State& state) const {
    NaryReduceNodeData* data = data_ptr<NaryReduceNodeData>(state);
    ssize_t new_size = this->size(state);
    ssize_t num_args = operands_.size();

    data->iterators.clear();
    for (const ArrayNode* array_ptr : operands_) {
        data->iterators.push_back(array_ptr->begin(state));
    }

    double val = initial;

    for (ssize_t index = 0; index < new_size; ++index) {
        // First input comes from the previous expression
        val = std::clamp(val, reduction_input()->min(), reduction_input()->max());
        reduction_input()->assign(data->register_, std::span(&val, 1));

        for (ssize_t arg_index = 0; arg_index < num_args; ++arg_index) {
            double arg_val = *data->iterators[arg_index];
            operand_inputs()[arg_index]->assign(data->register_, std::span(&arg_val, 1));
            data->iterators[arg_index]++;
        }
        val = evaluate_expression(data->register_);
        data->set(index, val);
    }

    if (data->diff().size()) Node::propagate(state);
}

const InputNode* const NaryReduceNode::reduction_input() const { return expression_.inputs()[0]; }

void NaryReduceNode::revert(State& state) const { data_ptr<NaryReduceNodeData>(state)->revert(); }

std::span<const ssize_t> NaryReduceNode::shape(const State& state) const {
    return operands_[0]->shape(state);
}

ssize_t NaryReduceNode::size(const State& state) const { return operands_[0]->size(state); }

SizeInfo NaryReduceNode::sizeinfo() const { return operands_[0]->sizeinfo(); }

ssize_t NaryReduceNode::size_diff(const State& state) const {
    return data_ptr<NaryReduceNodeData>(state)->size_diff();
}

}  // namespace dwave::optimization
