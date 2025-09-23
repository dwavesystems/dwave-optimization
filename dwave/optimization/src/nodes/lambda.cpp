// Copyright 2025 D-Wave Inc.
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

namespace dwave::optimization {

class AccumulateZipNodeData : public ArrayNodeStateData {
 public:
    explicit AccumulateZipNodeData(std::vector<double>&& values,
                                   std::vector<Array::const_iterator>&& iterators, State&& state)
            : ArrayNodeStateData(std::move(values)),
              iterators(std::move(iterators)),
              register_(std::move(state)) {}

    // used to avoid reallocating memory for predecessor iterators every propagation
    std::vector<Array::const_iterator> iterators;

    State register_;
};

AccumulateZipNode::AccumulateZipNode(std::shared_ptr<Graph> expression_ptr,
                                     const std::vector<ArrayNode*>& operands,
                                     array_or_double initial)
        : ArrayOutputMixin(operands.empty() ? std::span<ssize_t, 0>() : operands[0]->shape()),
          initial(initial),
          expression_ptr_(std::move(expression_ptr)),
          operands_(operands),
          sizeinfo_(operands.empty()? SizeInfo(0) : operands_[0]->sizeinfo()) {
    check(*expression_ptr_, operands, initial);

    if (std::holds_alternative<ArrayNode*>(initial)) {
        // was checked in check() method
        add_predecessor(std::get<ArrayNode*>(initial));
    }
    for (const auto& op : operands_) {
        add_predecessor(op);
    }
}

double const* AccumulateZipNode::buff(const State& state) const {
    return data_ptr<AccumulateZipNodeData>(state)->buff();
}

void AccumulateZipNode::check(const Graph& expression, std::span<const ArrayNode* const> operands,
                              array_or_double initial) {
    // First, let's check that the expression is valid
    {
        if (!expression.topologically_sorted()) {
            throw std::invalid_argument("expression must be topologically sorted");
        }

        if (!expression.objective()) {
            throw std::invalid_argument("expression must have an objective set");
        }

        if (expression.num_decisions()) {
            throw std::invalid_argument("expression must not have any decisions");
        }

        if (expression.num_inputs() < 2) {
            throw std::invalid_argument("expression must have at least two inputs");
        }

        for (const auto& node_uptr : expression.nodes()) {
            const Node* const node_ptr = node_uptr.get();
            const ArrayNode* const array_ptr = dynamic_cast<const ArrayNode*>(node_uptr.get());

            if (!array_ptr) {
                throw std::invalid_argument(
                        std::string("expression must contain only array nodes, ") +
                        node_ptr->repr() + " is not supported");
            }

            if (!supported_node_types::add_const::add_pointer::check(node_ptr)) {
                throw std::invalid_argument(
                        std::string("expression contains an unsupported node type, ") +
                        node_ptr->repr() + " is not supported");
            }

            if (array_ptr->ndim() != 0) {
                throw std::invalid_argument(
                        std::string("expression nodes should all be scalars, ") + node_ptr->repr() +
                        " is " + std::to_string(array_ptr->ndim()) + " dimensional");
            }
        }
    }

    // Check the shape of the initial value
    if (std::holds_alternative<ArrayNode*>(initial)) {
        ArrayNode* initial_node = std::get<ArrayNode*>(initial);
        if (initial_node->ndim() != 0) {
            throw std::invalid_argument(
                    "when using a node for the initial value, it must have scalar output");
        }
    }

    // Now we need to check whether the expression is consistent with the operands
    // and the operands are consistent with eachother
    {
        if (operands.empty()) {
            throw std::invalid_argument("must have at least one operands");
        }

        if (static_cast<ssize_t>(operands.size()) + 1 != expression.num_inputs()) {
            throw std::invalid_argument(
                    std::string("expression must have one more input than operands, ") +
                    "expression.num_inputs()=" + std::to_string(expression.num_inputs()) +
                    ", operands.size()=" + std::to_string(operands.size()));
        }

        // For array_shape_equal we need Array*, not ArrayNode* so we do a copy.
        // In the future we should consider another overload
        std::vector<const Array*> array_operands;
        for (const ArrayNode* op : operands) array_operands.emplace_back(op);
        if (!array_shape_equal(array_operands)) {
            throw std::invalid_argument("all operands must have the same shape");
        }

        // Make sure the min/max are consistent
        auto operand_inputs = expression.inputs().subspan(1);
        for (ssize_t op_idx = 0, stop = operands.size(); op_idx < stop; ++op_idx) {
            // make sure the values provided by the array don't exceed the values allowed by
            // the expression
            const auto outmin = operands[op_idx]->min();
            const auto outmax = operands[op_idx]->max();
            const auto inmin = operand_inputs[op_idx]->min();
            const auto inmax = operand_inputs[op_idx]->max();

            if (outmin < inmin) {
                throw std::invalid_argument(std::string("the ") + std::to_string(op_idx) +
                                            "th operand has minimum smaller than the corresponding "
                                            "input in the expression");
            } else if (outmax > inmax) {
                throw std::invalid_argument(std::string("the ") + std::to_string(op_idx) +
                                            "th operand has maximum larger than the corresponding "
                                            "input in the expression");
            } else if (operand_inputs[op_idx]->integral() && !operands[op_idx]->integral()) {
                throw std::invalid_argument(
                        std::string("the ") + std::to_string(op_idx) +
                        "th operand is not integral, but the corresponding input is integral");
            }
        }

        const auto expmin = expression.objective()->min();
        const auto expmax = expression.objective()->max();
        const auto accmin = expression.inputs()[0]->min();
        const auto accmax = expression.inputs()[0]->max();
        if (expression.inputs()[0]->integral() && !expression.objective()->integral()) {
            throw std::invalid_argument(
                    "if expression output can be non-integral, first input must not be integral");
        } else if (expmin < accmin) {
            throw std::invalid_argument(
                    "expression output must not have a lower min than the first input");
        } else if (expmax > accmax) {
            throw std::invalid_argument(
                    "expression output must not have a higher max than the first input");
        }

        if (std::holds_alternative<ArrayNode*>(initial)) {
            const auto initmin = std::get<ArrayNode*>(initial)->min();
            const auto initmax = std::get<ArrayNode*>(initial)->max();
            if (initmin < accmin) {
                throw std::invalid_argument(
                        "initial value must not have a lower min than the first input");
            }
            if (initmax > accmax) {
                throw std::invalid_argument(
                        "initial value must not have a higher max than the first input");
            }
        } else {
            assert(std::holds_alternative<double>(initial));
            if (std::get<double>(initial) < accmin) {
                throw std::invalid_argument("initial value must not be lower than the first input");
            }
            if (std::get<double>(initial) > accmax) {
                throw std::invalid_argument(
                        "initial value must not be greater than the first input");
            }
        }
    }
}

void AccumulateZipNode::commit(State& state) const {
    data_ptr<AccumulateZipNodeData>(state)->commit();
}

std::span<const Update> AccumulateZipNode::diff(const State& state) const {
    return data_ptr<AccumulateZipNodeData>(state)->diff();
}

double AccumulateZipNode::evaluate_expression(State& register_) const {
    // First propagate all the nodes
    for (const auto& node_ptr : expression_ptr_->nodes()) {
        node_ptr->propagate(register_);
    }
    // Then commit to clear the diffs
    for (const auto& node_ptr : expression_ptr_->nodes()) {
        node_ptr->commit(register_);
    }
    return expression_ptr_->objective()->view(register_)[0];
}

double AccumulateZipNode::get_initial_value(const State& state) const {
    if (std::holds_alternative<double>(initial)) {
        return std::get<double>(initial);
    } else {
        return std::get<ArrayNode*>(initial)->view(state)[0];
    }
}

void AccumulateZipNode::initialize_state(State& state) const {
    int node_idx = topological_index();
    assert(node_idx >= 0 && "must be topologically sorted");
    assert(state[node_idx] == nullptr && "already initialized state");

    ssize_t start_size = this->size(state);
    ssize_t num_args = operands_.size();
    std::vector<double> values;
    State reg;
    reg = expression_ptr_->empty_state();

    std::vector<Array::const_iterator> iterators;
    for (const ArrayNode* array_ptr : operands_) {
        iterators.push_back(array_ptr->begin(state));
    }

    // Initialize the inputs
    for (const auto inp : expression_ptr_->inputs()) {
        double val = inp->min();
        inp->initialize_state(reg, std::span(&val, 1));
    }

    // Finish the initialization after the input states have been set
    expression_ptr_->initialize_state(reg);

    double val = get_initial_value(state);

    // Compute the expression for each subsequent index
    for (ssize_t index = 0; index < start_size; ++index) {
        // First input comes from the previous expression
        val = std::clamp(val, accumulate_input()->min(), accumulate_input()->max());
        accumulate_input()->assign(reg, std::span(&val, 1));

        for (ssize_t arg_index = 0; arg_index < num_args; ++arg_index) {
            double input_val = *iterators[arg_index];
            operand_inputs()[arg_index]->assign(reg, std::span<double>(&input_val, 1));
            iterators[arg_index]++;
        }
        val = evaluate_expression(reg);
        values.push_back(val);
    }

    state[node_idx] = std::make_unique<AccumulateZipNodeData>(std::move(values),
                                                              std::move(iterators), std::move(reg));
}

bool AccumulateZipNode::integral() const { return expression_ptr_->objective()->integral(); }

double AccumulateZipNode::max() const { return expression_ptr_->objective()->max(); }

double AccumulateZipNode::min() const { return expression_ptr_->objective()->min(); }

std::span<const InputNode* const> AccumulateZipNode::operand_inputs() const {
    return expression_ptr_->inputs().subspan(1);
}

void AccumulateZipNode::propagate(State& state) const {
    AccumulateZipNodeData* data = data_ptr<AccumulateZipNodeData>(state);
    ssize_t new_size = this->size(state);
    ssize_t num_args = operands_.size();

    data->iterators.clear();
    for (const ArrayNode* array_ptr : operands_) {
        data->iterators.push_back(array_ptr->begin(state));
    }

    double val = get_initial_value(state);

    for (ssize_t index = 0; index < new_size; ++index) {
        // First input comes from the previous expression
        val = std::clamp(val, accumulate_input()->min(), accumulate_input()->max());
        accumulate_input()->assign(data->register_, std::span(&val, 1));

        for (ssize_t arg_index = 0; arg_index < num_args; ++arg_index) {
            double arg_val = *data->iterators[arg_index];
            operand_inputs()[arg_index]->assign(data->register_, std::span(&arg_val, 1));
            data->iterators[arg_index]++;
        }
        val = evaluate_expression(data->register_);

        if (index < data->size()) {
            data->set(index, val);
        } else if (index == data->size()) {
            data->emplace_back(val);
        } else {
            assert(false && "index is too large for current buffer");
            unreachable();
        }
    }

    data->trim_to(new_size);

    if (data->diff().size()) Node::propagate(state);
}

const InputNode* const AccumulateZipNode::accumulate_input() const {
    return expression_ptr_->inputs()[0];
}

void AccumulateZipNode::revert(State& state) const {
    data_ptr<AccumulateZipNodeData>(state)->revert();
}

std::span<const ssize_t> AccumulateZipNode::shape(const State& state) const {
    return operands_[0]->shape(state);
}

ssize_t AccumulateZipNode::size(const State& state) const { return operands_[0]->size(state); }

SizeInfo AccumulateZipNode::sizeinfo() const { return this->sizeinfo_; }

ssize_t AccumulateZipNode::size_diff(const State& state) const {
    return data_ptr<AccumulateZipNodeData>(state)->size_diff();
}

}  // namespace dwave::optimization
