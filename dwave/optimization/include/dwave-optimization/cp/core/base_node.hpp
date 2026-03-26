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

#include <unordered_map>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/cp/core/cpvar.hpp"
#include "dwave-optimization/cp/core/engine.hpp"
#include "dwave-optimization/cp/core/interval_array.hpp"
#include "dwave-optimization/cp/core/propagator.hpp"
#include "dwave-optimization/cp/propagators/binaryop.hpp"
#include "dwave-optimization/cp/state/cpvar_state.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/nodes.hpp"

namespace dwave::optimization::cp {

enum class CPConversionSatus { OK, Failed };

CPConversionSatus parse_graph(const dwave::optimization::Graph& graph, CPModel& model) {
    CPConversionSatus status = CPConversionSatus::OK;

    // We need to have a map from node to related CPVar.
    std::unordered_map<const Node*, CPVar*> node_map;

    for (const auto& uptr : graph.nodes()) {
        const Node* ptr = uptr.get();

        if (auto iptr = dynamic_cast<const IntegerNode*>(ptr); iptr) {
            // Integer node
            CPVar* i = model.emplace_variable<CPVar>(model, ptr, ptr->topological_index());
            node_map.insert({iptr, i});

        } else if (auto aptr = dynamic_cast<const AddNode*>(ptr); aptr) {
            // Add node
            CPVar* out = model.emplace_variable<CPVar>(model, ptr, ptr->topological_index());
            node_map.insert({aptr, out});

            auto lhs_it = node_map.find(dynamic_cast<const ArrayNode*>(aptr->operands()[0]));
            auto rhs_it = node_map.find(dynamic_cast<const ArrayNode*>(aptr->operands()[1]));

            if (lhs_it == node_map.end() or rhs_it == node_map.end()){
                throw std::runtime_error("Accessing CP variables before their definition!");
            }
            model.emplace_propagator<AddPropagator>(model.num_propagators(), lhs_it->second, rhs_it->second, out);
        } else if (auto aptr = dynamic_cast<const LessEqualNode*>(ptr); aptr) {
            // Add node
            CPVar* out = model.emplace_variable<CPVar>(model, ptr, ptr->topological_index());
            node_map.insert({aptr, out});

            auto lhs_it = node_map.find(dynamic_cast<const ArrayNode*>(aptr->operands()[0]));
            auto rhs_it = node_map.find(dynamic_cast<const ArrayNode*>(aptr->operands()[1]));

            if (lhs_it == node_map.end() or rhs_it == node_map.end()){
                throw std::runtime_error("Accessing CP variables before their definition!");
            }
            model.emplace_propagator<LessEqualPropagator>(model.num_propagators(), lhs_it->second, rhs_it->second, out);
        
        } else {
            status = CPConversionSatus::Failed;
            break;
        }
    }

    return status;
}

}  // namespace dwave::optimization::cp
