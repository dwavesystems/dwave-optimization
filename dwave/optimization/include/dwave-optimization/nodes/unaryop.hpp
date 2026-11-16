// Copyright 2025 D-Wave
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

#include "dwave-optimization/nodes/operator.hpp"
#include "dwave-optimization/operator.hpp"

namespace dwave::optimization {

using AbsoluteNode = OperatorNode<op::absolute>;
extern template class OperatorNode<op::absolute>;

using CosNode = OperatorNode<op::cos>;
extern template class OperatorNode<op::cos>;

using ExpNode = OperatorNode<op::exp>;
extern template class OperatorNode<op::exp>;

using ExpitNode = OperatorNode<op::expit>;
extern template class OperatorNode<op::expit>;

using LogNode = OperatorNode<op::log>;
extern template class OperatorNode<op::log>;

using LogicalNode = OperatorNode<op::logical>;
extern template class OperatorNode<op::logical>;

using NegativeNode = OperatorNode<op::negative>;
extern template class OperatorNode<op::negative>;

using NotNode = OperatorNode<op::logical_not>;
extern template class OperatorNode<op::logical_not>;

using RintNode = OperatorNode<op::rint>;
extern template class OperatorNode<op::rint>;

using SinNode = OperatorNode<op::sin>;
extern template class OperatorNode<op::sin>;

using SqrtNode = OperatorNode<op::sqrt>;
extern template class OperatorNode<op::sqrt>;

using SquareNode = OperatorNode<op::square>;
extern template class OperatorNode<op::square>;

using TanhNode = OperatorNode<op::tanh>;
extern template class OperatorNode<op::tanh>;

}  // namespace dwave::optimization
