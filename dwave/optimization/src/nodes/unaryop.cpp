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

#include "dwave-optimization/nodes/unaryop.hpp"

namespace dwave::optimization {

template class OperatorNode<op::absolute>;
template class OperatorNode<op::cos>;
template class OperatorNode<op::exp>;
template class OperatorNode<op::expit>;
template class OperatorNode<op::log>;
template class OperatorNode<op::logical>;
template class OperatorNode<op::logical_not>;
template class OperatorNode<op::negative>;
template class OperatorNode<op::rint>;
template class OperatorNode<op::sin>;
template class OperatorNode<op::square>;
template class OperatorNode<op::sqrt>;
template class OperatorNode<op::tanh>;

}  // namespace dwave::optimization
