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

template class UnaryOpNode<functional::absolute>;
template class UnaryOpNode<functional::cos>;
template class UnaryOpNode<functional::exp>;
template class UnaryOpNode<functional::expit>;
template class UnaryOpNode<functional::log>;
template class UnaryOpNode<functional::logical>;
template class UnaryOpNode<functional::logical_not>;
template class UnaryOpNode<functional::negative>;
template class UnaryOpNode<functional::rint>;
template class UnaryOpNode<functional::sin>;
template class UnaryOpNode<functional::square>;
template class UnaryOpNode<functional::square_root>;
template class UnaryOpNode<functional::tanh>;

}  // namespace dwave::optimization
