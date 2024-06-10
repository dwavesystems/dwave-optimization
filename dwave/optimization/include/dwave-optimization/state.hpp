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

using State = typename std::vector<std::unique_ptr<NodeStateData>>;

}  // namespace dwave::optimization
