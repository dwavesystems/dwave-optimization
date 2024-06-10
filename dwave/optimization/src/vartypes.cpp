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

#include <stdexcept>
#include <unordered_map>

#include "dwave-optimization/vartypes.hpp"

namespace dwave::optimization {

Vartype as_vartype(const std::string vartype)  {
    static std::unordered_map<std::string, Vartype> lookup = {
            {"BINARY", Vartype::BINARY},
            {"INTEGER", Vartype::INTEGER},
            {"REAL", Vartype::REAL},
    };

    auto it = lookup.find(vartype);
    if (it == lookup.end()) {
        throw std::domain_error("unknown vartype: " + vartype);
    }
    return it->second;
}

}  // namespace dwave::optimization
