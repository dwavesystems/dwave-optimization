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

namespace dwave::optimization::cp {
class DomainListener {
 public:
    virtual ~DomainListener() = default;

    // TODO: check whether this should be removed or not
    // virtual void empty() = 0;
    virtual void bind() = 0;
    virtual void change() = 0;
    virtual void change_max() = 0;
    virtual void change_min() = 0;
};
}  // namespace dwave::optimization::cp
