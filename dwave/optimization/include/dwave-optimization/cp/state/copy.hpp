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
#include <memory>

#include "dwave-optimization/cp/state/state.hpp"
#include "dwave-optimization/cp/state/state_entry.hpp"
#include "dwave-optimization/cp/state/storage.hpp"

namespace dwave::optimization::cp {
template <class T>
class Copy : virtual public State<T>, public Storage {
 protected:
    class CopyStateEntry : public StateEntry {
     private:
        T v_;
        Copy* copy_;

     public:
        CopyStateEntry(T v, Copy* copy) : v_(v), copy_(copy) {}
        void restore() override { copy_->set_value(v_); }
    };

 public:
    ~Copy() = default;

    // override of storage

    virtual std::unique_ptr<StateEntry> save() override;
};

class CopyInt : virtual public StateInt, virtual public Copy<int64_t> {
 public:
    // constructor
    CopyInt(int initial) : StateInt(initial) {}
    using StateInt::decrement;
    using StateInt::get_value;
    using StateInt::increment;
    using StateInt::set_value;
};

class CopyBool : virtual public StateBool, virtual public Copy<bool> {
 public:
    CopyBool(bool initial) : StateBool(initial) {}
    using StateBool::get_value;
    using StateBool::set_value;
};

class CopyReal : virtual public StateReal, virtual public Copy<double> {
 public:
    CopyReal(double initial) : StateReal(initial) {}

    using StateReal::get_value;
    using StateReal::set_value;
};

}  // namespace dwave::optimization::cp
