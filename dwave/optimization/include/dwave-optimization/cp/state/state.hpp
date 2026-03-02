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

#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace dwave::optimization::cp {
template <class T>
class State {
 public:
    virtual ~State() = default;

    virtual T get_value() = 0;
    virtual void set_value(T v) = 0;
};

class StateInt : virtual public State<ssize_t> {
 public:
    StateInt() : v_(0) {}
    StateInt(ssize_t value) : v_(value) {}

    ssize_t get_value() override { return v_; }
    void set_value(int64_t value) override { v_ = value; }

    virtual void increment() { v_++; }
    virtual void decrement() { v_--; }

 protected:
    ssize_t v_;
};

class StateBool : virtual public State<bool> {
 public:
    StateBool() : v_(false) {}
    StateBool(bool value) : v_(value) {}

    bool get_value() override { return v_; }
    void set_value(bool value) override { v_ = value; }

 protected:
    bool v_;
};

class StateReal : virtual public State<double> {
 public:
    StateReal() : v_(0.0) {}

    StateReal(double value) : v_(value) {}

    double get_value() override { return v_; }
    void set_value(double value) override { v_ = value; }

 protected:
    double v_;
};

}  // namespace dwave::optimization::cp
