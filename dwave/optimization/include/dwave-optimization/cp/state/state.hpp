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

class StateInt : virtual public State<int64_t> {
 public:
    StateInt() : v_(0) {}
    StateInt(int value) : v_(value) {}

    int64_t get_value() override { return v_; }
    void set_value(int64_t value) override { v_ = value; }

    virtual void increment() { v_++; }
    virtual void decrement() { v_--; }

 protected:
    int v_;
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
