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
