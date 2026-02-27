#pragma once

#include "dwave-optimization/cp/state/state.hpp"
#include "dwave-optimization/cp/state/state_manager.hpp"

namespace dwave::optimization::cp {
template <class T>
class StateStack {
 public:
    // constructor
    StateStack(StateManager* sm);

    void push(T* elem);

    int size() const;
    T* get(int index) const;

 protected:
    // ownership of the objects T is shared between the objects owning the stack_
    //  std::vector<std::shared_ptr<T>> stack_;
    std::vector<T*> stack_;
    // the size_ int is owned by the state manager
    StateInt* size_;
};
}  // namespace dwave::optimization::cp
