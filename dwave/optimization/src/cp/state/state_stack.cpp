#include "dwave-optimization/cp/state/state_stack.hpp"

#include "dwave-optimization/cp/core/cpvar.hpp"
#include "dwave-optimization/cp/core/propagator.hpp"

namespace dwave::optimization::cp {
template <class T>
StateStack<T>::StateStack(StateManager* sm_ptr) {
    size_ = sm_ptr->make_state_int(0);
}

template <class T>
void StateStack<T>::push(T* elem) {
    int s = size_->get_value();
    if (static_cast<int>(stack_.size()) > s) {
        stack_[s] = elem;
    } else {
        stack_.push_back(elem);
    }
    size_->increment();
}

template <class T>
int StateStack<T>::size() const {
    return size_->get_value();
}

template <class T>
T* StateStack<T>::get(int index) const {
    return stack_[index];
}

template class StateStack<Propagator>;
template class StateStack<CPVar>;

}  // namespace dwave::optimization::cp