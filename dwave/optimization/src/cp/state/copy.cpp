#include "dwave-optimization/cp/state/copy.hpp"

namespace dwave::optimization::cp {

/// TODO: I have to think about ownership of this part here
template <class T>
std::unique_ptr<StateEntry> Copy<T>::save() {
    return std::make_unique<CopyStateEntry>(this->get_value(), this);
}

template class Copy<int64_t>;
template class Copy<bool>;
template class Copy<double>;
}  // namespace dwave::optimization::cp
