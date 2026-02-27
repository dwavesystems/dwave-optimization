#pragma once

#include <memory>

#include "dwave-optimization/cp/state/state_entry.hpp"

namespace dwave::optimization::cp {
class Storage {
 public:
    // constructor should be default
    // destructor
    virtual ~Storage() = default;

    virtual std::unique_ptr<StateEntry> save() = 0;
};

}  // namespace dwave::optimization::cp
