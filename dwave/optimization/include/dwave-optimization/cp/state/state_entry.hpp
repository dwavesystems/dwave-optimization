#pragma once

namespace dwave::optimization::cp {
class StateEntry {
 public:
    virtual ~StateEntry() = default;
    virtual void restore() = 0;
};

}  // namespace dwave::optimization::cp