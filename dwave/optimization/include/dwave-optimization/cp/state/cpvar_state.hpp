#pragma once
#include <memory>
#include <vector>

#include "dwave-optimization/cp/core/domain_array.hpp"
#include "dwave-optimization/cp/state/state_manager.hpp"

namespace dwave::optimization::cp {

class CPVarData {
 public:
    CPVarData(StateManager* sm, ssize_t size, double lb, double ub,
              std::unique_ptr<DomainListener> listener, bool integral);

    // actions on the underlying domain
    size_t num_domains() const;

    double min(int index) const;
    double max(int index) const;
    double size(int index) const;
    bool is_bound(int index) const;
    bool contains(double value, int index) const;

    CPStatus remove(double value, int index);
    CPStatus remove_above(double value, int index);
    CPStatus remove_below(double value, int index);
    CPStatus remove_all_but(double value, int index);

 protected:
    // keeping it as a unique pointer in case we wanna have different types of domains..
    std::unique_ptr<DomainArray> domains_;
    std::unique_ptr<DomainListener> listen_;
};

using CPVarsState = std::vector<std::unique_ptr<CPVarData>>;

}  // namespace dwave::optimization::cp
