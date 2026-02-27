#pragma once

#include "dwave-optimization/cp/core/domain_listener.hpp"
#include "dwave-optimization/cp/core/status.hpp"

namespace dwave::optimization::cp {
class DomainArray {
 public:
    virtual ~DomainArray() = default;

    virtual size_t num_domains() const = 0;

    virtual double min(int index) const = 0;
    virtual double max(int index) const = 0;
    virtual double size(int index) const = 0;
    virtual bool is_bound(int index) const = 0;
    virtual bool contains(double value, int index) const = 0;

    virtual CPStatus remove(double value, int index, DomainListener* l) = 0;
    virtual CPStatus remove_above(double value, int index, DomainListener* l) = 0;
    virtual CPStatus remove_below(double value, int index, DomainListener* l) = 0;
    virtual CPStatus remove_all_but(double value, int index, DomainListener* l) = 0;

 protected:
    // TODO: needed?
    static constexpr double PRECISION = 1e-6;
};

}  // namespace dwave::optimization::cp
