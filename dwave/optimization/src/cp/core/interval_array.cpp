#include "dwave-optimization/cp/core/interval_array.hpp"

#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace dwave::optimization::cp {
template <>
IntervalArray<double>::IntervalArray(StateManager* sm, ssize_t size) {
    min_.resize(size);
    max_.resize(size);
    for (size_t i = 0; i < min_.size(); ++i) {
        min_[i] = sm->make_state_real(-std::numeric_limits<double>::max() / 2);
        max_[i] = sm->make_state_real(std::numeric_limits<double>::max() / 2);
    }
}

template <>
IntervalArray<int64_t>::IntervalArray(StateManager* sm, ssize_t size) {
    min_.resize(size, sm->make_state_int(0));
    max_.resize(size, sm->make_state_int((int64_t)1 << 51));

    min_.resize(size);
    max_.resize(size);
    for (size_t i = 0; i < min_.size(); ++i) {
        min_[i] = sm->make_state_int(0);
        max_[i] = sm->make_state_int((int64_t)1 << 51);
    }
}

template <>
IntervalArray<double>::IntervalArray(StateManager* sm, ssize_t size, double lb, double ub) {
    if (lb > ub) {
        throw std::invalid_argument("lower bound larger than upper bound");
    }

    min_.resize(size);
    max_.resize(size);

    for (size_t i = 0; i < min_.size(); ++i) {
        min_[i] = sm->make_state_real(lb);
        max_[i] = sm->make_state_real(ub);
    }
}

template <>
IntervalArray<int64_t>::IntervalArray(StateManager* sm, ssize_t size, double lb, double ub) {
    if (lb < std::numeric_limits<int64_t>::min())
        throw std::invalid_argument("lower bound too small for int64_t");
    if (ub > std::numeric_limits<int64_t>::max())
        throw std::invalid_argument("upper bound too big for int64_t");

    if (std::ceil(lb) > std::floor(ub)) {
        throw std::invalid_argument("lower bound larger than upper bound");
    }

    min_.resize(size);
    max_.resize(size);

    for (size_t i = 0; i < min_.size(); ++i) {
        min_[i] = sm->make_state_int(std::ceil(lb));
        max_[i] = sm->make_state_int(std::floor(ub));
    }
}

template <>
IntervalArray<double>::IntervalArray(StateManager* sm, std::vector<double> lb,
                                     std::vector<double> ub) {
    if (lb.size() != ub.size()) {
        throw std::invalid_argument("lower bounds and upper bounds have different sizes");
    }

    min_.resize(lb.size());
    max_.resize(ub.size());
    for (size_t i = 0; i < lb.size(); ++i) {
        if (lb[i] > ub[i]) {
            throw std::invalid_argument("lower bound larger than upper bound");
        }

        min_[i] = sm->make_state_real(lb[i]);
        max_[i] = sm->make_state_real(ub[i]);
    }
}

template <>
IntervalArray<int64_t>::IntervalArray(StateManager* sm, std::vector<double> lb,
                                      std::vector<double> ub) {
    if (lb.size() != ub.size()) {
        throw std::invalid_argument("lower bounds and upper bounds have different sizes");
    }

    min_.resize(lb.size());
    max_.resize(ub.size());

    for (size_t i = 0; i < lb.size(); ++i) {
        if (lb[i] < std::numeric_limits<int64_t>::min())
            throw std::invalid_argument("lower bound too small for int64_t");
        if (ub[i] > std::numeric_limits<int64_t>::max())
            throw std::invalid_argument("upper bound too big for int64_t");
        if (lb[i] > ub[i]) throw std::invalid_argument("lower bound larger than upper bound");
        min_[i] = sm->make_state_int(std::ceil(lb[i]));
        max_[i] = sm->make_state_int(std::floor(ub[i]));
    }
}

template <typename T>
double IntervalArray<T>::min(int index) const {
    assert(index < min_.size());
    return min_[index]->get_value();
}

template <typename T>
double IntervalArray<T>::max(int index) const {
    assert(index < max_.size());
    return max_[index]->get_value();
}

template <>
double IntervalArray<double>::size(int index) const {
    assert(index < min_.size());
    return max_[index]->get_value() - min_[index]->get_value();
}

template <>
double IntervalArray<int64_t>::size(int index) const {
    assert(index < min_.size());
    return max_[index]->get_value() - min_[index]->get_value() + 1;
}

template <>
bool IntervalArray<double>::is_bound(int index) const {
    return (this->size(index) == 0);
}

template <>
bool IntervalArray<int64_t>::is_bound(int index) const {
    return (this->size(index) == 1);
}

template <>
bool IntervalArray<double>::contains(double value, int index) const {
    if (value > max_[index]->get_value()) return false;
    if (value < min_[index]->get_value()) return false;
    return true;
}

template <>
bool IntervalArray<int64_t>::contains(double value, int index) const {
    if (value > max_[index]->get_value()) return false;
    if (value < min_[index]->get_value()) return false;
    if (std::floor(value) != std::ceil(value)) return false;
    return true;
}

template <>
CPStatus IntervalArray<double>::remove(double value, int index, DomainListener* l) {
    // can't really remove a double, even from the boundary
    return CPStatus::OK;
}

template <>
CPStatus IntervalArray<int64_t>::remove(double value, int index, DomainListener* l) {
    // can only remove from the boundary
    if (this->contains(value, index)) {
        // if there's only this value, then domain is wiped out
        if (this->is_bound(index)) return CPStatus::Inconsistency;

        bool change_max = value == max_[index]->get_value();
        bool change_min = value == min_[index]->get_value();

        if (change_max) {
            max_[index]->set_value(value - 1);
            l->change_max();
        } else if (change_min) {
            min_[index]->set_value(value + 1);
            l->change_min();
        }
    }
    return CPStatus::OK;
}

template <>
CPStatus IntervalArray<double>::remove_above(double value, int index, DomainListener* l) {
    // wipe-out all domain
    if (min_[index]->get_value() > value) return CPStatus::Inconsistency;

    // nothing to do
    if (max_[index]->get_value() <= value) return CPStatus::OK;

    max_[index]->set_value(value);
    l->change_max();
    return CPStatus::OK;
}

template <>
CPStatus IntervalArray<int64_t>::remove_above(double value, int index, DomainListener* l) {
    // wipe-out all domain
    if (min_[index]->get_value() > value) return CPStatus::Inconsistency;

    // nothing to do
    if (max_[index]->get_value() <= value) return CPStatus::OK;

    max_[index]->set_value(std::floor(value));
    l->change_max();
    return CPStatus::OK;
}

template <>
CPStatus IntervalArray<double>::remove_below(double value, int index, DomainListener* l) {
    // wipe-out all domain
    if (max_[index]->get_value() < value) return CPStatus::Inconsistency;

    // nothing to do
    if (min_[index]->get_value() >= value) return CPStatus::OK;

    min_[index]->set_value(value);
    l->change_min();
    return CPStatus::OK;
}

template <>
CPStatus IntervalArray<int64_t>::remove_below(double value, int index, DomainListener* l) {
    // wipe-out all domain
    if (max_[index]->get_value() < value) return CPStatus::Inconsistency;

    // nothing to do
    if (min_[index]->get_value() >= value) return CPStatus::OK;

    min_[index]->set_value(std::ceil(value));
    l->change_min();
    return CPStatus::OK;
}

template <typename T>
CPStatus IntervalArray<T>::remove_all_but(double value, int index, DomainListener* l) {
    // if the value is not contained, wipe-out
    if (not this->contains(value, index)) return CPStatus::Inconsistency;

    bool changed_min = value = min_[index]->get_value();
    bool changed_max = value = max_[index]->get_value();

    min_[index]->set_value(value);
    max_[index]->set_value(value);

    if (changed_max) l->change_max();
    if (changed_min) l->change_min();

    return CPStatus::OK;
}

template class IntervalArray<double>;
template class IntervalArray<int64_t>;

}  // namespace dwave::optimization::cp
