// Copyright 2026 D-Wave Systems Inc.
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#include "dwave-optimization/cp/core/sparse_set_array.hpp"

#include <cassert>

namespace dwave::optimization::cp {
SparseSetArray::SparseSetArray(StateManager* sm, ssize_t min_size, ssize_t max_size, ssize_t lb,
                               ssize_t ub) {
    assert(min_size <= max_size);
    assert(lb <= lb);

    n_.resize(max_size, ub - lb + 1);
    offsets_.resize(max_size, lb);
    indices_.resize(max_size);
    values_.resize(max_size);

    for (ssize_t i = 0; i < max_size; ++i) {
        min_.push_back(sm->make_state_int(0));
        max_.push_back(sm->make_state_int(ub - lb));
        size_.push_back(sm->make_state_int(ub - lb + 1));
        indices_[i].resize(n_[i]);
        values_[i].resize(n_[i]);
        std::iota(values_[i].begin(), values_[i].end(), 0);
        std::iota(indices_[i].begin(), indices_[i].end(), 0);
    }

    min_size_ = sm->make_state_int(min_size);
    max_size_ = sm->make_state_int(max_size);
}

double SparseSetArray::min(int index) const {
    assert(index < static_cast<ssize_t>(num_domains()));
    return offsets_[index] + min_[index]->get_value();
}

double SparseSetArray::max(int index) const {
    assert(index < static_cast<ssize_t>(num_domains()));
    return offsets_[index] + max_[index]->get_value();
}

double SparseSetArray::size(int index) const {
    assert(index < static_cast<ssize_t>(num_domains()));
    return size_[index]->get_value();
}

bool SparseSetArray::is_bound(int index) const { return size(index) == 1 and is_active(index); }

bool SparseSetArray::contains(double val, int index) const {
    assert(index < static_cast<ssize_t>(num_domains()));
    val -= offsets_[index];
    if (val > max_[index]->get_value()) return false;
    if (val < min_[index]->get_value()) return false;
    if (std::floor(val) != std::ceil(val)) return false;
    ssize_t value = static_cast<ssize_t>(val);
    return indices_[index][value] < size_[index]->get_value();
}

bool SparseSetArray::is_empty(int index) const { return size(index) == 0; }

bool SparseSetArray::is_active(int index) const {
    assert(index < static_cast<ssize_t>(num_domains()));
    return index < min_size_->get_value();
}

bool SparseSetArray::maybe_active(int index) const {
    assert(index < static_cast<ssize_t>(num_domains()));
    return index < max_size_->get_value();
}

void SparseSetArray::exchange_positions(int index, ssize_t val1, ssize_t val2) {
    assert(index < static_cast<ssize_t>(num_domains()));
    int v1 = val1;
    int v2 = val2;
    int i1 = indices_[index][v1];
    int i2 = indices_[index][v2];

    values_[index][i1] = v2;
    values_[index][i2] = v1;
    indices_[index][v1] = i2;
    indices_[index][v2] = i1;
}

CPStatus SparseSetArray::remove(double value, int index, DomainListener* l) {
    assert(index < static_cast<ssize_t>(num_domains()));

    if (not this->contains(value, index)) return CPStatus::OK;

    bool max_changed = this->max(index) == value;
    bool min_changed = this->min(index) == value;

    ssize_t val = static_cast<ssize_t>(value);

    // Now remove the element from the sparse set
    val -= offsets_[index];

    ssize_t s = size(index);
    exchange_positions(index, val, values_[index][s - 1]);
    size_[index]->decrement();
    update_bounds_val_removed(index, val);

    // This is an inconsistency
    if (size_[index]->get_value() == 0 and this->is_active(index)) return CPStatus::Inconsistency;

    // If we removed optional values, then update the max size
    if (size_[index]->get_value() == 0) return this->update_max_size(index, l);

    l->change(index);

    if (max_changed) l->change_max(index);
    if (min_changed) l->change_min(index);

    if (size_[index]->get_value() == 1 and this->is_active(index)) l->bind(index);

    return CPStatus::OK;
}

CPStatus SparseSetArray::remove_above(double value, int index, DomainListener* l) {
    ssize_t val = static_cast<ssize_t>(std::floor(value));

    if (this->max(index) > val) {
        if (this->min(index) > val) {
            size_[index]->set_value(0);
        } else {
            ssize_t vmx = this->max(index);
            for (ssize_t v = vmx; v > val; --v) {
                if (CPStatus status = this->remove(v, index, l); not status)
                    return CPStatus::Inconsistency;
            }
        }

        int s = size_[index]->get_value();
        switch (s) {
            case 0:
                if (this->is_active(index)) return CPStatus::Inconsistency;
                if (this->maybe_active(index)) return this->update_max_size(index, l);
                break;

            case 1:
                l->bind(index);
                [[fallthrough]];

            default:
                l->change_min(index);
                l->change(index);
                break;
        }
    }

    return CPStatus::OK;
}

CPStatus SparseSetArray::remove_below(double value, int index, DomainListener* l) {
    ssize_t val = static_cast<ssize_t>(std::ceil(value));
    if (this->min(index) < val) {
        if (this->max(index) < val) {
            // remove all values!
            size_[index]->set_value(0);
        } else {
            for (ssize_t v = this->min(index); v < val; ++v) {
                if (CPStatus status = this->remove(v, index, l); not status)
                    return CPStatus::Inconsistency;
            }
        }

        int s = size_[index]->get_value();
        switch (s) {
            case 0:
                if (this->is_active(index)) return CPStatus::Inconsistency;
                if (this->maybe_active(index)) return this->update_max_size(index, l);
                break;

            case 1:
                l->bind(index);
                [[fallthrough]];

            default:
                l->change_min(index);
                l->change(index);
                break;
        }
    }

    return CPStatus::OK;
}

CPStatus SparseSetArray::remove_all_but(double value, int index, DomainListener* l) {
    if (not contains(value, index) and this->is_active(index)) return CPStatus::Inconsistency;
    if (not contains(value, index) and this->maybe_active(index))
        return this->update_max_size(index, l);

    ssize_t val = static_cast<ssize_t>(value);
    val -= offsets_[index];

    if (size(index) != 1) {
        bool max_changed = max_[index]->get_value() == val;
        bool min_changed = min_[index]->get_value() == val;

        // Perform the sparse set operation to remove everything but the value
        int v = this->values_[index][0];
        int i = this->indices_[index][val];
        this->indices_[index][val] = 0;
        this->values_[index][0] = val;
        this->indices_[index][v] = i;
        this->values_[index][i] = v;

        this->min_[index]->set_value(val);
        this->max_[index]->set_value(val);
        this->size_[index]->set_value(1);

        // Note: should not happen
        if (this->size(index) == 0 and this->is_active(index)) return CPStatus::Inconsistency;
        if (this->size(index) == 0 and this->maybe_active(index))
            return this->update_max_size(index, l);

        l->bind(index);
        l->change(index);

        if (max_changed) l->change_max(index);
        if (min_changed) l->change_min(index);
    }

    return CPStatus::OK;
}

void SparseSetArray::update_bounds_val_removed(int index, ssize_t val) {
    this->update_max_val_removed(index, val);
    this->update_min_val_removed(index, val);
}

void SparseSetArray::update_max_val_removed(int index, ssize_t val) {
    if (not this->is_empty(index) and max_[index]->get_value() == val) {
        assert(not this->internal_contains(index, val));
        for (int v = val - 1; v >= min_[index]->get_value(); --v) {
            if (this->internal_contains(index, v)) {
                this->max_[index]->set_value(v);
                return;
            }
        }
    }
}

void SparseSetArray::update_min_val_removed(int index, ssize_t val) {
    if (not this->is_empty(index) and (min_[index]->get_value() == val)) {
        assert(not this->internal_contains(index, val));
        for (int v = val + 1; v <= max_[index]->get_value(); ++v) {
            if (this->internal_contains(index, v)) {
                this->min_[index]->set_value(v);
                return;
            }
        }
    }
}

bool SparseSetArray::internal_contains(int index, ssize_t val) const {
    if (val < 0 or val >= this->n_[index]) return false;
    return this->indices_[index][val] < this->size(index);
}

CPStatus SparseSetArray::update_min_size(int new_min_size, DomainListener* l) {
    // Simple case, nothing to do
    if (new_min_size < min_size_->get_value()) return CPStatus::OK;

    // Wipeout of the domain for the size variable
    if (new_min_size > max_size_->get_value()) return CPStatus::Inconsistency;

    if (new_min_size != min_size_->get_value()) {
        min_size_->set_value(new_min_size);
        l->change_array_size(new_min_size);
    }
    return CPStatus::OK;
}

CPStatus SparseSetArray::update_max_size(int new_max_size, DomainListener* l) {
    // Simple case, nothing to do
    if (new_max_size > max_size_->get_value()) return CPStatus::OK;

    // Wipeout of the domain for the size variable
    if (new_max_size < min_size_->get_value()) return CPStatus::Inconsistency;

    if (new_max_size != max_size_->get_value()) {
        max_size_->set_value(new_max_size);
        l->change_array_size(new_max_size);
    }
    return CPStatus::OK;
}

}  // namespace dwave::optimization::cp
