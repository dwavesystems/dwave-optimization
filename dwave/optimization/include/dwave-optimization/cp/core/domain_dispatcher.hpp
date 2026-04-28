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

#pragma once
#include <variant>

#include "dwave-optimization/cp/core/domain_array.hpp"
#include "dwave-optimization/cp/core/interval_array.hpp"
#include "dwave-optimization/cp/core/sparse_set_array.hpp"

namespace dwave::optimization::cp {

using DomainArrayVariant = std::variant<IntIntervalArray, RealIntervalArray, SparseSetArray>;

struct DomainDispatcher {
    static size_t num_domains(const DomainArrayVariant& d) {
        return std::visit([&](const auto& dom) { return dom.num_domains(); }, d);
    }

    static ssize_t min_size(const DomainArrayVariant& d) {
        return std::visit([&](const auto& dom) { return dom.min_size(); }, d);
    }

    static ssize_t max_size(const DomainArrayVariant& d) {
        return std::visit([&](const auto& dom) { return dom.max_size(); }, d);
    }

    static double min(const DomainArrayVariant& d, int index) {
        return std::visit([&](const auto& dom) { return dom.min(index); }, d);
    }

    static double max(const DomainArrayVariant& d, int index) {
        return std::visit([&](const auto& dom) { return dom.max(index); }, d);
    }

    static double size(const DomainArrayVariant& d, int index) {
        return std::visit([&](const auto& dom) { return dom.size(index); }, d);
    }

    static bool is_bound(const DomainArrayVariant& d, int index) {
        return std::visit([&](const auto& dom) { return dom.is_bound(index); }, d);
    }

    static bool is_active(const DomainArrayVariant& d, int index) {
        return std::visit([&](const auto& dom) { return dom.is_active(index); }, d);
    }

    static bool maybe_active(const DomainArrayVariant& d, int index) {
        return std::visit([&](const auto& dom) { return dom.maybe_active(index); }, d);
    }

    static bool contains(const DomainArrayVariant& d, double value, int index) {
        return std::visit([&](const auto& dom) { return dom.contains(value, index); }, d);
    }

    static CPStatus remove(DomainArrayVariant& d, double value, int index, DomainListener* l) {
        return std::visit([&](auto& dom) { return dom.remove(value, index, l); }, d);
    }

    static CPStatus remove_above(DomainArrayVariant& d, double value, int index,
                                 DomainListener* l) {
        return std::visit([&](auto& dom) { return dom.remove_above(value, index, l); }, d);
    }

    static CPStatus remove_below(DomainArrayVariant& d, double value, int index,
                                 DomainListener* l) {
        return std::visit([&](auto& dom) { return dom.remove_below(value, index, l); }, d);
    }

    static CPStatus remove_all_but(DomainArrayVariant& d, double value, int index,
                                   DomainListener* l) {
        return std::visit([&](auto& dom) { return dom.remove_all_but(value, index, l); }, d);
    }

    static CPStatus update_min_size(DomainArrayVariant& d, int new_min_size, DomainListener* l) {
        return std::visit([&](auto& dom) { return dom.update_min_size(new_min_size, l); }, d);
    }

    static CPStatus update_max_size(DomainArrayVariant& d, int new_max_size, DomainListener* l) {
        return std::visit([&](auto& dom) { return dom.update_max_size(new_max_size, l); }, d);
    }
};

}  // namespace dwave::optimization::cp
