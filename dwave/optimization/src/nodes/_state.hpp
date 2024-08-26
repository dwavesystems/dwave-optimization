// Copyright 2024 D-Wave Inc.
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

#include <memory>
#include <ranges>
#include <utility>
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/state.hpp"

namespace dwave::optimization {

// We don't current distinguish between dynamic and constant-sized nodes.
// This means that nodes that always have a constant shape are doing extra
// work. We could add a separate class, or perhaps template, but for now the
// difference should be quite minimal.
class ArrayNodeStateData : public NodeStateData {
 public:
    explicit ArrayNodeStateData(std::vector<double>&& values) noexcept
            : buffer(std::move(values)), previous_size_(buffer.size()) {}

    const double* buff() const noexcept { return buffer.data(); }

    void commit() noexcept {
        updates.clear();
        previous_size_ = buffer.size();
    }

    std::unique_ptr<NodeStateData> copy() const override {
        return std::make_unique<ArrayNodeStateData>(*this);
    }

    std::span<const Update> diff() const noexcept { return updates; }

    // Exchange the values in the buffer at index i and j and track the update.
    // Return whether a change was made.
    bool exchange(ssize_t i, ssize_t j) {
        assert(i >= 0 && static_cast<std::size_t>(i) < buffer.size());
        assert(j >= 0 && static_cast<std::size_t>(j) < buffer.size());

        // check whether there is any change to be made
        if (i == j) return false;
        if (buffer[i] == buffer[j]) return false;

        std::swap(buffer[i], buffer[j]);
        updates.emplace_back(i, buffer[j], buffer[i]);
        updates.emplace_back(j, buffer[i], buffer[j]);
        return true;
    }

    // Get the value at index i.
    const double& get(ssize_t i) const {
        assert(i >= 0 && static_cast<std::size_t>(i) < buffer.size());
        return buffer[i];
    }

    void revert() {
        if (previous_size_ > buffer.size()) buffer.resize(previous_size_);
        for (const auto& [index, old, _] : updates | std::views::reverse) {
            buffer[index] = old;
        }
        if (previous_size_ < buffer.size()) buffer.resize(previous_size_);
        updates.clear();
    }

    // Set the value at index, tracking the change in the diff.
    bool set(ssize_t i, double value) {
        assert(i >= 0 && static_cast<std::size_t>(i) < buffer.size());

        double& old = buffer[i];

        if (old == value) return false;

        std::swap(old, value);
        updates.emplace_back(i, value, old);
        return true;
    }

    ssize_t size_diff() const noexcept {
        return static_cast<ssize_t>(buffer.size()) - previous_size_;
    }

    // Changes made directly to the buffer/update must be reflected in both!
    std::vector<double> buffer;
    std::vector<Update> updates;

 private:
    // We use size_t to be consistent with the size of the buffer etc, but we
    // need to be a bit careful with subtraction etc
    std::size_t previous_size_ = 0;
};

class ScalarNodeStateData : public NodeStateData {
 public:
    explicit ScalarNodeStateData(double value) : update(0, value, value) {}

    const double* buff() const { return &update.value; }
    void commit() { update.old = update.value; }
    std::span<const Update> diff() const {
        return std::span<const Update>(&update, update.old != update.value);
    }
    void revert() { update.value = update.old; }
    void set(double value) { update.value = value; }

    Update update;
};

}  // namespace dwave::optimization
