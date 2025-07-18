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

#include <algorithm>
#include <memory>
#include <ranges>
#include <utility>
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/state.hpp"

namespace dwave::optimization {

// Hold the state of a contiguous array of values.
// We don't current distinguish between dynamic and constant-sized arrays.
// This means that arrays that always have a constant shape are doing extra
// work. We could add a separate class, or perhaps template, but for now the
// difference should be quite minimal.
class ArrayStateData {
 public:
    explicit ArrayStateData(std::vector<double>&& values) noexcept
            : buffer(std::move(values)), size_(buffer.size()), previous_size_(buffer.size()) {}

    template <std::ranges::range Range>
    explicit ArrayStateData(Range&& values) noexcept
            : ArrayStateData(std::vector<double>(values.begin(), values.end())) {}

    // Assign new values to the state starting from an offset, tracking the changes from the
    // previous state to the new. If the original buffer extends past the new range of values,
    // trim it by adding removal updates.
    bool assign(std::ranges::sized_range auto&& values, ssize_t offset = 0) {
        // dev note: we could implement a version of this that doesn't need sized_range.
        const ssize_t overlap_length =
                std::min<ssize_t>(buffer.size(), std::ranges::size(values) + offset) - offset;

        auto vit = std::ranges::begin(values);

        // first walk through the overlap, updating the buffer and the diff accordingly
        {
            auto bit = buffer.begin() + offset;
            for (ssize_t index = offset; index < overlap_length + offset; ++index, ++bit, ++vit) {
                if (*bit == *vit) continue;  // no change
                updates.emplace_back(index, *bit, *vit);
                *bit = *vit;
            }
        }

        // trim the excess buffer
        this->trim_to(overlap_length + offset);

        // finally walk forward through the excess values, if there are any, adding them to the
        // buffer
        {
            buffer.reserve(std::ranges::size(values));
            for (ssize_t index = buffer.size(), stop = std::ranges::size(values) + offset;
                 index < stop; ++index, ++vit) {
                updates.emplace_back(Update::placement(index, *vit));
                buffer.emplace_back(*vit);
            }
        }

        size_ = buffer.size();

        return !updates.empty();
    }

    const double& back() { return buffer.back(); }

    const double* buff() const noexcept { return buffer.data(); }

    void commit() noexcept {
        updates.clear();
        previous_size_ = buffer.size();
        assert(size_ >= 0 && static_cast<std::size_t>(size_) == buffer.size());
    }

    std::span<const Update> diff() const noexcept { return updates; }

    // Append a new value to the buffer, tracking the addition in the diff
    // Return whether a change was made.
    bool emplace_back(double value) {
        updates.emplace_back(Update::placement(buffer.size(), value));
        buffer.emplace_back(value);
        size_ += 1;
        assert(size_ >= 0 && static_cast<std::size_t>(size_) == buffer.size());
        return true;
    }

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

    // Pop an item from the buffer
    // Return whether a change was made.
    bool pop_back() {
        assert(buffer.size() >= 1);
        updates.emplace_back(Update::removal(buffer.size() - 1, buffer.back()));
        buffer.pop_back();
        size_ -= 1;
        assert(size_ >= 0 && static_cast<std::size_t>(size_) == buffer.size());
        return true;
    }

    void revert() {
        assert(previous_size_ >= 0);
        buffer.resize(previous_size_);
        const ssize_t size = buffer.size();
        for (const auto& [index, old, _] : updates | std::views::reverse) {
            assert(index >= 0);
            if (index >= size) continue;
            buffer[index] = old;
        }
        updates.clear();
        size_ = buffer.size();
    }

    // Set the value at index, tracking the change in the diff.
    // If allow_emplace is true, do an emplace_back iff the index is equal to the current size.
    bool set(ssize_t i, double value, bool allow_emplace = false) {
        if (allow_emplace && i == this->size()) {
            return this->emplace_back(value);
        }

        assert(i >= 0 && static_cast<std::size_t>(i) < buffer.size());

        double& old = buffer[i];

        if (old == value) return false;

        std::swap(old, value);
        updates.emplace_back(i, value, old);
        return true;
    }

    const ssize_t& size() const {
        assert(size_ >= 0 && static_cast<std::size_t>(size_) == buffer.size());
        return size_;
    }

    ssize_t size_diff() const noexcept {
        assert(size_ >= 0 && static_cast<std::size_t>(size_) == buffer.size());
        return size_ - previous_size_;
    }

    bool trim_to(ssize_t new_size) {
        assert(new_size >= 0);
        if (new_size >= size()) return false;

        for (ssize_t index = buffer.size() - 1; index >= new_size; --index) {
            updates.emplace_back(Update::removal(index, buffer[index]));
        }
        buffer.resize(new_size);

        size_ = new_size;

        return true;
    }

    // Update the state according to the given updates. Includes resizing.
    // Note that the old value of each update is ignored, and the buffer is
    // used as the source of truth.
    bool update(std::ranges::input_range auto&& updates) {
        for (const Update& update : updates) {
            const auto& [index, _, new_] = update;
            if (update.removed()) {
                assert(static_cast<std::size_t>(index) + 1 == buffer.size());
                this->updates.emplace_back(Update::removal(index, buffer[index]));
                buffer.pop_back();
            } else if (update.placed()) {
                assert(static_cast<std::size_t>(index) == buffer.size());
                this->updates.emplace_back(Update::placement(index, new_));
                buffer.emplace_back(new_);
            } else {
                assert(0 <= index && static_cast<std::size_t>(index) < buffer.size());
                this->set(index, new_);
            }
        }

        size_ = buffer.size();

        return !this->updates.empty();
    }

 private:
    // Changes made directly to the buffer/update must be reflected in both!
    std::vector<double> buffer;
    std::vector<Update> updates;

    // We need to be able to calculate a size diff, and to have a referencable
    // size. So we keep an addition source of truth for the size and we assert
    // it absolutely everywhere.
    ssize_t size_;
    ssize_t previous_size_;
};

class ArrayNodeStateData : public ArrayStateData, public NodeStateData {
 public:
    explicit ArrayNodeStateData(std::vector<double>&& values) noexcept
            : ArrayStateData(std::move(values)), NodeStateData() {}

    template <std::ranges::range Range>
    explicit ArrayNodeStateData(Range&& values) noexcept
            : ArrayNodeStateData(std::vector<double>(values.begin(), values.end())) {}

    std::unique_ptr<NodeStateData> copy() const override {
        return std::make_unique<ArrayNodeStateData>(*this);
    }
};

}  // namespace dwave::optimization
