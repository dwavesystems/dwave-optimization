// Copyright 2023 D-Wave Systems Inc.
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
#include <span>
#include <utility>
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"

namespace dwave::optimization {

/// A contiguous block of numbers.
class ConstantNode : public ArrayOutputMixin<ArrayNode> {
 public:
    struct DataSource {
        DataSource() = default;
        virtual ~DataSource() = default;
        // Technically should satisfy the rule of five here, and it's good
        // to be on the safe side by deleting the unnecessary constructors
        DataSource(const DataSource&) = delete;
        DataSource& operator=(const DataSource&) = delete;
        DataSource(DataSource&& other) noexcept = default;
        DataSource& operator=(DataSource&& other) noexcept = delete;
    };

    // Default constructor - defaults to an empty 1d array.
    ConstantNode() noexcept : ConstantNode(std::vector<double>{}, std::vector<ssize_t>{0}) {}

    // A single scalar value
    explicit ConstantNode(double value)
            : ConstantNode(std::vector{value}, std::vector<ssize_t>{}) {}

    // A pointer to another array or similar. In this case the ConstantNode will be a *view*
    // rather than a container. That is it does not manage the lifespan of the array data.
    ConstantNode(const double* data_ptr, std::initializer_list<ssize_t> shape);
    ConstantNode(const double* data_ptr, const std::span<const ssize_t> shape);

    // For use from Python, where we will pass in a PyDataSource which manages the python reference
    // to the original object (a numpy array) that holds the data.
    ConstantNode(std::unique_ptr<DataSource> data_source, const double* data_ptr,
                 const std::span<const ssize_t> shape);

    /// Create a ConstantNode by copying the contents of a range
    template <std::ranges::sized_range Range>
    explicit ConstantNode(Range&& values)
            : ConstantNode(std::forward<Range>(values), {static_cast<ssize_t>(values.size())}) {}

    /// Create a ConstantNode by copying the contents of a range and interpreting
    /// it as the given shape
    template <std::ranges::sized_range Range>
    ConstantNode(Range&& values, std::initializer_list<ssize_t> shape)
            : ConstantNode(from_range(std::forward<Range>(values), shape), shape) {}
    template <std::ranges::sized_range Range>
    ConstantNode(Range&& values, const std::span<const ssize_t> shape)
            : ConstantNode(from_range(std::forward<Range>(values), shape), shape) {}

    // Stateless access to the underlying data.
    std::span<const double> data() const noexcept {
        return std::span<const double>(buffer_ptr_, this->size());
    }

    // Overload of the buff() method. The first one is a stateless overload
    double const* buff() const { return buffer_ptr_; }
    double const* buff(const State&) const override { return buffer_ptr_; }

    // Overloads needed by the Array ABC **************************************

    // There are never any updates to propagate
    std::span<const Update> diff(const State& state) const noexcept override { return {}; }

    /// @copydoc Array::integral()
    // Returns ``true`` for an empty array. This has the slightly odd effect of
    // making empty arrays logical, because we also return 0 for min/max.
    bool integral() const override;

    /// @copydoc Array::min()
    double min() const override;

    /// @copydoc Array::max()
    double max() const override;

    // Overloads required by the Node ABC *************************************

    // This node never needs to update its successors
    void propagate(State&) const noexcept override {}

    // Constants don't have predecessors so no one should be calling update().
    // Always throws a logic_error.
    [[noreturn]] void update(State& state, int index) const override;

    // Never any changes to revert or commit
    void commit(State&) const noexcept override {}
    void revert(State&) const noexcept override {}

 private:
    struct OwningDataSource : public DataSource {
        OwningDataSource(std::unique_ptr<const double[]>&& owning_ptr)
                : data_(std::move(owning_ptr)) {}
        const double* get() const { return data_.get(); }

        std::unique_ptr<const double[]> data_;
    };

    // An owning pointer to an array. In this case the ConstantNode will manage the lifespan
    // of the array.
    ConstantNode(OwningDataSource&& data_source, std::initializer_list<ssize_t> shape);

    ConstantNode(OwningDataSource&& data_source, const std::span<const ssize_t> shape);

    // Create a unique_ptr<double[]> built from the given range of values holding
    // the values of the array.
    template <std::ranges::sized_range Range>
    static OwningDataSource from_range(Range&& values, const std::span<const ssize_t> shape) {
        const ssize_t size = shape_to_size(shape);
        if (size < 0) {
            throw std::invalid_argument("ConstantNode cannot be dynamic");
        }
        if (values.size() < static_cast<std::size_t>(size)) {
            throw std::out_of_range("too few values for the given shape");
        }

        // allocate the vector and populate it from the range
        std::unique_ptr<double[]> ptr = std::make_unique<double[]>(size);
        std::copy(values.begin(), std::next(values.begin(), size), ptr.get());
        return OwningDataSource(std::move(ptr));
    }

    // The beginning of the array data. The ConstantNode is unusual because it
    // holds its values on the object itself rather than in a State.
    const double* buffer_ptr_;

    const ValuesInfo values_info_;

    std::unique_ptr<DataSource> data_source_;
};

}  // namespace dwave::optimization
