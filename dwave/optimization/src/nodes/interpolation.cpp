// Copyright 2025 D-Wave
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

#include <tuple>

#include "dwave-optimization/nodes/interpolation.hpp"
#include "_state.hpp"

namespace dwave::optimization {

BSplineNode::BSplineNode(ArrayNode* array_ptr, const int k, const std::vector<double> t, const std::vector<double> c)
            :ArrayOutputMixin(array_ptr->size()), array_ptr_(array_ptr), k_(k), t_(std::move(t)), c_(std::move(c)) {
                if (!array_ptr) throw std::invalid_argument("node pointer cannot be nullptr");
                if (array_ptr->ndim() > 1) throw std::invalid_argument("node pointer cannot be multi-d array");
                // conservative upper limits to avoid expensive calculations
                if (k >= 5) throw std::invalid_argument("bspline degree should be smaller than 5");
                if (t.size() >= 20) throw std::invalid_argument("number of knots should be smaller than 20");
                if (t.size() != k + c.size() + 1){
                    throw std::invalid_argument("number of knots should be equal to sum of"
                                                "degree, number of coefficients and 1");
                }
            this->add_predecessor(array_ptr);
            }

std::vector<double> BSplineNode::bspline_basis(double state) const {

    int n = t_.size() - 1; // number of degree-0 basis functions
    std::vector<std::vector<double>> B(k_ + 1, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i){
        if ((t_[i] <= state && state < t_[i + 1])){
            B[0][i] = 1.0;
        }
    }

    for (int d = 1; d <= k_; ++d){
        for (int i = 0; i < n - d; ++i){

            double c1_denom = t_[i + d] - t_[i];
            double c1 = (c1_denom != 0) ? (B[d - 1][i] * (state - t_[i]) / c1_denom): 0.0;

            double c2_denom = t_[i + d + 1] - t_[i + 1];
            double c2 = (c2_denom != 0) ? (B[d - 1][i + 1] * (t_[i + d + 1] - state) / c2_denom): 0.0;

            B[d][i] = c1 + c2;
        }
    }
    return B[k_];
}

double BSplineNode::compute_value(double state) const {
    int m = c_.size();
    // return nans outside of the base interval, no extrapolation.
    if (state < t_[k_] || state > t_[m]) { return std::numeric_limits<double>::quiet_NaN(); }
    std::vector<double> B_k = bspline_basis(state);

    double sum = 0.0;
    for (int i = 0; i < m; ++i){
        sum += c_[i] * B_k[i];
    }
    return sum;
}

int BSplineNode::get_k() const { return k_; }

const std::vector<double>& BSplineNode::get_t() const { return t_; }

const std::vector<double>& BSplineNode::get_c() const { return c_; }


ssize_t BSplineNode::size(const State& state) const { return array_ptr_->size(state); }

double const* BSplineNode::buff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->buff();
}

void BSplineNode::commit(State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->commit();
}

std::span<const Update> BSplineNode::diff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->diff();
}
bool BSplineNode::integral() const { return false; }


void BSplineNode::revert(State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->revert();
}

std::pair<double, double> BSplineNode::minmax(
        optional_cache_type<std::pair<double, double>> cache) const {
    return memoize(cache, [&]() {

        double low = std::reduce(c_.begin(), c_.end(), c_[0], [](int a, int b) { return std::min(a, b); });
        double high = std::reduce(c_.begin(), c_.end(), c_[0], [](int a, int b) { return std::max(a, b); });

        return std::make_pair(low, high);
    });
}

void BSplineNode::initialize_state(State& state) const {
    std::vector<double> bspline_values;
    auto state_data = dynamic_cast<ArrayNode*>(predecessors()[0])->view(state);

    for (int i = 0, stop = array_ptr_->size(); i < stop; ++i) {
        bspline_values.push_back(compute_value(state_data[i]));
    }
    emplace_data_ptr<ArrayNodeStateData>(state, std::move(bspline_values));
}

void BSplineNode::propagate(State& state) const {
    auto node_data_ptr = dynamic_cast<ArrayNode*>(predecessors()[0]);
    auto diff = node_data_ptr->diff(state);
    auto state_data = node_data_ptr->view(state);

    if (diff.size()) {
        for (auto& update : diff) {
            auto index = update.index;
            data_ptr<ArrayNodeStateData>(state)->set(index, compute_value(state_data[index]));
        }
    }
}

}  // namespace dwave::optimization