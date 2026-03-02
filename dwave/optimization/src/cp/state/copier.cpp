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

#include "dwave-optimization/cp/state/copier.hpp"

#include "dwave-optimization/cp/state/copy.hpp"
namespace dwave::optimization::cp {
int Copier::get_level() { return prior.size() - 1; }

void Copier::restore_state() {
    prior.back().restore();
    prior.pop_back();
    this->notify_restore();
}

void Copier::save_state() { prior.emplace_back(Backup(*this)); }

void Copier::restore_state_until(int level) {
    while (get_level() > level) {
        this->restore_state();
    }
}

void Copier::with_new_state(std::function<void()> body) {
    int level = this->get_level();
    this->save_state();
    body();
    this->restore_state_until(level);
}

StateInt* Copier::make_state_int(ssize_t init_value) {
    store.emplace_back(std::make_unique<CopyInt>(init_value));
    return dynamic_cast<StateInt*>(store.back().get());
}

StateBool* Copier::make_state_bool(bool init_value) {
    store.emplace_back(std::make_unique<CopyBool>(init_value));
    return dynamic_cast<StateBool*>(store.back().get());
}

StateReal* Copier::make_state_real(double init_value) {
    store.emplace_back(std::make_unique<CopyReal>(init_value));
    return dynamic_cast<StateReal*>(store.back().get());
}

void Copier::notify_restore() {
    for (auto l : on_restore_listeners) {
        l();
    }
}
}  // namespace dwave::optimization::cp
