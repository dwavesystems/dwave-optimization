#pragma once

#include <cassert>
#include <memory>
#include <vector>

#include "dwave-optimization/cp/state/state.hpp"
#include "dwave-optimization/cp/state/state_entry.hpp"
#include "dwave-optimization/cp/state/state_manager.hpp"
#include "dwave-optimization/cp/state/storage.hpp"

namespace dwave::optimization::cp {
class Copier : public StateManager {
 private:
    class Backup {
     private:
        Copier& copier_;
        ssize_t size_;
        std::vector<std::unique_ptr<StateEntry>> backup_store_;

     public:
        Backup(Copier& outer) : copier_(outer) {
            size_ = copier_.store.size();
            for (const auto& st_ptr : copier_.store) {
                backup_store_.emplace_back(std::move(st_ptr->save()));
            }
        }

        void restore() {
            // Safety assertion
            assert(static_cast<int>(copier_.store.size()) >= size_);

            copier_.store.resize(size_);
            for (const auto& cse_ptr : backup_store_) {
                cse_ptr->restore();
            }
        }
    };

 public:
    // We store pointers of storage because we inherit from that class.
    // also I give the state manager the ownership of the state..
    // might need to make sure that it is the correct way
    std::vector<std::unique_ptr<Storage>> store;
    std::vector<Backup> prior;
    std::vector<std::function<void()>> on_restore_listeners;

    Copier() = default;

    /// state manager overrides

    int get_level() override;

    void restore_state() override;

    void save_state() override;

    void restore_state_until(int level) override;

    void with_new_state(std::function<void()> body) override;

    // TODO: the next two methods have a dynamic cast that doesn't make me really comfortable
    StateInt* make_state_int(int init_value) override;

    StateBool* make_state_bool(bool init_value) override;

    StateReal* make_state_real(double init_value) override;

    /// other methods

    int store_size() { return store.size(); }

    void on_restore(std::function<void()> listener) { on_restore_listeners.push_back(listener); }

 private:
    void notify_restore();
};

}  // namespace dwave::optimization::cp
