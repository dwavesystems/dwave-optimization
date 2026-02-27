#pragma once

#include "dwave-optimization/cp/core/domain_listener.hpp"

namespace dwave::optimization::cp
{
    class TestListener : public DomainListener {
        public:
        // void empty() override {}
        void bind() override {}
        void change() override {}
        void change_max() override {}
        void change_min() override {}
    };
} // namespace dwave::optimization::cp
