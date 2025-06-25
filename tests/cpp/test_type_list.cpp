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

#include "catch2/catch_test_macros.hpp"
#include "dwave-optimization/type_list.hpp"

namespace dwave::optimization {

TEST_CASE("Test type_list") {
    SECTION("static asserts") {
        static_assert(type_list<float, int>::count<float>() == 1);
        static_assert(type_list<float, int, float>::count<float>() == 2);
        static_assert(type_list<float, int>::count<bool>() == 0);
        static_assert(type_list<float, int>::count<const float>() == 0);

        static_assert(type_list<float, int>::contains<float>());
        static_assert(type_list<float, int, float>::contains<float>());
        static_assert(!type_list<float, int>::contains<bool>());

        static_assert(type_list<float, int>::size() == 2);
        static_assert(type_list<float, int, float>::size() == 3);

        static_assert(type_list<float>::issubset<type_list<int, float>>());
        static_assert(type_list<float, int>::issubset<type_list<int, float>>());
        static_assert(!type_list<float, int>::issubset<type_list<float>>());

        static_assert(!type_list<float>::issuperset<type_list<int, float>>());
        static_assert(type_list<float, int>::issuperset<type_list<int, float>>());
        static_assert(type_list<float, int>::issuperset<type_list<float>>());

        static_assert(type_list<float, int>::add_pointer::count<float*>() == 1);
        static_assert(type_list<float, float*>::add_pointer::count<float*>() == 1);
        static_assert(type_list<float, float*>::add_pointer::count<float**>() == 1);

        static_assert(type_list<float, float*>::remove_pointer::count<float>() == 2);
    }

    SECTION("type_list<...>::check() with pointers") {
        struct PolymorphicBase {
            virtual ~PolymorphicBase() = default;
        };
        struct PolymorphicDerived : PolymorphicBase {};
        struct NotPolymorphicBase {};
        struct NotPolymorphicDerived : NotPolymorphicBase {};

        using tl = type_list<PolymorphicDerived, NotPolymorphicDerived>;

        PolymorphicDerived pd;

        // direct casts
        CHECK(tl::add_pointer::check(&pd));
        CHECK(!tl::add_pointer::check(&std::as_const(pd)));
        CHECK(tl::add_const::add_pointer::check(&pd));
        CHECK(tl::add_const::add_pointer::check(&std::as_const(pd)));

        // down casts
        CHECK(tl::add_pointer::check(static_cast<PolymorphicBase*>(&pd)));
        CHECK(!tl::add_pointer::check(static_cast<const PolymorphicBase*>(&pd)));
        CHECK(tl::add_const::add_pointer::check(static_cast<PolymorphicBase*>(&pd)));
        CHECK(tl::add_const::add_pointer::check(static_cast<const PolymorphicBase*>(&pd)));

        NotPolymorphicDerived npd;

        // direct casts
        CHECK(tl::add_pointer::check(&npd));
        CHECK(!tl::add_pointer::check(&std::as_const(npd)));
        CHECK(tl::add_const::add_pointer::check(&npd));
        CHECK(tl::add_const::add_pointer::check(&std::as_const(npd)));

        // down casts - not polymorphic so not possible
        CHECK(!tl::add_pointer::check(static_cast<NotPolymorphicBase*>(&npd)));
        CHECK(!tl::add_pointer::check(static_cast<const NotPolymorphicBase*>(&npd)));
        CHECK(!tl::add_const::add_pointer::check(static_cast<NotPolymorphicBase*>(&npd)));
        CHECK(!tl::add_const::add_pointer::check(static_cast<const NotPolymorphicBase*>(&npd)));
    }

    SECTION("type_list<...>::make_variant() with pointers") {
        struct PolymorphicBase {
            virtual ~PolymorphicBase() = default;
        };
        struct PolymorphicDerived : PolymorphicBase {};
        struct NotPolymorphicBase {};
        struct NotPolymorphicDerived : NotPolymorphicBase {};

        using tl = type_list<PolymorphicDerived, NotPolymorphicDerived>;

        PolymorphicDerived pd;
        {
            auto var = tl::add_pointer::make_variant(static_cast<PolymorphicBase*>(&pd));
            REQUIRE(std::holds_alternative<PolymorphicDerived*>(var));
            CHECK(std::get<PolymorphicDerived*>(var) == &pd);
        }
        {
            auto var = tl::add_const::add_pointer::make_variant(&pd);
            REQUIRE(std::holds_alternative<const PolymorphicDerived*>(var));
            CHECK(std::get<const PolymorphicDerived*>(var) == &pd);
        }

        NotPolymorphicDerived npd;
        {
            auto var = tl::add_pointer::make_variant(static_cast<NotPolymorphicBase*>(&npd));
            CHECK(std::holds_alternative<std::monostate>(var));
        }
        {
            auto var = tl::add_const::add_pointer::make_variant(&npd);
            CHECK(std::holds_alternative<const NotPolymorphicDerived*>(var));
            CHECK(std::get<const NotPolymorphicDerived*>(var) == &npd);
        }
    }

    SECTION("type_list<...>::to") {
        // can convert to a tuple (also works for variant but this is easier to test)
        type_list<bool, int>::to<std::tuple> tpl{10, 10};
        CHECK(std::get<0>(tpl) == 1);
        CHECK(std::get<1>(tpl) == 10);
    }
}

}  // namespace dwave::optimization
