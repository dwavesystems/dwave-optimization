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

// type_list is partially based on
// https://github.com/lipk/cpp-typelist/blob/02eff2292fdf1ca73e8c09866c720d4b39473ed1
// under an MIT license.
//
//     MIT License
//
//     Copyright (c) 2018
//
//     Permission is hereby granted, free of charge, to any person obtaining a copy
//     of this software and associated documentation files (the "Software"), to deal
//     in the Software without restriction, including without limitation the rights
//     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//     copies of the Software, and to permit persons to whom the Software is
//     furnished to do so, subject to the following conditions:
//
//     The above copyright notice and this permission notice shall be included in all
//     copies or substantial portions of the Software.
//
//     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//     SOFTWARE.

#pragma once

#include <memory>
#include <type_traits>
#include <utility>
#include <variant>

namespace dwave::optimization {

/// A ``type_list`` encodes a collection of types.
template <typename Type, typename... Types>
struct type_list {
    /// A new ``type_list`` created by adding const to every type in the ``type_list``.
    using add_const = type_list<std::add_const_t<Type>, std::add_const_t<Types>...>;

    /// A new ``type_list`` created by adding a pointer to every type in the ``type_list``.
    using add_pointer = type_list<std::add_pointer_t<Type>, std::add_pointer_t<Types>...>;

    /// Return if ``T*`` can be cast to one of the types in the ``type_list``. Includes
    /// dynamic casts.
    template <class T>
    static bool check(T* ptr) {
        // If convertable, we're done
        if constexpr (std::convertible_to<T*, Type>) return true;

        // Next check if we could do a dynamic down cast. Do as much at compile-time as possible
        if constexpr (std::is_pointer_v<Type>                               //
                      && std::is_polymorphic_v<T>                           //
                      && std::derived_from<std::remove_pointer_t<Type>, T>  //
                      && (std::is_const_v<std::remove_pointer_t<Type>> || !std::is_const_v<T>)) {
            if (typeid(std::remove_pointer_t<Type>) == typeid(*ptr)) return true;
        }

        if constexpr (sizeof...(Types)) return type_list<Types...>::check(ptr);
        return false;
    }
    // Could add a check(T& obj) overload when/if needed.

    /// Return true of ``T`` is included in the ``type_list``.
    template <typename T>
    static constexpr bool contains() {
        if constexpr (std::same_as<Type, T>) return true;
        if constexpr (sizeof...(Types)) return type_list<Types...>::template contains<T>();
        return false;
    }

    /// Return the number of times that ``T`` appears in the ``type_list``.
    template <typename T>
    static constexpr std::size_t count() {
        constexpr std::size_t count = std::same_as<Type, T>;
        if constexpr (sizeof...(Types)) {
            return count + type_list<Types...>::template count<T>();
        }
        return count;
    }

    /// Return whether every type in the ``type_list`` is present in ``OtherTypeList``.
    template <typename OtherTypeList>   // this could be better specified
    static constexpr bool issubset() {  // match Python name
        bool subset = OtherTypeList::template contains<Type>();
        if constexpr (sizeof...(Types)) {
            return subset && type_list<Types...>::template issubset<OtherTypeList>();
        }
        return subset;
    }

    /// Return whether every type in the ``OtherTypeList`` is present in ``type_list``.
    template <typename OtherTypeList>     // this could be better specified
    static constexpr bool issuperset() {  // match Python name
        return OtherTypeList::template issubset<type_list<Type, Types...>>();
    }

    /// Cast the given pointer to a variant defined by the set of types available
    /// or a ``std::monostate`` if it cannot be cast.
    template <typename T>
    static std::variant<std::monostate, Type, Types...> make_variant(T* ptr) {
        // If convertable, we're done
        if constexpr (std::convertible_to<T*, Type>) return ptr;

        // Next check if we could do a dynamic down cast. Do as much at compile-time as possible
        if constexpr (std::is_pointer_v<Type>                               //
                      && std::is_polymorphic_v<T>                           //
                      && std::derived_from<std::remove_pointer_t<Type>, T>  //
                      && (std::is_const_v<std::remove_pointer_t<Type>> || !std::is_const_v<T>)) {
            if (auto dyn_ptr = dynamic_cast<Type>(ptr); dyn_ptr) return dyn_ptr;
        }

        // If there are still types left to check, we need to recurse. Unfortunately
        // recursing returns a variant that's a subset of what we want to support,
        // so we need to add supported types to the variant on the way back out.
        if constexpr (sizeof...(Types)) {
            return std::visit(
                    [](auto&& obj) -> std::variant<std::monostate, Type, Types...> {
                        return std::forward<decltype(obj)>(obj);
                    },
                    type_list<Types...>::template make_variant<T>(ptr));
        }

        // We didn't find any matches, so return the monostate
        return std::monostate();
    }
    // Could add a make_variant(T&& obj) overload when/if needed.

    /// A new ``type_list`` created by removing a pointer from every type in the ``type_list``.
    using remove_pointer = type_list<std::remove_pointer_t<Type>, std::remove_pointer_t<Types>...>;

    /// Return the number of types.
    static constexpr std::size_t size() { return 1 + sizeof...(Types); }

    /// Convert the type_list to another class such as ``std::variant`` or
    /// ``std::tuple``.
    template <template <typename...> class U>
    using to = U<Type, Types...>;
};

}  // namespace dwave::optimization
