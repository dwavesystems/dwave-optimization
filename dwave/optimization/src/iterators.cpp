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

#include "dwave-optimization/iterators.hpp"

namespace dwave::optimization {

// temporary experimental namespace for testing
namespace exp {

template class BufferIterator<float>;
template class BufferIterator<double>;
template class BufferIterator<std::int8_t>;
template class BufferIterator<std::int16_t>;
template class BufferIterator<std::int32_t>;
template class BufferIterator<std::int64_t>;

// There are a lot of combinations...
template class BufferIterator<float, float>;
template class BufferIterator<float, double>;
template class BufferIterator<float, std::int8_t>;
template class BufferIterator<float, std::int16_t>;
template class BufferIterator<float, std::int32_t>;
template class BufferIterator<float, std::int64_t>;

template class BufferIterator<double, float>;
template class BufferIterator<double, double>;
template class BufferIterator<double, std::int8_t>;
template class BufferIterator<double, std::int16_t>;
template class BufferIterator<double, std::int32_t>;
template class BufferIterator<double, std::int64_t>;

template class BufferIterator<std::int8_t, float>;
template class BufferIterator<std::int8_t, double>;
template class BufferIterator<std::int8_t, std::int8_t>;
template class BufferIterator<std::int8_t, std::int16_t>;
template class BufferIterator<std::int8_t, std::int32_t>;
template class BufferIterator<std::int8_t, std::int64_t>;

template class BufferIterator<std::int16_t, float>;
template class BufferIterator<std::int16_t, double>;
template class BufferIterator<std::int16_t, std::int8_t>;
template class BufferIterator<std::int16_t, std::int16_t>;
template class BufferIterator<std::int16_t, std::int32_t>;
template class BufferIterator<std::int16_t, std::int64_t>;

template class BufferIterator<std::int32_t, float>;
template class BufferIterator<std::int32_t, double>;
template class BufferIterator<std::int32_t, std::int8_t>;
template class BufferIterator<std::int32_t, std::int16_t>;
template class BufferIterator<std::int32_t, std::int32_t>;
template class BufferIterator<std::int32_t, std::int64_t>;

template class BufferIterator<std::int64_t, float>;
template class BufferIterator<std::int64_t, double>;
template class BufferIterator<std::int64_t, std::int8_t>;
template class BufferIterator<std::int64_t, std::int16_t>;
template class BufferIterator<std::int64_t, std::int32_t>;
template class BufferIterator<std::int64_t, std::int64_t>;

}  // namespace exp

}  // namespace dwave::optimization
