#pragma once

#include <cstddef>
#include <vector>
#include <unordered_map>
#include <string>

namespace lh{
    template<class T>
    using Param = std::pair<std::vector<std::size_t>, T *>;
    template<class T>
    using Graph = std::unordered_map<std::string, Param<T>>;
}