#pragma once

namespace dwave::optimization::cp {
class DomainListener {
 public:
    virtual ~DomainListener() = default;

    // TODO: check whether this should be removed or not
    // virtual void empty() = 0;
    virtual void bind() = 0;
    virtual void change() = 0;
    virtual void change_max() = 0;
    virtual void change_min() = 0;
};
}  // namespace dwave::optimization::cp
