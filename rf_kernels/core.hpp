// Not much in this file for now... !

#ifndef _RF_KERNELS_CORE_HPP
#define _RF_KERNELS_CORE_HPP

#include <string>
#include <iostream>

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++0x support (g++ -std=c++0x)"
#endif

namespace rf_kernels {
#if 0
}  // emacs pacifier
#endif


enum axis_type {
    AXIS_FREQ = 0,
    AXIS_TIME = 1,
    AXIS_NONE = 2
};

extern std::ostream &operator<<(std::ostream &os, axis_type axis);

extern std::string axis_type_to_string(axis_type axis);

extern axis_type axis_type_from_string(const std::string &s, const char *where="string");


}  // namespace rf_kernels
    
#endif  // _RF_KERNELS_CORE_HPP
