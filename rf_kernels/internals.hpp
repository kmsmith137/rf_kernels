#include <cstdlib>
#include <cstring>
#include <sstream>
#include <iostream>
#include <stdexcept>

#ifndef _RF_KERNELS_INTERNALS_HPP
#define _RF_KERNELS_INTERNALS_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++0x support (g++ -std=c++0x)"
#endif

// Branch predictor hint
#ifndef _unlikely
#define _unlikely(cond)  (__builtin_expect(cond,0))
#endif

// rf_assert(): like assert, but throws an exception in order to work smoothly with python.
#define rf_assert(cond) rf_assert2(cond, __LINE__)

#define rf_assert2(cond,line) \
    do { \
        if (_unlikely(!(cond))) { \
	    const char *msg = "rf_pipelines: assertion '" __STRING(cond) "' failed (" __FILE__ ":" __STRING(line) ")\n"; \
	    throw std::runtime_error(msg); \
	} \
    } while (0)


namespace rf_kernels {
#if 0
}  // emacs pacifier
#endif


// We align to 128 bytes by default (size of an L3 cache line "pair")
template<typename T>
inline T *aligned_alloc(size_t nelts, ssize_t nalign=128)
{
    rf_assert(nelts >= 0);
    rf_assert(nalign > 0);

    if (nelts == 0)
	return NULL;

    void *p = NULL;
    if (posix_memalign(&p, nalign, nelts * sizeof(T)) != 0)
	throw std::runtime_error("couldn't allocate memory");

    memset(p, 0, nelts * sizeof(T));
    return reinterpret_cast<T *> (p);
}


// Rounds 'nbytes' up to the nearest multiple of 'nalign'.
inline ssize_t _align(ssize_t nbytes, ssize_t nalign=128)
{
    rf_assert(nbytes >= 0);
    rf_assert(nalign > 0);
    return ((nbytes + nalign - 1) / nalign) * nalign;
}


}  // namespace rf_kernels


// Hmm, the C++11 standard library doesn't define hash functions for 
// composite types such as pair<S,T> or array<T,N>.
//
// FIXME: I may want to generalize from array<T,2> to array<T,N> some day.
// FIXME: Is it kosher to add hash<> specializations to the std:: namespace?

namespace std {
    template<typename T>
    struct hash<array<T,2>>
    {
	inline size_t operator()(const array<T,2> &v) const
	{
	    hash<T> h;
	    size_t h0 = h(v[0]);
	    size_t h1 = h(v[1]);

	    // from boost::hash_combine()
	    h0 ^= (h1 + 0x9e3779b9 + (h0 << 6) + (h0 >> 2));

	    return h0;
	}
    };
}


#endif  // _RF_KERNELS_INTERNALS_HPP
