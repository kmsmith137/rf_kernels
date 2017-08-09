#include <cstdlib>
#include <cstring>
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

template<typename T>
inline T *aligned_alloc(size_t nelts)
{
    if (nelts == 0)
	return NULL;

    // align to 128 bytes (size of an L3 cache line "pair")
    void *p = NULL;
    if (posix_memalign(&p, 128, nelts * sizeof(T)) != 0)
	throw std::runtime_error("couldn't allocate memory");

    memset(p, 0, nelts * sizeof(T));
    return reinterpret_cast<T *> (p);
}

}  // namespace rf_kernels

#endif  // _RF_KERNELS_INTERNALS_HPP
