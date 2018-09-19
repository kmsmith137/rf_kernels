#include <array>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <iostream>
#include <stdexcept>

#ifndef _RF_KERNELS_INTERNALS_HPP
#define _RF_KERNELS_INTERNALS_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
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
inline T *aligned_alloc(size_t nelts, ssize_t nalign=128, bool zero=true)
{
    rf_assert(nelts >= 0);
    rf_assert(nalign > 0);

    if (nelts == 0)
	return NULL;

    void *p = NULL;
    if (posix_memalign(&p, nalign, nelts * sizeof(T)) != 0)
	throw std::runtime_error("couldn't allocate memory");

    if (zero)
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


// uptr<T> usage:
//
//   uptr<float> p = make_uptr<float> (nelts);

struct uptr_deleter {
    inline void operator()(const void *p) { free(const_cast<void *> (p)); }
};

template<typename T>
using uptr = std::unique_ptr<T[], uptr_deleter>;

template<typename T>
inline uptr<T> make_uptr(size_t nelts, size_t nalign=128, bool zero=true)
{
    T *p = aligned_alloc<T> (nelts, nalign, zero);
    return uptr<T> (p);
}


template<typename T>
inline T square(T x)
{
    return x*x;
}

// Returns (m/n), in a situation where we want to assert that n evenly divides m.
inline ssize_t xdiv(ssize_t m, ssize_t n)
{
    rf_assert(m >= 0);
    rf_assert(n > 0);
    rf_assert(m % n == 0);
    return m / n;
}

// Returns (m % n), in a situation where we want to assert that the % operation makes sense
inline ssize_t xmod(ssize_t m, ssize_t n)
{
    rf_assert(m >= 0);
    rf_assert(n > 0);
    return m % n;
}


}  // namespace rf_kernels


// Hmm, the C++11 standard library doesn't define hash functions for 
// composite types such as pair<S,T> or array<T,N>.
//
// FIXME: Is it kosher to add hash<> specializations to the std:: namespace?

namespace std {
    template<size_t N, typename H, typename T, typename enable_if<(N==1),int>::type = 0>
    inline size_t kms_hash(const H &h, const T &t)
    {
	return h(t[0]);
    }
    
    template<size_t N, typename H, typename T, typename enable_if<(N>1),int>::type = 0>
    inline size_t kms_hash(const H &h, const T &t)
    {
	size_t h0 = kms_hash<N-1> (h,t);
	size_t h1 = h(t[N-1]);

	// from boost::hash_combine()
	h0 ^= (h1 + 0x9e3779b9 + (h0 << 6) + (h0 >> 2));
	return h0;
    }

    template<typename T, size_t N>
    struct hash<array<T,N>>
    {
	inline size_t operator()(const array<T,N> &v) const
	{
	    hash<T> h;
	    return kms_hash<N> (h,v);
	}
    };
}


#endif  // _RF_KERNELS_INTERNALS_HPP
