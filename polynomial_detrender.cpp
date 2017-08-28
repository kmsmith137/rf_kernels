// FIXME (low-priority) a nuisance issue when working with this code is that functions
// which are very similar have different argument orderings, e.g.
//
//          make_polynomial_detrender(nt_chunk, axis, polydeg, epsilon)
//   calls _make_polynomial_detrender(axis, nt_chunk, polydeg, epsilon)

#include <vector>
#include <string>

#include "rf_kernels/internals.hpp"
#include "rf_kernels/polynomial_detrender.hpp"
#include "rf_kernels/polynomial_detrender_internals.hpp"

using namespace std;

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


#ifdef __AVX__
constexpr int Sfid = 8;
#else
constexpr int Sfid = 4;
#endif

// To increase the maximum allowed polynomial degree, edit this line and rebuild rf_kernels ('make all install').
constexpr int polynomial_detrender_max_deg = 20;

// Usage: kernel(nfreq, nt, intensity, weights, stride, epsilon)
using detrending_kernel_t = void (*)(int, int, float *, float *, int, double);


// -------------------------------------------------------------------------------------------------
//
// _fill_detrending_kernel_table<S,N>(): fills shape (N,2) array with kernels.
// The outer index is a polynomial degree 0 <= polydeg < N, and the inner index is the axis.


template<int S, int N, typename std::enable_if<(N==0),int>::type = 0>
inline void fill_detrending_kernel_table(detrending_kernel_t *out) { }

template<int S, int N, typename std::enable_if<(N>0),int>::type = 0>
inline void fill_detrending_kernel_table(detrending_kernel_t *out)
{
    fill_detrending_kernel_table<S,N-1> (out);
    out[2*(N-1)] = _kernel_detrend_f<float,S,N>;
    out[2*(N-1)+1] = _kernel_detrend_t<float,S,N>;
}


struct detrending_kernel_table {
    static constexpr int MaxDeg = polynomial_detrender_max_deg;

    std::vector<detrending_kernel_t> entries;

    detrending_kernel_table() : entries(2*MaxDeg+2)
    {
	fill_detrending_kernel_table<Sfid,MaxDeg+1> (&entries[0]);
    }

    // Reminder: the 'axis' argument should be 0 to fit along the frequency axis, or 1 to fit along the time axis.
    inline detrending_kernel_t get_kernel(int axis, int polydeg)
    {
	if (_unlikely((axis < 0) || (axis > 1)))
	    throw runtime_error("rf_kernels::polynomial_detrender: axis=" + to_string(axis) + " is not defined for this transform");

	if (_unlikely(polydeg < 0))
	    throw runtime_error("rf_kernels::polynomial_detrender: polydeg=" + to_string(polydeg) + ", positive number expected");

	if (_unlikely(polydeg > MaxDeg))
	    throw runtime_error("rf_kernels::polynomial_detrender: polydeg=" + to_string(polydeg) + " is too large (the limit can be changed in rf_kernels/polynomial_detrender.cpp)");

	return entries[2*polydeg + axis];
    }
};


static detrending_kernel_table global_detrending_kernel_table;


// -------------------------------------------------------------------------------------------------


polynomial_detrender::polynomial_detrender(int axis_, int polydeg_) :
    axis(axis_),
    polydeg(polydeg_),
    _f(global_detrending_kernel_table.get_kernel(axis_, polydeg_))
{ }


void polynomial_detrender::detrend(int nfreq, int nt, float *intensity, float *weights, int stride, double epsilon)
{
    if (_unlikely(nfreq <= 0))
	throw runtime_error("rf_kernels::polynomial_detrender: nfreq=" + to_string(nfreq) + ", positive number expected");

    if (_unlikely(nt <= 0))
	throw runtime_error("rf_kernels::polynomial_detrender: nt_chunk=" + to_string(nt) + ", positive number expected");
    
    if (_unlikely((nt % Sfid) != 0))
	throw runtime_error("rf_kernels::polynomial_detrender: nt_chunk=" + to_string(nt) + " must be a multiple of " + to_string(Sfid));
    
    if (_unlikely(abs(stride) < nt))
	throw runtime_error("rf_kernels::polynomial_detrender: stride=" + to_string(stride) + " is too small");

    if (_unlikely(epsilon <= 0.0))
	throw runtime_error("rf_kernels::polynomial_detrender: epsilon=" + to_string(epsilon) + ", positive number expected");

    if (_unlikely(!intensity || !weights))
	throw runtime_error("rf_kernels::polynomial_detrender: null pointer passed to detrend()");

    this->_f(nfreq, nt, intensity, weights, stride, epsilon);
}


}  // namespace rf_kernels
