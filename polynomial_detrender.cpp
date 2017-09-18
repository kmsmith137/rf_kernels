// FIXME (low-priority) a nuisance issue when working with this code is that functions
// which are very similar have different argument orderings, e.g.
//
//          make_polynomial_detrender(nt_chunk, axis, polydeg, epsilon)
//   calls _make_polynomial_detrender(axis, nt_chunk, polydeg, epsilon)

#include <vector>
#include <string>
#include <unordered_map>

#include "rf_kernels/core.hpp"
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


// -------------------------------------------------------------------------------------------------


// Inner namespace for the kernel table.
// Must have a different name for each kernel, otherwise gcc is happy but clang has trouble!
namespace polynomial_detrender_kernel_table {
#if 0
}; // pacify emacs c-mode
#endif

// Usage: kernel(nfreq, nt, intensity, istride, weights, wstride, epsilon)
using kernel_t = void (*)(int, int, float *, int, float *, int, double);
 
// (axis, polydeg) -> kernel
static unordered_map<array<int,2>, kernel_t> kernel_table;


static kernel_t get_kernel(axis_type axis, int polydeg)
{
    auto p = kernel_table.find({{axis,polydeg}});
    
    if (_unlikely(p == kernel_table.end())) {
	stringstream ss;
	ss << "rf_kernels::polynomial_detrender: (axis,polydeg)=(" << axis << "," << polydeg << ") is invalid or unimplemented";
	throw runtime_error(ss.str());
    }
    
    return p->second;
}


template<int D, typename std::enable_if<(D<0),int>::type = 0>
inline void _populate() { }

template<int D, typename std::enable_if<(D>=0),int>::type = 0>
inline void _populate() 
{
    _populate<D-1> ();
    kernel_table[{{AXIS_FREQ,D}}] = _kernel_detrend_f<float,Sfid,D+1>;
    kernel_table[{{AXIS_TIME,D}}] = _kernel_detrend_t<float,Sfid,D+1>;
}


struct _initializer {
    _initializer() { _populate<20> (); }
} _init;

}  // namespace polynomial_detrender_kernel_table


// -------------------------------------------------------------------------------------------------


polynomial_detrender::polynomial_detrender(axis_type axis_, int polydeg_) :
    axis(axis_),
    polydeg(polydeg_),
    _f(polynomial_detrender_kernel_table::get_kernel(axis_, polydeg_))
{ }


void polynomial_detrender::detrend(int nfreq, int nt, float *intensity, int istride, float *weights, int wstride, double epsilon)
{
    if (_unlikely(nfreq <= 0))
	throw runtime_error("rf_kernels::polynomial_detrender: nfreq=" + to_string(nfreq) + ", positive number expected");

    if (_unlikely(nt <= 0))
	throw runtime_error("rf_kernels::polynomial_detrender: nt_chunk=" + to_string(nt) + ", positive number expected");
    
    if (_unlikely((nt % Sfid) != 0))
	throw runtime_error("rf_kernels::polynomial_detrender: nt_chunk=" + to_string(nt) + " must be a multiple of " + to_string(Sfid));
    
    if (_unlikely(abs(istride) < nt))
	throw runtime_error("rf_kernels::polynomial_detrender: istride=" + to_string(istride) + " is too small");
    
    if (_unlikely(abs(wstride) < nt))
	throw runtime_error("rf_kernels::polynomial_detrender: wstride=" + to_string(wstride) + " is too small");

    if (_unlikely(epsilon <= 0.0))
	throw runtime_error("rf_kernels::polynomial_detrender: epsilon=" + to_string(epsilon) + ", positive number expected");

    if (_unlikely(!intensity || !weights))
	throw runtime_error("rf_kernels::polynomial_detrender: null pointer passed to detrend()");

    this->_f(nfreq, nt, intensity, istride, weights, wstride, epsilon);
}


}  // namespace rf_kernels
