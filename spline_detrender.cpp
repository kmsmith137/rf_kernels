#include <iostream>

#include "rf_kernels/internals.hpp"
#include "rf_kernels/spline_detrender.hpp"

#if 0
#ifdef __AVX__
constexpr unsigned int S = 8;
#else
constexpr unsigned int S = 4;
#endif
#endif

using namespace std;

namespace rf_kernels {
#if 0
};  // pacify emacs c-mode
#endif


void _spline_detrender_init(int *bin_delim, float *poly_vals, int nx, int nbins)
{
    for (int b = 0; b <= nbins; b++) {
	double t = double(b) / double(nbins) * double(nx);
	bin_delim[b] = int(t + 0.5);
    }
    
    rf_assert(bin_delim[0] == 0);
    rf_assert(bin_delim[nbins] == nx);

    for (int b = 0; b < nbins; b++)
	rf_assert(bin_delim[b] < bin_delim[b+1]);

    for (int b = 0; b < nbins; b++) {
	for (int i = bin_delim[b]; i < bin_delim[b+1]; i++) {
	    double x = nbins * double(i+0.5) / double(nx) - b;

	    rf_assert(x > -1.0e-10);
	    rf_assert(x < 1.0 + 1.0e-10);
	    
	    poly_vals[4*i] = (1-x) * (1-x) * (1+2*x);
	    poly_vals[4*i+1] = (1-x) * (1-x) * x;
	    poly_vals[4*i+2] = x*x * (3 - 2*x);
	    poly_vals[4*i+3] = x*x * (x - 1);
	}
    }
}


}  // namespace rf_kernels
