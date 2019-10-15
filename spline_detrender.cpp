#include <iostream>

#include "rf_kernels/core.hpp"
#include "rf_kernels/internals.hpp"
#include "rf_kernels/spline_detrender.hpp"
#include "rf_kernels/spline_detrender_internals.hpp"

using namespace std;

namespace rf_kernels {
#if 0
};  // pacify emacs c-mode
#endif

// FIXME assumes AVX2
constexpr int S = 8;


spline_detrender::spline_detrender(int nfreq_, int nbins_, float epsilon_) :
    nfreq(nfreq_),
    nbins(nbins_),
    epsilon(epsilon_)
{
    if (nfreq <= 0)
	throw runtime_error("rf_kernels::spline_detrender: expected nfreq > 0");
    if (nbins <= 0)
	throw runtime_error("rf_kernels::spline_detrender: expected nbins > 0");
    if (nfreq < 16 * nbins)
	throw runtime_error("rf_kernels::spline_detrender: expected nfreq >= 16 * nbins");

    // FIXME improve the epsilon asserts
    if (epsilon <= 0)
	throw runtime_error("rf_kernels::spline_detrender: expected epsilon >= 0");

    // Allocate

    int bd_size = (nbins+1) * sizeof(int);
    int pv_size = nfreq * 4 * sizeof(float);
    int ninv_size = nbins * 10 * S * sizeof(float);
    int ninvx_size = nbins * 4 * S * sizeof(float);
    int cdiag_size = (nbins+1) * 3 * S * sizeof(float);
    int csdiag_size = nbins * 4 * S * sizeof(float);
    int coeffs_size = (nbins+1) * 2 * S * sizeof(float);

    int pv_offset = _align(bd_size, 64);
    int ninv_offset = pv_offset + _align(pv_size, 64);
    int ninvx_offset = ninv_offset + _align(ninv_size, 64);
    int cdiag_offset = ninvx_offset + _align(ninvx_size, 64);
    int csdiag_offset = cdiag_offset + _align(cdiag_size, 64);
    int coeffs_offset = csdiag_offset + _align(csdiag_size, 64);
    int end_offset = coeffs_offset + _align(coeffs_size, 64);

    this->allocated_memory = aligned_alloc<uint8_t> (end_offset);
    this->bin_delim = reinterpret_cast<int *> (allocated_memory);
    this->poly_vals = reinterpret_cast<float *> (allocated_memory + pv_offset);
    this->ninv = reinterpret_cast<float *> (allocated_memory + ninv_offset);
    this->ninvx = reinterpret_cast<float *> (allocated_memory + ninvx_offset);
    this->cholesky_invdiag = reinterpret_cast<float *> (allocated_memory + cdiag_offset);
    this->cholesky_subdiag = reinterpret_cast<float *> (allocated_memory + csdiag_offset);
    this->coeffs = reinterpret_cast<float *> (allocated_memory + coeffs_offset);

    _spline_detrender_init(bin_delim, poly_vals, nfreq, nbins);
}


spline_detrender::~spline_detrender()
{
    free(allocated_memory);

    this->bin_delim = nullptr;
    this->poly_vals = nullptr;
    this->ninv = nullptr;
    this->ninvx = nullptr;
    this->cholesky_invdiag = nullptr;
    this->cholesky_subdiag = nullptr;
    this->coeffs = nullptr;
    this->allocated_memory = nullptr;
}


void spline_detrender::detrend(int nt_chunk, float *intensity, int istride, const float *weights, int wstride, float* save_coeffs)
{
    if (nt_chunk <= 0)
	throw runtime_error("rf_kernels::spline_detrender: nt_chunk must be > 0");
    if (nt_chunk % 8 != 0)
	throw runtime_error("rf_kernels::spline_detrender: nt_chunk must be divisible by 8");
    if (!intensity || !weights)
	throw runtime_error("rf_kernels: null pointer passed to spline_detrender::detrend()");
    if (abs(istride) < nt_chunk)
	throw runtime_error("rf_kernels::spline_detrender: expected abs(istride) >= nt_chunk");
    if (abs(wstride) < nt_chunk)
	throw runtime_error("rf_kernels::spline_detrender: expected abs(wstride) >= nt_chunk");

    for (int it = 0; it < nt_chunk; it += 8) {
	_kernel_ninv(intensity + it, istride, weights + it, wstride);
	_kernel_fit_pass1();
	_kernel_fit_pass2();
	_kernel_fit_pass3();
	_kernel_detrend(intensity + it, istride);

        if (save_coeffs) {
            int ncoeffs = (nbins+1)*2;
            memcpy(save_coeffs + it*ncoeffs, coeffs, ncoeffs * 8);
        }
    }
}


// Helper function for spline_detrender constructor
void _spline_detrender_init(int *bin_delim, float *poly_vals, int nx, int nbins)
{
    if (nx <= 0)
	throw runtime_error("rf_kernels::_spline_detrender_init: expected nx > 0");
    if (nbins <= 0)
	throw runtime_error("rf_kernels::_spline_detrender_init: expected nbins > 0");
    if (nx < 16 * nbins)
	throw runtime_error("rf_kernels::_spline_detrender_init: expected nx >= 16 * nbins");

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
