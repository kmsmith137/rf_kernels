#include <stdexcept>
#include "rf_kernels/quantize.hpp"
#include "rf_kernels/internals.hpp"
#include "simd_helpers/quantize.hpp"

using namespace std;
using namespace simd_helpers;

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


// -------------------------------------------------------------------------------------------------
//
// Quantizer


template<int S, int B>
void _quantize_kernel(int nfreq, int nt, uint8_t *out, int ostride, const float *in, int istride)
{
    // Assumes all arguments have already been checked.
    // See quantizer::quantize() below.

    constexpr int kstep_out = (4 * S);      // Output uint8's per kernel.
    constexpr int kstep_in = (32 * S) / B;  // Input float32's per kernel.

    simd_quantizer<float,S,B> q;
    int nk = nt / kstep_in;

    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	for (int i = 0; i < nk; i++) {
	    simd_t<int,S> t = q.quantize(in + ifreq*istride + i*kstep_in);
	    simd_store(reinterpret_cast<int *> (out + ifreq*ostride + i*kstep_out), t);
	}
    }
}


// Helper for quantizer constructor.
inline int safe_kernel_size(int nbits)
{
    if (nbits != 1)
	throw runtime_error("rf_kernels::(de)quantizer: currently only nbits=1 is implemented!");

#ifdef __AVX__
    return 256 / nbits;
#else
    return 128 / nbits;
#endif
}


quantizer::quantizer(int nbits_) :
    nbits(nbits_),
    kernel_size(safe_kernel_size(nbits_))
{
    // Currently, only nbits=1 is implemented.
    // (This is checked in safe_kernel_size().)
#ifdef __AVX__
    this->_f = _quantize_kernel<8,1>;
#else
    this->_f = _quantize_kernel<4,1>;
#endif
}


// Called by quantizer::quantize() and quantizer::slow_reference_quantize(), to check arguments.
inline void _quantize_argument_checks(const quantizer *self, int nfreq, int nt, uint8_t *out, int ostride, const float *in, int istride)
{
    if (_unlikely(nfreq <= 0))
	throw runtime_error("rf_kernels::quantizer: expected nfreq > 0");

    if (_unlikely(nt <= 0))
	throw runtime_error("rf_kernels::quantizer: expected nt > 0");

    if (_unlikely(nt % self->kernel_size))
	throw runtime_error("rf_kernels::quantizer: expected nt divisible by kernel_size (=" + to_string(self->kernel_size) + ")");

    if (_unlikely(abs(ostride) < (nt * self->nbits) / 8))
	throw runtime_error("rf_kernels::quantizer: ostride is too small");

    // FIXME not sure whether this one is actually necessary.
    if (_unlikely(ostride % 4))
	throw runtime_error("rf_kernels::quantizer: expected ostride divisible by 4");

    if (_unlikely(abs(istride) < nt))
	throw runtime_error("rf_kernels::quantizer: istride is too small");

    if (_unlikely(!out))
	throw runtime_error("rf_kernels::quantizer: null 'out' pointer passed to quantizer::quantize()");

    if (_unlikely(!in))
	throw runtime_error("rf_kernels::quantizer: null 'in' pointer passed to quantizer::quantize()");    
}


void quantizer::quantize(int nfreq, int nt, uint8_t *out, int ostride, const float *in, int istride) const
{
    _quantize_argument_checks(this, nfreq, nt, out, ostride, in, istride);
    this->_f(nfreq, nt, out, ostride, in, istride);
}


void quantizer::slow_reference_quantize(int nfreq, int nt, uint8_t *out, int ostride, const float *in, int istride) const
{
    _quantize_argument_checks(this, nfreq, nt, out, ostride, in, istride);

    rf_assert(nbits == 1);
    rf_assert(nt % 8 == 0);

    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	for (int i = 0; i < nt/8; i++) {
	    uint8_t iout = 0;
	    for (int j = 0; j < 8; j++)
		if (in[ifreq*istride + 8*i + j] > 0.0f)
		    iout |= (1 << j);

	    out[ifreq*ostride + i] = iout;
	}
    }
}


// -------------------------------------------------------------------------------------------------
//
// Dequantizer


template<int S>
void _apply_bitmask_kernel(int nfreq, int nt, float *out, int ostride, const uint8_t *in, int istride)
{
    // Assumes all arguments have already been checked.
    // See dequantizer::apply_bitmask() below.

    constexpr int kstep_in = 4*S;     // Input uint8's per kernel.
    constexpr int kstep_out = 32*S;   // Output float32's per kernel.
    
    simd_dequantizer<float,S,1> dq;
    int nk = nt / kstep_out;

    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	for (int i = 0; i < nk; i++) {
	    dq.put(simd_load<int,S> (reinterpret_cast<const int *> (in + ifreq*istride + i*kstep_in)));
	    dq.apply_bitmask(out + ifreq*ostride + i*kstep_out);
	}
    }
}


dequantizer::dequantizer(int nbits_) :
    nbits(nbits_),
    kernel_size(safe_kernel_size(nbits_))
{
    // Currently, only nbits=1 is implemented.
    // (This is checked in safe_kernel_size().)
#ifdef __AVX__
    this->_f_bm = _apply_bitmask_kernel<8>;
#else
    this->_f_bm = _apply_bitmask_kernel<4>;
#endif    
}


void dequantizer::apply_bitmask(int nfreq, int nt, float *out, int ostride, const uint8_t *in, int istride) const
{
    if (_unlikely(nfreq <= 0))
	throw runtime_error("rf_kernels::dequantizer: expected nfreq > 0");

    if (_unlikely(nt <= 0))
	throw runtime_error("rf_kernels::dequantizer: expected nt > 0");

    if (_unlikely(nt % kernel_size))
	throw runtime_error("rf_kernels::dequantizer: expected nt divisible by kernel_size (=" + to_string(kernel_size) + ")");

    if (_unlikely(abs(ostride) < nt))
	throw runtime_error("rf_kernels::dequantizer: ostride is too small");

    if (_unlikely(abs(istride) < nt/8))
	throw runtime_error("rf_kernels::dequantizer: istride is too small");

    // Not sure whether this one is actually necessary.
    if (_unlikely(istride % 4))
	throw runtime_error("rf_kernels::dequantizer: expected istride divisible by 4");

    if (_unlikely(!out))
	throw runtime_error("rf_kernels: null 'out' pointer passed to dequantizer::apply_bitmask()");

    if (_unlikely(!in))
	throw runtime_error("rf_kernels: null 'out' pointer passed to dequantizer::apply_bitmask()");

    this->_f_bm(nfreq, nt, out, ostride, in, istride);
}


}  // namespace rf_kernels
