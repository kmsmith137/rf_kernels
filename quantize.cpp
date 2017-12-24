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


template<int S, int B>
void _quantize_kernel(int nfreq, int nt, uint8_t *out, int ostride, const float *in, int istride)
{
    // Assumes all arguments have already been checked.
    // See quantizer::quantize() below.

    constexpr int kstep_out = (4 * S);      // Output uint8's per kernel
    constexpr int kstep_in = (32 * S) / B;  // Input float32's per kernel

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
static int safe_kernel_size(int nbits)
{
    if (nbits != 1)
	throw runtime_error("rf_kernels::quantizer: currently only nbits=1 is implemented!");

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


void quantizer::quantize(int nfreq, int nt, uint8_t *out, int ostride, const float *in, int istride) const
{
    if (_unlikely(nfreq <= 0))
	throw runtime_error("rf_kernels::quantizer: expected nfreq > 0");

    if (_unlikely(nt <= 0))
	throw runtime_error("rf_kernels::quantizer: expected nt > 0");

    if (_unlikely(nt % kernel_size))
	throw runtime_error("rf_kernels::quantizer: expected nt divisible by kernel_size (=" + to_string(kernel_size) + ")");

    if (_unlikely(abs(ostride) < (nt*nbits)/8))
	throw runtime_error("rf_kernels::quantizer: ostride is too small");

    // Not sure whether this one is actually necessary.
    if (_unlikely(ostride % 4))
	throw runtime_error("rf_kernels::quantizer: expected ostride divisible by 4");

    if (_unlikely(abs(istride) < nt))
	throw runtime_error("rf_kernels::quantizer: istride is too small");

    if (_unlikely(!out))
	throw runtime_error("rf_kernels: null 'out' pointer passed to quantizer::quantize()");

    if (_unlikely(!in))
	throw runtime_error("rf_kernels: null 'out' pointer passed to quantizer::quantize()");

    this->_f(nfreq, nt, out, ostride, in, istride);
}


}  // namespace rf_kernels
