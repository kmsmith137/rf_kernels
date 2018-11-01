#include "rf_kernels/internals.hpp"
#include "rf_kernels/mask_counter.hpp"
#include <simd_helpers/quantize.hpp>

using namespace std;
using namespace simd_helpers;

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


constexpr int S = simd_size<float> ();


// -------------------------------------------------------------------------------------------------
//
// mask_count_without_bm(): implements mask_counter_data::mask_count(), in the case where the bitmask
// is not being computed (mask_counter_data::out_bitmask is a null pointer).


static int mask_count_without_bm(const mask_counter_data &d)
{
    int nfreq = d.nfreq;
    int nt = d.nt_chunk;
    int istride = d.istride;

    const float *in_2d = d.in;
    int *out_fcounts = d.out_fcounts;

    rf_assert(nt % S == 0);

    simd_t<float,S> zero = simd_t<float,S>::zero();
    simd_t<int,S> total = simd_t<int,S>::zero();

    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	const float *in_1d = in_2d + ifreq * istride;
	simd_t<int,S> subtotal = simd_t<int,S>::zero();

	for (int it = 0; it < nt; it += S) {
	    simd_t<float,S> x = simd_load<float,S> (in_1d + it);
	    subtotal += simd_cast<int,S> (x > zero);
	}

	if (out_fcounts != nullptr)
	    out_fcounts[ifreq] = -(subtotal.sum());

	total += subtotal;
    }
    
    return -(total.sum());
}


// -------------------------------------------------------------------------------------------------
//
// mask_count_with_bm(): implements mask_counter_data::mask_count(), in the case where the bitmask
// is being computed (mask_counter_data::out_bitmask is a non-null pointer).
//
// The bm_state helper class is a version of the 1-bit simd_quantizer (see simd_helpers/quantize.hpp)
// which keeps a 'subtotal' of unmasked entries.


struct bm_state 
{
    simd_downsampler<int,S,32,simd_bitwise_or<int,S>> ds;
    const simd_t<int,S> c;
    const simd_t<float,S> zero;
    simd_t<int,S> subtotal;

    bm_state() :
	c(_get_qmask<S>()),  // defined in simd_helpers/quantize.hpp
	zero(simd_t<float,S>::zero()),
	subtotal(simd_t<int,S>::zero())
    { }
};


template<int N, typename std::enable_if<(N==0),int>::type = 0>
inline void bm_put(bm_state &bm, const float *in)
{ }

template<int N, typename std::enable_if<(N>0),int>::type = 0>
inline void bm_put(bm_state &bm, const float *in)
{ 
    bm_put<N-1> (bm, in);

    simd_t<float,S> x = simd_load<float,S> (in + (N-1)*S);
    simd_t<int,S> y = simd_cast<int,S> (x > bm.zero);

    constexpr int L = ((N-1) % (32/S)) * S;
    simd_t<int,S> cs = bm.c << L;

    bm.ds.template put<N-1> (y & cs);
    bm.subtotal += y;
}


static int mask_count_with_bm(const mask_counter_data &d)
{
    int nfreq = d.nfreq;
    int nt8 = d.nt_chunk / 8;
    int istride = d.istride;
    int ostride = d.out_bmstride;

    const float *in_2d = d.in;
    uint8_t *out_2d = d.out_bitmask;
    int *out_fc = d.out_fcounts;
    
    rf_assert(out_2d != nullptr);
    rf_assert(d.nt_chunk % (32*S) == 0);

    bm_state bm;
    simd_t<int,S> total = simd_t<int,S>::zero();

    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	const float *in_1d = in_2d + ifreq * istride;
	uint8_t *out_1d = out_2d + ifreq * ostride;

	bm.subtotal = simd_t<int,S>::zero();

	for (int j = 0; j < nt8; j += 4*S) {
	    bm_put<32> (bm, in_1d + 8*j);
	    simd_store(reinterpret_cast<int *> (out_1d + j), bm.ds.get());
	}

	if (out_fc)
	    out_fc[ifreq] = -(bm.subtotal.sum());

	total += bm.subtotal;
    }    

    return -(total.sum());
}


// -------------------------------------------------------------------------------------------------


void mask_counter_data::check_args() const
{
    if (_unlikely(nfreq <= 0))
	throw runtime_error("rf_kernels::mask_counter: 'nfreq' was negative or unspecified");
    if (_unlikely(nt_chunk <= 0))
	throw runtime_error("rf_kernels::mask_counter: 'nt_chunk' was negative or unspecified");
    if (_unlikely(nt_chunk % (32*S) != 0))
	throw runtime_error("rf_kernels::mask_counter: 'nt_chunk' must be a multiple of " + to_string(32*S));
    if (_unlikely(!in))
	throw runtime_error("rf_kernels::mask_counter: 'in' argument is a null pointer");
    if (_unlikely(abs(istride) < nt_chunk))
	throw runtime_error("rf_kernels::mask_counter: 'istride' argument is uninitialized, or too small");
    if (_unlikely(out_bitmask && (abs(out_bmstride) < nt_chunk/8)))
	throw runtime_error("rf_kernels::mask_counter: 'out_bmstride' argument is uninitialized, or too small");
}


int mask_counter_data::mask_count() const
{
    check_args();
    return out_bitmask ? mask_count_with_bm(*this) : mask_count_without_bm(*this);
}


int mask_counter_data::slow_reference_mask_count() const
{
    check_args();

    if (out_bitmask != nullptr) {
	for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	    for (int ibyte = 0; ibyte < nt_chunk/8; ibyte++) {
		uint8_t byte = 0;
		for (int ibit = 0; ibit < 8; ibit++)
		    if (in[ifreq*istride + 8*ibyte + ibit] > 0.0f)
			byte |= (1 << ibit);

		out_bitmask[ifreq*out_bmstride + ibyte] = byte;
	    }
	}
    }

    if (out_fcounts != nullptr) {
	for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	    int subtotal = 0;
	    for (int it = 0; it < nt_chunk; it++)
		if (in[ifreq*istride + it] > 0.0f)
		    subtotal++;
	    out_fcounts[ifreq] = subtotal;
	}
    }

    int total = 0;
    for (int ifreq = 0; ifreq < nfreq; ifreq++)
	for (int it = 0; it < nt_chunk; it++)
	    if (in[ifreq*istride + it] > 0.0f)
		total++;

    return total;
}


}  // namespace rf_kernels
