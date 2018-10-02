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
// Based on simd_helpers/quantize.hpp.


template<bool Tcounts>
struct first_pass_state 
{
    simd_downsampler<int,S,32,simd_bitwise_or<int,S>> ds;
    const simd_t<int,S> c;
    const simd_t<float,S> zero;
    simd_t<int,S> tcounts;  // only used if Tcounts=false

    first_pass_state() :
	c(_get_qmask<S>()),  // defined in simd_helpers/quantize.hpp
	zero(simd_t<float,S>::zero()),
	tcounts(simd_t<int,S>::zero())
    { }
};


template<int N, bool Tcounts, typename std::enable_if<(N==0),int>::type = 0>
inline void fp_put(first_pass_state<Tcounts> &fp, const float *in)
{ }

template<int N, bool Tcounts, typename std::enable_if<(N>0),int>::type = 0>
inline void fp_put(first_pass_state<Tcounts> &fp, const float *in)
{ 
    fp_put<N-1> (fp, in);

    simd_t<float,S> x = simd_load<float,S> (in + (N-1)*S);
    simd_t<int,S> y = simd_cast<int,S> (x > fp.zero);

    constexpr int L = ((N-1) % (32/S)) * S;
    simd_t<int,S> cs = fp.c << L;

    fp.ds.template put<N-1> (y & cs);

    if (Tcounts)
	fp.tcounts += y;
}


// -------------------------------------------------------------------------------------------------


template<bool Tcounts>
void first_pass_with_bm(const mask_counter &mc, const mask_counter::kernel_args &args)
{
    uint8_t *out_2d = mc.ini_params.save_bitmask ? args.out_bitmask : mc._bm_workspace.get();
    const int ostride = mc.ini_params.save_bitmask ? args.out_bmstride : (mc.ini_params.nt_chunk / 8);
    const int istride = args.istride;
    const int nfreq = mc.ini_params.nfreq;
    const int nt8 = mc.ini_params.nt_chunk / 8;
    
    rf_assert(out_2d != nullptr);
    rf_assert(mc.ini_params.nt_chunk % (32*S) == 0);

    first_pass_state<Tcounts> fp;

    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	const float *in_1d = args.in + ifreq * istride;
	uint8_t *out_1d = out_2d + ifreq * ostride;
	fp.tcounts = simd_t<int,S>::zero();

	for (int j = 0; j < nt8; j += 4*S) {
	    fp_put<32> (fp, in_1d + 8*j);
	    simd_store(reinterpret_cast<int *> (out_1d + j), fp.ds.get());
	}

	if (Tcounts)
	    args.out_tcounts[ifreq] = -(fp.tcounts.sum());
    }    
}


void first_pass_no_bm(const mask_counter &mc, const mask_counter::kernel_args &args)
{
    const int istride = args.istride;
    const int nfreq = mc.ini_params.nfreq;
    const int nt = mc.ini_params.nt_chunk;

    rf_assert(nt % S == 0);

    simd_t<float,S> zero = simd_t<float,S>::zero();

    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	const float *in = args.in + ifreq * istride;
	simd_t<int,S> counts = simd_t<int,S>::zero();

	for (int it = 0; it < nt; it += S) {
	    simd_t<float,S> x = simd_load<float,S> (in+it);
	    simd_t<float,S> y = (x > zero);
	    counts += simd_cast<int,S> (y);
	}

	args.out_tcounts[ifreq] = -(counts.sum());
    }
}


void second_pass(const mask_counter &mc, const mask_counter::kernel_args &args)
{
    return;
}


// -------------------------------------------------------------------------------------------------


mask_counter::mask_counter(const initializer &ini_params_) :
    ini_params(ini_params_)
{
    if (ini_params.nfreq <= 0)
	throw runtime_error("rf_kernels::mask_counter constructor: 'nfreq' was negative or unspecified");
    if (ini_params.nt_chunk <= 0)
	throw runtime_error("rf_kernels::mask_counter constructor: 'nt_chunk' was negative or unspecified");
    if (ini_params.nt_chunk % (32*S) != 0)
	throw runtime_error("rf_kernels::mask_counter constructor: 'nt_chunk' must be a multiple of " + to_string(32*S));
    if (!ini_params.save_bitmask && !ini_params.save_tcounts && !ini_params.save_fcounts)
	throw runtime_error("rf_kernels::mask_counter constructor: at least one of the flags 'save_bitmask', 'save_tcounts', 'save_fcounts' must be set");

    if (!ini_params.save_bitmask && !ini_params.save_fcounts)
	this->f_first_pass = first_pass_no_bm;
    else if (ini_params.save_tcounts)
	this->f_first_pass = first_pass_with_bm<true>;
    else
	this->f_first_pass = first_pass_with_bm<false>;
    
    if (!ini_params.save_bitmask && ini_params.save_fcounts)
	this->_bm_workspace = make_sptr<uint8_t> (ini_params.nfreq * (ini_params.nt_chunk / 8));
}


inline void check_args(const mask_counter::initializer &cargs, const mask_counter::kernel_args &kargs)
{
    if (_unlikely(!kargs.in))
	throw runtime_error("rf_kernels::mask_counter: 'in' argument is a null pointer");
    if (_unlikely(abs(kargs.istride) < cargs.nt_chunk))
	throw runtime_error("rf_kernels::mask_counter: 'istride' argument is uninitialized, or too small");
    
    if (_unlikely(cargs.save_bitmask && !kargs.out_bitmask))
	throw runtime_error("rf_kernels::mask_counter: 'save_bitmask' flag was set at construction, but 'out_bitmask' is a null pointer");
    if (_unlikely(!cargs.save_bitmask && kargs.out_bitmask))
	throw runtime_error("rf_kernels::mask_counter: 'save_bitmask' flag was false at construction, but 'out_bitmask' is a non-null pointer");
    if (_unlikely(cargs.save_bitmask && (abs(kargs.out_bmstride) < cargs.nt_chunk/8)))
	throw runtime_error("rf_kernels::mask_counter: 'out_bmstride' argument is uninitialized, or too small");
    
    if (_unlikely(cargs.save_tcounts && !kargs.out_tcounts))
	throw runtime_error("rf_kernels::mask_counter: 'save_tcounts' flag was set at construction, but 'out_tcounts' is a null pointer");
    if (_unlikely(!cargs.save_tcounts && kargs.out_tcounts))
	throw runtime_error("rf_kernels::mask_counter: 'save_tcounts' flag was false at construction, but 'out_tcounts' is a non-null pointer");

    if (_unlikely(cargs.save_fcounts && !kargs.out_fcounts))
	throw runtime_error("rf_kernels::mask_counter: 'save_fcounts' flag was set at construction, but 'out_fcounts' is a null pointer");
    if (_unlikely(!cargs.save_fcounts && kargs.out_fcounts))
	throw runtime_error("rf_kernels::mask_counter: 'save_fcounts' flag was false at construction, but 'out_fcounts' is a non-null pointer");
}


void mask_counter::mask_count(const kernel_args &args) const
{
    check_args(ini_params, args);

    this->f_first_pass(*this, args);
    this->f_second_pass(*this, args);
}


void mask_counter::slow_reference_mask_count(const kernel_args &args) const
{
    const int nfreq = ini_params.nfreq;
    const int nt = ini_params.nt_chunk;
    const int istride = args.istride;
    const int bmstride = args.out_bmstride;
    const float *in = args.in;

    uint8_t *out_bm = args.out_bitmask;
    int *out_tc = args.out_tcounts;
    int *out_fc = args.out_fcounts;

    check_args(ini_params, args);

    if (out_bm != nullptr) {
	for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	    for (int i = 0; i < nt/8; i++) {
		uint8_t iout = 0;
		for (int j = 0; j < 8; j++)
		    if (in[ifreq*istride + 8*i + j] > 0.0f)
			iout |= (1 << j);

		out_bm[ifreq*bmstride + i] = iout;
	    }
	}
    }

    if (out_tc != nullptr) {
	for (int it = 0; it < nt; it++) {
	    int count = 0;
	    for (int ifreq = 0; ifreq < nfreq; ifreq++)
		if (in[ifreq*istride + it] > 0.0f)
		    count++;
	    out_tc[it] = count;
	}
    }

    if (out_fc != nullptr) {
	for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	    int count = 0;
	    for (int it = 0; it < nt; it++)
		if (in[ifreq*istride + it] > 0.0f)
		    count++;
	    out_fc[ifreq] = count;
	}
    }
}


}  // namespace rf_kernels
