#ifndef _RF_KERNELS_DOWNSAMPLE_INTERNALS_HPP
#define _RF_KERNELS_DOWNSAMPLE_INTERNALS_HPP

#include <simd_helpers/simd_float32.hpp>
#include <simd_helpers/simd_ntuple.hpp>
#include <simd_helpers/udsample.hpp>
#include <simd_helpers/downsample.hpp>


namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif

template<typename T, int S> using simd_t = simd_helpers::simd_t<T,S>;
template<typename T, int S, int D> using simd_ntuple = simd_helpers::simd_ntuple<T,S,D>;
template<typename T, int S, int D> using simd_downsampler = simd_helpers::simd_downsampler<T,S,D>;


// -------------------------------------------------------------------------------------------------
//
// _kernel_downsample1<T,S,R,N> (simd_t<T,S> &ds_wi, simd_t<T,S> &ds_w, const T *intensity, const T *weights, int stride)
//
// Reads a strided array of shape (R,N*S), and sums the result over outer index r
// and middle index N, returning a simd_t<T,S>.


// The "1a" variant accumulates its result
template<typename T, int S, int R, int N, typename std::enable_if<(R==0 || N==0),int>::type = 0>
inline void _kernel_downsample1a(simd_t<T,S> &ds_wi, simd_t<T,S> &ds_w, const T *intensity, const T *weights, int stride)
{
    return;
}


template<typename T, int S, int R, int N, typename std::enable_if<(R > 0 && N > 0),int>::type = 0>
inline void _kernel_downsample1a(simd_t<T,S> &ds_wi, simd_t<T,S> &ds_w, const T *intensity, const T *weights, int stride)
{
    simd_t<T,S> ival = simd_helpers::simd_load<T,S> (intensity);
    simd_t<T,S> wval = simd_helpers::simd_load<T,S> (weights);

    ds_wi += wval * ival;
    ds_w += wval;

    _kernel_downsample1a<T,S,R-1,1> (ds_wi, ds_w, intensity+stride, weights+stride, stride);
    _kernel_downsample1a<T,S,R,N-1> (ds_wi, ds_w, intensity+S, weights+S, stride);
}


template<typename T, int S, int R, int N, typename std::enable_if<(R > 0 && N > 0),int>::type = 0>
inline void _kernel_downsample1(simd_t<T,S> &ds_wi, simd_t<T,S> &ds_w, const T *intensity, const T *weights, int stride)
{
    simd_t<T,S> ival = simd_helpers::simd_load<T,S> (intensity);
    simd_t<T,S> wval = simd_helpers::simd_load<T,S> (weights);
    
    ds_wi = wval * ival;
    ds_w = wval;
    
    _kernel_downsample1a<T,S,R-1,1> (ds_wi, ds_w, intensity+stride, weights+stride, stride);
    _kernel_downsample1a<T,S,R,N-1> (ds_wi, ds_w, intensity+S, weights+S, stride);
}


// -------------------------------------------------------------------------------------------------
//
// _kernel_downsample2<T,S,R,D,N> (simd_ntuple<T,S,D> &ds_wi, simd_ntuple<T,S,D> &ds_w, const T *intensity, const T *weights, int stride)
//
// Reads a strided array of shape (R,D*N*S), and sums the result over outer index r
// and middle index n, returning a simd_ntuple<T,S,D>.


template<typename T, int S, int R, int D, int N, typename std::enable_if<(D==0),int>::type = 0>
inline void _kernel_downsample2(simd_ntuple<T,S,D> &ds_wi, simd_ntuple<T,S,D> &ds_w, const T *intensity, const T *weights, int stride)
{
    return;
}


template<typename T, int S, int R, int D, int N, typename std::enable_if<(D>0),int>::type = 0>
inline void _kernel_downsample2(simd_ntuple<T,S,D> &ds_wi, simd_ntuple<T,S,D> &ds_w, const T *intensity, const T *weights, int stride)
{
    _kernel_downsample2<T,S,R,D-1,N> (ds_wi.v, ds_w.v, intensity, weights, stride);
    _kernel_downsample1<T,S,R,N> (ds_wi.x, ds_w.x, intensity + (D-1)*N*S, weights + (D-1)*N*S, stride);
}


// -------------------------------------------------------------------------------------------------
//
// _kernel_downsample<T,S,R,D> (simd_t<T,S> &ds_wi, simd_t<T,S> &ds_w, const T *intensity, const T *weights, int stride)
//
// Reads a strided array of shape (R,D*S), and sums the result over outer index r and inner index d, 
// returning a simd_t<T,S>.

 
template<typename T, int S, int R, int D, typename std::enable_if<(D==1),int>::type = 0>
inline void _kernel_downsample(simd_t<T,S> &ds_wi, simd_t<T,S> &ds_w, const T *intensity, const T *weights, int stride)
{
    _kernel_downsample1<T,S,R,1> (ds_wi, ds_w, intensity, weights, stride);
}


template<typename T, int S, int R, int D, typename std::enable_if<(D>1 && D<=S),int>::type = 0>
inline void _kernel_downsample(simd_t<T,S> &ds_wi, simd_t<T,S> &ds_w, const T *intensity, const T *weights, int stride)
{
    simd_ntuple<T,S,D> dsn_wi, dsn_w;
    _kernel_downsample2<T,S,R,D,1> (dsn_wi, dsn_w, intensity, weights, stride);

    ds_wi = simd_helpers::downsample(dsn_wi);   // defined in simd_helpers/udsample.hpp
    ds_w = simd_helpers::downsample(dsn_w);
}


template<typename T, int S, int R, int D, typename std::enable_if<(D>S),int>::type = 0>
inline void _kernel_downsample(simd_t<T,S> &ds_wi, simd_t<T,S> &ds_w, const T *intensity, const T *weights, int stride)
{
    simd_ntuple<T,S,S> dsn_wi, dsn_w;
    _kernel_downsample2<T,S,R,S,D/S> (dsn_wi, dsn_w, intensity, weights, stride);

    ds_wi = simd_helpers::downsample(dsn_wi);
    ds_w = simd_helpers::downsample(dsn_w);
}


// -------------------------------------------------------------------------------------------------
//
// _kernel_downsample_2d<T,S,Df,Dt> (out_intensity, out_weights, out_stride, in_intensity, in_nfreq, in_nt, in_stride)
//
// Caller must check that nfreq is divisible by Df, and nt is divisible by (Dt*S).
//
// This is the kernel which gets called in the externally visible function wi_downsample().


template<typename T, int S, int Df, int Dt>
inline void _kernel_downsample_2d(T *out_intensity, T *out_weights, int out_stride, const T *in_intensity, const T *in_weights, int in_nfreq, int in_nt, int in_stride)
{
    const simd_t<T,S> zero = simd_t<T,S>::zero();
    const simd_t<T,S> one = simd_t<T,S> (1.0);

    int out_nfreq = in_nfreq / Df;
    int out_nt = in_nt / Dt;

    for (int ifreq = 0; ifreq < out_nfreq; ifreq++) {
	T *out_irow = out_intensity + ifreq * out_stride;
	T *out_wrow = out_weights + ifreq * out_stride;

	const T *in_irow = in_intensity + (ifreq*Df) * in_stride;
	const T *in_wrow = in_weights + (ifreq*Df) * in_stride;

	for (int it = 0; it < out_nt; it += S) {
	    simd_t<T,S> ds_wival, ds_wval;
	    _kernel_downsample<T,S,Df,Dt> (ds_wival, ds_wval, in_irow + it*Dt, in_wrow + it*Dt, in_stride);

	    simd_t<T,S> ds_ival = ds_wival / blendv(ds_wval.compare_gt(zero), ds_wval, one);
	    ds_ival.storeu(out_irow + it);
	    ds_wval.storeu(out_wrow + it);
	}
    }
}


// -------------------------------------------------------------------------------------------------
//
// _wi_downsample_0a<Df> (wi_acc, w_acc, intensity, weights, stride)
// _wi_downsample_0a<Df> (wi_acc, w_acc, intensity, weights, stride, N)
//
// These sum over a shape-(Df,S) and shape-(Df,N) array respectively.
// In the latter case, caller must check that N is divisible by the simd size S!
// No downsampling kernels are used -- the sum is purely "vertical".


template<int Df, typename T, int S, typename std::enable_if<(Df==0),int>::type = 0>
inline void _wi_downsample_0a(simd_t<T,S> &wi_acc, simd_t<T,S> &w_acc, const T *in_i, const T *in_w, int stride)
{ }

template<int Df, typename T, int S, typename std::enable_if<(Df>0),int>::type = 0>
inline void _wi_downsample_0a(simd_t<T,S> &wi_acc, simd_t<T,S> &w_acc, const T *in_i, const T *in_w, int stride)
{
    _wi_downsample_0a<Df-1> (wi_acc, w_acc, in_i, in_w, stride);
    
    simd_t<T,S> ival = simd_helpers::simd_load<T,S> (in_i + (Df-1)*stride);
    simd_t<T,S> wval = simd_helpers::simd_load<T,S> (in_w + (Df-1)*stride);

    wi_acc += wval * ival;
    w_acc += wval;
}

template<int Df, typename T, int S>
inline void _wi_downsample_0a(simd_t<T,S> &wi_acc, simd_t<T,S> &w_acc, const T *in_i, const T *in_w, int stride, int N)
{
    for (int it = 0; it < N; it += S)
	_wi_downsample_0a<Df> (wi_acc, w_acc, in_i + it, in_w + it, stride);
}
			      
// ----------------------------------------------------------------------------------------------------
//
// _wi_downsampler_0d_Dtsm<T,S,Df,Dt>::get(wi_out, w_out, intensity, weights, stride, Dt)
// _wi_downsampler_0d_Dtlg<T,S,Df>::get(wi_out, w_out, intensity, weights, stride, Dt)
//
// These ingest shape-(Df,Dt*S) intensity and weights arrays, downsample, and
// output length-S intensity and weights vectors.
//
// In the first case, Dt is a compile-time parameter, and Dt <= S.
// In the second case, Dt is a runtime parameter, and Dt is a multiple of S.
//
// NOTE: they now accumulate their output!


template<typename T_, int S_, int Df_, int Dt>
struct _wi_downsampler_0d_Dtsm {
    using T = T_;
    static constexpr int S = S_;
    static constexpr int Df = Df_;

    constexpr int get_Dt() const { return Dt; }

    
    template<int P, typename std::enable_if<(P==0),int>::type = 0>
    inline void _get_partial(simd_downsampler<T,S,Dt> &wi_ds, simd_downsampler<T,S,Dt> &w_ds, const T *in_i, const T *in_w, int stride) const
    { }

    template<int P, typename std::enable_if<(P>0),int>::type = 0>
    inline void _get_partial(simd_downsampler<T,S,Dt> &wi_ds, simd_downsampler<T,S,Dt> &w_ds, const T *in_i, const T *in_w, int stride) const
    {
	_get_partial<P-1> (wi_ds, w_ds, in_i, in_w, stride);
	
	simd_t<T,S> wi_acc = simd_t<T,S>::zero();
	simd_t<T,S> w_acc = simd_t<T,S>::zero();
	_wi_downsample_0a<Df> (wi_acc, w_acc, in_i + (P-1)*S, in_w + (P-1)*S, stride);
	
	wi_ds.template put<P-1> (wi_acc);
	w_ds.template put<P-1> (w_acc);
    }

    
    inline void get(simd_t<T,S> &wi_out, simd_t<T,S> &w_out, const T *i_in, const T *w_in, int stride) const
    {
	simd_downsampler<T,S,Dt> wi_ds, w_ds;
	_get_partial<Dt> (wi_ds, w_ds, i_in, w_in, stride);

	wi_out += wi_ds.get();
	w_out += w_ds.get();
    }
};


template<typename T_, int S_, int Df_>
struct _wi_downsampler_0d_Dtlg
{
    using T = T_;
    static constexpr int S = S_;
    static constexpr int Df = Df_;
    
    const int Dt;
    _wi_downsampler_0d_Dtlg(int Dt_) : Dt(Dt_) { }

    inline int get_Dt() const { return Dt; }

    
    template<int P, typename std::enable_if<(P==0),int>::type = 0>
    inline void _get_partial(simd_downsampler<T,S,S> &wi_ds, simd_downsampler<T,S,S> &w_ds, const T *in_i, const T *in_w, int stride) const
    { }
    
    template<int P, typename std::enable_if<(P>0),int>::type = 0>
    inline void _get_partial(simd_downsampler<T,S,S> &wi_ds, simd_downsampler<T,S,S> &w_ds, const T *in_i, const T *in_w, int stride) const
    {
	_get_partial<P-1> (wi_ds, w_ds, in_i, in_w, stride);
	
	simd_t<T,S> wi_acc = simd_t<T,S>::zero();
	simd_t<T,S> w_acc = simd_t<T,S>::zero();
	_wi_downsample_0a<Df> (wi_acc, w_acc, in_i + (P-1)*Dt, in_w + (P-1)*Dt, stride, Dt);
	
	wi_ds.template put<P-1> (wi_acc);
	w_ds.template put<P-1> (w_acc);    
    }

    
    inline void get(simd_t<T,S> &wi_out, simd_t<T,S> &w_out, const T *i_in, const T *w_in, int stride) const
    {
	simd_downsampler<T,S,S> wi_ds, w_ds;
	_get_partial<S> (wi_ds, w_ds, i_in, w_in, stride);
	
	wi_out += wi_ds.get();
	w_out += w_ds.get();
    }
};


// -------------------------------------------------------------------------------------------------
//
// _wi_downsample_1d<T,S,Df,Dt,Iflag,Fflag> (i_out, w_out, nt_out, i_in, w_in, istride)
// _wi_downsample_1d<T,S,Df,Iflag,Fflag> (i_out, w_out, nt_out, i_in, w_in, istride, Dt)
//
// These ingest shape-(Df,Dt*nt_out) intensity arrays, downsample, and output
// a length-nt_out array.
//
// In the first case, Dt is a compile-time parameter, and Dt <= S.
// In the second case, Dt is a runtime parameter, and Dt is a multiple of S.
//
// The Iflag, Fflag arguments can be used if this is a "multi-pass" downsampling kernel.
// In this case, Iflag is true if this is the first pass, and Fflag is true if this is
// the last pass.
//
// Caller must check that nt_out is a multiple of S.  (Note that nt_in = nt_out * Dt).
//
// FIXME there is a lot of cut-and-paste between these two kernels, can this be improved?


template<typename Tds0>
struct _wi_downsampler_1d_Dfsm {    
    using T = typename Tds0::T;
    static constexpr int S = Tds0::S;

    const Tds0 ds0;
    
    _wi_downsampler_1d_Dfsm(const Tds0 &ds0_) : ds0(ds0_) { }

    constexpr int get_Df() const { return Tds0::Df; }

    inline void downsample_1d(T *i_out, T *w_out, int nt_out, const T *i_in, const T *w_in, int istride)
    {
	const int Dt = ds0.get_Dt();
	
	simd_t<T,S> wival;
	simd_t<T,S> wval;
	simd_t<T,S> zero(0);
	simd_t<T,S> one(1);
    
	for (int it = 0; it < nt_out; it += S) {
	    wival = simd_t<T,S>::zero();
	    wval = simd_t<T,S>::zero();

	    ds0.get(wival, wval, i_in + it*Dt, w_in + it*Dt, istride);
	    
	    // FIXME revisit after smask cleanup.
	    wival /= blendv(wval.compare_gt(zero), wval, one);
	    
	    wival.storeu(i_out + it);
	    wval.storeu(w_out + it);
	}
    }
};


template<typename Tds0>
struct _wi_downsampler_1d_Dflg {
    using T = typename Tds0::T;
    static constexpr int S = Tds0::S;
    static constexpr int Df0 = Tds0::Df;

    const Tds0 ds0;
    const int Df;
    
    _wi_downsampler_1d_Dflg(const Tds0 &ds0_, int Df_) : ds0(ds0_), Df(Df_) { }

    inline int get_Df() const { return Df; }
    
    inline void downsample_1d(T *i_out, T *w_out, int nt_out, const T *i_in, const T *w_in, int istride)
    {
	const int Dt = ds0.get_Dt();
	
	simd_t<T,S> wival;
	simd_t<T,S> wval;
	simd_t<T,S> zero(0);
	simd_t<T,S> one(1);

	// First pass
	for (int it = 0; it < nt_out; it += S) {
	    wival = simd_t<T,S>::zero();
	    wval = simd_t<T,S>::zero();

	    ds0.get(wival, wval, i_in + it*Dt, w_in + it*Dt, istride);
	    wival.storeu(i_out + it);
	    wval.storeu(w_out + it);
	}

	i_in += Df0*istride;
	w_in += Df0*istride;
	
	// Middle passes
	for (int i = Df0; i < (Df-Df0); i += Df0) {
	    for (int it = 0; it < nt_out; it += S) {
		wival.loadu(i_out + it);
		wval.loadu(w_out + it);
		
		ds0.get(wival, wval, i_in + it*Dt, w_in + it*Dt, istride);
		wival.storeu(i_out + it);
		wval.storeu(w_out + it);
	    }
	    
	    i_in += Df0*istride;
	    w_in += Df0*istride;
	}

	// Last pass
	for (int it = 0; it < nt_out; it += S) {
	    wival.loadu(i_out + it);
	    wval.loadu(w_out + it);
		
	    ds0.get(wival, wval, i_in + it*Dt, w_in + it*Dt, istride);
	    	    
	    // FIXME revisit after smask cleanup.
	    wival /= blendv(wval.compare_gt(zero), wval, one);

	    wival.storeu(i_out + it);
	    wval.storeu(w_out + it);
	}
    }
};


// -------------------------------------------------------------------------------------------------


template<typename Tds1, typename T = typename Tds1::T>
inline void _wi_downsample_2d(Tds1 &ds1, int nfreq_out, int nt_out, T *out_i, T *out_w, int ostride, const T *in_i, const T *in_w, int istride)
{
    const int Df = ds1.get_Df();
	
    for (int ifreq = 0; ifreq < nfreq_out; ifreq++) {
	T *out_i2 = out_i + ifreq*ostride;
	T *out_w2 = out_w + ifreq*ostride;
	const T *in_i2 = in_i + ifreq*Df*istride;
	const T *in_w2 = in_w + ifreq*Df*istride;

	ds1.downsample_1d(out_i2, out_w2, nt_out, in_i2, in_w2, istride);
    }    
}



// -------------------------------------------------------------------------------------------------
//
// _wi_downsample_2d_Df_Dt<T,S,Df,Dt> (nfreq_out, nt_out, out_i, out_w, ostride, in_i, in_w, istride, Df, Dt)
// _wi_downsample_2d_Df<T,S,Df> (nfreq_out, nt_out, out_i, out_w, ostride, in_i, in_w, istride, Df, Dt)
// _wi_downsample_2d_Dt<T,S,Dt> (nfreq_out, nt_out, out_i, out_w, ostride, in_i, in_w, istride, Df, Dt)
// _wi_downsample_2d<T,S> (nfreq_out, nt_out, out_i, out_w, ostride, in_i, in_w, istride, Df, Dt)
//
// Caller must check that nt_out is divisible by S.


template<typename T, int S, int Df, int Dt>
inline void _wi_downsample_2d_Df_Dt(int nfreq_out, int nt_out, T *out_i, T *out_w, int ostride, const T *in_i, const T *in_w, int istride, int Df_, int Dt_)
{
    _wi_downsampler_0d_Dtsm<T,S,Df,Dt> ds0;
    _wi_downsampler_1d_Dfsm<decltype(ds0)> ds1(ds0);
    _wi_downsample_2d(ds1, nfreq_out, nt_out, out_i, out_w, ostride, in_i, in_w, istride);
}


template<typename T, int S, int Df>
inline void _wi_downsample_2d_Df(int nfreq_out, int nt_out, T *out_i, T *out_w, int ostride, const T *in_i, const T *in_w, int istride, int Df_, int Dt)
{
    _wi_downsampler_0d_Dtlg<T,S,Df> ds0(Dt);
    _wi_downsampler_1d_Dfsm<decltype(ds0)> ds1(ds0);
    _wi_downsample_2d(ds1, nfreq_out, nt_out, out_i, out_w, ostride, in_i, in_w, istride);
}


template<typename T, int S, int Dt>
inline void _wi_downsample_2d_Dt(int nfreq_out, int nt_out, T *out_i, T *out_w, int ostride, const T *in_i, const T *in_w, int istride, int Df, int Dt_)
{
    _wi_downsampler_0d_Dtsm<T,S,S,Dt> ds0;
    _wi_downsampler_1d_Dflg<decltype(ds0)> ds1(ds0, Df);
    _wi_downsample_2d(ds1, nfreq_out, nt_out, out_i, out_w, ostride, in_i, in_w, istride);
}


template<typename T, int S>
inline void _wi_downsample_2d(int nfreq_out, int nt_out, T *out_i, T *out_w, int ostride, const T *in_i, const T *in_w, int istride, int Df, int Dt)
{
    _wi_downsampler_0d_Dtlg<T,S,S> ds0(Dt);
    _wi_downsampler_1d_Dflg<decltype(ds0)> ds1(ds0, Df);
    _wi_downsample_2d(ds1, nfreq_out, nt_out, out_i, out_w, ostride, in_i, in_w, istride);
}


template<typename T>
inline void _wi_downsample_2d_1_1(int nfreq_out, int nt_out, T *out_i, T *out_w, int ostride, const T *in_i, const T *in_w, int istride, int Df, int Dt)
{
    for (int ifreq = 0; ifreq < nfreq_out; ifreq++)
	memcpy(out_i + ifreq*ostride, in_i + ifreq*istride, nt_out * sizeof(T));
    
    for (int ifreq = 0; ifreq < nfreq_out; ifreq++)
	memcpy(out_w + ifreq*ostride, in_w + ifreq*istride, nt_out * sizeof(T));
}


}  // namespace rf_kernels

#endif  // _RF_KERNELS_DOWNSAMPLE_INTERNALS_HPP
