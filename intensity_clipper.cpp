// FIXME: currently we need to compile a new kernel for every (Df,Dt) pair, where
// Df,Dt are the frequency/time downsampling factors.  Eventually I'd like to 
// improve this by having special kernels to handle the large-Df and large-Dt cases.

#include <unordered_map>

#include "rf_kernels/core.hpp"
#include "rf_kernels/internals.hpp"
#include "rf_kernels/downsample_internals.hpp"
#include "rf_kernels/clipper_internals.hpp"
#include "rf_kernels/intensity_clipper_internals.hpp"
#include "rf_kernels/intensity_clipper.hpp"

using namespace std;

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


// kernel(intensity, weights, nfreq, nt_chunk, stride, niter, sigma, iter_sigma, ds_intensity, ds_weights)
using kernel_t = void (*)(const float *, float *, int, int, int, int, double, double, float *, float *);


// -------------------------------------------------------------------------------------------------
//
// Depending on the arguments to the intensity_clipper, we may or may not need to allocate
// temporary buffers for downsampled intensity and weights.  The functions below contain
// logic for deciding whether this allocation is necessary.
//
// FIXME: the details of this logic are opaque and depend on chasing through kernels/*.hpp!
// It would be nice to have comments in these files which make it more transparent.


#if 0
inline int get_nds(int nfreq, int nt, int axis, int Df, int Dt)
{
    static constexpr int S = constants::single_precision_simd_length;

    if (axis == AXIS_FREQ)
	return (nfreq/Df) * S;
    if (axis == AXIS_TIME)
	return (nt/Dt);
    if (axis == AXIS_NONE)
	return (nfreq*nt) / (Df*Dt);

    throw std::runtime_error("rf_kernels internal error: bad axis in _intensity_clipper_alloc()");
}


inline float *alloc_ds_intensity(int nfreq, int nt, axis_type axis, int niter, int Df, int Dt, bool two_pass)
{
    if ((Df==1) && (Dt==1))
	return nullptr;

    int nds = get_nds(nfreq, nt, axis, Df, Dt);
    return aligned_alloc<float> (nds);
}


inline float *alloc_ds_weights(int nfreq, int nt, axis_type axis, int niter, int Df, int Dt, bool two_pass)
{
    if ((Df==1) && (Dt==1))
	return nullptr;

    if ((niter == 1) && !two_pass)
	return nullptr;

    int nds = get_nds(nfreq, nt, axis, Df, Dt);
    return aligned_alloc<float> (nds);
}
#endif


// -------------------------------------------------------------------------------------------------
//
// global kernel table


// (axis, Df, Dt, two_pass)
static unordered_map<array<int,4>, kernel_t> global_kernel_table;


static kernel_t get_kernel(axis_type axis, int Df, int Dt, bool two_pass)
{
    int t = two_pass ? 1 : 0;
    auto p = global_kernel_table.find({axis,Df,Dt,t});
    
    if (_unlikely(p == global_kernel_table.end())) {
	stringstream ss;
	ss << "rf_kernels::intensity_clipper: (axis,Dt,Dt,two_pass)=("
	   << axis << "," << Df << "," << Dt << "," << t << ") is invalid or unimplemented";
	throw runtime_error(ss.str());
    }
    
    return p->second;
}


template<int Df, int Dt, typename enable_if<(Dt==0),int>::type = 0>
inline void _populate1() { }

template<int Df, int Dt, typename enable_if<(Dt>0),int>::type = 0>
inline void _populate1()
{
    _populate1<Df,(Dt/2)> ();
    
    global_kernel_table[{AXIS_FREQ,Df,Dt,0}] = _kernel_clip_1d_f<float,8,Df,Dt,false>;
    global_kernel_table[{AXIS_FREQ,Df,Dt,1}] = _kernel_clip_1d_f<float,8,Df,Dt,true>;
    global_kernel_table[{AXIS_TIME,Df,Dt,0}] = _kernel_clip_1d_t<float,8,Df,Dt,false>;
    global_kernel_table[{AXIS_TIME,Df,Dt,1}] = _kernel_clip_1d_t<float,8,Df,Dt,true>;
    global_kernel_table[{AXIS_NONE,Df,Dt,0}] = _kernel_clip_2d<float,8,Df,Dt,false>;
    global_kernel_table[{AXIS_NONE,Df,Dt,1}] = _kernel_clip_2d<float,8,Df,Dt,true>;
}


template<int Df, int Dt, typename enable_if<(Df==0),int>::type = 0>
inline void _populate2() { }

template<int Df, int Dt, typename enable_if<(Df>0),int>::type = 0>
inline void _populate2()
{
    _populate2<(Df/2),Dt> ();
    _populate1<Df,Dt> ();
}


namespace {
    struct X {
	X() { _populate2<16,16>(); }
    } x;
}


// -------------------------------------------------------------------------------------------------


intensity_clipper::intensity_clipper(int nfreq_, int nt_chunk_, axis_type axis_, int Df_, int Dt_, bool two_pass_) :
    nfreq(nfreq_),
    nt_chunk(nt_chunk_),
    axis(axis_),
    Df(Df_),
    Dt(Dt_),
    two_pass(two_pass_),
    _f(get_kernel(axis_,Df_,Dt_,two_pass_))
{ }


// -------------------------------------------------------------------------------------------------



#if 0

static void check_params(const char *name, int Df, int Dt, axis_type axis, int nfreq, int nt, int stride, double sigma, int niter, double iter_sigma)
{
    static constexpr int S = constants::single_precision_simd_length;
    static constexpr int MaxDf = constants::max_frequency_downsampling;
    static constexpr int MaxDt = constants::max_time_downsampling;

    if (_unlikely((Df <= 0) || !is_power_of_two(Df)))
	throw runtime_error(string(name) + ": Df=" + to_string(Df) + " must be a power of two");

    if (_unlikely((Dt <= 0) || !is_power_of_two(Dt)))
	throw runtime_error(string(name) + ": Dt=" + to_string(Dt) + " must be a power of two");

    if (_unlikely((axis != AXIS_FREQ) && (axis != AXIS_TIME) && (axis != AXIS_NONE)))
	throw runtime_error(string(name) + ": axis=" + stringify(axis) + " is not defined for this transform");

    if (_unlikely(nfreq <= 0))
	throw runtime_error(string(name) + ": nfreq=" + to_string(nfreq) + ", positive value was expected");

    if (_unlikely(nt <= 0))
	throw runtime_error(string(name) + ": nt=" + to_string(nt) + ", positive value was expected");

    if (_unlikely(abs(stride) < nt))
	throw runtime_error(string(name) + ": stride=" + to_string(stride) + " must be >= nt");

    if (_unlikely(sigma < 1.0))
	throw runtime_error(string(name) + ": sigma=" + to_string(sigma) + " must be >= 1.0");

    if (_unlikely(niter < 1))
	throw runtime_error(string(name) + ": niter=" + to_string(niter) + " must be >= 1");

    if (_unlikely((nfreq % Df) != 0))
	throw runtime_error(string(name) + ": nfreq=" + to_string(nfreq)
			    + " must be a multiple of the downsampling factor Df=" + to_string(Df));
    
    if (_unlikely((nt % (Dt*S)) != 0))
	throw runtime_error(string(name) + ": nt=" + to_string(nt)
			    + " must be a multiple of the downsampling factor Dt=" + to_string(Dt)
			    + " multiplied by constants::single_precision_simd_length=" + to_string(S));

    if (_unlikely((Df > MaxDf) || (Dt > MaxDt)))
	throw runtime_error(string(name) + ": (Df,Dt)=(" + to_string(Df) + "," + to_string(Dt) + ")"
			    + " exceeds compile time limits; to fix this see 'constants' in rf_kernels.hpp");
}


// externally visible
shared_ptr<wi_transform> make_intensity_clipper(int nt_chunk, axis_type axis, double sigma, int niter, double iter_sigma, int Df, int Dt, bool two_pass)
{
    int dummy_nfreq = Df;         // arbitrary
    int dummy_stride = nt_chunk;  // arbitrary

    check_params("rf_kernels: make_intensity_clipper()", Df, Dt, axis, dummy_nfreq, nt_chunk, dummy_stride, sigma, niter, iter_sigma);
    
    auto kernel = global_intensity_clipper_kernel_table.get_kernel(axis, Df, Dt, two_pass);
    return make_shared<clipper_transform> (Df, Dt, axis, nt_chunk, sigma, niter, iter_sigma, two_pass, kernel);
}


// externally visible
void apply_intensity_clipper(const float *intensity, float *weights, int nfreq, int nt, int stride, axis_type axis, double sigma, int niter, double iter_sigma, int Df, int Dt, bool two_pass)
{
    check_params("rf_pipeliens: apply_intensity_clipper()", Df, Dt, axis, nfreq, nt, stride, sigma, niter, iter_sigma);

    if (_unlikely(!intensity))
	throw runtime_error("rf_kernels: apply_intensity_clipper(): NULL intensity pointer");
    if (_unlikely(!weights))
	throw runtime_error("rf_kernels: apply_intensity_clipper(): NULL weights pointer");

    float *ds_intensity = alloc_ds_intensity(nfreq, nt, axis, niter, Df, Dt, two_pass);
    float *ds_weights = alloc_ds_weights(nfreq, nt, axis, niter, Df, Dt, two_pass);

    auto kernel = global_intensity_clipper_kernel_table.get_kernel(axis, Df, Dt, two_pass);

    kernel(intensity, weights, nfreq, nt, stride, niter, sigma, iter_sigma, ds_intensity, ds_weights);

    free(ds_intensity);
    free(ds_weights);
}


template<typename T, int S>
inline void _weighted_mean_and_rms(simd_t<T,S> &mean, simd_t<T,S> &rms, const float *intensity, const float *weights, int nfreq, int nt, int stride, int niter, double sigma, bool two_pass)
{
    check_params("rf_kernels: weighted_mean_and_rms()", 1, 1, AXIS_NONE, nfreq, nt, stride, sigma, niter, sigma);

    if (two_pass)
	_kernel_noniterative_wrms_2d<T,S,1,1,false,false,true> (mean, rms, intensity, weights, nfreq, nt, stride, NULL, NULL);
    else
	_kernel_noniterative_wrms_2d<T,S,1,1,false,false,false> (mean, rms, intensity, weights, nfreq, nt, stride, NULL, NULL);

    _kernel_wrms_iterate_2d<T,S> (mean, rms, intensity, weights, nfreq, nt, stride, niter, sigma);
}


void weighted_mean_and_rms(float &mean, float &rms, const float *intensity, const float *weights, int nfreq, int nt, int stride, int niter, double sigma, bool two_pass)
{
    static constexpr int S = constants::single_precision_simd_length;

    if (_unlikely(!intensity))
	throw runtime_error("rf_kernels: weighted_mean_and_rms(): NULL intensity pointer");
    if (_unlikely(!weights))
	throw runtime_error("rf_kernels: weighted_mean_and_rms(): NULL weights pointer");

    simd_t<float,S> mean_x, rms_x;
    _weighted_mean_and_rms(mean_x, rms_x, intensity, weights, nfreq, nt, stride, niter, sigma, two_pass);

    mean = mean_x.template extract<0> ();
    rms = rms_x.template extract<0> ();
}


// The "wrms_hack_for_testing" is explained in test-cpp-python-equivalence.py

void _wrms_hack_for_testing1(vector<float> &mean_hint, const float *intensity, const float *weights, int nfreq, int nt, int stride, int niter, double sigma, bool two_pass)
{
    static constexpr int S = constants::single_precision_simd_length;

    simd_t<float,S> mean_x, rms_x;
    _weighted_mean_and_rms(mean_x, rms_x, intensity, weights, nfreq, nt, stride, niter, sigma, two_pass);

    mean_hint.resize(S);
    mean_x.storeu(&mean_hint[0]);
}


void _wrms_hack_for_testing2(float &mean, float &rms, const float *intensity, const float *weights, int nfreq, int nt, int stride, const vector<float> &mean_hint)
{
    static constexpr int S = constants::single_precision_simd_length;

    if (mean_hint.size() != S)
	throw runtime_error("rf_kernels: wrong mean_hint size in _wrms_hack_for_testing2()");

    simd_t<float,S> mh = simd_helpers::simd_load<float,S> (&mean_hint[0]);
    _mean_variance_iterator<float,S> v(mh, simd_t<float,S>(1.0e10));
    _kernel_visit_2d<1,1> (v, intensity, weights, nfreq, nt, stride);

    simd_t<float,S> mean_x, rms_x;
    v.get_mean_rms(mean_x, rms_x);

    mean = mean_x.template extract<0> ();
    rms = rms_x.template extract<0> ();
}

#endif


}  // namespace rf_kernels
