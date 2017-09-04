#include <unordered_map>
// #include <simd_helpers/simd_debug.hpp>

#include "rf_kernels/core.hpp"
#include "rf_kernels/internals.hpp"
#include "rf_kernels/mean_rms_internals.hpp"
#include "rf_kernels/mean_rms.hpp"

using namespace std;

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


struct wrms_kernel_table {
    // (wp, intensity, weights, stride)
    using kernel_t = void (*)(const weighted_mean_rms *, const float *, const float *, int);

    // (axis, Df, Dt, two_pass)
    // Currently assume (axis, two_pass) = (AXIS_TIME, true).
    unordered_map<array<int,4>, kernel_t> kernel_table;


    inline kernel_t get(axis_type axis, int Df, int Dt, bool two_pass)
    {
	if ((axis == AXIS_FREQ) && (Df > 0))
	    Df = 1;
	else if ((Df > 8) && (Df % 8 == 0))
	    Df = 16;

	if ((Dt > 8) && (Dt % 8 == 0))
	    Dt = 16;

	int t = two_pass ? 1 : 0;	
	auto p = kernel_table.find({{axis,Df,Dt,t}});
	
	if (_unlikely(p == kernel_table.end())) {
	    stringstream ss;
	    ss << "rf_kernels::wrms_kernel_table: (axis,Df,Dt,niter,two_pass)=("
	       << axis << "," << Df << "," << Dt << "," << two_pass
	       << ") is invalid or unimplemented";
	    throw runtime_error(ss.str());
	}
    
	return p->second;
    }

	
    template<int Df, int Dt, typename enable_if<(Df==0),int>::type = 0>
    inline void _populate1() 
    { 
	kernel_table[{{AXIS_FREQ,1,Dt,1}}] = kernel_wrms_faxis<float,8,Dt>;
    }

    template<int Df, int Dt, typename enable_if<(Df>0),int>::type = 0>
    inline void _populate1()
    {
	_populate1<(Df/2),Dt> ();
	kernel_table[{{AXIS_TIME,Df,Dt,1}}] = kernel_wrms_taxis<float,8,Df,Dt>;
    }


    template<int Df, int Dt, typename enable_if<(Dt==0),int>::type = 0>
    inline void _populate2() { }
    
    template<int Df, int Dt, typename enable_if<(Dt>0),int>::type = 0>
    inline void _populate2()
    {
	_populate2<Df,(Dt/2)> ();
	_populate1<Df,Dt> ();
    }


    wrms_kernel_table()
    {
	_populate2<16,16> ();
    }
} global_wrms_kernel_table;


weighted_mean_rms::weighted_mean_rms(int nfreq_, int nt_chunk_, axis_type axis_, int Df_, int Dt_, int niter_, double sigma_, bool two_pass_) :
    nfreq(nfreq_),
    nt_chunk(nt_chunk_),
    axis(axis_),
    Df(Df_),
    Dt(Dt_),
    niter(niter_),
    sigma(sigma_),
    two_pass(two_pass_),
    _f(global_wrms_kernel_table.get(axis,Df,Dt,two_pass))
{
    if (_unlikely(nfreq <= 0))
	throw runtime_error("rf_kernels::weighted_mean_rms: expected nfreq > 0");

    if (_unlikely(nt_chunk <= 0))
	throw runtime_error("rf_kernels::weighted_mean_rms: expected nt_chunk > 0");

    if (_unlikely((nfreq % Df) != 0))
	throw runtime_error("rf_kernels::weighted_mean_rms: expected nfreq to be a multiple of Df");
	
    if (_unlikely((nt_chunk % (8*Dt)) != 0))
	throw runtime_error("rf_kernels::weighted_mean_rms: expected nt_chunk to be a multiple of 8*Dt");

    if (_unlikely(niter < 1))
	throw runtime_error("rf_kernels::weighted_mean_rms: expected niter < 1");
    
    if (_unlikely((niter > 1) && sigma < 1.0))
	throw runtime_error("rf_kernels::weighted_mean_rms: expected sigma >= 1.0");
    
    this->nfreq_ds = xdiv(nfreq, Df);
    this->nt_ds = xdiv(nt_chunk, Dt);

    if (axis == AXIS_FREQ) {
	this->nout = nt_ds;
	this->ntmp = nfreq_ds * 8;
    }
    else if (axis == AXIS_TIME) {
	this->nout = nfreq_ds;
	this->ntmp = nt_ds;
    }
    else if (axis == AXIS_NONE) {
	this->nout = 1;
	this->ntmp = nfreq_ds * nt_ds;
    }
    else
	throw runtime_error("rf_kernels internal error: bad axis in mean_rms constructor");

    // FIXME overallocated?
    this->out_mean = aligned_alloc<float> (nout);
    this->out_rms = aligned_alloc<float> (nout);
    this->tmp_i = aligned_alloc<float> (ntmp);
    this->tmp_w = aligned_alloc<float> (ntmp);
}


weighted_mean_rms::~weighted_mean_rms()
{
    free(out_mean);
    free(out_rms);
    free(tmp_i);
    free(tmp_w);

    out_mean = out_rms = tmp_i = tmp_w = nullptr;
}


void weighted_mean_rms::compute_wrms(const float *intensity, const float *weights, int stride)
{
    if (_unlikely(!intensity || !weights))
	throw runtime_error("rf_kernels: null pointer passed to weighted_mean_rms::compute_wrms()");

    if (_unlikely(abs(stride) < nt_chunk))
	throw runtime_error("rf_kernels::weighed_mean_rms: stride is too small");

    this->_f(this, intensity, weights, stride);
}


}  // namespace rf_kernels
