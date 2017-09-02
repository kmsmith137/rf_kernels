#include <unordered_map>

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
    
    // Usage: kernel(out_mean, nfreq_ds, nt_ds, in_i, in_w, istride, Df, Dt, niter, sigma, tmp_i, tmp_w)
    using kernel_t = void (*)(float *, int, int, const float *, const float *, int, int, int, int, float, float *, float *);

    // (axis, Df, Dt, iterative, two_pass)
    // Currently assume (axis, iterative, two_pass) = (AXIS_TIME, false, true).
    unordered_map<array<int,5>, kernel_t> kernel_table;


    inline kernel_t get(axis_type axis, int Df, int Dt, int niter, bool two_pass)
    {
	int t1 = min(niter-1, 1);
	int t2 = two_pass ? 1 : 0;
	
	auto p = kernel_table.find({{axis,Df,Dt,t1,t2}});
	    
	if (_unlikely(p == kernel_table.end())) {
	    stringstream ss;
	    ss << "rf_kernels::wrms_kernel_table: (axis,Df,Dt,niter,two_pass)=("
	       << axis << "," << Df << "," << Dt << "," << niter
	       << "," << two_pass << ") is invalid or unimplemented";
	    throw runtime_error(ss.str());
	}
    
	return p->second;
    }

	
    // Helpers for constructor
    template<int Df, int Dt, typename enable_if<(Dt==0),int>::type = 0>
    inline void _populate1() { }

    template<int Df, int Dt, typename enable_if<(Dt>0),int>::type = 0>
    inline void _populate1()
    {
	_populate1<Df,(Dt/2)> ();

	kernel_table[{{AXIS_TIME,Df,Dt,0,1}}] = kernel_wrms_Dfsm_Dtsm<float,8,Df,Dt>;
    }

    template<int Df, int Dt, typename enable_if<(Df==0),int>::type = 0>
    inline void _populate2() { }
    
    template<int Df, int Dt, typename enable_if<(Df>0),int>::type = 0>
    inline void _populate2()
    {
	_populate2<(Df/2),Dt> ();
	_populate1<Df,Dt> ();
    }


    wrms_kernel_table()
    {
	_populate2<2,2> ();
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
    _f(global_wrms_kernel_table.get(axis,Df,Dt,niter,two_pass))
{ 
    if (_unlikely(nfreq <= 0))
	throw runtime_error("rf_kernels::weighted_mean_rms: expected nfreq > 0");

    if (_unlikely(nt_chunk <= 0))
	throw runtime_error("rf_kernels::weighted_mean_rms: expected nt_chunk > 0");

    if (_unlikely((nfreq % Df) != 0))
	throw runtime_error("rf_kernels::weighted_mean_rms: expected nfreq to be a multiple of Df");
	
    if (_unlikely((nt_chunk % (8*Dt)) != 0))
	throw runtime_error("rf_kernels::weighted_mean_rms: expected nt_chunk to be a multiple of 8*Dt");

    if (_unlikely((niter > 1) && sigma < 1.0))
	throw runtime_error("rf_kernels::weighted_mean_rms: expected sigma >= 1.0");

    // Temporary!
    rf_assert(axis == AXIS_TIME);
    rf_assert(niter == 1);
    rf_assert(two_pass == true);
    
    this->nfreq_ds = xdiv(nfreq, Df);
    this->nt_ds = xdiv(nt_chunk, Dt);
    this->nout = nfreq_ds;

    this->out_mean = aligned_alloc<float> (nout);
    this->tmp_i = aligned_alloc<float> (nfreq_ds * nt_ds);
    this->tmp_w = aligned_alloc<float> (nfreq_ds * nt_ds);
}


weighted_mean_rms::~weighted_mean_rms()
{
    free(out_mean);
    free(tmp_i);
    free(tmp_w);

    out_mean = tmp_i = tmp_w = nullptr;
}


void weighted_mean_rms::compute_wrms(const float *intensity, const float *weights, int stride)
{
    if (_unlikely(!intensity || !weights))
	throw runtime_error("rf_kernels: null pointer passed to weighted_mean_rms::compute_wrms()");

    if (_unlikely(abs(stride) < nt_chunk))
	throw runtime_error("rf_kernels::weighed_mean_rms: stride is too small");

    this->_f(out_mean, nfreq_ds, nt_ds, intensity, weights, stride, Df, Dt, niter, sigma, tmp_i, tmp_w);
}


}  // namespace rf_kernels
