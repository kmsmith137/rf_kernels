#include <unordered_map>

#include "rf_kernels/core.hpp"
#include "rf_kernels/internals.hpp"
#include "rf_kernels/downsample.hpp"
#include "rf_kernels/downsample_internals.hpp"


using namespace std;

namespace rf_kernels {
#if 0
}; // pacify emacs c-mode
#endif


struct downsampling_kernel_table {
    // Usage: kernel(ds, nfreq_out, nt_out, out_i, out_istride, out_w, out_wstride, in_i, in_istride, in_w, in_wstride)
    using kernel_t = void (*)(const wi_downsampler *, int, int, float *, int, float *, int, const float *, int, const float *, int);

    unordered_map<array<int,2>, kernel_t> kernel_table;   // (Df,Dt) -> kernel


    inline kernel_t get(int Df, int Dt)
    {
    	if ((Df > 8) && (Df % 8 == 0))
	    Df = 16;
	
	if ((Dt > 8) && (Dt % 8 == 0))
	    Dt = 16;

	auto p = kernel_table.find({{Df,Dt}});
    
	if (_unlikely(p == kernel_table.end())) {
	    stringstream ss;
	    ss << "rf_kernels::wi_downsampler: (Df,Dt)=(" << Df << "," << Dt << ") is not supported";
	    throw runtime_error(ss.str());
	}

	return p->second;
    }
    

    template<int Df, int Dt, typename enable_if<(Dt==0),int>::type=0>
    inline void _populate1() { }

    template<int Df, int Dt, typename enable_if<(Dt>0),int>::type=0>
    inline void _populate1()
    {
	_populate1<Df,(Dt/2)> ();
	kernel_table[{{Df,Dt}}] = kernel_wi_downsample<float,8,Df,Dt>;
    }


    template<int Df, int Dt, typename enable_if<(Df==0),int>::type=0>
    inline void _populate2() { }
    
    template<int Df, int Dt, typename enable_if<(Df>0),int>::type=0>
    inline void _populate2()
    {
	_populate2<(Df/2),Dt> ();
	_populate1<Df,Dt> ();
    }


    downsampling_kernel_table()
    {
	_populate2<16,16>();
    }
} global_downsampling_kernel_table;


// -------------------------------------------------------------------------------------------------


wi_downsampler::wi_downsampler(int Df_, int Dt_) :
    Df(Df_),
    Dt(Dt_),
    _f(global_downsampling_kernel_table.get(Df_,Dt_))
{ }
    

void wi_downsampler::downsample(int nfreq_out, int nt_out,
				float *out_i, int out_istride,
				float *out_w, int out_wstride,
				const float *in_i, int in_istride,
				const float *in_w, int in_wstride)
{
    if (_unlikely(nfreq_out <= 0))
	throw runtime_error("rf_kernels::wi_downsampler: expected nfreq_out > 0");
    
    if (_unlikely(nt_out <= 0))
	throw runtime_error("rf_kernels::wi_downsampler: expected nt_out > 0");
    
    if (_unlikely(nt_out % 8))
	throw runtime_error("rf_kernels::wi_downsampler: expected nt_out divisible by 8");

    if (_unlikely(abs(in_istride) < Dt * nt_out))
	throw runtime_error("rf_kernels::wi_downsampler: in_istride is too small");
    
    if (_unlikely(abs(in_wstride) < Dt * nt_out))
	throw runtime_error("rf_kernels::wi_downsampler: in_wstride is too small");

    if (_unlikely(abs(out_istride) < nt_out))
	throw runtime_error("rf_kernels::wi_downsampler: out_istride is too small");
    
    if (_unlikely(abs(out_wstride) < nt_out))
	throw runtime_error("rf_kernels::wi_downsampler: out_wstride is too small");
    
    if (_unlikely(!out_i || !out_w || !in_i || !in_w))
	throw runtime_error("rf_kernels: null pointer passed to wi_downsampler::downsample()");

    this->_f(this, nfreq_out, nt_out, out_i, out_istride, out_w, out_wstride, in_i, in_istride, in_w, in_wstride);
}


}  // namespace rf_kernels
