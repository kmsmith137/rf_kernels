// Slow code, to be replaced by fast Maya code!

#include "rf_kernels.hpp"

using namespace std;

namespace rf_kernels {
#if 0
};  // pacify emacs c-mode
#endif


inline bool get_v1(const float *intensity, const float *weights, float &v1)
{
    int zerocount = 0;
    float vsum = 0;
    float wsum = 0;

    for (int i=0; i < 32; ++i)
    {
        // I assume this is okay for checking whether the weight is 0?
        if (weights[i] < 1e-7)
	  ++zerocount;
        vsum += intensity[i] * intensity[i] * weights[i];
        wsum += weights[i];
    }

    wsum = max(wsum, 1.0f);

    // Check whether enough valid values were passed
    if (zerocount >= 23.9)
    {  
	v1 = 0;
        return false;
    }
    v1 = vsum / wsum;
    return true;
}


// Based on Maya code in rf_pipelines (scalar_mask_fill() in online_mask_filler.cpp)
// with a few changes as follows which were helpful when incorporating it into bonsai.
//
//   - Uses std::random, there is no good reason for this and it should use fast Maya RNG
//
//   - We don't actually modify the weights array (we do update the running_weights)
//     Not sure what's best here, maybe define a boolean flag for maximum flexibility?
//
//   - We do modify the intensity array (in the case where the intensity is unmasked),
//     by applying (i.e. multiplying by) the running_weight.
//
//   - For consistency, in the case where an intensity value is masked, we apply the
//     running_weight to the random value which is simulated.  I.e., the simulated
//     random intensity has variance (3 * running_weight^2 * running_variance).
//
//   - See "KMS" below for two more minor changes...


void online_mask_fill(const online_mask_filler_params &params, int nfreq, int nt_chunk, int stride,
		      float *intensity, const float *weights, float *running_var, float *running_weights, 
		      std::mt19937 &rng)
{
    const int v1_chunk = params.v1_chunk;
    const float w_clamp = params.w_clamp;
    const float var_weight = params.var_weight;
    const float var_clamp_add = params.var_clamp_add;
    const float var_clamp_mult = params.var_clamp_mult;
    const float w_cutoff = params.w_cutoff;

    // outer looop over frequency channels
    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	// (running_variance, running_weights) for this frequency channel
	float rv = running_var[ifreq];
	float rw = running_weights[ifreq];
	float v1;

	// middle loop over v1_chunks
	for (int ichunk = 0; ichunk < nt_chunk; ichunk += v1_chunk) {
	    // (intensity, weights) pointers for this v1_chunk
	    float *iacc = &intensity[ifreq*stride + ichunk];
            const float *wacc = &weights[ifreq*stride + ichunk];

	    if (!get_v1(iacc, wacc, v1)) {
		// For an unsuccessful v1, we decrease the weight if possible. We do not modify the running variance
		rw = max(0.0f, rw - w_clamp);
	    }
	    else if (rw <= 0.0f) {
		// KMS: changed logic slightly here!  If the v1 is successful but the running_weight is zero,
		// we set the running_variance equal to the value of 'v1' (ignoring the old value of running_variance),
		// rather than slowly transitioning from the old variance estimate to the new one.
		rv = v1;
		rw = w_clamp;
	    }
	    else {
		// If the v1 was succesful, try to increase the weight, if possible
		// KMS: changed max weight from 2.0f to 1.0f here, since taking max_weight=2 was confusing one of my
		// bonsai unit tests, and max_weight=2 was just catering to a bug of mine in rf_pipelines anyway!
		rw = min(1.0f, rw + w_clamp);

		// Then, restrict the change in variance estimate defined by the clamp parameters
		v1 = (1 - var_weight) * rv + var_weight * v1;
		v1 = min(v1, rv + var_clamp_add + rv*var_clamp_mult);
                v1 = max(v1, rv - var_clamp_add - rv*var_clamp_mult);
		rv = v1;
	    }

	    // Scaling factor for random numbers.
	    // KMS note factor of the running_weight here!
	    float scale = rw * sqrt(3*rv);

	    for (int i = 0; i < v1_chunk; i++) {
		// r = uniform random number in [-scale,scale]
		float r = std::uniform_real_distribution<float>(-scale,scale)(rng);

		// if unmasked, then apply running_weight
		// if masked, then set intensity to random.
		iacc[i] = (wacc[i] >= w_cutoff) ? (rw * iacc[i]) : r;
	    }
	}

	running_var[ifreq] = rv;
	running_weights[ifreq] = rw;
    }    
}


}  // namespace rf_kernels
