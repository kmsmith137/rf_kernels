#include <simd_helpers/core.hpp>     // simd_size()
#include "rf_kernels/internals.hpp"  // uptr
#include "rf_kernels/mask_counter.hpp"
#include "rf_kernels/unit_testing.hpp"

using namespace std;
using namespace rf_kernels;


static void test_mask_counter(std::mt19937 &rng, const mask_counter::initializer &ini_params, int istride, int ostride)
{
    int nfreq = ini_params.nfreq;
    int nt = ini_params.nt_chunk;

    uptr<float> data = make_uptr<float> (nfreq * istride);
    uptr<uint8_t> bm1 = make_uptr<uint8_t> (nfreq * ostride);
    uptr<uint8_t> bm2 = make_uptr<uint8_t> (nfreq * ostride);
    uptr<int> fc1 = make_uptr<int> (nfreq);
    uptr<int> fc2 = make_uptr<int> (nfreq);
    uptr<int> tc1 = make_uptr<int> (nt);
    uptr<int> tc2 = make_uptr<int> (nt);

    for (int i = 0; i < nfreq * istride; i++)
	data[i] = randint(rng,0,4) ? uniform_rand(rng,-1.0,1.0) : 0.0;
    
    mask_counter::kernel_args args1;
    args1.in = data.get();
    args1.istride = istride;
    args1.out_bmstride = ostride;
    
    mask_counter::kernel_args args2;
    args2.in = data.get();
    args2.istride = istride;
    args2.out_bmstride = ostride;

    if (ini_params.save_bitmask) {
	args1.out_bitmask = bm1.get();
	args2.out_bitmask = bm2.get();
    }

    if (ini_params.save_tcounts) {
	args1.out_tcounts = tc1.get();
	args2.out_tcounts = tc2.get();
    }

    if (ini_params.save_fcounts) {
	args1.out_fcounts = fc1.get();
	args2.out_fcounts = fc2.get();
    }

    mask_counter mc(ini_params);
    mc.mask_count(args1);
    mc.slow_reference_mask_count(args2);

    if (ini_params.save_bitmask) {
	for (int ifreq = 0; ifreq < nfreq; ifreq++)
	    for (int it = 0; it < nt; it++)
		rf_assert(bm1[ifreq*ostride+it] == bm2[ifreq*ostride+it]);
    }

    if (ini_params.save_tcounts) {
	for (int it = 0; it < nt; it++)
	    rf_assert(tc1[it] == tc2[it]);
    }

    if (ini_params.save_fcounts && 0) {   // XXXXXX!!!!!!
	for (int ifreq = 0; ifreq < nfreq; ifreq++)
	    rf_assert(fc1[ifreq] == fc2[ifreq]);
    }
}


static void test_mask_counter(std::mt19937 &rng)
{
    constexpr int S = simd_helpers::simd_size<float>();

    mask_counter::initializer ini_params;
    ini_params.nfreq = randint(rng, 1, 10);
    ini_params.nt_chunk = (32*S) * randint(rng, 1, 10);

    for (;;) {
	ini_params.save_bitmask = randint(rng, 0, 2);
	ini_params.save_tcounts = randint(rng, 0, 2);
	ini_params.save_fcounts = randint(rng, 0, 2);

	if (ini_params.save_bitmask || ini_params.save_tcounts || ini_params.save_fcounts)
	    break;
    }

    int nt = ini_params.nt_chunk;
    int istride = randint(rng, nt, 2*nt);
    int ostride = randint(rng, nt/8, nt/4);

    test_mask_counter(rng, ini_params, istride, ostride);
}


int main(int argc, char **argv)
{
    // std::random_device rd;
    // std::mt19937 rng(rd());
    std::mt19937 rng(23);

    cout << "test-mask-counter: start" << endl;

    for (int iter = 0; iter < 1000; iter++)
	test_mask_counter(rng);

    cout << "test-mask-counter: pass" << endl;
    return 0;
}
