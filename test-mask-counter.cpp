#include <simd_helpers/core.hpp>     // simd_size()
#include "rf_kernels/internals.hpp"  // uptr
#include "rf_kernels/mask_counter.hpp"
#include "rf_kernels/unit_testing.hpp"

using namespace std;
using namespace rf_kernels;


// Caller has initialized all integer members of 'd', but pointer members are null pointers.
static void test_mask_counter(std::mt19937 &rng, const mask_counter_data &d, bool have_bitmask, bool have_fcounts)
{
#if 1
    cout << "test_mask_counter: nfreq=" << d.nfreq
	 << ", nt_chunk=" << d.nt_chunk
	 << ", istride=" << d.istride
	 << ", ostride=" << d.out_bmstride
	 << ", have_bitmask=" << have_bitmask
	 << ", have_fcounts=" << have_fcounts
	 << endl;
#endif

    int nfreq = d.nfreq;
    int nt = d.nt_chunk;
    int istride = d.istride;
    int ostride = d.out_bmstride;

    uptr<float> data = make_uptr<float>(nfreq*istride);
    uptr<uint8_t> bm1 = have_bitmask ? make_uptr<uint8_t>(nfreq*ostride) : uptr<uint8_t>();
    uptr<uint8_t> bm2 = have_bitmask ? make_uptr<uint8_t>(nfreq*ostride) : uptr<uint8_t>();
    uptr<int> fc1 = have_fcounts ? make_uptr<int>(nfreq) : uptr<int>();
    uptr<int> fc2 = have_fcounts ? make_uptr<int>(nfreq) : uptr<int>();

    for (int i = 0; i < nfreq * istride; i++)
	data[i] = randint(rng,0,4) ? uniform_rand(rng,-1.0,1.0) : 0.0;

    mask_counter_data d1 = d;
    d1.in = data.get();
    d1.out_bitmask = bm1.get();
    d1.out_fcounts = fc1.get();

    mask_counter_data d2 = d;
    d2.in = data.get();
    d2.out_bitmask = bm2.get();
    d2.out_fcounts = fc2.get();
    
    int total1 = d1.mask_count();
    int total2 = d1.slow_reference_mask_count();

    rf_assert(total1 == total2);

    if (have_bitmask) {
	for (int ifreq = 0; ifreq < nfreq; ifreq++)
	    for (int it = 0; it < nt; it++)
		rf_assert(bm1[ifreq*ostride+it] == bm2[ifreq*ostride+it]);
    }

    if (have_fcounts) {
	for (int ifreq = 0; ifreq < nfreq; ifreq++)
	    rf_assert(fc1[ifreq] == fc2[ifreq]);
    }
}


static void test_mask_counter(std::mt19937 &rng)
{
    constexpr int S = simd_helpers::simd_size<float>();

    mask_counter_data d;
    d.nfreq = randint(rng, 1, 10);
    d.nt_chunk = (32*S) * randint(rng, 1, 10);
    d.istride = randint(rng, d.nt_chunk, 2*d.nt_chunk);
    d.out_bmstride = randint(rng, d.nt_chunk/8, d.nt_chunk/4);

    bool have_bitmask = randint(rng, 0, 2);
    bool have_fcounts = randint(rng, 0, 2);

    test_mask_counter(rng, d, have_bitmask, have_fcounts);
}


int main(int argc, char **argv)
{
    // std::random_device rd;
    // std::mt19937 rng(rd());
    std::mt19937 rng(23); // XXXXXX

    cout << "test-mask-counter: start" << endl;

    for (int iter = 0; iter < 1000; iter++)
	test_mask_counter(rng);

    cout << "test-mask-counter: pass" << endl;
    return 0;
}
