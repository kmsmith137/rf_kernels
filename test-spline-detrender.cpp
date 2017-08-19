#include "rf_kernels/unit_testing.hpp"
#include "rf_kernels/spline_detrender.hpp"

using namespace std;
using namespace rf_kernels;


// -------------------------------------------------------------------------------------------------
//
// This source file will contain a reference implementation of spline detrending...


static void test_reference_spline_detrender(std::mt19937 &rng, int nx, int nbins)
{
    vector<int> bin_delim(nbins+1, 0);
    vector<float> poly_vals(4*nx, 0.0);

    _spline_detrender_init(&bin_delim[0], &poly_vals[0], nx, nbins);
}


static void test_reference_spline_detrender(std::mt19937 &rng)
{
    for (int iter = 0; iter < 1000; iter++) {
	int nx = randint(rng, 4, 200);
	int nbins = randint(rng, 1, min(nx/4+1,20));
	test_reference_spline_detrender(rng, nx, nbins);
    }

    cout << "test_reference_spline_detrender: pass" << endl;
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    
    test_reference_spline_detrender(rng);

    return 0;
}
