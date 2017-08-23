#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"
#include "rf_kernels/spline_detrender.hpp"
#include "rf_kernels/spline_detrender_internals.hpp"

using namespace std;
using namespace rf_kernels;


// -------------------------------------------------------------------------------------------------
//
// Helpers


// The meaning of this matrix needs a little explanation!
//
// We represent cubic polynomials p(x) on the unit interval [0,1]
// by a four-component vector v_i = (p(0), p'(0), p(1), p'(1)).
//
// In this basis, the quadratic form Q = int_0^1 (dp/dx)^2 dx
// is represented by a 4-by-4 symmetric matrix Q_{ij} v_i v_j.

static float reg_matrix[16] = {
       (6./5.),  (1./10.),  (-6./5.),  (1./10.),
      (1./10.),  (2./15.), (-1./10.), (-1./30.),
      (-6./5.), (-1./10.),   (6./5.), (-1./10.),
      (1./10.), (-1./30.), (-1./10.),  (2./15.)
};


inline float eval_cubic(float lval, float lderiv, float rval, float rderiv, float x)
{
    // P(x) = x^2 Q(x) + (1-x)^2 R(x)
    //
    // Q(1) = rval
    // Q'(1) = rderiv - 2*rval
    // R(0) = lval
    // R'(0) = lderiv + 2*lval

    float q = (rderiv-2*rval)*(x-1) + rval;
    float r = (lderiv+2*lval)*x + lval;
    return x*x*q + (1-x)*(1-x)*r;
}


void test_eval_cubic(std::mt19937 &rng)
{
    for (int iter = 0; iter < 1000; iter++) {
	float x = uniform_rand(rng, 0.0, 1.0);

	float eps1 = 1.0 - eval_cubic(1,0,1,0,x);
	float eps2 = x - eval_cubic(0,1,1,1,x);
	float eps3 = x*x - eval_cubic(0,0,1,2,x);
	float eps4 = x*x*x - eval_cubic(0,0,1,3,x);

	rf_assert(abs(eps1) < 1.0e-6);
	rf_assert(abs(eps2) < 1.0e-6);
	rf_assert(abs(eps3) < 1.0e-6);
	rf_assert(abs(eps4) < 1.0e-6);
    }

    cout << "test_eval_cubic: pass" << endl;
}



// -------------------------------------------------------------------------------------------------


struct vec2 {
    float v0 = 0.0;
    float v1 = 0.0;

    vec2() { }
    vec2(float v0_, float v1_) : v0(v0_), v1(v1_) { }

    vec2 &operator+=(const vec2 &v) { v0 += v.v0; v1 += v.v1; return *this; }
    vec2 &operator-=(const vec2 &v) { v0 -= v.v0; v1 -= v.v1; return *this; }
    vec2 operator-(const vec2 &v) const { return vec2(v0-v.v0, v1-v.v1); }
};


struct mat2 {
    float a00 = 0.0;
    float a01 = 0.0;
    float a10 = 0.0;
    float a11 = 0.0;

    mat2() { }
    mat2(float a00_, float a01_, float a10_, float a11_) : a00(a00_), a01(a01_), a10(a10_), a11(a11_) { }

    mat2 &operator+=(const mat2 &m)  { a00 += m.a00; a01 += m.a01; a10 += m.a10; a11 += m.a11; return *this; }
    mat2 &operator*=(float t)        { a00 *= t; a01 *= t; a10 *= t; a11 *= t; return *this; }
    mat2 operator-(const mat2 &m)    { return mat2(a00-m.a00, a01-m.a01, a10-m.a10, a11-m.a11); }

    inline mat2 transpose() const { return mat2(a00,a10,a01,a11); }

    inline vec2 operator*(const vec2 &v) const { return vec2(a00*v.v0 + a01*v.v1, a10*v.v0 + a11*v.v1); }

    inline mat2 operator*(const mat2 &m) const
    {
	return mat2(a00 * m.a00 + a01 * m.a10,   // (0,0)
		    a00 * m.a01 + a01 * m.a11,   // (0,1)
		    a10 * m.a00 + a11 * m.a10,   // (1,0)
		    a10 * m.a01 + a11 * m.a11);  // (1,1)
    }
};


struct chol2 {
    mat2 lmat;
    mat2 linv;

    chol2() { }

    explicit chol2(const mat2 &a)
    {
	rf_assert(a.a00 > 0.0);
	rf_assert(a.a11 > 0.0);
	rf_assert(a.a00 * a.a11 > 1.0001 * a.a01 * a.a01);

	lmat.a00 = xsqrt(a.a00);
	lmat.a01 = 0.0;
	lmat.a10 = a.a01 / lmat.a00;
	lmat.a11 = xsqrt(a.a11 - lmat.a10*lmat.a10);
	
	linv.a00 = 1.0 / lmat.a00;
	linv.a01 = 0.0;
	linv.a10 = -lmat.a10 / (lmat.a00 * lmat.a11);
	linv.a11 = 1.0 / lmat.a11;
    }
};


inline ostream &operator<<(ostream &os, const vec2 &v) 
{ 
    os << "vec2(" << v.v0 << ", " << v.v1 << ")";
    return os;
}

inline ostream &operator<<(ostream &os, const mat2 &m)
{
    os << "mat2(" << m.a00 << ", " << m.a01 << ", " << m.a10 << ", " << m.a11 << ")";
    return os;
}

inline ostream &operator<<(ostream &os, const chol2 &c)
{
    os << "chol2(lmat=" << c.lmat << ", linv=" << c.linv << ")";
    return os;
}


// -------------------------------------------------------------------------------------------------


struct big_cholesky {
    int nbins = 0;
    vector<mat2> a_diag;            // length (nbins+1)
    vector<mat2> a_subdiag;         // length (nbins)
    vector<chol2> cholesky_diag;    // length (nbins+1)
    vector<mat2> cholesky_subdiag;  // length (nbins)


    big_cholesky(const vector<mat2> &a_diag_, const vector<mat2> &a_subdiag_)
	: a_diag(a_diag_), a_subdiag(a_subdiag_)
    {
	rf_assert(a_diag.size() == a_subdiag.size()+1);

	this->nbins = a_subdiag.size();
	this->cholesky_diag.resize(nbins+1);
	this->cholesky_subdiag.resize(nbins);

	cholesky_diag[0] = chol2(a_diag[0]);
	
	for (int b = 0; b < nbins; b++) {
	    cholesky_subdiag[b] = a_subdiag[b] * cholesky_diag[b].linv.transpose();
	    cholesky_diag[b+1] = chol2(a_diag[b+1] - cholesky_subdiag[b] * cholesky_subdiag[b].transpose());
	}
    }


    // This constructor is a convenient sentinel for unit tests.
    explicit big_cholesky(int nbins_) :
	nbins(nbins_),
	a_diag(nbins+1),
	a_subdiag(nbins),
	cholesky_diag(nbins+1),
	cholesky_subdiag(nbins)
    { }


    vector<vec2> apply_a(const vector<vec2> &v)
    {
	rf_assert((int)v.size() == nbins+1);

	vector<vec2> ret(nbins+1);
	
	for (int b = 0; b <= nbins; b++)
	    ret[b] = a_diag[b] * v[b];

	for (int b = 0; b < nbins; b++) {
	    ret[b+1] += a_subdiag[b] * v[b];
	    ret[b] += a_subdiag[b].transpose() * v[b+1];
	}
	
	return ret;
    }

    vector<vec2> apply_linv(const vector<vec2> &v)
    {
	rf_assert((int)v.size() == nbins+1);

	vector<vec2> ret(nbins+1);

	ret[0] = cholesky_diag[0].linv * v[0];
	for (int b = 1; b <= nbins; b++)
	    ret[b] = cholesky_diag[b].linv * (v[b] - cholesky_subdiag[b-1] * ret[b-1]);

	return ret;
    }

    vector<vec2> apply_ltinv(const vector<vec2> &v)
    {
	rf_assert((int)v.size() == nbins+1);

	vector<vec2> ret(nbins+1);
	
	ret[nbins] = cholesky_diag[nbins].linv.transpose() * v[nbins];
	for (int b = nbins-1; b >= 0; b--)
	    ret[b] = cholesky_diag[b].linv.transpose() * (v[b] - cholesky_subdiag[b].transpose() * ret[b+1]);

	return ret;
    }

    vector<vec2> apply_ainv(const vector<vec2> &v)
    {
	rf_assert((int)v.size() == nbins+1);
	return apply_ltinv(apply_linv(v));
    }
};


// -------------------------------------------------------------------------------------------------
//
// refsd_*: a reference implementation of spline detrending (1D).


struct refsd_params {
    const int nx;
    const int nbins;

    vector<int> bin_delim;
    vector<float> poly_vals;
    
    refsd_params(int nx, int nbins);

    // 'out' has length nx, 'coeffs' has length (2*nbins+2).
    void eval_model(float *out, const float *coeffs);

    // Like eval_model(), except that instead of using precomputed poly_vals,
    // we evaluate polynomials "from scratch" using eval_cubic().
    void eval_model_from_scratch(float *out, const float *coeffs);

    // 'ninv' has length 16, 'ninvx' has length 4, intensity+weights have length nx.
    void analyze_bin(float *ninv, float *ninvx, const float *intensity, const float *weights, int b);

    // 'coeffs' has length (2*nbins+2).
    // Returns the 'big_cholesky' object which contains intermediate quantities (caller may ignore return value)
    big_cholesky fit_model(float *coeffs, const float *intensity, const float *weights, float epsilon_reg);

    void detrend(float *intensity, const float *weights, float epsilon_reg);

    // Helper function for testing.
    void make_sparse_weights(std::mt19937 &rng, float *weights);
};


refsd_params::refsd_params(int nx_, int nbins_) :
    nx(nx_), 
    nbins(nbins_),
    bin_delim(nbins_+1, 0),
    poly_vals(4*nx_, 0.0)
{
    _spline_detrender_init(&bin_delim[0], &poly_vals[0], nx, nbins);    
}


void refsd_params::eval_model(float *out, const float *coeffs)
{
    memset(out, 0, nx * sizeof(*out));

    for (int b = 0; b < nbins; b++)
	for (int i = bin_delim[b]; i < bin_delim[b+1]; i++)
	    for (int p = 0; p < 4; p++)
		out[i] += coeffs[2*b+p] * poly_vals[4*i+p];
}


void refsd_params::eval_model_from_scratch(float *out, const float *coeffs)
{
    for (int i = 0; i < nx; i++) {
	float x = (i+0.5) / float(nx) * float(nbins);   // note: 0 < x < nbins
	int b = int(x);
	out[i] = eval_cubic(coeffs[2*b], coeffs[2*b+1], coeffs[2*b+2], coeffs[2*b+3], x-b);
    }
}


void refsd_params::analyze_bin(float ninv[16], float ninvx[4], const float *intensity, const float *weights, int b)
{
    memset(ninv, 0, 16 * sizeof(ninv[0]));
    memset(ninvx, 0, 4 * sizeof(ninvx[0]));

    for (int i = bin_delim[b]; i < bin_delim[b+1]; i++) {
	for (int p = 0; p < 4; p++)
	    for (int q = 0; q < 4; q++)
		ninv[4*p+q] += weights[i] * poly_vals[4*i+p] * poly_vals[4*i+q];
	
	for (int p = 0; p < 4; p++)
	    ninvx[p] += weights[i] * intensity[i] * poly_vals[4*i+p];
    }
}

big_cholesky refsd_params::fit_model(float *coeffs, const float *intensity, const float *weights, float epsilon_reg)
{
    float wsum = 0.0;
    for (int i = 0; i < nx; i++)
	wsum += weights[i];

    if (wsum <= 0.0) {
	memset(coeffs, 0, (2*nbins+2) * sizeof(*coeffs));
	return big_cholesky(nbins);
    }

    float w = wsum * epsilon_reg / float(nbins);    
    vector<mat2> ninv_diag(nbins+1);
    vector<mat2> ninv_subdiag(nbins);
    vector<vec2> ninv_x(nbins+1);
    
    for (int b = 0; b < nbins; b++) {
	float ninv_bin[16];
	float ninvx_bin[4];
	analyze_bin(ninv_bin, ninvx_bin, intensity, weights, b);

	for (int i = 0; i < 16; i++)
	    ninv_bin[i] += w * reg_matrix[i];
	
	ninv_diag[b] += mat2(ninv_bin[0], ninv_bin[1], ninv_bin[4], ninv_bin[5]);
	ninv_diag[b+1] += mat2(ninv_bin[10], ninv_bin[11], ninv_bin[14], ninv_bin[15]);
	ninv_subdiag[b] += mat2(ninv_bin[8], ninv_bin[9], ninv_bin[12], ninv_bin[13]);
	ninv_x[b] += vec2(ninvx_bin[0], ninvx_bin[1]);
	ninv_x[b+1] += vec2(ninvx_bin[2], ninvx_bin[3]);
    }

    big_cholesky bc(ninv_diag, ninv_subdiag);
    vector<vec2> v = bc.apply_ainv(ninv_x);

    for (int b = 0; b <= nbins; b++) {
	coeffs[2*b] = v[b].v0;
	coeffs[2*b+1] = v[b].v1;
    }

    return bc;
}


void refsd_params::detrend(float *intensity, const float *weights, float epsilon_reg)
{
    vector<float> coeffs(2*nbins+2);
    fit_model(&coeffs[0], intensity, weights, epsilon_reg);
    
    vector<float> intensity_fit(nx);
    eval_model(&intensity_fit[0], &coeffs[0]);

    for (int i = 0; i < nx; i++)
	intensity[i] -= intensity_fit[i];
}


void refsd_params::make_sparse_weights(std::mt19937 &rng, float *weights)
{
    memset(&weights[0], 0, nx * sizeof(float));

    for (int b = 0; b < nbins; b++) {
	int n = bin_delim[b+1] - bin_delim[b];

	// Number of nonzero weights, chosen to sample "interesting" matrix ranks.
	int nz = randint(rng,0,2) ? randint(rng,0,4) : randint(rng,0,n+1);
	
	for (int iz = 0; iz < nz; iz++) {
	    int i = randint(rng, bin_delim[b], bin_delim[b+1]);
	    weights[i] = uniform_rand(rng);
	}
    }
}


// -------------------------------------------------------------------------------------------------


static void test_reference_spline_detrender(std::mt19937 &rng, int nx, int nbins)
{
    refsd_params params(nx, nbins);

    // Test 1: compare refsd_params::eval_model() and refsd_params::eval_model_from_scratch().
    // This indirectly the poly_vals generation in _spline_detrender_init().
    
    vector<float> intensity(nx, 0.0);
    vector<float> in_coeffs = uniform_randvec(rng, 2*nbins+2, -1.0, 1.0);
    params.eval_model(&intensity[0], &in_coeffs[0]);

    vector<float> intensity2(nx, 0.0);
    params.eval_model_from_scratch(&intensity2[0], &in_coeffs[0]);

    float eps1 = maxdiff(intensity, intensity2);
    if (eps1 > 1.0e-4) {
	cout << "refsd::eval_model() failed: nx=" << nx << ", nbins=" << nbins << ", epsilon=" << eps1 << endl;
	exit(1);
    }

    // Test 2: test detrending in "well-conditioned" case where all weights are positive.
    //
    // In this case, the strongest test is to take epsilon_reg=0, and check that
    // refsd_params::fit_model() recovers the model coefficients to high accuracy.

    vector<float> weights = uniform_randvec(rng, nx, 0.1, 1.0);
    
    vector<float> out_coeffs(2*nbins+2, 0.0);
    params.fit_model(&out_coeffs[0], &intensity[0], &weights[0], 0);  // epsilon_reg=0

    float eps2 = maxdiff(in_coeffs, out_coeffs);
    if (eps2 > 1.0e-4) {
	cout << "refsd::fit_model() failed: nx=" << nx << ", nbins=" << nbins << ", epsilon=" << eps2 << endl;
	exit(1);
    }

    // Test 3: test detrending in poorly conditioned cases with sparse weights.
    //
    // In this case, it's hard to figure out the best unit test!  I decided to test
    // that the weighted RMS intensity after detrending is < 0.01, after detrending
    // with epsilon_reg=1.0e-4.

    params.make_sparse_weights(rng, &weights[0]);
    params.detrend(&intensity[0], &weights[0], 1.0e-4);  // epsilon_reg = 10^(-4)

    float eps3 = weighted_rms_1d(intensity, weights);
    if (eps3 > 1.0e-2) {
	cout << "refsd::detrend() failed: nx=" << nx << ", nbins=" << nbins << ", epsilon=" << eps3 << endl;
	exit(1);
    }

#if 0
    // This print statement is sometimes interesting to look at.
    // Note that maxabs(intensity) can be of order one!
    cout << "detrend: weighted_rms=" << weighted_rms_1d(intensity,weights) << ", maxabs=" << maxabs(intensity) << endl;
#endif
}


static void test_reference_spline_detrender(std::mt19937 &rng)
{
    for (int iter = 0; iter < 1000; iter++) {
	// Number of bins is chosen in a way which tests both low-nbins and high-nbins cases.
	int nbins = randint(rng,0,2) ? randint(rng,1,100) : randint(rng,1,8);
	int nx = randint(rng, 64*nbins, 128*nbins);

	test_reference_spline_detrender(rng, nx, nbins);
    }

    cout << "test_reference_spline_detrender: pass" << endl;
}


// -------------------------------------------------------------------------------------------------


// Helper for test_fast_kernels
inline void _compare(const char *str, float x_ref, float x_fast, float epsilon)
{
    float delta = abs(x_ref - x_fast);

    if (delta > epsilon) {
	cout << "test_fast_kernels failed (" << str << ") ref=" << x_ref << ", fast=" << x_fast << ", delta=" << delta << ", epsilon=" << epsilon << endl;
	exit(1);
    }

#if 0
    if (delta > epsilon/3.)
	cout << "test_fast_kernels close call (" << str << ") ref=" << x_ref << ", fast=" << x_fast << ", delta=" << delta << ", epsilon=" << epsilon << endl;
#endif
}


static void test_fast_kernels(std::mt19937 &rng, int nfreq, int nbins, int stride, float epsilon_reg)
{
    spline_detrender fast_sd(nfreq, nbins, epsilon_reg);
    refsd_params ref_sd(nfreq, nbins);

    vector<float> ref_intensity = uniform_randvec(rng, 8*nfreq, -1.0, 1.0);
    vector<float> ref_weights = uniform_randvec(rng, 8*nfreq, 0.1, 1.0);

#if 1
    for (int s = 0; s < 8; s++)
	ref_sd.make_sparse_weights(rng, &ref_weights[s*nfreq]);
#endif

    float *fast_intensity = aligned_alloc<float> (nfreq * stride);
    float *fast_weights = aligned_alloc<float> (nfreq * stride);

    for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	for (int s = 0; s < 8; s++) {
	    fast_intensity[ifreq*stride + s] = ref_intensity[s*nfreq + ifreq];
	    fast_weights[ifreq*stride + s] = ref_weights[s*nfreq + ifreq];
	}
    }

    fast_sd._kernel_ninv(stride, fast_intensity, fast_weights);

    for (int s = 0; s < 8; s++) {
	for (int b = 0; b < nbins; b++) {
	    float ref_ninv[16];
	    float ref_ninvx[4];
	    
	    ref_sd.analyze_bin(ref_ninv, ref_ninvx, &ref_intensity[s*nfreq], &ref_weights[s*nfreq], b);

	    _compare("wi0", ref_ninvx[0], fast_sd.ninvx[b*32+s], 1.0e-5);
	    _compare("wi1", ref_ninvx[1], fast_sd.ninvx[b*32+8+s], 1.0e-5);
	    _compare("wi2", ref_ninvx[2], fast_sd.ninvx[b*32+16+s], 1.0e-5);
	    _compare("wi3", ref_ninvx[3], fast_sd.ninvx[b*32+24+s], 1.0e-5);

	    _compare("w00", ref_ninv[0], fast_sd.ninv[b*80+s], 1.0e-5);
	    _compare("w01", ref_ninv[1], fast_sd.ninv[b*80+8+s], 1.0e-5);
	    _compare("w02", ref_ninv[2], fast_sd.ninv[b*80+16+s], 1.0e-5);
	    _compare("w03", ref_ninv[3], fast_sd.ninv[b*80+24+s], 1.0e-5);
	    _compare("w11", ref_ninv[5], fast_sd.ninv[b*80+32+s], 1.0e-5);
	    _compare("w12", ref_ninv[6], fast_sd.ninv[b*80+40+s], 1.0e-5);
	    _compare("w13", ref_ninv[7], fast_sd.ninv[b*80+48+s], 1.0e-5);
	    _compare("w22", ref_ninv[10], fast_sd.ninv[b*80+56+s], 1.0e-5);
	    _compare("w23", ref_ninv[11], fast_sd.ninv[b*80+64+s], 1.0e-5);
	    _compare("w33", ref_ninv[15], fast_sd.ninv[b*80+72+s], 1.0e-5);
	}
    }

    fast_sd._kernel_fit_pass1();
    fast_sd._kernel_fit_pass2();
    fast_sd._kernel_fit_pass3();
    
    for (int s = 0; s < 8; s++) {
	vector<float> ref_coeffs(2*nbins+2, 0.0);
	big_cholesky ref_bc = ref_sd.fit_model(&ref_coeffs[0], &ref_intensity[s*nfreq], &ref_weights[s*nfreq], epsilon_reg);

	for (int b = 0; b <= nbins; b++) {
	    float eps00 = 1.0e-3 * ref_bc.cholesky_diag[b].linv.a00;
	    float eps11 = 1.0e-3 * ref_bc.cholesky_diag[b].linv.a11;
	    _compare("linv00", ref_bc.cholesky_diag[b].linv.a00, fast_sd.cholesky_invdiag[24*b+s], eps00);
	    _compare("linv10", ref_bc.cholesky_diag[b].linv.a10, fast_sd.cholesky_invdiag[24*b+8+s], sqrt(eps00*eps11));
	    _compare("linv11", ref_bc.cholesky_diag[b].linv.a11, fast_sd.cholesky_invdiag[24*b+16+s], eps11);
	}

	for (int b = 0; b < nbins; b++) {
	    _compare("ls00", ref_bc.cholesky_subdiag[b].a00, fast_sd.cholesky_subdiag[32*b+s], 1.0e-4);
	    _compare("ls01", ref_bc.cholesky_subdiag[b].a01, fast_sd.cholesky_subdiag[32*b+8+s], 1.0e-4);
	    _compare("ls10", ref_bc.cholesky_subdiag[b].a10, fast_sd.cholesky_subdiag[32*b+16+s], 1.0e-4);
	    _compare("ls11", ref_bc.cholesky_subdiag[b].a11, fast_sd.cholesky_subdiag[32*b+24+s], 1.0e-4);
	}

	for (int i = 0; i < 2*nbins+2; i++)
	    _compare("coeff", ref_coeffs[i], fast_sd.coeffs[8*i+s], 3.0e-3);
    }

    fast_sd._kernel_detrend(stride, fast_intensity);

    float maxdiff = 0.0;
    float wrmsdiff_num = 0.0;
    float wrmsdiff_den = 0.0;

    for (int s = 0; s < 8; s++) {
	ref_sd.detrend(&ref_intensity[s*nfreq], &ref_weights[s*nfreq], epsilon_reg);

	for (int ifreq = 0; ifreq < nfreq; ifreq++) {
	    float w = ref_weights[s*nfreq+ifreq];
	    float i_ref = ref_intensity[s*nfreq+ifreq];
	    float i_fast = fast_intensity[ifreq*stride+s];

	    maxdiff = max(maxdiff, abs(i_ref-i_fast));
	    wrmsdiff_num += w * (i_ref-i_fast) * (i_ref-i_fast);
	    wrmsdiff_den += w;
	}
    }
	
    float wrmsdiff = (wrmsdiff_den > 0.0) ? sqrt(wrmsdiff_num / wrmsdiff_den) : 0.0;

    if ((maxdiff > 1.0e-3) || (wrmsdiff > 1.0e-5)) {
	cout << "test_fast_kernels failed (detrend comparison): maxdiff=" << maxdiff << ", wrmsdiff=" << wrmsdiff << endl;
	exit(1);
    }
	
    free(fast_intensity);
    free(fast_weights);
}


static void test_fast_kernels(std::mt19937 &rng)
{
    for (int iter = 0; iter < 1000; iter++) {
	// Sometimes I run with more than 1000 iterations.
	if (iter % 2000 == 1999)
	    cout << "iteration " << (iter+1) << endl;

	// Number of bins is chosen in a way which tests both low-nbins and high-nbins cases.
	int nbins = randint(rng,0,2) ? randint(rng,1,100) : randint(rng,1,8);
	int nfreq = randint(rng, 64*nbins, 128*nbins);
	int stride = 8 * randint(rng, 1, 5);
	float epsilon_reg = exp(uniform_rand(rng, log(1.0e-3), 0.0));

	test_fast_kernels(rng, nfreq, nbins, stride, epsilon_reg);
    }

    cout << "test_fast_kernels: pass" << endl;
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    test_eval_cubic(rng);
    test_reference_spline_detrender(rng);
    // test_fast_kernels(rng, 32, 1, 8, 1.0);
    test_fast_kernels(rng);

    return 0;
}
