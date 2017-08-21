#include "rf_kernels/internals.hpp"
#include "rf_kernels/unit_testing.hpp"
#include "rf_kernels/spline_detrender.hpp"

using namespace std;
using namespace rf_kernels;


// -------------------------------------------------------------------------------------------------
//
// Helpers


inline float xsqrt(float x)
{
    rf_assert(x >= 0.0);
    return sqrt(x);
}


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
    for (int iter = 0; iter < 10; iter++) {
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
	lmat.a00 = xsqrt(a.a00);
	lmat.a01 = 0.0;
	lmat.a10 = a.a01 / lmat.a00;
	lmat.a11 = xsqrt(a.a11 - lmat.a10*lmat.a10);
	
	// These checks should apply if all weights are positive.
	rf_assert(a.a00 > 0.0);
	rf_assert(a.a11 > 0.0);
	rf_assert(lmat.a10*lmat.a10 <= 0.999 * a.a11);
	
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

    void fit_model(float *coeffs, const float *intensity, const float *weights);
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


void refsd_params::fit_model(float *coeffs, const float *intensity, const float *weights)
{
    vector<mat2> ninv_diag(nbins+1);
    vector<mat2> ninv_subdiag(nbins);
    vector<vec2> ninv_x(nbins+1);
    
    for (int b = 0; b < nbins; b++) {
	float ninv_bin[16];
	float ninvx_bin[4];
	analyze_bin(ninv_bin, ninvx_bin, intensity, weights, b);

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
}


// -------------------------------------------------------------------------------------------------


static void test_reference_spline_detrender(std::mt19937 &rng, int nx, int nbins)
{
    refsd_params params(nx, nbins);
    
    vector<float> intensity(nx, 0.0);
    vector<float> intensity2(nx, 0.0);
    vector<float> weights = uniform_randvec(rng, nx, 0.1, 1.0);

    vector<float> in_coeffs = uniform_randvec(rng, 2*nbins+2, -1.0, 1.0);
    params.eval_model(&intensity[0], &in_coeffs[0]);
    params.eval_model_from_scratch(&intensity2[0], &in_coeffs[0]);

    float eps1 = maxdiff(intensity, intensity2);
    if (eps1 > 1.0e-4) {
	cout << "refsd::eval_model comparison failed: nx=" << nx << ", nbins=" << nbins << ", epsilon=" << eps1 << endl;
	exit(2);
    }
    
    vector<float> out_coeffs(2*nbins+2, 0.0);
    params.fit_model(&out_coeffs[0], &intensity[0], &weights[0]);

    float eps2 = maxdiff(intensity, intensity2);
    if (eps2 > 1.0e-4) {
	cout << "refsd::fit_model comparison failed: nx=" << nx << ", nbins=" << nbins << ", epsilon=" << eps2 << endl;
	exit(2);
    }
}


static void test_reference_spline_detrender(std::mt19937 &rng)
{
    for (int iter = 0; iter < 1000; iter++) {
	int nbins = randint(rng, 1, 100);
	int nx = randint(rng, 32*nbins, 128*nbins);
	test_reference_spline_detrender(rng, nx, nbins);
    }

    cout << "test_reference_spline_detrender: pass" << endl;
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    test_eval_cubic(rng);
    test_reference_spline_detrender(rng);

    return 0;
}
