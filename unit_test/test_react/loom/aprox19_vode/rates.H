// Reaction rates for the Loom burn kernel (see README.md), aprox19:
// interfaces/tfactors.H and the 46 rates/aprox_rates.H rate functions
// networks/aprox19/actual_network.H evaluate_analytical_rate reaches, for both
// number types the burn uses:
//   *_r  number_t = amrex::Real      (networks/rhs.H rhs())
//   *_d  number_t = autodiff::dual   (networks/rhs.H jac(); d/dT only)
// plus the two Real-only weak-rate routines ecapnuc (p -> n) and langanke
// (ni56 electron capture) and langanke's tables (rv, tv, datn and the
// rates_init() cubic-interpolation factors rfd*/tfd*) as namespace-scope
// constexpr arrays behind 1-based accessors with the stock names.
// The _r functions mirror the stock arithmetic exactly; the _d functions are
// the dual.H call sequences autodiff evaluates (dual.H header, section 3).
// Rates return rate_t<T>{fr, rr} by value instead of writing fr/rr through
// references; tf and params come in by const reference. Runtime parameters
// come in through RatesParams (filled by fill_rates_params in
// unit_test/test_react/loom_params.H).
#ifndef MPL_RATES_H
#define MPL_RATES_H

#include "support.H"
#include "dual.H"
#include "network_data.H"

namespace mpl {

// Runtime parameters read by rates.H and screen.H.
struct RatesParams {
    int network_use_c12ag_deboer17;             // network_rp::use_c12ag_deboer17 (bool)
    int screening_enable_debye_huckel_skip;     // screening_rp::enable_debye_huckel_skip (bool)
    Real screening_debye_huckel_skip_threshold; // screening_rp::debye_huckel_skip_threshold
};

// from interfaces/rhs_type.H:115 rate_t
template <typename T>
struct rate_t {
    T fr;
    T rr;
};

// Rates::NetworkRates indices and NumRates: network_data.H (generated).

// ---------------------------------------------------------------------------
// interfaces/tfactors.H
// ---------------------------------------------------------------------------

// from interfaces/tfactors.H:11 tf_t (same fields, same order)
template <typename T>
struct tf_t {
    T temp;
    T t9;
    T t92;
    T t93;
    T t95;
    T t912;
    T t932;
    T t952;
    T t972;
    T t913;
    T t923;
    T t943;
    T t953;
    T t9i;
    T t9i2;
    T t9i12;
    T t9i32;
    T t9i13;
    T t9i23;
    T t9i43;
    T t9i53;
    T lnt9;
};

// from interfaces/tfactors.H:75 get_tfactors (number_t = Real)
MPL_DEVICE tf_t<Real> get_tfactors_r(Real temp)
{
    tf_t<Real> tf{};

    tf.temp = temp;

    tf.t9    = temp * 1.0e-9;
    tf.t92   = tf.t9*tf.t9;
    tf.t93   = tf.t9*tf.t92;
    tf.t95   = tf.t92*tf.t93;

    tf.t912  = math::sqrt(tf.t9);
    tf.t932  = tf.t9*tf.t912;
    tf.t952  = tf.t9*tf.t932;
    tf.t972  = tf.t92*tf.t932;

    tf.t913  = math::cbrt(tf.t9);
    tf.t923  = tf.t913*tf.t913;
    tf.t943  = tf.t9*tf.t913;
    tf.t953  = tf.t9*tf.t923;

    tf.t9i   = 1.0e0/tf.t9;
    tf.t9i2  = tf.t9i*tf.t9i;

    tf.t9i12 = 1.0e0/tf.t912;
    tf.t9i32 = tf.t9i*tf.t9i12;

    tf.t9i13 = 1.0e0/tf.t913;
    tf.t9i23 = tf.t9i13*tf.t9i13;
    tf.t9i43 = tf.t9i*tf.t9i13;
    tf.t9i53 = tf.t9i*tf.t9i23;

    tf.lnt9 = math::log(tf.t9);

    return tf;
}

// from interfaces/tfactors.H:75 get_tfactors (number_t = autodiff::dual)
MPL_DEVICE tf_t<Dual> get_tfactors_d(Dual temp)
{
    tf_t<Dual> tf{};

    tf.temp = temp;

    tf.t9    = dual_mul_num(temp, 1.0e-9);
    tf.t92   = dual_mul(tf.t9, tf.t9);
    tf.t93   = dual_mul(tf.t9, tf.t92);
    tf.t95   = dual_mul(tf.t92, tf.t93);

    tf.t912  = dual_sqrt(tf.t9);
    tf.t932  = dual_mul(tf.t9, tf.t912);
    tf.t952  = dual_mul(tf.t9, tf.t932);
    tf.t972  = dual_mul(tf.t92, tf.t932);

    tf.t913  = dual_cbrt(tf.t9);
    tf.t923  = dual_mul(tf.t913, tf.t913);
    tf.t943  = dual_mul(tf.t9, tf.t913);
    tf.t953  = dual_mul(tf.t9, tf.t923);

    tf.t9i   = dual_div(dual_const(1.0e0), tf.t9);
    tf.t9i2  = dual_mul(tf.t9i, tf.t9i);

    tf.t9i12 = dual_div(dual_const(1.0e0), tf.t912);
    tf.t9i32 = dual_mul(tf.t9i, tf.t9i12);

    tf.t9i13 = dual_div(dual_const(1.0e0), tf.t913);
    tf.t9i23 = dual_mul(tf.t9i13, tf.t9i13);
    tf.t9i43 = dual_mul(tf.t9i, tf.t9i13);
    tf.t9i53 = dual_mul(tf.t9i, tf.t9i23);

    tf.lnt9 = dual_log(tf.t9);

    return tf;
}

// ---------------------------------------------------------------------------
// rates/aprox_rates.H, number_t = Real: the stock statements, with
// admath:: -> math::, fr/rr -> rates.fr/rates.rr.
// ---------------------------------------------------------------------------

// from rates/aprox_rates.H:71 rate_c12ag
MPL_DEVICE rate_t<Real> rate_c12ag_r(const tf_t<Real>& tf, const Real den, const RatesParams& params)
{
    rate_t<Real> rates{};
    if (params.network_use_c12ag_deboer17) {
        // This version computes the nuclear reaction rate for 12C(a,g)16O and its inverse
        // using fit parameters from Deboer et al. 2017 (https://doi.org/10.1103/RevModPhys.89.035007).

        // from Table XXVI of deboer + 2017
        // non-resonant contributions to the reaction
        constexpr Real a0_nr = 24.1e0;
        constexpr Real a1_nr = 0e0;
        constexpr Real a2_nr = -32e0;
        constexpr Real a3_nr = -5.9e0;
        constexpr Real a4_nr = 1.8e0;
        constexpr Real a5_nr = -0.17e0;
        constexpr Real a6_nr = -2.0/3.0;

        const Real term_a0_nr = math::exp(a0_nr);
        const Real term_a1_nr = math::exp(a1_nr*tf.t9i);
        const Real term_a2_nr = math::exp(a2_nr*tf.t9i13);
        const Real term_a3_nr = math::exp(a3_nr*tf.t913);
        const Real term_a4_nr = math::exp(a4_nr*tf.t9);
        const Real term_a5_nr = math::exp(a5_nr*tf.t953);
        const Real term_a6_nr = math::pow(tf.t9,a6_nr);

        const Real term_nr = term_a0_nr * term_a1_nr * term_a2_nr *
                                 term_a3_nr * term_a4_nr * term_a5_nr *
                                 term_a6_nr;

        // resonant contributions to the reaction
        constexpr Real a0_r = 7.4e0;
        constexpr Real a1_r = -30e0;
        constexpr Real a2_r = 0e0;
        constexpr Real a3_r = 0e0;
        constexpr Real a4_r = 0e0;
        constexpr Real a5_r = 0e0;
        constexpr Real a6_r = -3.0e0/2.0e0;

        const Real term_a0_r = math::exp(a0_r);
        const Real term_a1_r = math::exp(a1_r*tf.t9i);
        const Real term_a2_r = math::exp(a2_r*tf.t9i13);
        const Real term_a3_r = math::exp(a3_r*tf.t913);
        const Real term_a4_r = math::exp(a4_r*tf.t9);
        const Real term_a5_r = math::exp(a5_r*tf.t953);
        const Real term_a6_r = math::pow(tf.t9,a6_r);

        const Real term_r = term_a0_r * term_a1_r * term_a2_r *
                                term_a3_r * term_a4_r * term_a5_r *
                                term_a6_r;

        // full rate is the sum of resonant and non-resonant contributions
        const Real term = term_nr + term_r;

        rates.fr    = term * den;

        // first term is 9.8685e9 * T9**(2/3) * (M0*M1/M3)**(3/2)
        // see iliadis 2007 eqn. 3.44
        // ratio of partition functions are assumed to be unity
        const Real rev    = 5.1345573e10 * tf.t932 * math::exp(-83.114082*tf.t9i);

        rates.rr     = rev * term;

    }
    else {

        constexpr Real q1 = 1.0e0/12.222016e0;

        // c12(a,g)o16
        const Real aa   = 1.0e0 + 0.0489e0*tf.t9i23;

        const Real bb   = tf.t92*aa*aa;

        const Real cc   = math::exp(-32.120e0*tf.t9i13 - tf.t92*q1);

        const Real dd   = 1.0e0 + 0.2654e0*tf.t9i23;

        const Real ee   = tf.t92*dd*dd;

        const Real ff   = math::exp(-32.120e0*tf.t9i13);

        const Real gg   = 1.25e3 * tf.t9i32 * math::exp(-27.499*tf.t9i);

        const Real hh   = 1.43e-2 * tf.t95 * math::exp(-15.541*tf.t9i);

        Real zz   = 1.0e0/bb;
        const Real f1   = cc*zz;

        zz   = 1.0e0/ee;
        const Real f2   = ff*zz;

        Real term = 1.04e8*f1  + 1.76e8*f2 + gg + hh;

        // 1.7 times cf88 value
        term     *= 1.7e0;

        rates.fr    = term * den;

        const Real rev = 5.13e10 * tf.t932 * math::exp(-83.111*tf.t9i);

        rates.rr     = rev * term;

    }

    return rates;
}

// from rates/aprox_rates.H:177 rate_triplealf
MPL_DEVICE rate_t<Real> rate_triplealf_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    constexpr Real rc28 = 0.1;
    constexpr Real q1 = 1.0 / 0.009604e0;
    constexpr Real q2 = 1.0e0/0.055225e0;

    // triple alfa to c12
    // this is a(a,g)be8
    const Real aa    = 7.40e5 * tf.t9i32 * math::exp(-1.0663*tf.t9i);
    const Real bb    = 4.164e9 * tf.t9i23 * math::exp(-13.49*tf.t9i13 - tf.t92*q1);
    const Real cc    = 1.0e0 + 0.031*tf.t913 + 8.009*tf.t923 + 1.732*tf.t9  // NOLINT(modernize-use-std-numbers)
          + 49.883*tf.t943 + 27.426*tf.t953;
    const Real r2abe    = aa + bb * cc;

    // this is be8(a,g)c12
    const Real dd    = 130.0e0 * tf.t9i32 * math::exp(-3.3364*tf.t9i);
    const Real ee    = 2.510e7 * tf.t9i23 * math::exp(-23.57*tf.t9i13 - tf.t92*q2);
    const Real ff    = 1.0e0 + 0.018*tf.t913 + 5.249*tf.t923 + 0.650*tf.t9 +
         19.176*tf.t943 + 6.034*tf.t953;
    const Real rbeac    = dd + ee * ff;

    // a factor
    const Real xx    = rc28 * 1.35e-07 * tf.t9i32 * math::exp(-24.811*tf.t9i);

    Real term = 0.0;
    // high temperature rate
    if (tf.t9 > 0.08) {
        term    = 2.90e-16 * r2abe * rbeac + xx;

    // low temperature rate
    } else {
        const Real uu   = 0.8e0*math::exp(-math::pow(0.025*tf.t9i,3.263));
        const Real yy   = 0.2e0 + uu;
        //    ! fxt yy   = 0.01 + 0.2e0 + uu;
        const Real vv   = 4.0e0*math::exp(-math::pow(tf.t9/0.025, 9.227));
        const Real zz   = 1.0e0 + vv;
        const Real zzinv   = 1.0e0/zz;
        const Real f1   = 0.01e0 + yy * zzinv;
        //    ! fxt f1   = yy * 1/zz;
        term = 2.90e-16 * r2abe * rbeac * f1 +  xx;
    }

    // rates
    //      term    = 1.2e0 * term

    rates.fr    = (term * den) * den;

    const Real rev    = 2.00e20*tf.t93*math::exp(-84.424*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:231 rate_c12c12
MPL_DEVICE rate_t<Real> rate_c12c12_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // c12 + c12 reaction
    // this is the C12(C12,g)Mg24 rate from Caughlin & Fowler (1988)

    const Real aa      = 1.0e0 + 0.0396*tf.t9;
    const Real zz      = 1.0e0/aa;
    const Real t9a     = tf.t9*zz;
    const Real t9a13   = math::cbrt(t9a);
    const Real t9a56   = math::pow(t9a,5.0/6.0);
    const Real term    = 4.27e26 * t9a56 * tf.t9i32 *
         math::exp(-84.165/t9a13 - 2.12e-03*tf.t93);

    // rates
    rates.fr    = den * term;
    rates.rr    = 0.0e0;
    return rates;
}

// from rates/aprox_rates.H:251 rate_c12o16
MPL_DEVICE rate_t<Real> rate_c12o16_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // c12 + o16 reaction; see cf88 references 47-4
    Real term = 0.0;
    if (tf.t9 >= 0.5) {
        Real aa     = 1.0e0 + 0.055*tf.t9;
        Real zz     = 1.0e0/aa;
        Real t9a    = tf.t9*zz;
        Real t9a13  = math::cbrt(t9a);
        Real t9a23  = t9a13*t9a13;
        Real t9a56  = math::pow(t9a,5.0/6.0);

        aa      = math::exp(-0.18*t9a*t9a);
        Real bb      = 1.06e-03*math::exp(2.562*t9a23);
        Real cc      = aa + bb;
        zz      = 1.0e0/cc;
        term    = 1.72e31 * t9a56 * tf.t9i32 * math::exp(-106.594/t9a13) * zz;

    } else {
        // term    = 2.6288035e-29
        term    = 0.0e0;
    }

    // the rates
    rates.fr    = den * term;
    rates.rr    = 0.0e0;
    return rates;
}

// from rates/aprox_rates.H:281 rate_o16o16
MPL_DEVICE rate_t<Real> rate_o16o16_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // o16 + o16
    // this is the O16(O16,g)S32 rate from Caughlin & Fowler (1988)

    const Real term  = 7.10e36 * tf.t9i23 *
         math::exp(-135.93 * tf.t9i13 - 0.629*tf.t923
         - 0.445*tf.t943 + 0.0103*tf.t9*tf.t9);

    // rates
    rates.fr    = den * term;
    rates.rr    = 0.0e0;
    return rates;
}

// from rates/aprox_rates.H:297 rate_o16ag
MPL_DEVICE rate_t<Real> rate_o16ag_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    constexpr Real q1 = 1.0e0/2.515396e0;

    // o16(a,g)ne20
    const Real term1   = 9.37e9 * tf.t9i23 * math::exp(-39.757*tf.t9i13 - tf.t92*q1);

    const Real aa      = 62.1 * tf.t9i32 * math::exp(-10.297*tf.t9i);
    const Real bb      = 538.0e0 * tf.t9i32 * math::exp(-12.226*tf.t9i);
    const Real cc      = 13.0e0 * tf.t92 * math::exp(-20.093*tf.t9i);
    const Real term2   = aa + bb + cc;

    const Real term    = term1 + term2;

    // the rates
    rates.fr    = den * term;

    const Real rev      = 5.65e10*tf.t932*math::exp(-54.937*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:321 rate_ne20ag
MPL_DEVICE rate_t<Real> rate_ne20ag_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    constexpr Real rc102 = 0.1;
    constexpr Real q1 = 1.0e0/4.923961e0;

    // ne20(a,g)mg24
    Real aa   = 4.11e11 * tf.t9i23 * math::exp(-46.766*tf.t9i13 - tf.t92*q1);
    Real bb   = 1.0e0 + 0.009*tf.t913 + 0.882*tf.t923 + 0.055*tf.t9
         + 0.749*tf.t943 + 0.119*tf.t953;
    const Real term1  = aa * bb;

    aa   = 5.27e3 * tf.t9i32 * math::exp(-15.869*tf.t9i);
    bb   = 6.51e3 * tf.t912 * math::exp(-16.223*tf.t9i);
    const Real term2  = aa + bb;

    aa   = 42.1 * tf.t9i32 * math::exp(-9.115*tf.t9i);
    bb   =  32.0 * tf.t9i23 * math::exp(-9.383*tf.t9i);
    const Real term3  = rc102 * (aa + bb);

    aa  = 5.0e0*math::exp(-18.960*tf.t9i);
    bb  = 1.0e0 + aa;
    const Real zz      = 1.0e0/bb;
    const Real term    = (term1 + term2 + term3)*zz;

    // the rates
    rates.fr    = den * term;

    const Real rev      = 6.01e10 * tf.t932 * math::exp(-108.059*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:355 rate_mg24ag
MPL_DEVICE rate_t<Real> rate_mg24ag_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    constexpr Real rc121 = 0.1e0;

    // 24mg(a,g)28si
    const Real aa    = 4.78e1 * tf.t9i32 * math::exp(-13.506*tf.t9i);
    const Real bb    =  2.38e3 * tf.t9i32 * math::exp(-15.218*tf.t9i);
    const Real cc    = 2.47e2 * tf.t932 * math::exp(-15.147*tf.t9i);
    const Real dd    = rc121 * 1.72e-09 * tf.t9i32 * math::exp(-5.028*tf.t9i);
    const Real ee    = rc121* 1.25e-03 * tf.t9i32 * math::exp(-7.929*tf.t9i);
    const Real ff    = rc121 * 2.43e1 * tf.t9i * math::exp(-11.523*tf.t9i);

    const Real gg    = 5.0e0*math::exp(-15.882*tf.t9i);
    const Real hh    = 1.0e0 + gg;
    const Real hhi   = 1.0e0/hh;
    const Real term    = (aa + bb + cc + dd + ee + ff) * hhi;

    // the rates
    rates.fr    = den * term;

    const Real rev      = 6.27e10 * tf.t932 * math::exp(-115.862*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:382 rate_mg24ap
MPL_DEVICE rate_t<Real> rate_mg24ap_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    constexpr Real rc148 = 0.1;
    constexpr Real q1 = 1.0 / 0.024649e0;

    // 24mg(a,p)al27
    Real aa     = 1.10e8 * tf.t9i23 * math::exp(-23.261*tf.t9i13 - tf.t92*q1);
    Real bb     =  1.0e0 + 0.018*tf.t913 + 12.85*tf.t923 + 1.61*tf.t9
         + 89.87*tf.t943 + 28.66*tf.t953;
    const Real term1  = aa * bb;

    aa     = 129.0e0 * tf.t9i32 * math::exp(-2.517*tf.t9i);
    bb     = 5660.0e0 * tf.t972 * math::exp(-3.421*tf.t9i);
    const Real cc     = rc148 * 3.89e-08 * tf.t9i32 * math::exp(-0.853*tf.t9i);
    const Real dd     = rc148 * 8.18e-09 * tf.t9i32 * math::exp(-1.001*tf.t9i);
    const Real term2  = aa + bb + cc + dd;

    const Real ee     = 1.0/3.0*math::exp(-9.792*tf.t9i);
    const Real ff     =  2.0/3.0 * math::exp(-11.773*tf.t9i);
    const Real gg     = 1.0e0 + ee + ff;

    const Real term    = (term1 + term2)/gg;

    // the rates
    const Real rev      = 1.81 * math::exp(-18.572*tf.t9i);

    rates.fr    = den * rev * term;
    rates.rr    = den * term;
    return rates;
}

// from rates/aprox_rates.H:414 rate_al27pg
MPL_DEVICE rate_t<Real> rate_al27pg_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // al27(p,g)si28
    // champagne 1996

    const Real aa  = 1.32e9 * tf.t9i23 * math::exp(-23.26*tf.t9i13);
    const Real bb  = (3.22e-10 * tf.t9i32 * math::exp(-0.836*tf.t9i))*0.17;
    const Real cc  = 1.74e0 * tf.t9i32 * math::exp(-2.269*tf.t9i);
    const Real dd  = 9.92e0 * tf.t9i32 * math::exp(-2.492*tf.t9i);
    const Real ee  = 4.29e1 * tf.t9i32 * math::exp(-3.273*tf.t9i);
    const Real ff  = 1.34e2 * tf.t9i32 * math::exp(-3.654*tf.t9i);
    const Real gg  = 1.77e4 * math::pow(tf.t9,0.53) * math::exp(-4.588*tf.t9i);
    const Real term    = aa + bb + cc + dd + ee + ff + gg;

    // rates
    rates.fr    = den * term;

    const Real rev      = 1.13e11 * tf.t932 * math::exp(-134.434*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}


// from rates/aprox_rates.H:471 rate_si28ag
MPL_DEVICE rate_t<Real> rate_si28ag_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // si28(a,g)s32
    const Real z     = math::min(10.0e0, tf.t9);
    const Real z2    = z*z;
    const Real z3    = z2*z;
    const Real aa    = 1.0e0 + 6.340e-2*z + 2.541e-3*z2 - 2.900e-4*z3;

    const Real term    = 4.82e22 * tf.t9i23 * math::exp(-61.015 * tf.t9i13 * aa);

    // the rates
    rates.fr    = den * term;

    const Real rev      = 6.461e10 * tf.t932 * math::exp(-80.643*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:491 rate_si28ap
MPL_DEVICE rate_t<Real> rate_si28ap_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // si28(a,p)p31
    const Real z     = math::min(10.0e0, tf.t9);
    const Real z2    = z*z;
    const Real z3    = z2*z;
    const Real aa    = 1.0e0 + 2.798e-3*z + 2.763e-3*z2 - 2.341e-4*z3;

    const Real term    = 4.16e13 * tf.t9i23 * math::exp(-25.631 * tf.t9i13 * aa);

    // the rates
    const Real rev      = 0.5825e0 * math::exp(-22.224*tf.t9i);

    rates.fr    = den * rev * term;
    rates.rr    = den * term;
    return rates;
}

// from rates/aprox_rates.H:510 rate_p31pg
MPL_DEVICE rate_t<Real> rate_p31pg_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // p31(p,g)s32
    const Real z     = math::min(10.0e0, tf.t9);
    const Real z2    = z*z;
    const Real z3    = z2*z;
    const Real aa    = 1.0e0 + 1.928e-1*z - 1.540e-2*z2 + 6.444e-4*z3;

    const Real term    = 1.08e16 * tf.t9i23 * math::exp(-27.042 * tf.t9i13 * aa);

    // the rates
    rates.fr    = den * term;

    const Real rev      = 3.764e10 * tf.t932 * math::exp(-102.865*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:530 rate_s32ag
MPL_DEVICE rate_t<Real> rate_s32ag_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // s32(a,g)ar36
    const Real z     = math::min(10.0e0, tf.t9);
    const Real z2    = z*z;
    const Real z3    = z2*z;
    const Real aa    = 1.0e0 + 4.913e-2*z + 4.637e-3*z2 - 4.067e-4*z3;

    const Real term    = 1.16e24 * tf.t9i23 * math::exp(-66.690 * tf.t9i13 * aa);

    // the rates
    rates.fr    = den * term;

    const Real rev      = 6.616e10 * tf.t932 * math::exp(-77.080*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:550 rate_s32ap
MPL_DEVICE rate_t<Real> rate_s32ap_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // s32(a,p)cl35
    const Real z     = math::min(10.0e0, tf.t9);
    const Real z2    = z*z;
    const Real z3    = z2*z;
    const Real aa    = 1.0e0 + 1.041e-1*z - 1.368e-2*z2 + 6.969e-4*z3;

    const Real term    = 1.27e16 * tf.t9i23 * math::exp(-31.044 * tf.t9i13 * aa);

    // the rates
    const Real rev      = 1.144 * math::exp(-21.643*tf.t9i);

    rates.fr    = den * rev * term;
    rates.rr    = den * term;
    return rates;
}

// from rates/aprox_rates.H:569 rate_cl35pg
MPL_DEVICE rate_t<Real> rate_cl35pg_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // cl35(p,g)ar36
    const Real aa    = 1.0e0 + 1.761e-1*tf.t9 - 1.322e-2*tf.t92 + 5.245e-4*tf.t93;
    const Real term    =  4.48e16 * tf.t9i23 * math::exp(-29.483 * tf.t9i13 * aa);

    // the rates
    rates.fr    = den * term;

    const Real rev      = 7.568e10*tf.t932*math::exp(-98.722*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:585 rate_ar36ag
MPL_DEVICE rate_t<Real> rate_ar36ag_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // ar36(a,g)ca40
    const Real z     = math::min(10.0e0, tf.t9);
    const Real z2    = z*z;
    const Real z3    = z2*z;
    const Real aa    = 1.0e0 + 1.458e-1*z - 1.069e-2*z2 + 3.790e-4*z3;

    const Real term    = 2.81e30 * tf.t9i23 * math::exp(-78.271 * tf.t9i13 * aa);

    // the rates
    rates.fr    = den * term;

    const Real rev      = 6.740e10 * tf.t932 * math::exp(-81.711*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:605 rate_ar36ap
MPL_DEVICE rate_t<Real> rate_ar36ap_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // ar36(a,p)k39
    const Real z     = math::min(10.0e0, tf.t9);
    const Real z2    = z*z;
    const Real z3    = z2*z;
    const Real aa    = 1.0e0 + 4.826e-3*z - 5.534e-3*z2 + 4.021e-4*z3;

    const Real term    = 2.76e13 * tf.t9i23 * math::exp(-34.922 * tf.t9i13 * aa);

    // the rates
    const Real rev      = 1.128*math::exp(-14.959*tf.t9i);

    rates.fr    = den * rev * term;

    rates.rr    = den * term;
    return rates;
}

// from rates/aprox_rates.H:625 rate_k39pg
MPL_DEVICE rate_t<Real> rate_k39pg_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // k39(p,g)ca40
    const Real z     = math::min(10.0e0, tf.t9);
    const Real z2    = z*z;
    const Real z3    = z2*z;
    const Real aa    = 1.0e0 + 1.622e-1*z - 1.119e-2*z2 + 3.910e-4*z3;

    const Real term    = 4.09e16 * tf.t9i23 * math::exp(-31.727 * tf.t9i13 * aa);

    // the rates
    rates.fr    = den * term;

    const Real rev      = 7.600e10 * tf.t932 * math::exp(-96.657*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:645 rate_ca40ag
MPL_DEVICE rate_t<Real> rate_ca40ag_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // ca40(a,g)ti44
    const Real z     = math::min(10.0e0, tf.t9);
    const Real z2    = z*z;
    const Real z3    = z2*z;
    const Real aa    = 1.0e0 + 1.650e-2*z + 5.973e-3*z2 - 3.889e-04*z3;

    const Real term    = 4.66e24 * tf.t9i23 * math::exp(-76.435 * tf.t9i13 * aa);

    // the rates
    rates.fr    = den * term;

    const Real rev      = 6.843e10 * tf.t932 * math::exp(-59.510*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:665 rate_ca40ap
MPL_DEVICE rate_t<Real> rate_ca40ap_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // ca40(a,p)sc43
    const Real z     = math::min(10.0e0, tf.t9);
    const Real z2    = z*z;
    const Real z3    = z2*z;
    const Real aa    = 1.0e0 - 1.206e-2*z + 7.753e-3*z2 - 5.071e-4*z3;

    const Real term    = 4.54e14 * tf.t9i23 * math::exp(-32.177 * tf.t9i13 * aa);

    // the rates
    const Real rev      = 2.229 * math::exp(-40.966*tf.t9i);

    rates.fr    = den * rev * term;
    rates.rr    = den * term;
    return rates;
}

// from rates/aprox_rates.H:684 rate_sc43pg
MPL_DEVICE rate_t<Real> rate_sc43pg_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // sc43(p,g)ca40
    const Real z     = math::min(10.0e0, tf.t9);
    const Real z2    = z*z;
    const Real z3    = z2*z;
    const Real aa    = 1.0e0 + 1.023e-1*z - 2.242e-3*z2 - 5.463e-5*z3;

    const Real term    = 3.85e16 * tf.t9i23 * math::exp(-33.234 * tf.t9i13 * aa);

    // the rates
    rates.fr    = den * term;

    const Real rev      = 1.525e11 * tf.t932 * math::exp(-100.475*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:704 rate_ti44ag
MPL_DEVICE rate_t<Real> rate_ti44ag_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // ti44(a,g)cr48
    const Real z     = math::min(10.0e0, tf.t9);
    const Real z2    = z*z;
    const Real z3    = z2*z;
    const Real aa    = 1.0e0 + 1.066e-1*z - 1.102e-2*z2 + 5.324e-4*z3;

    const Real term    = 1.37e26 * tf.t9i23 * math::exp(-81.227 * tf.t9i13 * aa);

    // the rates
    rates.fr    = den * term;

    const Real rev      = 6.928e10*tf.t932*math::exp(-89.289*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:724 rate_ti44ap
MPL_DEVICE rate_t<Real> rate_ti44ap_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // ti44(a,p)v47
    const Real z     = math::min(10.0e0, tf.t9);
    const Real z2    = z*z;
    const Real z3    = z2*z;
    const Real aa    = 1.0e0 + 2.655e-2*z - 3.947e-3*z2 + 2.522e-4*z3;

    const Real term    = 6.54e20 * tf.t9i23 * math::exp(-66.678 * tf.t9i13 * aa);

    // the rates
    const Real rev      = 1.104 * math::exp(-4.723*tf.t9i);

    rates.fr    = den * rev * term;
    rates.rr    = den * term;
    return rates;
}

// from rates/aprox_rates.H:743 rate_v47pg
MPL_DEVICE rate_t<Real> rate_v47pg_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // v47(p,g)cr48
    const Real z     = math::min(10.0e0, tf.t9);
    const Real z2    = z*z;
    const Real z3    = z2*z;
    const Real aa    = 1.0e0 + 9.979e-2*z - 2.269e-3*z2 - 6.662e-5*z3;

    const Real term    = 2.05e17 * tf.t9i23 * math::exp(-35.568 * tf.t9i13 * aa);

    // the rates
    rates.fr    = den * term;

    const Real rev      = 7.649e10*tf.t932*math::exp(-93.999*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:763 rate_cr48ag
MPL_DEVICE rate_t<Real> rate_cr48ag_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // cr48(a,g)fe52
    const Real z     = math::min(10.0e0, tf.t9);
    const Real z2    = z*z;
    const Real z3    = z2*z;
    const Real aa    = 1.0e0 + 6.325e-2*z - 5.671e-3*z2 + 2.848e-4*z3;

    const Real term    = 1.04e23 * tf.t9i23 * math::exp(-81.420 * tf.t9i13 * aa);

    // the rates
    rates.fr    = den * term;

    const Real rev      = 7.001e10 * tf.t932 * math::exp(-92.177*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:783 rate_cr48ap
MPL_DEVICE rate_t<Real> rate_cr48ap_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // cr48(a,p)mn51
    const Real z     = math::min(10.0e0, tf.t9);
    const Real z2    = z*z;
    const Real z3    = z2*z;
    const Real aa    = 1.0e0 + 1.384e-2*z + 1.081e-3*z2 - 5.933e-5*z3;

    const Real term    = 1.83e26 * tf.t9i23 * math::exp(-86.741 * tf.t9i13 * aa);

    // the rates
    rates.fr    = den * term;

    const Real rev      = 0.6087*math::exp(-6.510*tf.t9i);

    rates.rr    = den * rev * term;
    return rates;
}

// from rates/aprox_rates.H:803 rate_mn51pg
MPL_DEVICE rate_t<Real> rate_mn51pg_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // mn51(p,g)fe52
    const Real z     = math::min(10.0e0, tf.t9);
    const Real z2    = z*z;
    const Real z3    = z2*z;
    const Real aa    = 1.0e0 + 8.922e-2*z - 1.256e-3*z2 - 9.453e-5*z3;

    const Real term    = 3.77e17 * tf.t9i23 * math::exp(-37.516 * tf.t9i13 * aa);

    // the rates
    rates.fr    = den * term;

    const Real rev      = 1.150e11*tf.t932*math::exp(-85.667*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:823 rate_fe52ag
MPL_DEVICE rate_t<Real> rate_fe52ag_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // fe52(a,g)ni56
    const Real z     = math::min(10.0e0, tf.t9);
    const Real z2    = z*z;
    const Real z3    = z2*z;
    const Real aa    = 1.0e0 + 7.846e-2*z - 7.430e-3*z2 + 3.723e-4*z3;

    const Real term    = 1.05e27 * tf.t9i23 * math::exp(-91.674 * tf.t9i13 * aa);

    // the rates
    rates.fr    = den * term;

    const Real rev      = 7.064e10*tf.t932*math::exp(-92.850*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:843 rate_fe52ap
MPL_DEVICE rate_t<Real> rate_fe52ap_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // fe52(a,p)co55
    const Real z     = math::min(10.0e0, tf.t9);
    const Real z2    = z*z;
    const Real z3    = z2*z;
    const Real aa    = 1.0e0 + 1.367e-2*z + 7.428e-4*z2 - 3.050e-5*z3;

    const Real term    = 1.30e27 * tf.t9i23 * math::exp(-91.674 * tf.t9i13 * aa);

    // the rates
    rates.fr    = den * term;

    const Real rev      = 0.4597*math::exp(-9.470*tf.t9i);

    rates.rr    = den * rev * term;
    return rates;
}

// from rates/aprox_rates.H:863 rate_co55pg
MPL_DEVICE rate_t<Real> rate_co55pg_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // co55(p,g)ni56
    const Real z     = math::min(10.0e0, tf.t9);
    const Real z2    = z*z;
    const Real z3    = z2*z;
    const Real aa    = 1.0e0 + 9.894e-2*z - 3.131e-3*z2 - 2.160e-5*z3;

    const Real term    = 1.21e18 * tf.t9i23 * math::exp(-39.604 * tf.t9i13 * aa);

    // the rates
    rates.fr    = den * term;

    const Real rev      = 1.537e11*tf.t932*math::exp(-83.382*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:883 rate_pp
MPL_DEVICE rate_t<Real> rate_pp_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // p(p,e+nu)d
    Real term;
    if (tf.t9 <= 3.0) {
       const Real aa   = 4.01e-15 * tf.t9i23 * math::exp(-3.380e0*tf.t9i13);
       const Real bb   = 1.0e0 + 0.123e0*tf.t913 + 1.09e0*tf.t923 + 0.938e0*tf.t9;
       term    = aa * bb;
    } else {
       term    = 1.1581136e-15;
    }

    // rate
    rates.fr    = den * term;
    rates.rr    = 0.0e0;
    return rates;
}

// from rates/aprox_rates.H:902 rate_png
MPL_DEVICE rate_t<Real> rate_png_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // p(n,g)d
    // smith,kawano,malany 1992

    const Real aa      = 1.0e0 - 0.8504*tf.t912 + 0.4895*tf.t9
         - 0.09623*tf.t932 + 8.471e-3*tf.t92
         - 2.80e-4*tf.t952;
    const Real term    = 4.742e4 * aa;

    // rates
    rates.fr    = den * term;

    const Real rev      = 4.71e9 * tf.t932 * math::exp(-25.82*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:926 rate_dpg
MPL_DEVICE rate_t<Real> rate_dpg_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // d(p,g)he3
    const Real aa      = 2.24e3 * tf.t9i23 * math::exp(-3.720*tf.t9i13);
    const Real bb      = 1.0e0 + 0.112*tf.t913 + 3.38*tf.t923 + 2.65*tf.t9;
    const Real term    = aa * bb;

    // rates
    rates.fr    = den * term;

    const Real rev      = 1.63e10 * tf.t932 * math::exp(-63.750*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:943 rate_he3ng
MPL_DEVICE rate_t<Real> rate_he3ng_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // he3(n,g)he4
    const Real term    = 6.62 * (1.0e0 + 905.0*tf.t9);

    // rates
    rates.fr    = den * term;

    const Real rev      = 2.61e10 * tf.t932 * math::exp(-238.81*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:958 rate_he3he3
MPL_DEVICE rate_t<Real> rate_he3he3_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // he3(he3,2p)he4
    const Real aa   = 6.04e10 * tf.t9i23 * math::exp(-12.276*tf.t9i13);
    const Real bb   = 1.0e0 + 0.034*tf.t913 - 0.522*tf.t923 - 0.124*tf.t9
         + 0.353*tf.t943 + 0.213*tf.t953;
    const Real term    = aa * bb;

    // rates
    rates.fr    = den * term;

    const Real rev      = 3.39e-10 * tf.t9i32 * math::exp(-149.230*tf.t9i);

    rates.rr    = den * den * rev * term;
    return rates;
}

// from rates/aprox_rates.H:976 rate_he3he4
MPL_DEVICE rate_t<Real> rate_he3he4_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // he3(he4,g)be7
    const Real aa      = 1.0e0 + 0.0495*tf.t9;

    Real zz      = 1.0e0/aa;
    const Real t9a     = tf.t9*zz;
    const Real t9a13   = math::cbrt(t9a);
    const Real t9a56   = math::pow(t9a,5.0/6.0);

    const Real term    = 5.61e6 * t9a56 * tf.t9i32 * math::exp(-12.826/t9a13);

    // rates
    rates.fr    = den * term;

    const Real rev      = 1.11e10 * tf.t932 * math::exp(-18.423*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:998 rate_c12pg
MPL_DEVICE rate_t<Real> rate_c12pg_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    constexpr Real q1 = 1.0e0/2.25e0;

    // c12(p,g)13n
    const Real aa   = 2.04e7 * tf.t9i23 * math::exp(-13.69*tf.t9i13 - tf.t92*q1);
    const Real bb   = 1.0e0 + 0.03*tf.t913 + 1.19*tf.t923 + 0.254*tf.t9
         + 2.06*tf.t943 + 1.12*tf.t953;
    const Real cc   = aa * bb;

    const Real dd   = 1.08e5 * tf.t9i32 * math::exp(-4.925*tf.t9i);

    const Real ee   = 2.15e5 * tf.t9i32 * math::exp(-18.179*tf.t9i);

    const Real term    = cc + dd + ee;

    // rates
    rates.fr    = den * term;

    const Real rev      = 8.84e9 * tf.t932 * math::exp(-22.553*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:1024 rate_n14pg
MPL_DEVICE rate_t<Real> rate_n14pg_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    constexpr Real q1 = 1.0e0/10.850436e0;

    // n14(p,g)o15
    const Real aa  = 4.90e7 * tf.t9i23 * math::exp(-15.228*tf.t9i13 - tf.t92*q1);
    const Real bb   = 1.0e0 + 0.027*tf.t913 - 0.778*tf.t923 - 0.149*tf.t9
         + 0.261*tf.t943 + 0.127*tf.t953;
    const Real cc   = aa * bb;

    const Real dd   = 2.37e3 * tf.t9i32 * math::exp(-3.011*tf.t9i);

    const Real ee   = 2.19e4 * math::exp(-12.530*tf.t9i);

    const Real term    = cc + dd + ee;

    // rates
    rates.fr    = den * term;

    const Real rev    = 2.70e10 * tf.t932 * math::exp(-84.678*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:1050 rate_n15pg
MPL_DEVICE rate_t<Real> rate_n15pg_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    constexpr Real q1 = 1.0 / 0.2025;

    // n15(p,g)o16
    const Real aa  = 9.78e8 * tf.t9i23 * math::exp(-15.251*tf.t9i13 - tf.t92*q1);
    const Real bb   = 1.0e0  + 0.027*tf.t913 + 0.219*tf.t923 + 0.042*tf.t9
         + 6.83*tf.t943 + 3.32*tf.t953;
    const Real cc   = aa * bb;

    const Real dd   = 1.11e4*tf.t9i32*math::exp(-3.328*tf.t9i);

    const Real ee   = 1.49e4*tf.t9i32*math::exp(-4.665*tf.t9i);

    const Real ff   = 3.8e6*tf.t9i32*math::exp(-11.048*tf.t9i);

    const Real term    = cc + dd + ee + ff;

    // rates
    rates.fr    = den * term;

    const Real rev      = 3.62e10 * tf.t932 * math::exp(-140.734*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:1078 rate_n15pa
MPL_DEVICE rate_t<Real> rate_n15pa_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    constexpr Real theta = 0.1;
    constexpr Real q1 = 1.0 / 0.272484;

    // n15(p,a)c12
    const Real aa  = 1.08e12*tf.t9i23*math::exp(-15.251*tf.t9i13 - tf.t92*q1);
    const Real bb   = 1.0e0 + 0.027*tf.t913 + 2.62*tf.t923 + 0.501*tf.t9
         + 5.36*tf.t943 + 2.60*tf.t953;
    const Real cc   = aa * bb;

    const Real dd   = 1.19e8 * tf.t9i32 * math::exp(-3.676*tf.t9i);

    const Real ee   = 5.41e8 * tf.t9i12 * math::exp(-8.926*tf.t9i);

    const Real ff   = theta * 4.72e8 * tf.t9i32 * math::exp(-7.721*tf.t9i);

    const Real gg   = theta * 2.20e9 * tf.t9i32 * math::exp(-11.418*tf.t9i);

    const Real term    = cc + dd + ee + ff + gg;

    // rates
    rates.fr    = den * term;

    const Real rev      = 7.06e-01*math::exp(-57.625*tf.t9i);

    rates.rr    = den * rev * term;
    return rates;
}

// from rates/aprox_rates.H:1109 rate_o16pg
MPL_DEVICE rate_t<Real> rate_o16pg_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // o16(p,g)f17
    const Real aa  = math::exp(-0.728*tf.t923);
    const Real bb  = 1.0e0 + 2.13 * (1.0e0 - aa);
    const Real cc  = tf.t923 * bb;
    const Real dd   = math::exp(-16.692*tf.t9i13);
    const Real zz   = 1.0e0/cc;
    const Real ee   = dd*zz;
    const Real term    = 1.50e8 * ee;

    // rates
    rates.fr    = den * term;

    const Real rev      = 3.03e9*tf.t932*math::exp(-6.968*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:1130 rate_n14ag
MPL_DEVICE rate_t<Real> rate_n14ag_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    constexpr Real q1 = 1.0e0/0.776161e0;

    // n14(a,g)f18
    const Real aa  = 7.78e9 * tf.t9i23 * math::exp(-36.031*tf.t9i13- tf.t92*q1);
    const Real bb   = 1.0e0 + 0.012*tf.t913 + 1.45*tf.t923 + 0.117*tf.t9
         + 1.97*tf.t943 + 0.406*tf.t953;
    const Real cc   = aa * bb;

    const Real dd   = 2.36e-10 * tf.t9i32 * math::exp(-2.798*tf.t9i);

    const Real ee   = 2.03 * tf.t9i32 * math::exp(-5.054*tf.t9i);

    const Real ff   = 1.15e4 * tf.t9i23 * math::exp(-12.310*tf.t9i);

    const Real term    = cc + dd + ee + ff;

    // rates
    rates.fr    = den * term;

    const Real rev      = 5.42e10 * tf.t932 * math::exp(-51.236*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:1158 rate_fe52ng
MPL_DEVICE rate_t<Real> rate_fe52ng_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // fe52(n,g)fe53
    const Real tq2     = tf.t9 - 0.348e0;
    const Real term    = 9.604e5 * math::exp(-0.0626*tq2);

    // the rates
    rates.fr    = den * term;

    const Real rev      = 2.43e9 * tf.t932 * math::exp(-123.951*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:1174 rate_fe53ng
MPL_DEVICE rate_t<Real> rate_fe53ng_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // fe53(n,g)fe54
    const Real tq1   = tf.t9/0.348;
    const Real tq10  = math::pow(tq1, 0.10);
    const Real tq2   = tf.t9 - 0.348e0;

    const Real term    = 1.817e6 * tq10 * math::exp(-0.06319*tq2);

    // the rates
    rates.fr    = den * term;

    const Real rev      = 1.56e11 * tf.t932 * math::exp(-155.284*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// from rates/aprox_rates.H:1218 rate_fe54pg
MPL_DEVICE rate_t<Real> rate_fe54pg_r(const tf_t<Real>& tf, const Real den)
{
    rate_t<Real> rates{};
    // fe54(p,g)co55
    const Real z     = math::min(10.0e0,tf.t9);
    const Real z2    = z*z;
    const Real z3    = z2*z;
    const Real aa    = 1.0e0 + 9.593e-2*z - 3.445e-3*z2 + 8.594e-5*z3;

    const Real term    = 4.51e17 * tf.t9i23 * math::exp(-38.483 * tf.t9i13 * aa);

    // the rates
    rates.fr    = den * term;

    const Real rev      = 2.400e9 * tf.t932 * math::exp(-58.605*tf.t9i);

    rates.rr    = rev * term;
    return rates;
}

// ---------------------------------------------------------------------------
// rates/aprox_rates.H, number_t = autodiff::dual. Helpers name recurring
// expression shapes; each is the exact dual.H call sequence for that shape.
// ---------------------------------------------------------------------------

// `c * x * admath::exp(e * arg)`, c and e numbers, x and arg duals:
// Mul(NDM(c, x), Exp(NDM(e, arg)))
MPL_DEVICE Dual rates_cxexp_d(Real c, Dual x, Real e, Dual arg)
{
    return dual_mul(dual_mul_num(x, c), dual_exp(dual_mul_num(arg, e)));
}

// `c * admath::exp(e * arg)`: Mul(c, Exp(NDM(e, arg)))
MPL_DEVICE Dual rates_cexp_d(Real c, Real e, Dual arg)
{
    return dual_mul(dual_const(c), dual_exp(dual_mul_num(arg, e)));
}

// `c * tf.t9i23 * admath::exp(e*tf.t9i13 - tf.t92*q)`:
// Mul(NDM(c, t9i23), Exp(Add(NDM(e, t9i13), Neg(Mul(t92, q)))))
MPL_DEVICE Dual rates_cexp2_d(const tf_t<Dual>& tf, Real c, Real e, Real q)
{
    return dual_mul(dual_mul_num(tf.t9i23, c),
                    dual_exp(dual_sub(dual_mul_num(tf.t9i13, e),
                                      dual_mul_num(tf.t92, q))));
}

// `1.0 + c1*tf.t913 + c2*tf.t923 + c3*tf.t9 + c4*tf.t943 + c5*tf.t953`
MPL_DEVICE Dual rates_poly5_d(const tf_t<Dual>& tf, Real c1, Real c2, Real c3, Real c4, Real c5)
{
    Dual s = dual_const(1.0e0);
    s = dual_add_ndm(s, c1, tf.t913);
    s = dual_add_ndm(s, c2, tf.t923);
    s = dual_add_ndm(s, c3, tf.t9);
    s = dual_add_ndm(s, c4, tf.t943);
    s = dual_add_ndm(s, c5, tf.t953);
    return s;
}

// The z-polynomial rates (si28ag ... co55pg):
//   z = admath::min(10.0e0_rt, tf.t9); z2 = z*z; z3 = z2*z;
//   aa = 1.0e0_rt + c1*z + c2*z2 + c3*z3;   (a source `- c*zk` is c = -c)
//   term = ct * tf.t9i23 * admath::exp(et * tf.t9i13 * aa);
MPL_DEVICE Dual rates_zterm_d(const tf_t<Dual>& tf, Real c1, Real c2, Real c3, Real ct, Real et)
{
    const Dual z = dual_min(dual_const(10.0e0), tf.t9);
    const Dual z2 = dual_mul(z, z);
    const Dual z3 = dual_mul(z2, z);
    const Dual aa = dual_add_ndm(dual_add_ndm(dual_add_ndm(dual_const(1.0e0), c1, z), c2, z2), c3, z3);
    return dual_mul(dual_mul_num(tf.t9i23, ct),
                    dual_exp(dual_mul(dual_mul_num(tf.t9i13, et), aa)));
}

// fr = den * term; rev = cr * tf.t932 * exp(er*tf.t9i); rr = rev * term;
MPL_DEVICE rate_t<Dual> rates_fin_pg_d(const tf_t<Dual>& tf, Real den, Dual term, Real cr, Real er)
{
    rate_t<Dual> rates{};
    rates.fr = dual_mul_num(term, den);
    const Dual rev = rates_cxexp_d(cr, tf.t932, er, tf.t9i);
    rates.rr = dual_mul(rev, term);
    return rates;
}

// rev = cr * exp(er*tf.t9i); fr = den * rev * term; rr = den * term;
MPL_DEVICE rate_t<Dual> rates_fin_ap_d(const tf_t<Dual>& tf, Real den, Dual term, Real cr, Real er)
{
    rate_t<Dual> rates{};
    const Dual rev = rates_cexp_d(cr, er, tf.t9i);
    rates.fr = dual_mul(dual_mul_num(rev, den), term);
    rates.rr = dual_mul_num(term, den);
    return rates;
}

// fr = den * term; rev = cr * exp(er*tf.t9i); rr = den * rev * term;
MPL_DEVICE rate_t<Dual> rates_fin_ap_rev_d(const tf_t<Dual>& tf, Real den, Dual term, Real cr, Real er)
{
    rate_t<Dual> rates{};
    rates.fr = dual_mul_num(term, den);
    const Dual rev = rates_cexp_d(cr, er, tf.t9i);
    rates.rr = dual_mul(dual_mul_num(rev, den), term);
    return rates;
}

// from rates/aprox_rates.H:71 rate_c12ag
MPL_DEVICE rate_t<Dual> rate_c12ag_d(const tf_t<Dual>& tf, const Real den, const RatesParams& params)
{
    rate_t<Dual> rates{};
    if (params.network_use_c12ag_deboer17) {
        constexpr Real a0_nr = 24.1e0;
        constexpr Real a1_nr = 0e0;
        constexpr Real a2_nr = -32e0;
        constexpr Real a3_nr = -5.9e0;
        constexpr Real a4_nr = 1.8e0;
        constexpr Real a5_nr = -0.17e0;
        constexpr Real a6_nr = -2.0/3.0;

        const Real term_a0_nr = math::exp(a0_nr);
        const Dual term_a1_nr = dual_exp(dual_mul_num(tf.t9i, a1_nr));
        const Dual term_a2_nr = dual_exp(dual_mul_num(tf.t9i13, a2_nr));
        const Dual term_a3_nr = dual_exp(dual_mul_num(tf.t913, a3_nr));
        const Dual term_a4_nr = dual_exp(dual_mul_num(tf.t9, a4_nr));
        const Dual term_a5_nr = dual_exp(dual_mul_num(tf.t953, a5_nr));
        const Dual term_a6_nr = dual_pow_num(tf.t9, a6_nr);

        Dual term_nr = dual_mul_num(term_a1_nr, term_a0_nr);
        term_nr = dual_mul(term_nr, term_a2_nr);
        term_nr = dual_mul(term_nr, term_a3_nr);
        term_nr = dual_mul(term_nr, term_a4_nr);
        term_nr = dual_mul(term_nr, term_a5_nr);
        term_nr = dual_mul(term_nr, term_a6_nr);

        constexpr Real a0_r = 7.4e0;
        constexpr Real a1_r = -30e0;
        constexpr Real a2_r = 0e0;
        constexpr Real a3_r = 0e0;
        constexpr Real a4_r = 0e0;
        constexpr Real a5_r = 0e0;
        constexpr Real a6_r = -3.0e0/2.0e0;

        const Real term_a0_r = math::exp(a0_r);
        const Dual term_a1_r = dual_exp(dual_mul_num(tf.t9i, a1_r));
        const Dual term_a2_r = dual_exp(dual_mul_num(tf.t9i13, a2_r));
        const Dual term_a3_r = dual_exp(dual_mul_num(tf.t913, a3_r));
        const Dual term_a4_r = dual_exp(dual_mul_num(tf.t9, a4_r));
        const Dual term_a5_r = dual_exp(dual_mul_num(tf.t953, a5_r));
        const Dual term_a6_r = dual_pow_num(tf.t9, a6_r);

        Dual term_r = dual_mul_num(term_a1_r, term_a0_r);
        term_r = dual_mul(term_r, term_a2_r);
        term_r = dual_mul(term_r, term_a3_r);
        term_r = dual_mul(term_r, term_a4_r);
        term_r = dual_mul(term_r, term_a5_r);
        term_r = dual_mul(term_r, term_a6_r);

        const Dual term = dual_add(term_nr, term_r);

        rates.fr = dual_mul_num(term, den);

        const Dual rev = rates_cxexp_d(5.1345573e10, tf.t932, -83.114082, tf.t9i);

        rates.rr = dual_mul(rev, term);
    } else {
        constexpr Real q1 = 1.0e0/12.222016e0;

        const Dual aa = dual_add_ndm(dual_const(1.0e0), 0.0489e0, tf.t9i23);
        const Dual bb = dual_mul(dual_mul(tf.t92, aa), aa);
        const Dual cc = dual_exp(dual_sub(dual_mul_num(tf.t9i13, -32.120e0),
                                          dual_mul_num(tf.t92, q1)));
        const Dual dd = dual_add_ndm(dual_const(1.0e0), 0.2654e0, tf.t9i23);
        const Dual ee = dual_mul(dual_mul(tf.t92, dd), dd);
        const Dual ff = dual_exp(dual_mul_num(tf.t9i13, -32.120e0));
        const Dual gg = rates_cxexp_d(1.25e3, tf.t9i32, -27.499, tf.t9i);
        const Dual hh = rates_cxexp_d(1.43e-2, tf.t95, -15.541, tf.t9i);

        Dual zz = dual_div(dual_const(1.0e0), bb);
        const Dual f1 = dual_mul(cc, zz);

        zz = dual_div(dual_const(1.0e0), ee);
        const Dual f2 = dual_mul(ff, zz);

        Dual term = dual_add_ndm(dual_mul_num(f1, 1.04e8), 1.76e8, f2);
        term = dual_add(term, gg);
        term = dual_add(term, hh);

        term = dual_mul(term, dual_const(1.7e0));

        rates.fr = dual_mul_num(term, den);

        const Dual rev = rates_cxexp_d(5.13e10, tf.t932, -83.111, tf.t9i);

        rates.rr = dual_mul(rev, term);
    }
    return rates;
}

// from rates/aprox_rates.H:177 rate_triplealf
MPL_DEVICE rate_t<Dual> rate_triplealf_d(const tf_t<Dual>& tf, const Real den)
{
    rate_t<Dual> rates{};
    constexpr Real rc28 = 0.1;
    constexpr Real q1 = 1.0 / 0.009604e0;
    constexpr Real q2 = 1.0e0/0.055225e0;

    const Dual aa = rates_cxexp_d(7.40e5, tf.t9i32, -1.0663, tf.t9i);
    const Dual bb = rates_cexp2_d(tf, 4.164e9, -13.49, q1);
    const Dual cc = rates_poly5_d(tf, 0.031, 8.009, 1.732, 49.883, 27.426);
    const Dual r2abe = dual_add(aa, dual_mul(bb, cc));

    const Dual dd = rates_cxexp_d(130.0e0, tf.t9i32, -3.3364, tf.t9i);
    const Dual ee = rates_cexp2_d(tf, 2.510e7, -23.57, q2);
    const Dual ff = rates_poly5_d(tf, 0.018, 5.249, 0.650, 19.176, 6.034);
    const Dual rbeac = dual_add(dd, dual_mul(ee, ff));

    const Dual xx = rates_cxexp_d(rc28 * 1.35e-07, tf.t9i32, -24.811, tf.t9i);

    Dual term{};
    if (tf.t9.val > 0.08) {
        term = dual_add(dual_mul(dual_mul_num(r2abe, 2.90e-16), rbeac), xx);
    } else {
        const Dual uu = dual_mul(dual_const(0.8e0),
            dual_exp(dual_neg(dual_pow_num(dual_mul_num(tf.t9i, 0.025), 3.263))));
        const Dual yy = dual_add(dual_const(0.2e0), uu);
        const Dual vv = dual_mul(dual_const(4.0e0),
            dual_exp(dual_neg(dual_pow_num(dual_div_num(tf.t9, 0.025), 9.227))));
        const Dual zz = dual_add(dual_const(1.0e0), vv);
        const Dual zzinv = dual_div(dual_const(1.0e0), zz);
        const Dual f1 = dual_add(dual_const(0.01e0), dual_mul(yy, zzinv));
        term = dual_add(dual_mul(dual_mul(dual_mul_num(r2abe, 2.90e-16), rbeac), f1), xx);
    }

    rates.fr = dual_mul_num(dual_mul_num(term, den), den);

    const Dual rev = rates_cxexp_d(2.00e20, tf.t93, -84.424, tf.t9i);

    rates.rr = dual_mul(rev, term);
    return rates;
}

// from rates/aprox_rates.H:231 rate_c12c12
MPL_DEVICE rate_t<Dual> rate_c12c12_d(const tf_t<Dual>& tf, const Real den)
{
    rate_t<Dual> rates{};
    const Dual aa = dual_add_ndm(dual_const(1.0e0), 0.0396, tf.t9);
    const Dual zz = dual_div(dual_const(1.0e0), aa);
    const Dual t9a = dual_mul(tf.t9, zz);
    const Dual t9a13 = dual_cbrt(t9a);
    const Dual t9a56 = dual_pow_num(t9a, 5.0/6.0);
    const Dual arg = dual_add_ndm(dual_div(dual_const(-84.165), t9a13), -2.12e-03, tf.t93);
    const Dual term = dual_mul(dual_mul(dual_mul_num(t9a56, 4.27e26), tf.t9i32), dual_exp(arg));

    rates.fr = dual_mul_num(term, den);
    rates.rr = dual_const(0.0e0);
    return rates;
}

// from rates/aprox_rates.H:251 rate_c12o16
MPL_DEVICE rate_t<Dual> rate_c12o16_d(const tf_t<Dual>& tf, const Real den)
{
    rate_t<Dual> rates{};
    Dual term{};
    if (tf.t9.val >= 0.5) {
        Dual aa = dual_add_ndm(dual_const(1.0e0), 0.055, tf.t9);
        Dual zz = dual_div(dual_const(1.0e0), aa);
        const Dual t9a = dual_mul(tf.t9, zz);
        const Dual t9a13 = dual_cbrt(t9a);
        const Dual t9a23 = dual_mul(t9a13, t9a13);
        const Dual t9a56 = dual_pow_num(t9a, 5.0/6.0);

        aa = dual_exp(dual_mul(dual_mul_num(t9a, -0.18), t9a));
        const Dual bb = dual_mul(dual_const(1.06e-03), dual_exp(dual_mul_num(t9a23, 2.562)));
        const Dual cc = dual_add(aa, bb);
        zz = dual_div(dual_const(1.0e0), cc);
        term = dual_mul_num(t9a56, 1.72e31);
        term = dual_mul(term, tf.t9i32);
        term = dual_mul(term, dual_exp(dual_div(dual_const(-106.594), t9a13)));
        term = dual_mul(term, zz);
    } else {
        term = dual_const(0.0e0);
    }

    rates.fr = dual_mul_num(term, den);
    rates.rr = dual_const(0.0e0);
    return rates;
}

// from rates/aprox_rates.H:281 rate_o16o16
MPL_DEVICE rate_t<Dual> rate_o16o16_d(const tf_t<Dual>& tf, const Real den)
{
    rate_t<Dual> rates{};
    Dual s = dual_mul_num(tf.t9i13, -135.93);
    s = dual_add_ndm(s, -0.629, tf.t923);
    s = dual_add_ndm(s, -0.445, tf.t943);
    s = dual_add(s, dual_mul(dual_mul_num(tf.t9, 0.0103), tf.t9));
    const Dual term = dual_mul(dual_mul_num(tf.t9i23, 7.10e36), dual_exp(s));

    rates.fr = dual_mul_num(term, den);
    rates.rr = dual_const(0.0e0);
    return rates;
}

// from rates/aprox_rates.H:297 rate_o16ag
MPL_DEVICE rate_t<Dual> rate_o16ag_d(const tf_t<Dual>& tf, const Real den)
{
    constexpr Real q1 = 1.0e0/2.515396e0;

    const Dual term1 = rates_cexp2_d(tf, 9.37e9, -39.757, q1);

    const Dual aa = rates_cxexp_d(62.1, tf.t9i32, -10.297, tf.t9i);
    const Dual bb = rates_cxexp_d(538.0e0, tf.t9i32, -12.226, tf.t9i);
    const Dual cc = rates_cxexp_d(13.0e0, tf.t92, -20.093, tf.t9i);
    const Dual term2 = dual_add(dual_add(aa, bb), cc);

    const Dual term = dual_add(term1, term2);

    return rates_fin_pg_d(tf, den, term, 5.65e10, -54.937);
}

// from rates/aprox_rates.H:321 rate_ne20ag
MPL_DEVICE rate_t<Dual> rate_ne20ag_d(const tf_t<Dual>& tf, const Real den)
{
    constexpr Real rc102 = 0.1;
    constexpr Real q1 = 1.0e0/4.923961e0;

    Dual aa = rates_cexp2_d(tf, 4.11e11, -46.766, q1);
    Dual bb = rates_poly5_d(tf, 0.009, 0.882, 0.055, 0.749, 0.119);
    const Dual term1 = dual_mul(aa, bb);

    aa = rates_cxexp_d(5.27e3, tf.t9i32, -15.869, tf.t9i);
    bb = rates_cxexp_d(6.51e3, tf.t912, -16.223, tf.t9i);
    const Dual term2 = dual_add(aa, bb);

    aa = rates_cxexp_d(42.1, tf.t9i32, -9.115, tf.t9i);
    bb = rates_cxexp_d(32.0, tf.t9i23, -9.383, tf.t9i);
    const Dual term3 = dual_mul(dual_const(rc102), dual_add(aa, bb));

    aa = rates_cexp_d(5.0e0, -18.960, tf.t9i);
    bb = dual_add(dual_const(1.0e0), aa);
    const Dual zz = dual_div(dual_const(1.0e0), bb);
    const Dual term = dual_mul(dual_add(dual_add(term1, term2), term3), zz);

    return rates_fin_pg_d(tf, den, term, 6.01e10, -108.059);
}

// from rates/aprox_rates.H:355 rate_mg24ag
MPL_DEVICE rate_t<Dual> rate_mg24ag_d(const tf_t<Dual>& tf, const Real den)
{
    constexpr Real rc121 = 0.1e0;

    const Dual aa = rates_cxexp_d(4.78e1, tf.t9i32, -13.506, tf.t9i);
    const Dual bb = rates_cxexp_d(2.38e3, tf.t9i32, -15.218, tf.t9i);
    const Dual cc = rates_cxexp_d(2.47e2, tf.t932, -15.147, tf.t9i);
    const Dual dd = rates_cxexp_d(rc121 * 1.72e-09, tf.t9i32, -5.028, tf.t9i);
    const Dual ee = rates_cxexp_d(rc121* 1.25e-03, tf.t9i32, -7.929, tf.t9i);
    const Dual ff = rates_cxexp_d(rc121 * 2.43e1, tf.t9i, -11.523, tf.t9i);

    const Dual gg = rates_cexp_d(5.0e0, -15.882, tf.t9i);
    const Dual hh = dual_add(dual_const(1.0e0), gg);
    const Dual hhi = dual_div(dual_const(1.0e0), hh);
    Dual s = dual_add(aa, bb);
    s = dual_add(s, cc);
    s = dual_add(s, dd);
    s = dual_add(s, ee);
    s = dual_add(s, ff);
    const Dual term = dual_mul(s, hhi);

    return rates_fin_pg_d(tf, den, term, 6.27e10, -115.862);
}

// from rates/aprox_rates.H:382 rate_mg24ap
MPL_DEVICE rate_t<Dual> rate_mg24ap_d(const tf_t<Dual>& tf, const Real den)
{
    constexpr Real rc148 = 0.1;
    constexpr Real q1 = 1.0 / 0.024649e0;

    Dual aa = rates_cexp2_d(tf, 1.10e8, -23.261, q1);
    Dual bb = rates_poly5_d(tf, 0.018, 12.85, 1.61, 89.87, 28.66);
    const Dual term1 = dual_mul(aa, bb);

    aa = rates_cxexp_d(129.0e0, tf.t9i32, -2.517, tf.t9i);
    bb = rates_cxexp_d(5660.0e0, tf.t972, -3.421, tf.t9i);
    const Dual cc = rates_cxexp_d(rc148 * 3.89e-08, tf.t9i32, -0.853, tf.t9i);
    const Dual dd = rates_cxexp_d(rc148 * 8.18e-09, tf.t9i32, -1.001, tf.t9i);
    const Dual term2 = dual_add(dual_add(dual_add(aa, bb), cc), dd);

    const Dual ee = rates_cexp_d(1.0/3.0, -9.792, tf.t9i);
    const Dual ff = rates_cexp_d(2.0/3.0, -11.773, tf.t9i);
    const Dual gg = dual_add(dual_add(dual_const(1.0e0), ee), ff);

    const Dual term = dual_div(dual_add(term1, term2), gg);

    return rates_fin_ap_d(tf, den, term, 1.81, -18.572);
}

// from rates/aprox_rates.H:414 rate_al27pg
MPL_DEVICE rate_t<Dual> rate_al27pg_d(const tf_t<Dual>& tf, const Real den)
{
    const Dual aa = dual_mul(dual_mul_num(tf.t9i23, 1.32e9), dual_exp(dual_mul_num(tf.t9i13, -23.26)));
    const Dual bb = dual_mul_num(rates_cxexp_d(3.22e-10, tf.t9i32, -0.836, tf.t9i), 0.17);
    const Dual cc = rates_cxexp_d(1.74e0, tf.t9i32, -2.269, tf.t9i);
    const Dual dd = rates_cxexp_d(9.92e0, tf.t9i32, -2.492, tf.t9i);
    const Dual ee = rates_cxexp_d(4.29e1, tf.t9i32, -3.273, tf.t9i);
    const Dual ff = rates_cxexp_d(1.34e2, tf.t9i32, -3.654, tf.t9i);
    const Dual gg = dual_mul(dual_mul(dual_const(1.77e4), dual_pow_num(tf.t9, 0.53)),
                             dual_exp(dual_mul_num(tf.t9i, -4.588)));
    Dual term = dual_add(aa, bb);
    term = dual_add(term, cc);
    term = dual_add(term, dd);
    term = dual_add(term, ee);
    term = dual_add(term, ff);
    term = dual_add(term, gg);

    return rates_fin_pg_d(tf, den, term, 1.13e11, -134.434);
}

// from rates/aprox_rates.H:471 rate_si28ag
MPL_DEVICE rate_t<Dual> rate_si28ag_d(const tf_t<Dual>& tf, const Real den)
{
    const Dual term = rates_zterm_d(tf, 6.340e-2, 2.541e-3, -2.900e-4, 4.82e22, -61.015);
    return rates_fin_pg_d(tf, den, term, 6.461e10, -80.643);
}

// from rates/aprox_rates.H:491 rate_si28ap
MPL_DEVICE rate_t<Dual> rate_si28ap_d(const tf_t<Dual>& tf, const Real den)
{
    const Dual term = rates_zterm_d(tf, 2.798e-3, 2.763e-3, -2.341e-4, 4.16e13, -25.631);
    return rates_fin_ap_d(tf, den, term, 0.5825e0, -22.224);
}

// from rates/aprox_rates.H:510 rate_p31pg
MPL_DEVICE rate_t<Dual> rate_p31pg_d(const tf_t<Dual>& tf, const Real den)
{
    const Dual term = rates_zterm_d(tf, 1.928e-1, -1.540e-2, 6.444e-4, 1.08e16, -27.042);
    return rates_fin_pg_d(tf, den, term, 3.764e10, -102.865);
}

// from rates/aprox_rates.H:530 rate_s32ag
MPL_DEVICE rate_t<Dual> rate_s32ag_d(const tf_t<Dual>& tf, const Real den)
{
    const Dual term = rates_zterm_d(tf, 4.913e-2, 4.637e-3, -4.067e-4, 1.16e24, -66.690);
    return rates_fin_pg_d(tf, den, term, 6.616e10, -77.080);
}

// from rates/aprox_rates.H:550 rate_s32ap
MPL_DEVICE rate_t<Dual> rate_s32ap_d(const tf_t<Dual>& tf, const Real den)
{
    const Dual term = rates_zterm_d(tf, 1.041e-1, -1.368e-2, 6.969e-4, 1.27e16, -31.044);
    return rates_fin_ap_d(tf, den, term, 1.144, -21.643);
}

// from rates/aprox_rates.H:569 rate_cl35pg
MPL_DEVICE rate_t<Dual> rate_cl35pg_d(const tf_t<Dual>& tf, const Real den)
{
    const Dual aa = dual_add_ndm(dual_add_ndm(dual_add_ndm(dual_const(1.0e0), 1.761e-1, tf.t9),
                                              -1.322e-2, tf.t92), 5.245e-4, tf.t93);
    const Dual term = dual_mul(dual_mul_num(tf.t9i23, 4.48e16),
                               dual_exp(dual_mul(dual_mul_num(tf.t9i13, -29.483), aa)));
    return rates_fin_pg_d(tf, den, term, 7.568e10, -98.722);
}

// from rates/aprox_rates.H:585 rate_ar36ag
MPL_DEVICE rate_t<Dual> rate_ar36ag_d(const tf_t<Dual>& tf, const Real den)
{
    const Dual term = rates_zterm_d(tf, 1.458e-1, -1.069e-2, 3.790e-4, 2.81e30, -78.271);
    return rates_fin_pg_d(tf, den, term, 6.740e10, -81.711);
}

// from rates/aprox_rates.H:605 rate_ar36ap
MPL_DEVICE rate_t<Dual> rate_ar36ap_d(const tf_t<Dual>& tf, const Real den)
{
    const Dual term = rates_zterm_d(tf, 4.826e-3, -5.534e-3, 4.021e-4, 2.76e13, -34.922);
    return rates_fin_ap_d(tf, den, term, 1.128, -14.959);
}

// from rates/aprox_rates.H:625 rate_k39pg
MPL_DEVICE rate_t<Dual> rate_k39pg_d(const tf_t<Dual>& tf, const Real den)
{
    const Dual term = rates_zterm_d(tf, 1.622e-1, -1.119e-2, 3.910e-4, 4.09e16, -31.727);
    return rates_fin_pg_d(tf, den, term, 7.600e10, -96.657);
}

// from rates/aprox_rates.H:645 rate_ca40ag
MPL_DEVICE rate_t<Dual> rate_ca40ag_d(const tf_t<Dual>& tf, const Real den)
{
    const Dual term = rates_zterm_d(tf, 1.650e-2, 5.973e-3, -3.889e-04, 4.66e24, -76.435);
    return rates_fin_pg_d(tf, den, term, 6.843e10, -59.510);
}

// from rates/aprox_rates.H:665 rate_ca40ap
MPL_DEVICE rate_t<Dual> rate_ca40ap_d(const tf_t<Dual>& tf, const Real den)
{
    const Dual term = rates_zterm_d(tf, -1.206e-2, 7.753e-3, -5.071e-4, 4.54e14, -32.177);
    return rates_fin_ap_d(tf, den, term, 2.229, -40.966);
}

// from rates/aprox_rates.H:684 rate_sc43pg
MPL_DEVICE rate_t<Dual> rate_sc43pg_d(const tf_t<Dual>& tf, const Real den)
{
    const Dual term = rates_zterm_d(tf, 1.023e-1, -2.242e-3, -5.463e-5, 3.85e16, -33.234);
    return rates_fin_pg_d(tf, den, term, 1.525e11, -100.475);
}

// from rates/aprox_rates.H:704 rate_ti44ag
MPL_DEVICE rate_t<Dual> rate_ti44ag_d(const tf_t<Dual>& tf, const Real den)
{
    const Dual term = rates_zterm_d(tf, 1.066e-1, -1.102e-2, 5.324e-4, 1.37e26, -81.227);
    return rates_fin_pg_d(tf, den, term, 6.928e10, -89.289);
}

// from rates/aprox_rates.H:724 rate_ti44ap
MPL_DEVICE rate_t<Dual> rate_ti44ap_d(const tf_t<Dual>& tf, const Real den)
{
    const Dual term = rates_zterm_d(tf, 2.655e-2, -3.947e-3, 2.522e-4, 6.54e20, -66.678);
    return rates_fin_ap_d(tf, den, term, 1.104, -4.723);
}

// from rates/aprox_rates.H:743 rate_v47pg
MPL_DEVICE rate_t<Dual> rate_v47pg_d(const tf_t<Dual>& tf, const Real den)
{
    const Dual term = rates_zterm_d(tf, 9.979e-2, -2.269e-3, -6.662e-5, 2.05e17, -35.568);
    return rates_fin_pg_d(tf, den, term, 7.649e10, -93.999);
}

// from rates/aprox_rates.H:763 rate_cr48ag
MPL_DEVICE rate_t<Dual> rate_cr48ag_d(const tf_t<Dual>& tf, const Real den)
{
    const Dual term = rates_zterm_d(tf, 6.325e-2, -5.671e-3, 2.848e-4, 1.04e23, -81.420);
    return rates_fin_pg_d(tf, den, term, 7.001e10, -92.177);
}

// from rates/aprox_rates.H:783 rate_cr48ap
MPL_DEVICE rate_t<Dual> rate_cr48ap_d(const tf_t<Dual>& tf, const Real den)
{
    const Dual term = rates_zterm_d(tf, 1.384e-2, 1.081e-3, -5.933e-5, 1.83e26, -86.741);
    return rates_fin_ap_rev_d(tf, den, term, 0.6087, -6.510);
}

// from rates/aprox_rates.H:803 rate_mn51pg
MPL_DEVICE rate_t<Dual> rate_mn51pg_d(const tf_t<Dual>& tf, const Real den)
{
    const Dual term = rates_zterm_d(tf, 8.922e-2, -1.256e-3, -9.453e-5, 3.77e17, -37.516);
    return rates_fin_pg_d(tf, den, term, 1.150e11, -85.667);
}

// from rates/aprox_rates.H:823 rate_fe52ag
MPL_DEVICE rate_t<Dual> rate_fe52ag_d(const tf_t<Dual>& tf, const Real den)
{
    const Dual term = rates_zterm_d(tf, 7.846e-2, -7.430e-3, 3.723e-4, 1.05e27, -91.674);
    return rates_fin_pg_d(tf, den, term, 7.064e10, -92.850);
}

// from rates/aprox_rates.H:843 rate_fe52ap
MPL_DEVICE rate_t<Dual> rate_fe52ap_d(const tf_t<Dual>& tf, const Real den)
{
    const Dual term = rates_zterm_d(tf, 1.367e-2, 7.428e-4, -3.050e-5, 1.30e27, -91.674);
    return rates_fin_ap_rev_d(tf, den, term, 0.4597, -9.470);
}

// from rates/aprox_rates.H:863 rate_co55pg
MPL_DEVICE rate_t<Dual> rate_co55pg_d(const tf_t<Dual>& tf, const Real den)
{
    const Dual term = rates_zterm_d(tf, 9.894e-2, -3.131e-3, -2.160e-5, 1.21e18, -39.604);
    return rates_fin_pg_d(tf, den, term, 1.537e11, -83.382);
}

// from rates/aprox_rates.H:883 rate_pp
MPL_DEVICE rate_t<Dual> rate_pp_d(const tf_t<Dual>& tf, const Real den)
{
    rate_t<Dual> rates{};
    // p(p,e+nu)d
    Dual term{};
    if (tf.t9.val <= 3.0) {
       // 4.01e-15 * t9i23 * exp(-3.380*t9i13)
       const Dual aa = rates_cxexp_d(4.01e-15, tf.t9i23, -3.380e0, tf.t9i13);
       // 1.0 + 0.123*t913 + 1.09*t923 + 0.938*t9
       const Dual bb = dual_add_ndm(dual_add_ndm(dual_add_ndm(dual_const(1.0e0), 0.123e0, tf.t913),
                                                 1.09e0, tf.t923), 0.938e0, tf.t9);
       term = dual_mul(aa, bb);
    } else {
       term = dual_const(1.1581136e-15);
    }

    // rate
    rates.fr = dual_mul_num(term, den);
    rates.rr = dual_const(0.0e0);
    return rates;
}

// from rates/aprox_rates.H:902 rate_png
MPL_DEVICE rate_t<Dual> rate_png_d(const tf_t<Dual>& tf, const Real den)
{
    // aa = 1.0 - 0.8504*t912 + 0.4895*t9 - 0.09623*t932 + 8.471e-3*t92 - 2.80e-4*t952
    //   -> Add(...Add(1.0, NDM(-0.8504, t912))..., NDM(-2.80e-4, t952))
    Dual aa = dual_add_ndm(dual_const(1.0e0), -0.8504, tf.t912);
    aa = dual_add_ndm(aa, 0.4895, tf.t9);
    aa = dual_add_ndm(aa, -0.09623, tf.t932);
    aa = dual_add_ndm(aa, 8.471e-3, tf.t92);
    aa = dual_add_ndm(aa, -2.80e-4, tf.t952);
    // term = 4.742e4 * aa  (number * dual: NDM)
    const Dual term = dual_mul_num(aa, 4.742e4);

    return rates_fin_pg_d(tf, den, term, 4.71e9, -25.82);
}

// from rates/aprox_rates.H:926 rate_dpg
MPL_DEVICE rate_t<Dual> rate_dpg_d(const tf_t<Dual>& tf, const Real den)
{
    const Dual aa = rates_cxexp_d(2.24e3, tf.t9i23, -3.720, tf.t9i13);
    const Dual bb = dual_add_ndm(dual_add_ndm(dual_add_ndm(dual_const(1.0e0), 0.112, tf.t913),
                                              3.38, tf.t923), 2.65, tf.t9);
    const Dual term = dual_mul(aa, bb);
    return rates_fin_pg_d(tf, den, term, 1.63e10, -63.750);
}

// from rates/aprox_rates.H:943 rate_he3ng
MPL_DEVICE rate_t<Dual> rate_he3ng_d(const tf_t<Dual>& tf, const Real den)
{
    // 6.62 * (1.0 + 905.0*t9): number * expr -> Mul(6.62, Add(1.0, NDM(905.0, t9)))
    const Dual term = dual_mul(dual_const(6.62), dual_add_ndm(dual_const(1.0e0), 905.0, tf.t9));
    return rates_fin_pg_d(tf, den, term, 2.61e10, -238.81);
}

// from rates/aprox_rates.H:958 rate_he3he3
MPL_DEVICE rate_t<Dual> rate_he3he3_d(const tf_t<Dual>& tf, const Real den)
{
    rate_t<Dual> rates{};
    const Dual aa = rates_cxexp_d(6.04e10, tf.t9i23, -12.276, tf.t9i13);
    const Dual bb = rates_poly5_d(tf, 0.034, -0.522, -0.124, 0.353, 0.213);
    const Dual term = dual_mul(aa, bb);

    // rates
    rates.fr = dual_mul_num(term, den);

    const Dual rev = rates_cxexp_d(3.39e-10, tf.t9i32, -149.230, tf.t9i);

    // den * den * rev * term -> Mul(NDM(den*den, rev), term)
    rates.rr = dual_mul(dual_mul_num(rev, den * den), term);
    return rates;
}

// from rates/aprox_rates.H:976 rate_he3he4
MPL_DEVICE rate_t<Dual> rate_he3he4_d(const tf_t<Dual>& tf, const Real den)
{
    const Dual aa = dual_add_ndm(dual_const(1.0e0), 0.0495, tf.t9);

    Dual zz = dual_div(dual_const(1.0e0), aa);
    const Dual t9a = dual_mul(tf.t9, zz);
    const Dual t9a13 = dual_cbrt(t9a);
    const Dual t9a56 = dual_pow_num(t9a, 5.0/6.0);

    // 5.61e6 * t9a56 * t9i32 * exp(-12.826/t9a13)
    const Dual term = dual_mul(dual_mul(dual_mul_num(t9a56, 5.61e6), tf.t9i32),
                               dual_exp(dual_div(dual_const(-12.826), t9a13)));

    return rates_fin_pg_d(tf, den, term, 1.11e10, -18.423);
}

// from rates/aprox_rates.H:998 rate_c12pg
MPL_DEVICE rate_t<Dual> rate_c12pg_d(const tf_t<Dual>& tf, const Real den)
{
    constexpr Real q1 = 1.0e0/2.25e0;

    const Dual aa = rates_cexp2_d(tf, 2.04e7, -13.69, q1);
    const Dual bb = rates_poly5_d(tf, 0.03, 1.19, 0.254, 2.06, 1.12);
    const Dual cc = dual_mul(aa, bb);

    const Dual dd = rates_cxexp_d(1.08e5, tf.t9i32, -4.925, tf.t9i);

    const Dual ee = rates_cxexp_d(2.15e5, tf.t9i32, -18.179, tf.t9i);

    const Dual term = dual_add(dual_add(cc, dd), ee);

    return rates_fin_pg_d(tf, den, term, 8.84e9, -22.553);
}

// from rates/aprox_rates.H:1024 rate_n14pg
MPL_DEVICE rate_t<Dual> rate_n14pg_d(const tf_t<Dual>& tf, const Real den)
{
    constexpr Real q1 = 1.0e0/10.850436e0;

    const Dual aa = rates_cexp2_d(tf, 4.90e7, -15.228, q1);
    const Dual bb = rates_poly5_d(tf, 0.027, -0.778, -0.149, 0.261, 0.127);
    const Dual cc = dual_mul(aa, bb);

    const Dual dd = rates_cxexp_d(2.37e3, tf.t9i32, -3.011, tf.t9i);

    const Dual ee = rates_cexp_d(2.19e4, -12.530, tf.t9i);

    const Dual term = dual_add(dual_add(cc, dd), ee);

    return rates_fin_pg_d(tf, den, term, 2.70e10, -84.678);
}

// from rates/aprox_rates.H:1050 rate_n15pg
MPL_DEVICE rate_t<Dual> rate_n15pg_d(const tf_t<Dual>& tf, const Real den)
{
    constexpr Real q1 = 1.0 / 0.2025;

    const Dual aa = rates_cexp2_d(tf, 9.78e8, -15.251, q1);
    const Dual bb = rates_poly5_d(tf, 0.027, 0.219, 0.042, 6.83, 3.32);
    const Dual cc = dual_mul(aa, bb);

    const Dual dd = rates_cxexp_d(1.11e4, tf.t9i32, -3.328, tf.t9i);

    const Dual ee = rates_cxexp_d(1.49e4, tf.t9i32, -4.665, tf.t9i);

    const Dual ff = rates_cxexp_d(3.8e6, tf.t9i32, -11.048, tf.t9i);

    const Dual term = dual_add(dual_add(dual_add(cc, dd), ee), ff);

    return rates_fin_pg_d(tf, den, term, 3.62e10, -140.734);
}

// from rates/aprox_rates.H:1078 rate_n15pa
MPL_DEVICE rate_t<Dual> rate_n15pa_d(const tf_t<Dual>& tf, const Real den)
{
    constexpr Real theta = 0.1;
    constexpr Real q1 = 1.0 / 0.272484;

    const Dual aa = rates_cexp2_d(tf, 1.08e12, -15.251, q1);
    const Dual bb = rates_poly5_d(tf, 0.027, 2.62, 0.501, 5.36, 2.60);
    const Dual cc = dual_mul(aa, bb);

    const Dual dd = rates_cxexp_d(1.19e8, tf.t9i32, -3.676, tf.t9i);

    const Dual ee = rates_cxexp_d(5.41e8, tf.t9i12, -8.926, tf.t9i);

    // theta * 4.72e8 is a number sub-expression: NDM(theta * 4.72e8, t9i32)
    const Dual ff = rates_cxexp_d(theta * 4.72e8, tf.t9i32, -7.721, tf.t9i);

    const Dual gg = rates_cxexp_d(theta * 2.20e9, tf.t9i32, -11.418, tf.t9i);

    const Dual term = dual_add(dual_add(dual_add(dual_add(cc, dd), ee), ff), gg);

    // fr = den * term; rev = 7.06e-01*exp(-57.625*t9i); rr = den * rev * term
    return rates_fin_ap_rev_d(tf, den, term, 7.06e-01, -57.625);
}

// from rates/aprox_rates.H:1109 rate_o16pg
MPL_DEVICE rate_t<Dual> rate_o16pg_d(const tf_t<Dual>& tf, const Real den)
{
    const Dual aa = dual_exp(dual_mul_num(tf.t923, -0.728));
    // 1.0 + 2.13 * (1.0 - aa) -> Add(1.0, Mul(2.13, Add(1.0, Neg(aa))))
    const Dual bb = dual_add(dual_const(1.0e0),
                             dual_mul(dual_const(2.13), dual_sub(dual_const(1.0e0), aa)));
    const Dual cc = dual_mul(tf.t923, bb);
    const Dual dd = dual_exp(dual_mul_num(tf.t9i13, -16.692));
    const Dual zz = dual_div(dual_const(1.0e0), cc);
    const Dual ee = dual_mul(dd, zz);
    const Dual term = dual_mul_num(ee, 1.50e8);

    return rates_fin_pg_d(tf, den, term, 3.03e9, -6.968);
}

// from rates/aprox_rates.H:1130 rate_n14ag
MPL_DEVICE rate_t<Dual> rate_n14ag_d(const tf_t<Dual>& tf, const Real den)
{
    constexpr Real q1 = 1.0e0/0.776161e0;

    const Dual aa = rates_cexp2_d(tf, 7.78e9, -36.031, q1);
    const Dual bb = rates_poly5_d(tf, 0.012, 1.45, 0.117, 1.97, 0.406);
    const Dual cc = dual_mul(aa, bb);

    const Dual dd = rates_cxexp_d(2.36e-10, tf.t9i32, -2.798, tf.t9i);

    const Dual ee = rates_cxexp_d(2.03, tf.t9i32, -5.054, tf.t9i);

    const Dual ff = rates_cxexp_d(1.15e4, tf.t9i23, -12.310, tf.t9i);

    const Dual term = dual_add(dual_add(dual_add(cc, dd), ee), ff);

    return rates_fin_pg_d(tf, den, term, 5.42e10, -51.236);
}

// from rates/aprox_rates.H:1158 rate_fe52ng
MPL_DEVICE rate_t<Dual> rate_fe52ng_d(const tf_t<Dual>& tf, const Real den)
{
    // t9 - 0.348 -> Add(t9, -0.348)
    const Dual tq2 = dual_add_num(tf.t9, -0.348e0);
    const Dual term = rates_cexp_d(9.604e5, -0.0626, tq2);

    return rates_fin_pg_d(tf, den, term, 2.43e9, -123.951);
}

// from rates/aprox_rates.H:1174 rate_fe53ng
MPL_DEVICE rate_t<Dual> rate_fe53ng_d(const tf_t<Dual>& tf, const Real den)
{
    // t9/0.348 -> Mul(t9, Inv(0.348)) -> DIV
    const Dual tq1 = dual_div_num(tf.t9, 0.348);
    const Dual tq10 = dual_pow_num(tq1, 0.10);
    const Dual tq2 = dual_add_num(tf.t9, -0.348e0);

    const Dual term = rates_cxexp_d(1.817e6, tq10, -0.06319, tq2);

    return rates_fin_pg_d(tf, den, term, 1.56e11, -155.284);
}

// from rates/aprox_rates.H:1218 rate_fe54pg
MPL_DEVICE rate_t<Dual> rate_fe54pg_d(const tf_t<Dual>& tf, const Real den)
{
    const Dual term = rates_zterm_d(tf, 9.593e-2, -3.445e-3, 8.594e-5, 4.51e17, -38.483);
    return rates_fin_pg_d(tf, den, term, 2.400e9, -58.605);
}

// ---------------------------------------------------------------------------
// Weak rates (Real only; networks/aprox19 calls them with amrex::Real).
// ---------------------------------------------------------------------------

// langanke's tables. Stock: AMREX_GPU_MANAGED amrex::Array1D/Array3D globals
// (rates/aprox_rates_data.cpp), datn and rfd*/tfd* filled at run time by
// rates_init() (rates/aprox_rates.H:12). Here: namespace-scope constexpr
// arrays (probes d01, d16, d19), stored 0-based and read through 1-based
// accessors with the stock names, so langanke keeps the stock index
// expressions: rv(k) == rv_tab[k - 1], datn(n, i, j) == datn_tab[...].

// rates/aprox_rates_data.cpp:5 rv(1..6), :6 tv(1..14)
constexpr Real rv_tab[6] = {6.0, 7.0, 8.0, 9.0, 10.0, 11.0};
constexpr Real tv_tab[14] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0};

// rates/aprox_rates.H:15 datn0 and :29 datn1, verbatim and in that order;
// rates_init() sets datn(1,i,j) = datn0[(i-1)+6*(j-1)] and
// datn(2,i,j) = datn1[(i-1)+6*(j-1)] (i = 1..6, j = 1..14).
constexpr Real datn_tab[2 * 84] = {
    // datn0
    -4.363, -3.091, -1.275, 1.073, 3.035, 4.825,
    -4.17, -2.964, -1.177, 1.085, 3.037, 4.826,
    -3.834, -2.727, -1.039, 1.104, 3.04, 4.826,
    -3.284, -2.418, -0.882, 1.129, 3.043, 4.827,
    -2.691, -2.093, -0.719, 1.159, 3.048, 4.827,
    -2.1675, -1.7668, -0.5573, 1.1947, 3.0527, 4.8272,
    -1.7095, -1.4462, -0.3991, 1.2358, 3.0577, 4.8276,
    -1.3119, -1.1451, -0.2495, 1.2818, 3.0648, 4.8284,
    -0.9812, -0.8612, -0.1084, 1.3336, 3.0738, 4.8295,
    -0.682, -0.595, 0.028, 1.386, 3.084, 4.831,
    -0.4046, -0.3523, 0.1605, 1.4364, 3.0957, 4.8333,
    -0.1636, -0.1352, 0.2879, 1.4861, 3.1092, 4.8365,
    0.0461, 0.0595, 0.4105, 1.5354, 3.1242, 4.8405,
    0.2295, 0.235, 0.5289, 1.5842, 3.1405, 4.845,
    // datn1
    -4.539, -3.097, -1.134, 1.525, 3.907, 6.078,
    -4.199, -2.905, -1.024, 1.545, 3.91, 6.079,
    -3.736, -2.602, -0.851, 1.578, 3.916, 6.08,
    -3.052, -2.206, -0.636, 1.623, 3.923, 6.081,
    -2.31, -1.766, -0.396, 1.678, 3.931, 6.082,
    -1.6631, -1.319, -0.1438, 1.7471, 3.9409, 6.0829,
    -1.1064, -0.8828, 0.1094, 1.8279, 3.9534, 6.0841,
    -0.6344, -0.496, 0.3395, 1.9168, 3.9699, 6.0862,
    -0.2568, -0.1555, 0.5489, 2.0163, 3.9906, 6.0893,
    0.081, 0.158, 0.746, 2.114, 4.013, 6.093,
    0.3961, 0.4448, 0.9304, 2.2026, 4.0363, 6.0976,
    0.6673, 0.6964, 1.0985, 2.2849, 4.0614, 6.1033,
    0.9009, 0.9175, 1.2525, 2.3619, 4.0882, 6.1099,
    1.1032, 1.113, 1.3947, 2.4345, 4.1161, 6.1171};

// rates_init()'s cubic-interpolation factors, evaluated at compile time with
// its formulas (rates/aprox_rates.H:52-65, same operands, same order; every
// operand is an exactly representable integer, so each value is the one
// correctly rounded division, as at run time). Host-only constexpr helpers:
// only the resulting tables are reachable from device code.
namespace rates_init_ct {
constexpr Real rv(int k) { return rv_tab[k - 1]; }
constexpr Real tv(int j) { return tv_tab[j - 1]; }
constexpr Real rfdm(int k) { return 1.0 / ((rv(k-1)-rv(k))*(rv(k-1)-rv(k+1))*(rv(k-1)-rv(k+2))); }
constexpr Real rfd0(int k) { return 1.0 / ((rv(k)-rv(k-1))*(rv(k)-rv(k+1))*(rv(k)-rv(k+2))); }
constexpr Real rfd1(int k) { return 1.0 / ((rv(k+1)-rv(k-1))*(rv(k+1)-rv(k))*(rv(k+1)-rv(k+2))); }
constexpr Real rfd2(int k) { return 1.0 / ((rv(k+2)-rv(k-1))*(rv(k+2)-rv(k))*(rv(k+2)-rv(k+1))); }
constexpr Real tfdm(int j) { return 1.0 / ((tv(j-1)-tv(j))*(tv(j-1)-tv(j+1))*(tv(j-1)-tv(j+2))); }
constexpr Real tfd0(int j) { return 1.0 / ((tv(j)-tv(j-1))*(tv(j)-tv(j+1))*(tv(j)-tv(j+2))); }
constexpr Real tfd1(int j) { return 1.0 / ((tv(j+1)-tv(j-1))*(tv(j+1)-tv(j))*(tv(j+1)-tv(j+2))); }
constexpr Real tfd2(int j) { return 1.0 / ((tv(j+2)-tv(j-1))*(tv(j+2)-tv(j))*(tv(j+2)-tv(j+1))); }
} // namespace rates_init_ct

// rfd*(1..4): rates_init() sets k = 2..4; slot k = 1 is never written (the
// stock global stays 0.0) and never read.
constexpr Real rfdm_tab[4] = {0.0, rates_init_ct::rfdm(2), rates_init_ct::rfdm(3), rates_init_ct::rfdm(4)};
constexpr Real rfd0_tab[4] = {0.0, rates_init_ct::rfd0(2), rates_init_ct::rfd0(3), rates_init_ct::rfd0(4)};
constexpr Real rfd1_tab[4] = {0.0, rates_init_ct::rfd1(2), rates_init_ct::rfd1(3), rates_init_ct::rfd1(4)};
constexpr Real rfd2_tab[4] = {0.0, rates_init_ct::rfd2(2), rates_init_ct::rfd2(3), rates_init_ct::rfd2(4)};

// tfd*(1..12): rates_init() sets j = 2..12; slot j = 1 stays 0.0.
#define MPL_RATES_TFD_TAB(f) {0.0, rates_init_ct::f(2), rates_init_ct::f(3), rates_init_ct::f(4), \
    rates_init_ct::f(5), rates_init_ct::f(6), rates_init_ct::f(7), rates_init_ct::f(8),           \
    rates_init_ct::f(9), rates_init_ct::f(10), rates_init_ct::f(11), rates_init_ct::f(12)}
constexpr Real tfdm_tab[12] = MPL_RATES_TFD_TAB(tfdm);
constexpr Real tfd0_tab[12] = MPL_RATES_TFD_TAB(tfd0);
constexpr Real tfd1_tab[12] = MPL_RATES_TFD_TAB(tfd1);
constexpr Real tfd2_tab[12] = MPL_RATES_TFD_TAB(tfd2);
#undef MPL_RATES_TFD_TAB

// 1-based device accessors with the stock names (stock: amrex::Array1D/3D
// operator()).
MPL_DEVICE Real rv(int k) { return rv_tab[k - 1]; }
MPL_DEVICE Real tv(int j) { return tv_tab[j - 1]; }
MPL_DEVICE Real datn(int n, int i, int j) { return datn_tab[84*(n-1) + (i-1) + 6*(j-1)]; }
MPL_DEVICE Real rfdm(int k) { return rfdm_tab[k - 1]; }
MPL_DEVICE Real rfd0(int k) { return rfd0_tab[k - 1]; }
MPL_DEVICE Real rfd1(int k) { return rfd1_tab[k - 1]; }
MPL_DEVICE Real rfd2(int k) { return rfd2_tab[k - 1]; }
MPL_DEVICE Real tfdm(int j) { return tfdm_tab[j - 1]; }
MPL_DEVICE Real tfd0(int j) { return tfd0_tab[j - 1]; }
MPL_DEVICE Real tfd1(int j) { return tfd1_tab[j - 1]; }
MPL_DEVICE Real tfd2(int j) { return tfd2_tab[j - 1]; }

// from rates/aprox_rates.H:1324 langanke
// this routine evaluates Langanke et al. 2000 fits for the ni56 electron
// capture rate rn56ec and neutrino loss rate sn56ec

// input:
// y56 = nickel56 molar abundance
// ye  = electron to baryon number, zbar/abar

// output:
// rn56ec = ni56 electron capture rate
// sn56ec = ni56 neutrino loss rate
MPL_DEVICE void langanke(const Real btemp, const Real bden,
                         const Real y56, const Real ye,
                         Real& rn56ec, Real& sn56ec)
{
    // calculate ni56 electron capture and neutrino loss rates
    rn56ec = 0.0;
    sn56ec = 0.0;

    if ( (btemp < 1.0e9) || (bden*ye < 1.0e6)) return;

    const Real t9    = math::min(btemp, 1.4e10) * 1.0e-9;
    const Real r     = math::max(6.0e0, math::min(11.0e0, math::log10(bden*ye)));
    const int jp    = math::min(math::max(2, int(t9)), 12);
    const int kp    = math::min(math::max(2, int(r)-5), 4);
    MPL_ASSUME(2 <= jp && jp <= 12);
    MPL_ASSUME(2 <= kp && kp <= 4);
    const Real rfm   = r - rv(kp-1);
    const Real rf0   = r - rv(kp);
    const Real rf1   = r - rv(kp+1);
    const Real rf2   = r - rv(kp+2);
    const Real dfacm = rf0*rf1*rf2*rfdm(kp);
    const Real dfac0 = rfm*rf1*rf2*rfd0(kp);
    const Real dfac1 = rfm*rf0*rf2*rfd1(kp);
    const Real dfac2 = rfm*rf0*rf1*rfd2(kp);
    const Real tfm   = t9 - tv(jp-1);
    const Real tf0   = t9 - tv(jp);
    const Real tf1   = t9 - tv(jp+1);
    const Real tf2   = t9 - tv(jp+2);
    const Real tfacm = tf0*tf1*tf2*tfdm(jp);
    const Real tfac0 = tfm*tf1*tf2*tfd0(jp);
    const Real tfac1 = tfm*tf0*tf2*tfd1(jp);
    const Real tfac2 = tfm*tf0*tf1*tfd2(jp);

    // stock: uninitialized amrex::Array1D/Array2D locals (struct locals need {})
    Array1D<Real, 1, 2> rnt{};
    Array2D<Real, 1, 2, 1, 14> rne{};

    // evaluate the spline fits (stock bounds jr <= 2 and jd <= jp+2, written
    // as strict < so Loom sees them)
    for (int jr = 1; jr < 2 + 1; ++jr) {
        for (int jd = jp-1; jd < jp+2 + 1; ++jd) {
            MPL_ASSUME(1 <= jd && jd <= 14);
            rne(jr,jd) = dfacm*datn(jr,kp-1,jd) + dfac0*datn(jr,kp,jd)
                + dfac1*datn(jr,kp+1,jd) + dfac2*datn(jr,kp+2,jd);
        }
        rnt(jr) =  tfacm*rne(jr,jp-1) + tfac0*rne(jr,jp)
                 + tfac1*rne(jr,jp+1) + tfac2*rne(jr,jp+2);
    }

    // set the output
    rn56ec = math::pow(10.0e0, rnt(1));
    sn56ec = 6.022548e23 * 1.60218e-6 * y56 * math::pow(10.0e0, rnt(2));
}

// from rates/aprox_rates.H:1381 ecapnuc
// given the electron degeneracy parameter etakep (chemical potential
// without the electron's rest mass divided by kt) and the temperature
// temp, this routine calculates rates for
// electron capture on protons rpen (captures/sec/proton),
// positron capture on neutrons rnep (captures/sec/neutron),
// and their associated neutrino energy loss rates
// spenc (erg/sec/proton) and snepc (erg/sec/neutron)
MPL_DEVICE void ecapnuc(const Real etakep, const Real temp,
                        Real& rpen, Real& rnep,
                        Real& spenc, Real& snepc)
{
    constexpr Real qn1    = -2.0716446e-06;
    constexpr Real ftinv  = 1.0e0/1083.9269e0;
    constexpr Real twoln  = 0.6931472e0;
    constexpr Real cmk5   = 1.3635675e-49;
    constexpr Real cmk6   = 2.2993864e-59;
    constexpr Real bk     = 1.38062e-16;
    constexpr Real qn2    = 2.0716446e-06;
    constexpr Real c2me   = 8.1872665e-07;
    constexpr Real pi     = 3.1415927e0;
    constexpr Real pi2    = pi * pi;

    // c2me is the constant used to convert the neutrino energy
    // loss rate from mec2/s (as in the paper) to ergs/particle/sec.

    // initialize
    rpen   = 0.0e0;
    rnep   = 0.0e0;
    Real bktinv = 1.0e0/(bk *temp);
    Real qn     = qn1;

    // chemical potential including the electron rest mass
    Real etaef = etakep + c2me*bktinv;
    Real eta, t5, rie1, rie2, rjv1, rjv2;

    // iflag=1 is for electrons,  iflag=2 is for positrons
    for (int iflag = 1; iflag <= 2; ++iflag) {
        Real etael;
        if (iflag == 1) {
            etael = qn2*bktinv;
        } else {
            // iflag == 2
            etael = c2me*bktinv;
            etaef = -etaef;
        }

        t5    = temp*temp*temp*temp*temp;
        const Real zetan = qn*bktinv;
        eta   = etaef - etael;

        // protect from overflowing with large eta values
        const Real exeta = (eta <= 6.8e2) ? math::exp(eta) : 0.0;
        const Real etael2 = etael*etael;
        const Real etael3 = etael2*etael;
        const Real etael4 = etael3*etael;
        const Real etael5 = etael4*etael;
        const Real zetan2 = zetan*zetan;
        const Real f0 = (eta <= 6.8e2) ? math::log(1.0e0 + exeta) : eta;

        // if eta <= 0., the following fermi integrals apply
        const Real f1l = exeta;
        const Real f2l = 2.0e0   * f1l;
        const Real f3l = 6.0e0   * f1l;
        const Real f4l = 24.0e0  * f1l;
        const Real f5l = 120.0e0 * f1l;

        // if eta > 0., the following fermi integrals apply:
        Real f1g = 0.0e0;
        Real f2g = 0.0e0;
        Real f3g = 0.0e0;
        Real f4g = 0.0e0;
        Real f5g = 0.0e0;
        if (eta > 0.0) {
            const Real exmeta = math::exp(-eta);
            const Real eta2   = eta*eta;
            const Real eta3   = eta2*eta;
            const Real eta4   = eta3*eta;
            f1g = 0.5e0*eta2 + 2.0e0 - exmeta;
            f2g = eta3*1.0/3.0 + 4.0e0*eta + 2.0e0*exmeta;
            f3g = 0.25e0*eta4 + 0.5e0*pi2*eta2 + 12.0e0 - 6.0e0*exmeta;
            f4g = 0.2e0*eta4*eta + 2.0e0*pi2*1.0/3.0*eta3 + 48.0e0*eta
                    + 24.0e0*exmeta;
            f5g = eta4*eta2*1.0/6.0 + 5.0e0*1.0/6.0*pi2*eta4
                    + 7.0e0*1.0/6.0*pi2*eta2  + 240.0e0 -120.e0*exmeta;
        }

        // factors which are multiplied by the fermi integrals
        const Real fac3 = 2.0e0*zetan + 4.0e0*etael;
        const Real fac2 = 6.0e0*etael2 + 6.0e0*etael*zetan + zetan2;
        const Real fac1 = 4.0e0*etael3 + 6.0e0*etael2*zetan + 2.0e0*etael*zetan2;
        const Real fac0 = etael4 + 2.0e0*zetan*etael3 + etael2*zetan2;

        // electron capture rates onto protons with no blocking
        rie1 = f4l + fac3*f3l + fac2*f2l + fac1*f1l + fac0*f0;
        rie2 = f4g + fac3*f3g + fac2*f2g + fac1*f1g + fac0*f0;

        // neutrino emission rate for electron capture:
        const Real facv4 = 5.0e0*etael + 3.0e0*zetan;
        const Real facv3 = 10.0e0*etael2 + 12.0e0*etael*zetan + 3.0e0*zetan2;
        const Real facv2 = 10.0e0*etael3 + 18.0e0*etael2*zetan
            + 9.0e0*etael*zetan2 + zetan2*zetan;
        const Real facv1 = 5.0e0*etael4 + 12.0e0*etael3*zetan
            + 9.0e0*etael2*zetan2 + 2.0e0*etael*zetan2*zetan;
        const Real facv0 = etael5 + 3.0e0*etael4*zetan
            + 3.0e0*etael3*zetan2 + etael2*zetan2*zetan;
        rjv1  = f5l + facv4*f4l + facv3*f3l
            + facv2*f2l + facv1*f1l + facv0*f0;
        rjv2  = f5g + facv4*f4g + facv3*f3g
            + facv2*f2g + facv1*f1g + facv0*f0;

        // for electrons capture onto protons
        if (iflag < 2) {
            if (eta <= 0.0) {
                rpen  = twoln*cmk5*t5*rie1*ftinv;
                spenc = twoln*cmk6*t5*temp*rjv1*ftinv*c2me;
            } else {
                rpen = twoln*cmk5*t5*rie2*ftinv;
                spenc = twoln*cmk6*t5*temp*rjv2*ftinv*c2me;
            }
            qn = qn2;
        }
    }

    // for positrons capture onto neutrons
    if (eta <= 0.0) {
        rnep  = twoln*cmk5*t5*rie1*ftinv;
        snepc = twoln*cmk6*t5*temp*rjv1*ftinv*c2me;
    } else {
        rnep  = twoln*cmk5*t5*rie2*ftinv;
        snepc = twoln*cmk6*t5*temp*rjv2*ftinv*c2me;
    }
}

// from networks/aprox19/actual_network.H:1330 evaluate_analytical_rate (number_t = Real)
// rates.fr/rates.rr start at 0 (construct_rate) and are overwritten by the
// rate. Rates the stock dispatch has no branch for (the postprocessed ones)
// keep the zeros.
template <int rate>
MPL_DEVICE rate_t<Real> evaluate_analytical_rate_r(const tf_t<Real>& tf, const RatesParams& params)
{
    rate_t<Real> rates{0.0, 0.0};
    if constexpr (rate == Rates::H1_H1_to_He3) {
        // stock passes a discarded tmp1 for rr: rates.rr stays 0
        const rate_t<Real> r = rate_pp_r(tf, 1.0);
        rates.fr = r.fr;
    } else if constexpr (rate == Rates::P_to_N) {
        // Stock: if (state.tf.temp >= 1.0e6 && state.rho >= 1.0e-9) ecapnuc(state.eta,
        // temp, ...) -> fr, rr. It needs rho and eta, which this state-free
        // interface does not have; the real evaluation lives in network.H's
        // net_analytical_rate_{r,d}. Here (and in the validator's stock call
        // with a zero rhs_state_t, rho = 0) the gate is false: zeros.
    } else if constexpr (rate == Rates::P_N_to_H2) {
        rates = rate_png_r(tf, 1.0);
    } else if constexpr (rate == Rates::H2_P_to_He3) {
        rates = rate_dpg_r(tf, 1.0);
    } else if constexpr (rate == Rates::He3_N_to_He4) {
        rates = rate_he3ng_r(tf, 1.0);
    } else if constexpr (rate == Rates::He3_He3_to_He4_2H1) {
        // stock passes a discarded tmp1 for rr: rates.rr stays 0
        const rate_t<Real> r = rate_he3he3_r(tf, 1.0);
        rates.fr = r.fr;
    } else if constexpr (rate == Rates::He3_He4_H1_to_2He4) {
        // stock passes a discarded tmp1 for rr: rates.rr stays 0
        const rate_t<Real> r = rate_he3he4_r(tf, 1.0);
        rates.fr = r.fr;
    } else if constexpr (rate == Rates::C12_He4_to_O16) {
        rates = rate_c12ag_r(tf, 1.0, params);
    } else if constexpr (rate == Rates::He4_He4_He4_to_C12) {
        rates = rate_triplealf_r(tf, 1.0);
    } else if constexpr (rate == Rates::C12_H1_to_N13) {
        rates = rate_c12pg_r(tf, 1.0);
    } else if constexpr (rate == Rates::C12_C12_to_Ne20_He4) {
        rates = rate_c12c12_r(tf, 1.0);
    } else if constexpr (rate == Rates::C12_O16_to_Mg24_He4) {
        rates = rate_c12o16_r(tf, 1.0);
    } else if constexpr (rate == Rates::N14_H1_to_O15) {
        rates = rate_n14pg_r(tf, 1.0);
    } else if constexpr (rate == Rates::N15_H1_to_O16) {
        rates = rate_n15pg_r(tf, 1.0);
    } else if constexpr (rate == Rates::N15_H1_to_C12_He4) {
        rates = rate_n15pa_r(tf, 1.0);
    } else if constexpr (rate == Rates::N14_He4_to_F18) {
        rates = rate_n14ag_r(tf, 1.0);
    } else if constexpr (rate == Rates::O16_H1_to_F17) {
        rates = rate_o16pg_r(tf, 1.0);
    } else if constexpr (rate == Rates::O16_O16_to_P31_P) {
        rates = rate_o16o16_r(tf, 1.0);
    } else if constexpr (rate == Rates::O16_He4_to_Ne20) {
        rates = rate_o16ag_r(tf, 1.0);
    } else if constexpr (rate == Rates::Ne20_He4_to_Mg24) {
        rates = rate_ne20ag_r(tf, 1.0);
    } else if constexpr (rate == Rates::Mg24_He4_to_Si28) {
        rates = rate_mg24ag_r(tf, 1.0);
    } else if constexpr (rate == Rates::Si28_He4_to_S32) {
        rates = rate_si28ag_r(tf, 1.0);
    } else if constexpr (rate == Rates::S32_He4_to_Ar36) {
        rates = rate_s32ag_r(tf, 1.0);
    } else if constexpr (rate == Rates::Ar36_He4_to_Ca40) {
        rates = rate_ar36ag_r(tf, 1.0);
    } else if constexpr (rate == Rates::Ca40_He4_to_Ti44) {
        rates = rate_ca40ag_r(tf, 1.0);
    } else if constexpr (rate == Rates::Ti44_He4_to_Cr48) {
        rates = rate_ti44ag_r(tf, 1.0);
    } else if constexpr (rate == Rates::Cr48_He4_to_Fe52) {
        rates = rate_cr48ag_r(tf, 1.0);
    } else if constexpr (rate == Rates::Fe52_He4_to_Ni56) {
        rates = rate_fe52ag_r(tf, 1.0);
    } else if constexpr (rate == Rates::Mg24_He4_to_Al27_P) {
        rates = rate_mg24ap_r(tf, 1.0);
    } else if constexpr (rate == Rates::Al27_P_to_Si28) {
        rates = rate_al27pg_r(tf, 1.0);
    } else if constexpr (rate == Rates::Si28_He4_to_P31_P) {
        rates = rate_si28ap_r(tf, 1.0);
    } else if constexpr (rate == Rates::P31_P_to_S32) {
        rates = rate_p31pg_r(tf, 1.0);
    } else if constexpr (rate == Rates::S32_He4_to_Cl35_P) {
        rates = rate_s32ap_r(tf, 1.0);
    } else if constexpr (rate == Rates::Cl35_P_to_Ar36) {
        rates = rate_cl35pg_r(tf, 1.0);
    } else if constexpr (rate == Rates::Ar36_He4_to_K39_P) {
        rates = rate_ar36ap_r(tf, 1.0);
    } else if constexpr (rate == Rates::K39_P_to_Ca40) {
        rates = rate_k39pg_r(tf, 1.0);
    } else if constexpr (rate == Rates::Ca40_He4_to_Sc43_P) {
        rates = rate_ca40ap_r(tf, 1.0);
    } else if constexpr (rate == Rates::Sc43_P_to_Ti44) {
        rates = rate_sc43pg_r(tf, 1.0);
    } else if constexpr (rate == Rates::Ti44_He4_to_V47_P) {
        rates = rate_ti44ap_r(tf, 1.0);
    } else if constexpr (rate == Rates::V47_P_to_Cr48) {
        rates = rate_v47pg_r(tf, 1.0);
    } else if constexpr (rate == Rates::Cr48_He4_to_Mn51_P) {
        rates = rate_cr48ap_r(tf, 1.0);
    } else if constexpr (rate == Rates::Mn51_P_to_Fe52) {
        rates = rate_mn51pg_r(tf, 1.0);
    } else if constexpr (rate == Rates::Fe52_He4_to_Co55_P) {
        rates = rate_fe52ap_r(tf, 1.0);
    } else if constexpr (rate == Rates::Co55_P_to_Ni56) {
        rates = rate_co55pg_r(tf, 1.0);
    } else if constexpr (rate == Rates::Fe52_N_to_Fe53) {
        rates = rate_fe52ng_r(tf, 1.0);
    } else if constexpr (rate == Rates::Fe53_N_to_Fe54) {
        rates = rate_fe53ng_r(tf, 1.0);
    } else if constexpr (rate == Rates::Fe54_P_to_Co55) {
        rates = rate_fe54pg_r(tf, 1.0);
    } else if constexpr (rate == Rates::Ni56_to_Fe54) {
        // Stock: if (temp >= 1.0e6 && state.rho >= 1.0e-9) langanke(temp, rho,
        // y(Ni56), y_e, ...) -> fr, then fr *= 1/54. It needs rho, y and y_e,
        // which this state-free interface does not have; the real evaluation
        // lives in network.H's net_analytical_rate_{r,d}. Here (and in the
        // validator's stock call with a zero rhs_state_t, rho = 0) the gate
        // is false: zeros.
    }
    (void) tf;
    (void) params;
    return rates;
}

// from networks/aprox19/actual_network.H:1330 evaluate_analytical_rate (number_t = autodiff::dual)
// rates.fr/rates.rr start at 0 (construct_rate) and are overwritten by the
// rate. Rates the stock dispatch has no branch for (the postprocessed ones)
// keep the zeros.
template <int rate>
MPL_DEVICE rate_t<Dual> evaluate_analytical_rate_d(const tf_t<Dual>& tf, const RatesParams& params)
{
    rate_t<Dual> rates{dual_const(0.0), dual_const(0.0)};
    if constexpr (rate == Rates::H1_H1_to_He3) {
        // stock passes a discarded tmp1 for rr: rates.rr stays 0
        const rate_t<Dual> r = rate_pp_d(tf, 1.0);
        rates.fr = r.fr;
    } else if constexpr (rate == Rates::P_to_N) {
        // Stock: if (state.tf.temp >= 1.0e6 && state.rho >= 1.0e-9) ecapnuc(state.eta,
        // temp, ...) -> fr, rr. It needs rho and eta, which this state-free
        // interface does not have; the real evaluation lives in network.H's
        // net_analytical_rate_{r,d}. Here (and in the validator's stock call
        // with a zero rhs_state_t, rho = 0) the gate is false: zeros.
    } else if constexpr (rate == Rates::P_N_to_H2) {
        rates = rate_png_d(tf, 1.0);
    } else if constexpr (rate == Rates::H2_P_to_He3) {
        rates = rate_dpg_d(tf, 1.0);
    } else if constexpr (rate == Rates::He3_N_to_He4) {
        rates = rate_he3ng_d(tf, 1.0);
    } else if constexpr (rate == Rates::He3_He3_to_He4_2H1) {
        // stock passes a discarded tmp1 for rr: rates.rr stays 0
        const rate_t<Dual> r = rate_he3he3_d(tf, 1.0);
        rates.fr = r.fr;
    } else if constexpr (rate == Rates::He3_He4_H1_to_2He4) {
        // stock passes a discarded tmp1 for rr: rates.rr stays 0
        const rate_t<Dual> r = rate_he3he4_d(tf, 1.0);
        rates.fr = r.fr;
    } else if constexpr (rate == Rates::C12_He4_to_O16) {
        rates = rate_c12ag_d(tf, 1.0, params);
    } else if constexpr (rate == Rates::He4_He4_He4_to_C12) {
        rates = rate_triplealf_d(tf, 1.0);
    } else if constexpr (rate == Rates::C12_H1_to_N13) {
        rates = rate_c12pg_d(tf, 1.0);
    } else if constexpr (rate == Rates::C12_C12_to_Ne20_He4) {
        rates = rate_c12c12_d(tf, 1.0);
    } else if constexpr (rate == Rates::C12_O16_to_Mg24_He4) {
        rates = rate_c12o16_d(tf, 1.0);
    } else if constexpr (rate == Rates::N14_H1_to_O15) {
        rates = rate_n14pg_d(tf, 1.0);
    } else if constexpr (rate == Rates::N15_H1_to_O16) {
        rates = rate_n15pg_d(tf, 1.0);
    } else if constexpr (rate == Rates::N15_H1_to_C12_He4) {
        rates = rate_n15pa_d(tf, 1.0);
    } else if constexpr (rate == Rates::N14_He4_to_F18) {
        rates = rate_n14ag_d(tf, 1.0);
    } else if constexpr (rate == Rates::O16_H1_to_F17) {
        rates = rate_o16pg_d(tf, 1.0);
    } else if constexpr (rate == Rates::O16_O16_to_P31_P) {
        rates = rate_o16o16_d(tf, 1.0);
    } else if constexpr (rate == Rates::O16_He4_to_Ne20) {
        rates = rate_o16ag_d(tf, 1.0);
    } else if constexpr (rate == Rates::Ne20_He4_to_Mg24) {
        rates = rate_ne20ag_d(tf, 1.0);
    } else if constexpr (rate == Rates::Mg24_He4_to_Si28) {
        rates = rate_mg24ag_d(tf, 1.0);
    } else if constexpr (rate == Rates::Si28_He4_to_S32) {
        rates = rate_si28ag_d(tf, 1.0);
    } else if constexpr (rate == Rates::S32_He4_to_Ar36) {
        rates = rate_s32ag_d(tf, 1.0);
    } else if constexpr (rate == Rates::Ar36_He4_to_Ca40) {
        rates = rate_ar36ag_d(tf, 1.0);
    } else if constexpr (rate == Rates::Ca40_He4_to_Ti44) {
        rates = rate_ca40ag_d(tf, 1.0);
    } else if constexpr (rate == Rates::Ti44_He4_to_Cr48) {
        rates = rate_ti44ag_d(tf, 1.0);
    } else if constexpr (rate == Rates::Cr48_He4_to_Fe52) {
        rates = rate_cr48ag_d(tf, 1.0);
    } else if constexpr (rate == Rates::Fe52_He4_to_Ni56) {
        rates = rate_fe52ag_d(tf, 1.0);
    } else if constexpr (rate == Rates::Mg24_He4_to_Al27_P) {
        rates = rate_mg24ap_d(tf, 1.0);
    } else if constexpr (rate == Rates::Al27_P_to_Si28) {
        rates = rate_al27pg_d(tf, 1.0);
    } else if constexpr (rate == Rates::Si28_He4_to_P31_P) {
        rates = rate_si28ap_d(tf, 1.0);
    } else if constexpr (rate == Rates::P31_P_to_S32) {
        rates = rate_p31pg_d(tf, 1.0);
    } else if constexpr (rate == Rates::S32_He4_to_Cl35_P) {
        rates = rate_s32ap_d(tf, 1.0);
    } else if constexpr (rate == Rates::Cl35_P_to_Ar36) {
        rates = rate_cl35pg_d(tf, 1.0);
    } else if constexpr (rate == Rates::Ar36_He4_to_K39_P) {
        rates = rate_ar36ap_d(tf, 1.0);
    } else if constexpr (rate == Rates::K39_P_to_Ca40) {
        rates = rate_k39pg_d(tf, 1.0);
    } else if constexpr (rate == Rates::Ca40_He4_to_Sc43_P) {
        rates = rate_ca40ap_d(tf, 1.0);
    } else if constexpr (rate == Rates::Sc43_P_to_Ti44) {
        rates = rate_sc43pg_d(tf, 1.0);
    } else if constexpr (rate == Rates::Ti44_He4_to_V47_P) {
        rates = rate_ti44ap_d(tf, 1.0);
    } else if constexpr (rate == Rates::V47_P_to_Cr48) {
        rates = rate_v47pg_d(tf, 1.0);
    } else if constexpr (rate == Rates::Cr48_He4_to_Mn51_P) {
        rates = rate_cr48ap_d(tf, 1.0);
    } else if constexpr (rate == Rates::Mn51_P_to_Fe52) {
        rates = rate_mn51pg_d(tf, 1.0);
    } else if constexpr (rate == Rates::Fe52_He4_to_Co55_P) {
        rates = rate_fe52ap_d(tf, 1.0);
    } else if constexpr (rate == Rates::Co55_P_to_Ni56) {
        rates = rate_co55pg_d(tf, 1.0);
    } else if constexpr (rate == Rates::Fe52_N_to_Fe53) {
        rates = rate_fe52ng_d(tf, 1.0);
    } else if constexpr (rate == Rates::Fe53_N_to_Fe54) {
        rates = rate_fe53ng_d(tf, 1.0);
    } else if constexpr (rate == Rates::Fe54_P_to_Co55) {
        rates = rate_fe54pg_d(tf, 1.0);
    } else if constexpr (rate == Rates::Ni56_to_Fe54) {
        // Stock: if (temp >= 1.0e6 && state.rho >= 1.0e-9) langanke(temp, rho,
        // y(Ni56), y_e, ...) -> fr, then fr *= 1/54. It needs rho, y and y_e,
        // which this state-free interface does not have; the real evaluation
        // lives in network.H's net_analytical_rate_{r,d}. Here (and in the
        // validator's stock call with a zero rhs_state_t, rho = 0) the gate
        // is false: zeros.
    }
    (void) tf;
    (void) params;
    return rates;
}

} // namespace mpl

#endif
