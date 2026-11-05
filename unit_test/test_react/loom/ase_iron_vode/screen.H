// Screening for the Loom burn kernel of a pynucastro network (see README.md):
// screening/screen.H (SCREEN_METHOD_chabrier1998) and screening/screen_data.H,
// util/approx_math/approx_math.H fast_atan, for both number types the burn uses:
//   *_r  number_t = amrex::Real      (actual_rhs.H evaluate_screening<0, rate_t>)
//   *_d  number_t = autodiff::dual   (evaluate_screening<1, rate_derivs_t>)
// The _r functions mirror the stock arithmetic exactly; the _d functions are
// the dual.H call sequences autodiff evaluates (dual.H header, section 3).
//
// fill_plasma_state and debye_huckel are the shipped screen5 module's
// translations (same stock functions). Screen factors: evaluate_screening
// folds scrn::calculate_screen_factor as a `constexpr auto`; the values come
// from network_data.H (net_screen_*(pair), generated from the stock constexpr
// evaluation and checked bit for bit against it). The gcem constants of
// chabrier1998 / fast_atan are folded at compile time in the stock build
// (double, also with an x87 long double GCEM_PI); their values are written
// below as exact hex literals.
#ifndef MPL_SCREEN_H
#define MPL_SCREEN_H

#include "support.H"
#include "dual.H"
#include "network_data.H"
#include "burn.H"

#if MPL_GEN_SCREENING && !defined(MPL_GEN_SCREEN_METHOD_chabrier1998)
#error "this screen.H translates SCREEN_METHOD=chabrier1998 only (network_data.H was generated for another method)"
#endif

namespace mpl {

// screening_rp::* read on the screening path (bool -> int)
struct ScreenParams {
    int screening_enable_debye_huckel_skip;
    Real screening_debye_huckel_skip_threshold;
    int screening_enable_chabrier1998_quantum_corr;
};

// from screening/screen.H:54 plasma_state_t
template <typename T>
struct plasma_state_t {
    T qlam0z;
    T taufac;
    T aa;
    T temp;
    T zbar;
    T z2bar;
    T n_e;
    T gamma_e_fac;
};


// from screening/screen_data.H:13 screen_factors_t (SCREEN_METHOD_chabrier1998
// fields; same order as the stock class)
struct screen_factors_t {
    Real z1;
    Real z2;
    Real a1;
    Real a2;
    Real zs53;
    Real z1_53;
    Real z2_53;
    Real zs13;
    Real zs13inv;
    Real aznut;
};

// scrn::calculate_screen_factor(Z1, A1, Z2, A2) as actual_rhs.H
// evaluate_screening folds it for Rates::ScreenPairs pair P (network_data.H).
template <int P>
MPL_DEVICE screen_factors_t screen_factor_pair()
{
    static_assert(1 <= P && P <= ScreenPairs::NumScreenPairs, "screen_factor_pair: unknown pair");

    constexpr Real z1 = net_screen_z1(P);
    constexpr Real z2 = net_screen_z2(P);
    constexpr Real a1 = net_screen_a1(P);
    constexpr Real a2 = net_screen_a2(P);
    constexpr Real zs53 = net_screen_zs53(P);
    constexpr Real z1_53 = net_screen_z1_53(P);
    constexpr Real z2_53 = net_screen_z2_53(P);
    constexpr Real zs13 = net_screen_zs13(P);
    constexpr Real zs13inv = net_screen_zs13inv(P);
    constexpr Real aznut = net_screen_aznut(P);

    screen_factors_t r{z1, z2, a1, a2, zs53, z1_53, z2_53, zs13, zs13inv, aznut};
    return r;
}

// from screening/screen.H:115 fill_plasma_state (number_t = temp_t = Real)
// y(1..NumSpec) = rhs_state.y
MPL_DEVICE void fill_plasma_state_r(plasma_state_t<Real>* state, Real temp,
                                    const Real dens, const NetYArray1D& y)
{
    Real sum = 0.0;
    for (int n = 1; n <= NumSpec; n++) {
        sum += y(n);
    }
    Real abar = 1.0 / sum;
    Real ytot = sum;

    Real m1 = 0.0;
    Real m2 = 0.0;
    for (int n = 1; n <= NumSpec; n++) {
        m1 += zion[n-1]*y(n);
        m2 += zion[n-1]*zion[n-1]*y(n);
    }

    Real temp_ad = temp;

    const Real co2 = (1.0/3.0) * 4.248719e3;

    Real zbar = m1 * abar;
    Real z2bar = m2 * abar;

    Real rr = dens * ytot;
    Real tempi = 1.0 / temp_ad;

    Real pp = math::sqrt(rr*tempi*(z2bar + zbar));

    state->qlam0z = 1.88e8 * tempi * pp;

    state->taufac = co2 * math::cbrt(tempi);

    Real xni = math::cbrt(rr * zbar);

    state->aa = 2.27493e5 * tempi * xni;

    state->temp = temp_ad;
    state->zbar = zbar;
    state->z2bar = z2bar;

    // static_cast<amrex::Real>(C::n_A)
    constexpr Real n_A = 6.02214076e+23;
    state->n_e = zbar * rr * n_A;

    // C::q_e * C::q_e * gcem::pow(4.0_rt * M_PI / 3.0_rt, 1.0_rt / 3.0_rt) / C::k_B
    constexpr Real gamma_e_constants = 0.0026936538176192486;
    state->gamma_e_fac = gamma_e_constants * math::cbrt(state->n_e);
}

// from screening/screen.H:115 fill_plasma_state (number_t = temp_t = autodiff::dual)
MPL_DEVICE void fill_plasma_state_d(plasma_state_t<Dual>* state, Dual temp,
                                    const Real dens, const NetYArray1D& y)
{
    Real sum = 0.0;
    for (int n = 1; n <= NumSpec; n++) {
        sum += y(n);
    }
    Real abar = 1.0 / sum;
    Real ytot = sum;

    Dual m1 = dual_const(0.0);
    Dual m2 = dual_const(0.0);
    for (int n = 1; n <= NumSpec; n++) {
        m1 = dual_add(m1, dual_const(zion[n-1]*y(n)));
        m2 = dual_add(m2, dual_const(zion[n-1]*zion[n-1]*y(n)));
    }

    Dual temp_ad = temp;

    const Real co2 = (1.0/3.0) * 4.248719e3;

    Dual zbar = dual_mul_num(m1, abar);
    Dual z2bar = dual_mul_num(m2, abar);

    Real rr = dens * ytot;
    Dual tempi = dual_div(dual_const(1.0), temp_ad);

    Dual pp = dual_sqrt(dual_mul(dual_mul_num(tempi, rr), dual_add(z2bar, zbar)));

    state->qlam0z = dual_mul(dual_mul_num(tempi, 1.88e8), pp);

    state->taufac = dual_mul(dual_const(co2), dual_cbrt(tempi));

    Dual xni = dual_cbrt(dual_mul_num(zbar, rr));

    state->aa = dual_mul(dual_mul_num(tempi, 2.27493e5), xni);

    state->temp = temp_ad;
    state->zbar = zbar;
    state->z2bar = z2bar;

    constexpr Real n_A = 6.02214076e+23;
    state->n_e = dual_mul_num(dual_mul_num(zbar, rr), n_A);

    constexpr Real gamma_e_constants = 0.0026936538176192486;
    state->gamma_e_fac = dual_mul(dual_const(gamma_e_constants), dual_cbrt(state->n_e));
}

// from screening/screen.H:187 debye_huckel (number_t = Real)
MPL_DEVICE Real debye_huckel_r(const plasma_state_t<Real>& state,
                               const screen_factors_t& scn_fac)
{
    Real z1z2 = scn_fac.z1 * scn_fac.z2;
    Real z_ratio = state.z2bar / state.zbar;

    Real Gamma_e = state.gamma_e_fac / state.temp;

    Real h_DH = z1z2 * math::sqrt(3 * math::powi<3>(Gamma_e) * z_ratio);

    return math::max(h_DH, 0.0);
}

// from screening/screen.H:187 debye_huckel (number_t = autodiff::dual)
MPL_DEVICE Dual debye_huckel_d(const plasma_state_t<Dual>& state,
                               const screen_factors_t& scn_fac)
{
    Real z1z2 = scn_fac.z1 * scn_fac.z2;
    Dual z_ratio = dual_div(state.z2bar, state.zbar);

    Dual Gamma_e = dual_div(state.gamma_e_fac, state.temp);

    // 3 * powi<3>(Gamma_e) is NDM(3, dual): the int 3 converts to 3.0
    Dual h_DH = dual_mul(dual_const(z1z2),
                         dual_sqrt(dual_mul(dual_mul_num(dual_powi<3>(Gamma_e), 3.0), z_ratio)));

    return dual_max(h_DH, dual_const(0.0));
}


// ---------------------------------------------------------------------------
// util/approx_math/approx_math.H
// ---------------------------------------------------------------------------

// from util/approx_math/approx_math.H:19 fast_atan_1
MPL_DEVICE Real fast_atan_1(const Real x)
{
    constexpr Real A = 0.0776509570923569;
    constexpr Real B = -0.287434475393028;
    constexpr Real C = 0x1.fd887410f1834p-1;   // GCEM_PI / 4.0_rt - A - B
    constexpr Real third = 1.0 / 3.0;

    Real result{};
    if (math::abs(x) < 0.378) {
        Real x2 = x*x;
        result = ((0.2*x2 - third)*x2 + 1.0)*x;
    } else {
        Real x2 = x*x;
        result = ((A*x2 + B)*x2 + C)*x;
    }
    return result;
}

// from util/approx_math/approx_math.H:67 fast_atan
MPL_DEVICE Real fast_atan(const Real x)
{
    constexpr Real PI_2 = 0x1.921fb54442d18p+0;   // 0.5_rt * GCEM_PI

    Real result{};
    if (x > 1.e8) {
        result = PI_2;
    } else if (x < -1.e8) {
        result = -PI_2;
    } else if (x > 1.0) {
        result = PI_2 - fast_atan_1(1.0 / x);
    } else if (x < -1.0) {
        result = -PI_2 - fast_atan_1(1.0 / x);
    } else {
        result = fast_atan_1(x);
    }
    return result;
}

// util/microphysics_autodiff.H:398 apply(Dual&, FastAtanOp):
//   aux = One<T>() / (1.0 + val * val); val = ::fast_atan(val); grad *= aux;
MPL_DEVICE Dual dual_fast_atan(Dual x)
{
    const Real aux = 1.0 / (1.0 + x.val * x.val);
    x.val = fast_atan(x.val);
    x.grad *= aux;
    return x;
}

// ---------------------------------------------------------------------------
// screening/screen.H SCREEN_METHOD_chabrier1998
// ---------------------------------------------------------------------------

// Fitted parameters, see Chabrier & Potekhin 1998 Sec.IV (stock constexpr locals)
constexpr Real chabrier_A_1 = -0.9052;
constexpr Real chabrier_A_2 = 0.6322;
constexpr Real chabrier_A_2i = 1.0 / chabrier_A_2;
constexpr Real chabrier_sqrt_A2 = 0x1.9718aa4841c5ap-1;     // gcem::sqrt(A_2)
constexpr Real chabrier_sqrt_A2i = 1.0 / chabrier_sqrt_A2;
constexpr Real chabrier_A_3 = 0x1.16f8c469183ep-2;          // -0.5 * gcem::sqrt(3.0) - A_1 * sqrt_A2i
constexpr Real chabrier_CBRT_2 = 0x1.428a2f98d728bp+0;      // gcem::pow(2.0, 1.0/3.0)

// from screening/screen.H:562 chabrier1998_helmholtz_F (number_t = Real)
MPL_DEVICE Real chabrier1998_helmholtz_F_r(const Real gamma)
{
    constexpr Real A_1 = chabrier_A_1;
    constexpr Real A_2 = chabrier_A_2;
    constexpr Real A_2i = chabrier_A_2i;
    constexpr Real sqrt_A2 = chabrier_sqrt_A2;
    constexpr Real sqrt_A2i = chabrier_sqrt_A2i;
    constexpr Real A_3 = chabrier_A_3;

    const Real sqrt_gamma = math::sqrt(gamma);
    const Real term3 = math::sqrt(1.0 + gamma * A_2i);
    const Real term1 = sqrt_gamma * sqrt_A2 * term3;
    const Real term2 = sqrt_gamma * sqrt_A2i;

    return A_1 * (term1 - A_2 * math::log(term2 + term3)) +
           2.0 * A_3 * (sqrt_gamma - fast_atan(sqrt_gamma));
}

// from screening/screen.H:562 chabrier1998_helmholtz_F (number_t = autodiff::dual)
//   term3: Sqrt(Add(1.0, Mul(gamma, A_2i)))
//   result: Add(Mul(A_1, Add(term1, Neg(Mul(A_2, Log(Add(term2, term3)))))),
//               Mul(2*A_3, Add(sqrt_gamma, Neg(FastAtan(sqrt_gamma)))))
MPL_DEVICE Dual chabrier1998_helmholtz_F_d(const Dual gamma)
{
    constexpr Real A_1 = chabrier_A_1;
    constexpr Real A_2 = chabrier_A_2;
    constexpr Real A_2i = chabrier_A_2i;
    constexpr Real sqrt_A2 = chabrier_sqrt_A2;
    constexpr Real sqrt_A2i = chabrier_sqrt_A2i;
    constexpr Real A_3 = chabrier_A_3;

    const Dual sqrt_gamma = dual_sqrt(gamma);
    const Dual term3 = dual_sqrt(dual_add(dual_const(1.0), dual_mul_num(gamma, A_2i)));
    const Dual term1 = dual_mul(dual_mul_num(sqrt_gamma, sqrt_A2), term3);
    const Dual term2 = dual_mul_num(sqrt_gamma, sqrt_A2i);

    const Dual t_log = dual_sub(term1, dual_mul(dual_const(A_2), dual_log(dual_add(term2, term3))));
    const Dual t_atan = dual_sub(sqrt_gamma, dual_fast_atan(sqrt_gamma));
    return dual_add(dual_mul(dual_const(A_1), t_log), dual_mul(dual_const(2.0 * A_3), t_atan));
}

// from screening/screen.H:586 chabrier1998 (number_t = Real)
MPL_DEVICE Real chabrier1998_r(const plasma_state_t<Real>& state,
                               const screen_factors_t& scn_fac,
                               const ScreenParams& params)
{
    Real Gamma_e = state.gamma_e_fac / state.temp;

    Real Gamma1 = Gamma_e * scn_fac.z1_53;
    Real Gamma2 = Gamma_e * scn_fac.z2_53;
    Real Gamma12 = Gamma_e * scn_fac.zs53;

    Real f1 = chabrier1998_helmholtz_F_r(Gamma1);
    Real f2 = chabrier1998_helmholtz_F_r(Gamma2);
    Real f12 = chabrier1998_helmholtz_F_r(Gamma12);

    Real quantum_corr_1 = 0.0;
    Real quantum_corr_2 = 0.0;

    if (params.screening_enable_chabrier1998_quantum_corr) {
        constexpr Real CBRT_2 = chabrier_CBRT_2;
        Real Gamma_eff = CBRT_2 * scn_fac.z1 * scn_fac.z2 *
                         scn_fac.zs13inv * Gamma_e;

        Real tau12 = state.taufac * scn_fac.aznut;

        Real b_fac = Gamma_eff / tau12;

        quantum_corr_1 = -tau12 * (5.0/32.0 * math::powi<3>(b_fac) -
                                   0.014 * math::powi<4>(b_fac) -
                                   0.128 * math::powi<5>(b_fac));

        quantum_corr_2 = -Gamma_eff * (0.0055 * math::powi<4>(b_fac) -
                                       0.0098 * math::powi<5>(b_fac) +
                                       0.0048 * math::powi<6>(b_fac));
    }

    Real h12 = f1 + f2 - f12 + quantum_corr_1 + quantum_corr_2;

    return math::max(h12, 0.0);
}

// from screening/screen.H:586 chabrier1998 (number_t = autodiff::dual)
MPL_DEVICE Dual chabrier1998_d(const plasma_state_t<Dual>& state,
                               const screen_factors_t& scn_fac,
                               const ScreenParams& params)
{
    Dual Gamma_e = dual_div(state.gamma_e_fac, state.temp);

    Dual Gamma1 = dual_mul_num(Gamma_e, scn_fac.z1_53);
    Dual Gamma2 = dual_mul_num(Gamma_e, scn_fac.z2_53);
    Dual Gamma12 = dual_mul_num(Gamma_e, scn_fac.zs53);

    Dual f1 = chabrier1998_helmholtz_F_d(Gamma1);
    Dual f2 = chabrier1998_helmholtz_F_d(Gamma2);
    Dual f12 = chabrier1998_helmholtz_F_d(Gamma12);

    Dual quantum_corr_1 = dual_const(0.0);
    Dual quantum_corr_2 = dual_const(0.0);

    if (params.screening_enable_chabrier1998_quantum_corr) {
        constexpr Real CBRT_2 = chabrier_CBRT_2;
        // CBRT_2 * z1 * z2 * zs13inv is a number product, then NDM(n, Gamma_e)
        Dual Gamma_eff = dual_mul_num(Gamma_e, CBRT_2 * scn_fac.z1 * scn_fac.z2 *
                                               scn_fac.zs13inv);

        Dual tau12 = dual_mul_num(state.taufac, scn_fac.aznut);

        Dual b_fac = dual_div(Gamma_eff, tau12);

        // -tau12 * (5/32 * P3 - 0.014 * P4 - 0.128 * P5):
        //   Mul(Neg(tau12), Add(Add(NDM(5/32, P3), NDM(-0.014, P4)), NDM(-0.128, P5)))
        const Dual s1 = dual_add_ndm(dual_add_ndm(dual_mul_num(dual_powi<3>(b_fac), 5.0/32.0),
                                                  -0.014, dual_powi<4>(b_fac)),
                                     -0.128, dual_powi<5>(b_fac));
        quantum_corr_1 = dual_mul(dual_neg(tau12), s1);

        // -Gamma_eff * (0.0055 * P4 - 0.0098 * P5 + 0.0048 * P6)
        const Dual s2 = dual_add_ndm(dual_add_ndm(dual_mul_num(dual_powi<4>(b_fac), 0.0055),
                                                  -0.0098, dual_powi<5>(b_fac)),
                                     0.0048, dual_powi<6>(b_fac));
        quantum_corr_2 = dual_mul(dual_neg(Gamma_eff), s2);
    }

    // f1 + f2 - f12 + qc1 + qc2: Add(Add(Add(Add(f1, f2), Neg(f12)), qc1), qc2)
    Dual h12 = dual_add(dual_add(dual_sub(dual_add(f1, f2), f12), quantum_corr_1), quantum_corr_2);

    return dual_max(h12, dual_const(0.0));
}

// from screening/screen.H:656 actual_log_screen (number_t = Real)
MPL_DEVICE Real actual_log_screen_r(const plasma_state_t<Real>& state,
                                    const screen_factors_t& scn_fac,
                                    const ScreenParams& params)
{
    Real log_scor = 0.0;
    bool done = false;
    if (params.screening_enable_debye_huckel_skip) {
        log_scor = debye_huckel_r(state, scn_fac);
        if (log_scor <= params.screening_debye_huckel_skip_threshold) {
            done = true;
        }
    }
    if (!done) {
        log_scor = chabrier1998_r(state, scn_fac, params);
    }
    return log_scor;
}

// from screening/screen.H:656 actual_log_screen (number_t = autodiff::dual)
MPL_DEVICE Dual actual_log_screen_d(const plasma_state_t<Dual>& state,
                                    const screen_factors_t& scn_fac,
                                    const ScreenParams& params)
{
    Dual log_scor = dual_const(0.0);
    bool done = false;
    if (params.screening_enable_debye_huckel_skip) {
        log_scor = debye_huckel_d(state, scn_fac);
        if (log_scor.val <= params.screening_debye_huckel_skip_threshold) {
            done = true;
        }
    }
    if (!done) {
        log_scor = chabrier1998_d(state, scn_fac, params);
    }
    return log_scor;
}

// from screening/screen.H:687 actual_log_screen(state, scn_fac, log_scor, log_scordt)
// (number_t = Real): log_scordt = quiet_NaN
MPL_DEVICE void actual_log_screen_r(const plasma_state_t<Real>& state,
                                    const screen_factors_t& scn_fac,
                                    const ScreenParams& params,
                                    Real& log_scor, Real& log_scordt)
{
    Real log_scor_dual = actual_log_screen_r(state, scn_fac, params);
    log_scor = log_scor_dual;
    log_scordt = __builtin_bit_cast(Real, 0x7ff8000000000000ULL);
}

// from screening/screen.H:687 actual_log_screen(state, scn_fac, log_scor, log_scordt)
// (number_t = autodiff::dual): value and d/dT
MPL_DEVICE void actual_log_screen_d(const plasma_state_t<Dual>& state,
                                    const screen_factors_t& scn_fac,
                                    const ScreenParams& params,
                                    Real& log_scor, Real& log_scordt)
{
    Dual log_scor_dual = actual_log_screen_d(state, scn_fac, params);
    log_scor = log_scor_dual.val;
    log_scordt = log_scor_dual.grad;
}

} // namespace mpl

#endif
