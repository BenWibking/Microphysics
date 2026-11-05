// Native bit-exact validation of a pynucastro network's translation
// (network.H + pyn_rates.H + pyn_network.H + screen.H) against the stock
// network (host only; included by loom_validate.cpp under
// -DLOOM_VALIDATE_NETWORK, and by validate_integrator.H for the burn_t layout
// static_asserts).
//
// For states built with the stock EOS (T 1e4..1e16, rho 1e2..1e12,
// compositions from NumSpec, random states) and every screening parameter
// variant (validate_rates.H screen_variants): every field of
// evaluate_rates<0, rate_t> vs evaluate_rates_r and of
// evaluate_rates<1, rate_derivs_t> vs evaluate_rates_d, actual_rhs and
// actual_jac on every element, and util/linpack.H dgefa / dgesl (pivoting on
// and off) vs net_dgefa / net_dgesl on I - gamma*J.
#ifndef VALIDATE_NETWORK_H
#define VALIDATE_NETWORK_H

#include <cmath>
#include <cstddef>
#include <cstring>
#include <random>
#include <vector>

#include <extern_parameters.H>
#include <eos.H>
#include <burn_type.H>
#include <actual_rhs.H>
#include <linpack.H>
#include <integrator_data.H>
#include <loom_validate.H>

#include "network.H"

// ---------------------------------------------------------------------------
// burn_t layout: stock vs mpl (validators memcpy between the two). These are
// the fields of the configurations gen_network_data.py supports (no aux
// species, no SDC); PORT: add an LV_OFF line for any field burn.H adds.
// ---------------------------------------------------------------------------
static_assert(sizeof(burn_t) == sizeof(mpl::burn_t));
#define LV_OFF(f) static_assert(offsetof(burn_t, f) == offsetof(mpl::burn_t, f), "burn_t." #f);
LV_OFF(time) LV_OFF(rho) LV_OFF(T) LV_OFF(e) LV_OFF(xn) LV_OFF(e_scale) LV_OFF(dedr)
LV_OFF(dedT) LV_OFF(dedA) LV_OFF(dedZ) LV_OFF(i) LV_OFF(j) LV_OFF(k) LV_OFF(y)
LV_OFF(rho_orig) LV_OFF(rhoe_orig) LV_OFF(ydot_a) LV_OFF(sdc_iter) LV_OFF(num_sdc_iters)
LV_OFF(T_fixed) LV_OFF(cv) LV_OFF(dx) LV_OFF(mu) LV_OFF(mu_e) LV_OFF(y_e) LV_OFF(eta)
LV_OFF(abar) LV_OFF(zbar) LV_OFF(n_rhs) LV_OFF(n_jac) LV_OFF(n_step) LV_OFF(success)
LV_OFF(error_code)
#undef LV_OFF
static_assert(NumSpec == mpl::NumSpec && neqs == mpl::neqs && net_ienuc == mpl::net_ienuc &&
              INT_NEQS == mpl::neqs);
static_assert(Rates::NumRates == mpl::Rates::NumRates);

#ifdef LOOM_VALIDATE_NETWORK
#include <loom_params.H>
#include <validate_rates.H>

namespace loomval_net {

using mpl::Real;

inline burn_t make_state(Real rho, Real T, const std::vector<Real>& X)
{
    burn_t s;
    s.rho = rho;
    s.T = T;
    for (int n = 0; n < NumSpec; ++n) { s.xn[n] = X[n]; }
    eos(eos_input_rt, s);
    return s;
}

template <typename S, typename L>
inline void cmp_rates_common(LoomCheck& c, const char* t, const S& s, const L& l)
{
    std::string w = std::string(t);
    for (int r = 1; r <= Rates::NumRates; ++r) {
        loom_check_bits(c, (w + ".screened_rates").c_str(), s.screened_rates(r), l.screened_rates(r));
    }
    for (int p = 1; p <= Rates::NumScreenPairs; ++p) {
        loom_check_bits(c, (w + ".log_screen").c_str(), s.log_screen(p), l.log_screen(p));
    }
    loom_check_bits(c, (w + ".enuc_weak").c_str(), s.enuc_weak, l.enuc_weak);
}

// noinline: see validate_rates.H (GCC constant-folding of libm calls)
__attribute__((noinline)) void one(LoomCheck& c, const burn_t& s, const mpl::NetParams* np)
{
    mpl::burn_t m;
    std::memcpy(&m, &s, sizeof m);
    const mpl::ScreenParams sp = mpl::net_screen_params(*np);

    amrex::Array1D<Real, 1, NumSpec> Y;
    mpl::Array1D<Real, 1, mpl::NumSpec> lY{};
    for (int i = 1; i <= NumSpec; ++i) {
        Y(i) = s.xn[i-1] * aion_inv[i-1];
        lY(i) = Y(i);
    }

    {
        rate_t re{};
        evaluate_rates<0, rate_t>(s, Y, re);
        mpl::rate_t lre{};
        mpl::evaluate_rates_r(m, lY, sp, lre);
        cmp_rates_common(c, "rates_r", re, lre);
    }
    {
        rate_derivs_t re{};
        evaluate_rates<1, rate_derivs_t>(s, Y, re);
        mpl::rate_derivs_t lre{};
        mpl::evaluate_rates_d(m, lY, sp, lre);
        cmp_rates_common(c, "rates_d", re, lre);
        for (int r = 1; r <= Rates::NumRates; ++r) {
            loom_check_bits(c, "rates_d.dscreened_rates_dT", re.dscreened_rates_dT(r), lre.dscreened_rates_dT(r));
        }
        for (int p = 1; p <= Rates::NumScreenPairs; ++p) {
            loom_check_bits(c, "rates_d.dlog_screen_dT", re.dlog_screen_dT(p), lre.dlog_screen_dT(p));
        }
        for (int n = 1; n <= NumSpec; ++n) {
            loom_check_bits(c, "rates_d.dweak_ydot_dYe", re.dweak_ydot_dYe(n), lre.dweak_ydot_dYe(n));
            loom_check_bits(c, "rates_d.denuc_weak_dY", re.denuc_weak_dY(n), lre.denuc_weak_dY(n));
        }
        loom_check_bits(c, "rates_d.denuc_weak_dT", re.denuc_weak_dT, lre.denuc_weak_dT);
        loom_check_bits(c, "rates_d.denuc_weak_dYe", re.denuc_weak_dYe, lre.denuc_weak_dYe);
        loom_check_bits(c, "rates_d.dYN.Fe52_n_n", re.drate_Fe52_n_n_to_Fe54_approx_dYN, lre.drate_Fe52_n_n_to_Fe54_approx_dYN);
        loom_check_bits(c, "rates_d.dYN.Fe54_Fe52", re.drate_Fe54_to_Fe52_n_n_approx_dYN, lre.drate_Fe54_to_Fe52_n_n_approx_dYN);
        loom_check_bits(c, "rates_d.dYN.Fe54_n_n", re.drate_Fe54_n_n_to_Fe56_approx_dYN, lre.drate_Fe54_n_n_to_Fe56_approx_dYN);
        loom_check_bits(c, "rates_d.dYN.Fe56_Fe54", re.drate_Fe56_to_Fe54_n_n_approx_dYN, lre.drate_Fe56_to_Fe54_n_n_approx_dYN);
        loom_check_bits(c, "rates_d.dYN.Ni56_n_n", re.drate_Ni56_n_n_to_Ni58_approx_dYN, lre.drate_Ni56_n_n_to_Ni58_approx_dYN);
        loom_check_bits(c, "rates_d.dYN.Ni58_Ni56", re.drate_Ni58_to_Ni56_n_n_approx_dYN, lre.drate_Ni58_to_Ni56_n_n_approx_dYN);
    }

    amrex::Array1D<Real, 1, neqs> ydot;
    for (int i = 1; i <= neqs; ++i) { ydot(i) = 0.0; }
    actual_rhs(s, ydot);
    mpl::Array1D<Real, 1, mpl::neqs> ly{};
    mpl::actual_rhs(m, *np, ly);
    for (int i = 1; i <= neqs; ++i) {
        loom_check_bits(c, "rhs", ydot(i), ly(i));
    }

    JacNetArray2D jac;
    jac.zero();
    actual_jac(s, jac);
    mpl::JacNetArray2D lj{};
    mpl::actual_jac(m, *np, lj);
    for (int i = 1; i <= neqs; ++i) {
        for (int j = 1; j <= neqs; ++j) {
            loom_check_bits(c, "jac", jac(i, j), lj(i, j));
        }
    }

    // LU on I - gamma*J (the shape VODE factors), then a solve; pivoting on and off
    for (Real gamma : {1.0e-12, 1.0e-6, 1.0e-2, 1.0}) {
        for (int piv = 0; piv <= 1; ++piv) {
            RArray2D a;
            IArray1D pivot;
            mpl::JacNetArray2D la{};
            mpl::NetPivotArray lpivot{};
            for (int i = 1; i <= neqs; ++i) {
                pivot(i) = 0;
                for (int j = 1; j <= neqs; ++j) {
                    a(i, j) = (i == j ? 1.0 : 0.0) - gamma * jac(i, j);
                    la(i, j) = a(i, j);
                }
            }
            int info_s = -5, info_l = -5;
            if (piv) {
                dgefa<INT_NEQS, true>(a, pivot, info_s);
                mpl::net_dgefa<true>(la, lpivot, info_l);
            } else {
                dgefa<INT_NEQS, false>(a, pivot, info_s);
                mpl::net_dgefa<false>(la, lpivot, info_l);
            }
            loom_check_int(c, "dgefa.info", info_s, info_l);
            for (int i = 1; i <= neqs; ++i) {
                loom_check_int(c, "dgefa.pivot", pivot(i), lpivot(i));
                for (int j = 1; j <= neqs; ++j) {
                    loom_check_bits(c, "dgefa", a(i, j), la(i, j));
                }
            }
            RArray1D b;
            mpl::Array1D<Real, 1, mpl::neqs> lb{};
            for (int i = 1; i <= neqs; ++i) { b(i) = ydot(i); lb(i) = ydot(i); }
            if (piv) {
                dgesl<INT_NEQS, true>(a, pivot, b);
                mpl::net_dgesl<true>(la, lpivot, lb);
            } else {
                dgesl<INT_NEQS, false>(a, pivot, b);
                mpl::net_dgesl<false>(la, lpivot, lb);
            }
            for (int i = 1; i <= neqs; ++i) { loom_check_bits(c, "dgesl", b(i), lb(i)); }
        }
    }
}

// one-hot per species, a subset of pairs (every pair for small networks),
// uniform, near-pure, random (12 decades)
inline std::vector<std::vector<Real>> compositions(std::mt19937_64& rng, int n_random)
{
    std::vector<std::vector<Real>> comps;
    for (int a = 0; a < NumSpec; ++a) {
        std::vector<Real> x(NumSpec, 0.0);
        x[a] = 1.0;
        comps.push_back(x);
        for (int b = a + 1; b < NumSpec; ++b) {
            if (NumSpec > 12 && (a + 3 * b) % 9 != 0) { continue; }
            std::vector<Real> y(NumSpec, 0.0);
            y[a] = 0.7;
            y[b] = 0.3;
            comps.push_back(y);
        }
    }
    comps.push_back(std::vector<Real>(NumSpec, 1.0 / NumSpec));
    {
        std::vector<Real> x(NumSpec, 1.0e-10);
        x[0] = 1.0 - (NumSpec - 1) * 1.0e-10;
        comps.push_back(x);
    }
    std::uniform_real_distribution<Real> u01(0.0, 1.0);
    for (int k = 0; k < n_random; ++k) {
        std::vector<Real> x(NumSpec);
        Real s = 0.0;
        for (auto& v : x) { v = std::pow(10.0, -12.0 * u01(rng)); s += v; }
        for (auto& v : x) { v /= s; }
        comps.push_back(x);
    }
    return comps;
}

} // namespace loomval_net

inline long validate_network()
{
    using namespace loomval_net;
    LoomCheck c;
    c.module = "network";

    std::mt19937_64 rng(4242);
    std::uniform_real_distribution<Real> u01(0.0, 1.0);
    const auto comps = compositions(rng, 10);
    const Real dlr = comps.size() > 40 ? 1.0 : 0.5;
    const Real dlt = comps.size() > 40 ? 0.4 : 0.2;

    const auto s0 = loomval_rates::save_screen();
    long nstates = 0;
    for (const auto& var : loomval_rates::screen_variants()) {
        loomval_rates::apply_screen(var);
        mpl::NetParams np{};
        fill_net_params(&np);
        for (const auto& X : comps) {
            for (Real lr = 2.0; lr <= 12.0; lr += dlr) {
                for (Real lt = 4.0; lt <= 16.0; lt += dlt) {
                    one(c, make_state(std::pow(10.0, lr), std::pow(10.0, lt), X), &np);
                    ++nstates;
                }
            }
        }
        for (int k = 0; k < 3000; ++k) {
            const auto& X = comps[k % comps.size()];
            one(c, make_state(std::pow(10.0, 2.0 + 10.0 * u01(rng)),
                              std::pow(10.0, 4.0 + 12.0 * u01(rng)), X), &np);
            ++nstates;
        }
        amrex::Print() << "LOOM_VALIDATE network variant=" << var.name << " done\n";
    }
    loomval_rates::restore_screen(s0);
    amrex::Print() << "LOOM_VALIDATE network states=" << nstates << " compositions=" << comps.size() << "\n";
    return loom_report(c);
}
#endif // LOOM_VALIDATE_NETWORK

#endif
