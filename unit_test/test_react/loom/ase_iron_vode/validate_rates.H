// Native bit-exact validation of the pieces of a pynucastro network's
// translation (pyn_rates.H, screen.H) against the stock network, one
// component at a time (host only; included by loom_validate.cpp under
// -DLOOM_VALIDATE_RATES). The whole evaluate_rates / rhs / jac comparison is
// validate_network.H; this file localizes a mismatch:
//   * evaluate_tfactors over T = 1e4..1e16 K, every partition-function and
//     weak-table knot +-4 ulps, and random points;
//   * fast_atan (Real and dual) near every branch point;
//   * math::exp10 (host: std::pow(10, x), as amrex::Math::exp10);
//   * interp_net::find_index / find_index_extrap on every 1-D table, at and
//     around every knot and outside the range;
//   * get_partition_function (fresh cache and cached index) and
//     get_partition_function_cached for every species 1..NumSpecTotal;
//   * tabular_evaluate<0/1> for every weak-rate table over a log(rho Ye) x
//     log(T) grid that covers every cell, every knot and both sides outside;
//   * screen factors of every pair (all fields), fill_plasma_state and
//     actual_log_screen (Real and dual) for every pair over rho x T x
//     composition, for each screening parameter variant (Debye-Hueckel skip
//     off / on with two thresholds, chabrier1998 quantum correction off / on).
// A sensitivity check at the end (T + 1 ulp on the Loom side must change some
// rate) is reported, not counted.
#ifndef VALIDATE_RATES_H
#define VALIDATE_RATES_H

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include <extern_parameters.H>
#include <actual_rhs.H>
#include <screen.H>
#include <approx_math.H>
#include <loom_validate.H>

#include "network.H"
#include <loom_params.H>

namespace loomval_rates {

using mpl::Real;

// ---------------------------------------------------------------------------
// screening parameter variants (stock globals and mpl::ScreenParams)
// ---------------------------------------------------------------------------

struct ScreenVariant { bool skip; Real threshold; bool qc; const char* name; };
inline const std::vector<ScreenVariant>& screen_variants()
{
    static const std::vector<ScreenVariant> v = {
        {false, 1.0e-2, false, "default"},
        {true, 1.0e-2, false, "dh_skip"},
        {true, 1.0, false, "dh_skip_thr1"},
        {false, 1.0e-2, true, "quantum_corr"},
        {true, 3.0e-1, true, "dh_skip_qc"},
    };
    return v;
}

struct ScreenSave { bool skip; Real thr; bool qc; };
inline ScreenSave save_screen()
{
    return {static_cast<bool>(screening_rp::enable_debye_huckel_skip), screening_rp::debye_huckel_skip_threshold,
            static_cast<bool>(screening_rp::enable_chabrier1998_quantum_corr)};
}
inline void restore_screen(const ScreenSave& s)
{
    screening_rp::enable_debye_huckel_skip = s.skip;
    screening_rp::debye_huckel_skip_threshold = s.thr;
    screening_rp::enable_chabrier1998_quantum_corr = s.qc;
}
inline void apply_screen(const ScreenVariant& v)
{
    screening_rp::enable_debye_huckel_skip = v.skip;
    screening_rp::debye_huckel_skip_threshold = v.threshold;
    screening_rp::enable_chabrier1998_quantum_corr = v.qc;
}

// ---------------------------------------------------------------------------
// per-sample checks (noinline: GCC constant-folds libm calls otherwise,
// references/footguns.md)
// ---------------------------------------------------------------------------

__attribute__((noinline)) void check_tf(LoomCheck& c, Real T)
{
    const tf_t s = evaluate_tfactors(T);
    const mpl::tf_t l = mpl::evaluate_tfactors(T);
    loom_check_bits(c, "tf.T9", s.T9, l.T9);
    loom_check_bits(c, "tf.T9i", s.T9i, l.T9i);
    loom_check_bits(c, "tf.T943i", s.T943i, l.T943i);
    loom_check_bits(c, "tf.T923i", s.T923i, l.T923i);
    loom_check_bits(c, "tf.T913i", s.T913i, l.T913i);
    loom_check_bits(c, "tf.T913", s.T913, l.T913);
    loom_check_bits(c, "tf.T923", s.T923, l.T923);
    loom_check_bits(c, "tf.T953", s.T953, l.T953);
    loom_check_bits(c, "tf.lnT9", s.lnT9, l.lnT9);
}

__attribute__((noinline)) void check_atan(LoomCheck& c, Real x)
{
    loom_check_bits(c, "fast_atan", ::fast_atan(x), mpl::fast_atan(x));
    autodiff::dual d = x;
    autodiff::seed(d);
    autodiff::dual sd = admath::fast_atan(d);
    mpl::Dual ld = mpl::dual_fast_atan(mpl::dual_seed(x));
    loom_check_bits(c, "dual_fast_atan.val", autodiff::val(sd), ld.val);
    loom_check_bits(c, "dual_fast_atan.grad", autodiff::derivative(sd), ld.grad);
    loom_check_bits(c, "exp10", amrex::Math::exp10(x), mpl::math::exp10(x));
}

// stock runtime-inuc cached lookup vs the kernel's compile-time-species one,
// species inuc .. 1 (fresh or indexed caches, two passes: miss then hit)
template <int inuc>
inline void check_pf_t(LoomCheck& c, Real T9, part_fun::pf_cache_t& sc, mpl::part_fun::pf_cache_t& lc)
{
    if constexpr (inuc >= 1) {
        check_pf_t<inuc - 1>(c, T9, sc, lc);
        Real s1{}, s2{}, l1{}, l2{};
        get_partition_function_cached(inuc, T9, sc, s1, s2);
        mpl::get_partition_function_cached<inuc>(T9, lc, l1, l2);
        loom_check_bits(c, "pf_t.logpf", s1, l1);
        loom_check_bits(c, "pf_t.dlogpf_dT9", s2, l2);
        Real g1{}, g2{};
        mpl::get_partition_function<inuc>(T9, lc, g1, g2);
        get_partition_function(inuc, T9, sc, s1, s2);
        loom_check_bits(c, "pf_t.uncached.logpf", s1, g1);
        loom_check_bits(c, "pf_t.uncached.dlogpf_dT9", s2, g2);
    }
}

__attribute__((noinline)) void check_pf(LoomCheck& c, Real T9)
{
    // fresh caches: index -1, so every interpolation searches
    part_fun::pf_cache_t sc;
    mpl::part_fun::pf_cache_t lc{};
    mpl::part_fun::pf_cache_init(lc);
    for (int inuc = 1; inuc <= NumSpecTotal; ++inuc) {
        Real s1{}, s2{}, l1{}, l2{};
        get_partition_function(inuc, T9, sc, s1, s2);
        mpl::get_partition_function(inuc, T9, lc, l1, l2);
        loom_check_bits(c, "pf.logpf", s1, l1);
        loom_check_bits(c, "pf.dlogpf_dT9", s2, l2);
    }
    // cached index (as derived_rates::fill_rates sets it), then the cache
    part_fun::pf_cache_t sc2;
    mpl::part_fun::pf_cache_t lc2{};
    mpl::part_fun::pf_cache_init(lc2);
    sc2.index_temp_array_1 = interp_net::find_index(T9, part_fun::temp_array_1);
    lc2.index_temp_array_1 = mpl::interp_net::find_index<mpl::TA_temp_array_1>(T9);
    loom_check_int(c, "pf.index", sc2.index_temp_array_1, lc2.index_temp_array_1);
    for (int pass = 0; pass < 2; ++pass) {
        for (int inuc = 1; inuc <= NumSpecTotal; ++inuc) {
            Real s1{}, s2{}, l1{}, l2{};
            get_partition_function_cached(inuc, T9, sc2, s1, s2);
            mpl::get_partition_function_cached(inuc, T9, lc2, l1, l2);
            loom_check_bits(c, "pf_cached.logpf", s1, l1);
            loom_check_bits(c, "pf_cached.dlogpf_dT9", s2, l2);
            loom_check_bits(c, "pf_cached.data1", sc2.data(inuc, 1), lc2.data(inuc, 1));
            loom_check_bits(c, "pf_cached.data2", sc2.data(inuc, 2), lc2.data(inuc, 2));
        }
    }
    for (int indexed = 0; indexed <= 1; ++indexed) {
        part_fun::pf_cache_t sc3;
        mpl::part_fun::pf_cache_t lc3{};
        mpl::part_fun::pf_cache_init(lc3);
        if (indexed) {
            sc3.index_temp_array_1 = interp_net::find_index(T9, part_fun::temp_array_1);
            lc3.index_temp_array_1 = mpl::interp_net::find_index<mpl::TA_temp_array_1>(T9);
        }
        check_pf_t<NumSpecTotal>(c, T9, sc3, lc3);
        check_pf_t<NumSpecTotal>(c, T9, sc3, lc3);
    }
}

template <int TAB, typename R, typename T, typename D>
__attribute__((noinline)) void check_weak(LoomCheck& c, const table_t& meta, const R& rhoy_t, const T& temp_t,
                                          const D& data, Real log_rhoy, Real log_temp)
{
    const Real rhoy = std::pow(10.0, log_rhoy);
    const Real temp = std::pow(10.0, log_temp);
    {
        weak_rate_t s{};
        mpl::weak_rate_t l{};
        tabular_evaluate<0>(meta, rhoy_t, temp_t, data, log_rhoy, rhoy, log_temp, temp, s);
        mpl::tabular_evaluate<0, TAB>(log_rhoy, rhoy, log_temp, temp, l);
        loom_check_bits(c, "weak0.rate", s.rate, l.rate);
        loom_check_bits(c, "weak0.enu", s.enu, l.enu);
        loom_check_bits(c, "weak0.gamma", s.gamma, l.gamma);
        loom_check_bits(c, "weak0.drate_dT", s.drate_dT, l.drate_dT);
        loom_check_bits(c, "weak0.drate_drhoye", s.drate_drhoye, l.drate_drhoye);
        loom_check_bits(c, "weak0.denu_dT", s.denu_dT, l.denu_dT);
        loom_check_bits(c, "weak0.denu_drhoye", s.denu_drhoye, l.denu_drhoye);
    }
    {
        weak_rate_t s{};
        mpl::weak_rate_t l{};
        tabular_evaluate<1>(meta, rhoy_t, temp_t, data, log_rhoy, rhoy, log_temp, temp, s);
        mpl::tabular_evaluate<1, TAB>(log_rhoy, rhoy, log_temp, temp, l);
        loom_check_bits(c, "weak1.rate", s.rate, l.rate);
        loom_check_bits(c, "weak1.enu", s.enu, l.enu);
        loom_check_bits(c, "weak1.gamma", s.gamma, l.gamma);
        loom_check_bits(c, "weak1.drate_dT", s.drate_dT, l.drate_dT);
        loom_check_bits(c, "weak1.drate_drhoye", s.drate_drhoye, l.drate_drhoye);
        loom_check_bits(c, "weak1.denu_dT", s.denu_dT, l.denu_dT);
        loom_check_bits(c, "weak1.denu_drhoye", s.denu_drhoye, l.denu_drhoye);
    }
}

template <int P>
__attribute__((noinline)) void check_screen_pair(LoomCheck& c, const plasma_state_t<Real>& sps,
                                                 const plasma_state_t<autodiff::dual>& spd,
                                                 const mpl::plasma_state_t<Real>& lps,
                                                 const mpl::plasma_state_t<mpl::Dual>& lpd,
                                                 const mpl::ScreenParams& sp)
{
    const mpl::screen_factors_t lf = mpl::screen_factor_pair<P>();
    const scrn::screen_factors_t sf = scrn::calculate_screen_factor(lf.z1, lf.a1, lf.z2, lf.a2);
    loom_check_bits(c, "scn.z1", sf.z1, lf.z1);
    loom_check_bits(c, "scn.z2", sf.z2, lf.z2);
    loom_check_bits(c, "scn.a1", sf.a1, lf.a1);
    loom_check_bits(c, "scn.a2", sf.a2, lf.a2);
    loom_check_bits(c, "scn.zs53", sf.zs53, lf.zs53);
    loom_check_bits(c, "scn.z1_53", sf.z1_53, lf.z1_53);
    loom_check_bits(c, "scn.z2_53", sf.z2_53, lf.z2_53);
    loom_check_bits(c, "scn.zs13", sf.zs13, lf.zs13);
    loom_check_bits(c, "scn.zs13inv", sf.zs13inv, lf.zs13inv);
    loom_check_bits(c, "scn.aznut", sf.aznut, lf.aznut);

    Real s1{}, s2{}, l1{}, l2{};
    actual_log_screen(sps, sf, s1, s2);
    mpl::actual_log_screen_r(lps, lf, sp, l1, l2);
    loom_check_bits(c, "log_screen_r", s1, l1);
    loom_check_bits(c, "log_screen_r.dT(NaN)", s2, l2);

    actual_log_screen(spd, sf, s1, s2);
    mpl::actual_log_screen_d(lpd, lf, sp, l1, l2);
    loom_check_bits(c, "log_screen_d", s1, l1);
    loom_check_bits(c, "log_screen_d.dT", s2, l2);

    loom_check_bits(c, "chabrier_r", chabrier1998(sps, sf), mpl::chabrier1998_r(lps, lf, sp));
    loom_check_bits(c, "debye_huckel_r", debye_huckel(sps, sf), mpl::debye_huckel_r(lps, lf));
}

#define LV_PS(f) loom_check_bits(c, "pstate_r." #f, sps.f, lps.f);
#define LV_PSD(f) loom_check_bits(c, "pstate_d." #f ".val", autodiff::val(spd.f), lpd.f.val); \
                  loom_check_bits(c, "pstate_d." #f ".grad", autodiff::derivative(spd.f), lpd.f.grad);
#define LV_PS_FIELDS(X) X(qlam0z) X(taufac) X(aa) X(temp) X(zbar) X(z2bar) X(n_e) X(gamma_e_fac)

__attribute__((noinline)) void check_screen(LoomCheck& c, Real rho, Real T, const std::vector<Real>& X,
                                            const mpl::ScreenParams& sp)
{
    amrex::Array1D<Real, 1, NumSpec> Y;
    mpl::Array1D<Real, 1, mpl::NumSpec> lY{};
    for (int i = 1; i <= NumSpec; ++i) {
        Y(i) = X[i-1] * aion_inv[i-1];
        lY(i) = Y(i);
    }
    plasma_state_t<Real> sps{};
    fill_plasma_state(sps, T, rho, Y);
    autodiff::dual td = T;
    autodiff::seed(td);
    plasma_state_t<autodiff::dual> spd{};
    fill_plasma_state(spd, td, rho, Y);

    mpl::plasma_state_t<Real> lps{};
    mpl::fill_plasma_state_r(&lps, T, rho, lY);
    mpl::plasma_state_t<mpl::Dual> lpd{};
    mpl::fill_plasma_state_d(&lpd, mpl::dual_seed(T), rho, lY);

    LV_PS_FIELDS(LV_PS)
    LV_PS_FIELDS(LV_PSD)

#define LV_PAIR(p) check_screen_pair<mpl::ScreenPairs::p>(c, sps, spd, lps, lpd, sp);
    MPL_PYN_SCREEN_PAIRS(LV_PAIR)
#undef LV_PAIR
}

// ---------------------------------------------------------------------------

inline std::vector<Real> around(Real x, int ulps)
{
    std::vector<Real> v{x};
    Real up = x, dn = x;
    for (int k = 0; k < ulps; ++k) {
        up = std::nextafter(up, 1.0e300);
        dn = std::nextafter(dn, -1.0e300);
        v.push_back(up);
        v.push_back(dn);
    }
    return v;
}

template <int ID>
inline void check_find_index(LoomCheck& c, const char* what)
{
    constexpr int lo = mpl::tab_lo<ID>();
    constexpr int hi = mpl::tab_hi<ID>();
    std::vector<Real> xs;
    for (int i = lo; i <= hi; ++i) {
        for (Real x : around(mpl::tab<ID>(i), 2)) { xs.push_back(x); }
        if (i < hi) { xs.push_back(0.5 * (mpl::tab<ID>(i) + mpl::tab<ID>(i + 1))); }
    }
    xs.push_back(mpl::tab<ID>(lo) - 1.0);
    xs.push_back(mpl::tab<ID>(hi) + 1.0);
    xs.push_back(-1.0e300);
    xs.push_back(1.0e300);
    xs.push_back(std::numeric_limits<Real>::quiet_NaN());
    std::string w = std::string("find_index.") + what;
    std::string we = std::string("find_index_extrap.") + what;
    for (Real x : xs) {
        // the stock search on the stock table (same values: compared below)
        int s = -2, se = -2;
        if constexpr (ID == mpl::TA_temp_array_1) {
            s = interp_net::find_index(x, part_fun::temp_array_1);
            se = interp_net::find_index_extrap(x, part_fun::temp_array_1);
        } else {
            // reference: the stock algorithm on an amrex::Array1D copy
            amrex::Array1D<Real, lo, hi> a;
            for (int i = lo; i <= hi; ++i) { a(i) = mpl::tab<ID>(i); }
            s = interp_net::find_index(x, a);
            se = interp_net::find_index_extrap(x, a);
        }
        loom_check_int(c, w.c_str(), s, mpl::interp_net::find_index<ID>(x));
        loom_check_int(c, we.c_str(), se, mpl::interp_net::find_index_extrap<ID>(x));
    }
}

template <int ID>
inline void check_all_find_index(LoomCheck& c)
{
    if constexpr (ID >= 1) {
        check_all_find_index<ID - 1>(c);
        check_find_index<ID>(c, std::to_string(ID).c_str());
    }
}

inline long validate_rates()
{
    LoomCheck c{"rates"};
    std::mt19937_64 rng(777);
    std::uniform_real_distribution<Real> u01(0.0, 1.0);

    // tables: translated constexpr arrays vs the stock device globals
#define LV_PFTAB(sp) \
    for (int i = 0; i < part_fun::npts_1; ++i) { \
        loom_check_bits(c, "table.pf." #sp, part_fun::sp##_pf_array(i), mpl::part_fun::sp##_pf_array[i]); \
    }
    MPL_PYN_PF_SPECIES(LV_PFTAB)
#undef LV_PFTAB
    for (int i = 0; i < part_fun::npts_1; ++i) {
        loom_check_bits(c, "table.temp_array_1", part_fun::temp_array_1(i), mpl::part_fun::temp_array_1[i]);
    }
#define LV_WTAB(t) \
    { \
        constexpr int TAB = mpl::TAB_##t; \
        const table_t& m = rate_tables::t##_meta; \
        for (int i = 1; i <= m.nrhoy; ++i) { \
            loom_check_bits(c, "table." #t ".rhoy", rate_tables::t##_rhoy(i), mpl::tab<mpl::wt_rhoy_id<TAB>()>(i)); \
        } \
        for (int i = 1; i <= m.ntemp; ++i) { \
            loom_check_bits(c, "table." #t ".temp", rate_tables::t##_temp(i), mpl::tab<mpl::wt_temp_id<TAB>()>(i)); \
        } \
        for (int k = 1; k <= num_vars; ++k) { \
            for (int j = 1; j <= m.nrhoy; ++j) { \
                for (int i = 1; i <= m.ntemp; ++i) { \
                    loom_check_bits(c, "table." #t ".data", rate_tables::t##_data(i, j, k), mpl::wt_data<TAB>(i, j, k)); \
                } \
            } \
        } \
        loom_check_int(c, "table." #t ".ntemp", m.ntemp, mpl::tab_hi<mpl::wt_temp_id<TAB>()>()); \
        loom_check_int(c, "table." #t ".nrhoy", m.nrhoy, mpl::tab_hi<mpl::wt_rhoy_id<TAB>()>()); \
    }
    MPL_PYN_WEAK_TABLES(LV_WTAB)
#undef LV_WTAB
    loom_check_int(c, "num_vars", num_vars, mpl::num_vars);

    // temperatures: log grid, knots of every table +-4 ulps, random
    std::vector<Real> temps;
    for (Real lt = 4.0; lt <= 16.0; lt += 0.01) { temps.push_back(std::pow(10.0, lt)); }
    for (int i = 0; i < part_fun::npts_1; ++i) {
        for (Real x : around(part_fun::temp_array_1(i) * 1.0e9, 4)) { temps.push_back(x); }
    }
    for (int k = 0; k < 20000; ++k) { temps.push_back(std::pow(10.0, 4.0 + 12.0 * u01(rng))); }

    for (Real T : temps) {
        check_tf(c, T);
        check_pf(c, T * 1.0e-9);
    }
    for (int i = 0; i < part_fun::npts_1; ++i) {
        for (Real x : around(part_fun::temp_array_1(i), 4)) { check_pf(c, x); }
    }
    amrex::Print() << "LOOM_VALIDATE rates tfactors/partition functions: temperatures=" << temps.size() << "\n";

    // fast_atan / exp10
    {
        std::vector<Real> xs = {0.0, -0.0, 1.0e-300, 1.0e300, -1.0e300};
        for (Real b : {0.378, 1.0, 1.0e8}) {
            for (Real x : around(b, 3)) { xs.push_back(x); xs.push_back(-x); }
        }
        for (Real lx = -12.0; lx <= 12.0; lx += 0.01) {
            xs.push_back(std::pow(10.0, lx));
            xs.push_back(-std::pow(10.0, lx));
        }
        for (Real x = -400.0; x <= 400.0; x += 0.37) { xs.push_back(x); }
        for (Real x : xs) { check_atan(c, x); }
    }

    // find_index on every 1-D table
    check_all_find_index<mpl::num_tab1d>(c);

    // weak-rate tables: every cell, knots, outside both ends
#define LV_WEAK(t) \
    { \
        constexpr int TAB = mpl::TAB_##t; \
        const auto& rt = rate_tables::t##_rhoy; \
        const auto& tt = rate_tables::t##_temp; \
        const table_t& m = rate_tables::t##_meta; \
        std::vector<Real> lr, lt; \
        for (int i = 1; i <= m.nrhoy; ++i) { for (Real x : around(rt(i), 2)) { lr.push_back(x); } \
            if (i < m.nrhoy) { for (Real f : {0.13, 0.5, 0.87}) { lr.push_back(rt(i) + f * (rt(i + 1) - rt(i))); } } } \
        for (int i = 1; i <= m.ntemp; ++i) { for (Real x : around(tt(i), 2)) { lt.push_back(x); } \
            if (i < m.ntemp) { for (Real f : {0.13, 0.5, 0.87}) { lt.push_back(tt(i) + f * (tt(i + 1) - tt(i))); } } } \
        lr.push_back(rt(1) - 0.7); lr.push_back(rt(m.nrhoy) + 0.7); \
        lt.push_back(tt(1) - 0.7); lt.push_back(tt(m.ntemp) + 0.7); \
        for (Real a : lr) { for (Real b : lt) { \
            check_weak<TAB>(c, m, rt, tt, rate_tables::t##_data, a, b); } } \
    }
    MPL_PYN_WEAK_TABLES(LV_WEAK)
#undef LV_WEAK

    // screening: every pair, every parameter variant
    {
        std::vector<std::vector<Real>> comps;
        for (int a = 0; a < NumSpec; ++a) {
            std::vector<Real> x(NumSpec, 1.0e-12);
            x[a] = 1.0;
            comps.push_back(x);
        }
        comps.push_back(std::vector<Real>(NumSpec, 1.0 / NumSpec));
        for (int k = 0; k < 4; ++k) {
            std::vector<Real> x(NumSpec);
            Real s = 0.0;
            for (auto& v : x) { v = std::pow(10.0, -12.0 * u01(rng)); s += v; }
            for (auto& v : x) { v /= s; }
            comps.push_back(x);
        }
        const ScreenSave s0 = save_screen();
        long n = 0;
        for (const auto& var : screen_variants()) {
            apply_screen(var);
            mpl::NetParams np{};
            fill_net_params(&np);
            const mpl::ScreenParams sp = mpl::net_screen_params(np);
            for (const auto& X : comps) {
                for (Real lr = -2.0; lr <= 12.0; lr += 0.5) {
                    for (Real lt = 4.0; lt <= 12.0; lt += 0.25) {
                        check_screen(c, std::pow(10.0, lr), std::pow(10.0, lt), X, sp);
                        ++n;
                    }
                }
            }
            for (int k = 0; k < 2000; ++k) {
                check_screen(c, std::pow(10.0, -2.0 + 14.0 * u01(rng)), std::pow(10.0, 4.0 + 8.0 * u01(rng)),
                             comps[k % comps.size()], sp);
                ++n;
            }
        }
        restore_screen(s0);
        amrex::Print() << "LOOM_VALIDATE rates screening: states=" << n << " (x" << mpl::ScreenPairs::NumScreenPairs
                       << " pairs)\n";
    }

    // sensitivity: T + 1 ulp on the Loom side must change something
    {
        LoomCheck cs{"rates_sensitivity"};
        cs.reported = 20;
        for (Real T : {1.0e8, 3.0e9, 6.0e9}) {
            const tf_t s = evaluate_tfactors(T);
            const mpl::tf_t l = mpl::evaluate_tfactors(std::nextafter(T, 1.0e300));
            loom_check_bits(cs, "sens.T913", s.T913, l.T913);
            loom_check_bits(cs, "sens.lnT9", s.lnT9, l.lnT9);
        }
        amrex::Print() << "LOOM_VALIDATE rates_sensitivity (T +1ulp, expected nonzero) checks=" << cs.checks
                       << " mismatches=" << cs.mismatches << "\n";
        if (cs.mismatches == 0) {
            amrex::Print() << "LOOM_MISMATCH rates_sensitivity: perturbation went undetected\n";
            c.mismatches += 1;
        }
    }
    return loom_report(c);
}

} // namespace loomval_rates

using loomval_rates::validate_rates;

#endif
