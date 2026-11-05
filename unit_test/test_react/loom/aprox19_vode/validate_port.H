// Per-port hooks for validate_rates.H and validate_network.H (host only).
// Template: copy into the kernel directory next to them and fill in the
// lines marked "PORT:". Everything else in those two validators is
// network-neutral (it loops over Rates::NumRates, NumSpec and the screened
// pairs from network_data.H).
#ifndef VALIDATE_PORT_H
#define VALIDATE_PORT_H

#include <string>
#include <vector>

#include <extern_parameters.H>
#include <microphysics_autodiff.H>
#include <loom_validate.H>

#include <aprox_rates.H>   // PORT: stock rate_*, langanke, ecapnuc and their tables

#include "rates.H"

namespace loomval_port {

using mpl::Real;

// stock autodiff dual vs mpl::Dual, value and derivative bit for bit
inline void cmp_dual(LoomCheck& c, const char* what, const autodiff::dual& s, const mpl::Dual& l)
{
    const std::string w(what);
    loom_check_bits(c, (w + ".val").c_str(), autodiff::val(s), l.val);
    loom_check_bits(c, (w + ".grad").c_str(), autodiff::derivative(s), l.grad);
}

// Runtime parameters that select rate or network branches (network_rp::...).
// The validators run every check once per variant: apply_param_variant(v)
// for v = 0 .. num_param_variants - 1, then restore_params(). Screening's
// enable_debye_huckel_skip is toggled by the validators themselves.
// PORT: e.g. for network.use_c12ag_deboer17:
//   constexpr int num_param_variants = 2;
//   inline bool saved_deboer = false;
//   inline void save_params() { saved_deboer = network_rp::use_c12ag_deboer17; }
//   inline void apply_param_variant(int v) { network_rp::use_c12ag_deboer17 = (v == 1); }
//   inline void restore_params() { network_rp::use_c12ag_deboer17 = saved_deboer; }
// PORT (aprox19): network.use_c12ag_deboer17 off / on (rate_c12ag).
constexpr int num_param_variants = 2;
inline bool saved_deboer = false;
inline void save_params() { saved_deboer = network_rp::use_c12ag_deboer17; }
inline void apply_param_variant(int v) { network_rp::use_c12ag_deboer17 = (v == 1); }
inline void restore_params() { network_rp::use_c12ag_deboer17 = saved_deboer; }

// Temperatures where a rate switches branch (each is bracketed by +-64 ulps
// by validate_rates.H). PORT: one entry per branch threshold of the
// translated rates, e.g. rate_triplealf's t9 = 0.08: t.push_back(0.08e9);
// PORT (aprox19): triplealf t9 > 0.08, c12o16 t9 >= 0.5, pp t9 <= 3.0, the
// z-polynomial rates' min(10, t9) (si28ag ... co55pg, fe54pg), langanke's
// btemp < 1e9 cutoff, its min(btemp, 1.4e10) clamp and its int(t9) table
// index jp (integer t9 = 2 .. 13), and the 1e6 K gate of the weak rates.
inline void branch_temperatures(std::vector<Real>& t)
{
    t.push_back(0.08e9);
    t.push_back(0.5e9);
    t.push_back(3.0e9);
    t.push_back(10.0e9);
    t.push_back(1.0e9);
    t.push_back(1.4e10);
    for (int k = 2; k <= 13; ++k) { t.push_back(static_cast<Real>(k) * 1.0e9); }
    t.push_back(1.0e6);
}

// Direct calls of the translated rate functions at several densities, for
// rates whose stock function takes a density argument that
// evaluate_analytical_rate fixes to 1. Called once per temperature and
// parameter variant with the stock and Loom tfactors (Real and dual).
// PORT: for each rate_<name>(tf, den, fr, rr) the network uses, e.g.
//   for (Real den : {1.0, 1.0e-3, 3.7e4, 1.0e8, 2.5e11}) {
//       Real fr = 0.0, rr = 0.0;
//       rate_triplealf(st, den, fr, rr);
//       const mpl::rate_t<Real> l = mpl::rate_triplealf_r(lt, den);
//       loom_check_bits(c, "triplealf_r.fr", fr, l.fr);
//       loom_check_bits(c, "triplealf_r.rr", rr, l.rr);
//       ... the same for the dual versions with cmp_dual() ...
//   }
// PORT (aprox19): one langanke / ecapnuc comparison each, noinline so g++
// cannot constant-fold a libm call on one side only (footguns.md).
__attribute__((noinline)) inline void check_langanke(LoomCheck& c, Real T, Real bden, Real ye, Real y56)
{
    Real rs = 0.0, ss = 0.0;
    ::langanke(T, bden, y56, ye, rs, ss);
    Real rl = 0.0, sl = 0.0;
    mpl::langanke(T, bden, y56, ye, rl, sl);
    loom_check_bits(c, "langanke.rn56ec", rs, rl);
    loom_check_bits(c, "langanke.sn56ec", ss, sl);
}

__attribute__((noinline)) inline void check_ecapnuc(LoomCheck& c, Real T, Real eta)
{
    Real a = 0.0, b = 0.0, d = 0.0, e = 0.0;
    ::ecapnuc(eta, T, a, b, d, e);
    Real al = 0.0, bl = 0.0, dl = 0.0, el = 0.0;
    mpl::ecapnuc(eta, T, al, bl, dl, el);
    loom_check_bits(c, "ecapnuc.rpen", a, al);
    loom_check_bits(c, "ecapnuc.rnep", b, bl);
    loom_check_bits(c, "ecapnuc.spenc", d, dl);
    loom_check_bits(c, "ecapnuc.snepc", e, el);
}

template <typename StockTf, typename StockTfD>
inline void direct_rate_checks(LoomCheck& c, const StockTf& st, const StockTfD& std_,
                               const mpl::tf_t<Real>& lt, const mpl::tf_t<mpl::Dual>& ltd,
                               const mpl::RatesParams& rp)
{
    // PORT (aprox19): the 46 rate functions, Real and dual, at several densities.
#define LOOMVAL_DIRECT_RATE(name, ...)                                                  \
    {                                                                                   \
        Real fr = 0.0, rr = 0.0;                                                        \
        ::rate_##name(st, den, fr, rr);                                                 \
        const mpl::rate_t<Real> l = mpl::rate_##name##_r(lt, den __VA_ARGS__);          \
        loom_check_bits(c, #name "_r.fr", fr, l.fr);                                    \
        loom_check_bits(c, #name "_r.rr", rr, l.rr);                                    \
        autodiff::dual frd = 0.0, rrd = 0.0;                                            \
        ::rate_##name(std_, den, frd, rrd);                                             \
        const mpl::rate_t<mpl::Dual> ld = mpl::rate_##name##_d(ltd, den __VA_ARGS__);   \
        cmp_dual(c, #name "_d.fr", frd, ld.fr);                                         \
        cmp_dual(c, #name "_d.rr", rrd, ld.rr);                                         \
    }
    for (Real den : {1.0, 1.0e-3, 3.7e4, 1.0e8, 2.5e11}) {
        LOOMVAL_DIRECT_RATE(c12ag, , rp)
        LOOMVAL_DIRECT_RATE(triplealf)
        LOOMVAL_DIRECT_RATE(c12c12)
        LOOMVAL_DIRECT_RATE(c12o16)
        LOOMVAL_DIRECT_RATE(o16o16)
        LOOMVAL_DIRECT_RATE(o16ag)
        LOOMVAL_DIRECT_RATE(ne20ag)
        LOOMVAL_DIRECT_RATE(mg24ag)
        LOOMVAL_DIRECT_RATE(mg24ap)
        LOOMVAL_DIRECT_RATE(al27pg)
        LOOMVAL_DIRECT_RATE(si28ag)
        LOOMVAL_DIRECT_RATE(si28ap)
        LOOMVAL_DIRECT_RATE(p31pg)
        LOOMVAL_DIRECT_RATE(s32ag)
        LOOMVAL_DIRECT_RATE(s32ap)
        LOOMVAL_DIRECT_RATE(cl35pg)
        LOOMVAL_DIRECT_RATE(ar36ag)
        LOOMVAL_DIRECT_RATE(ar36ap)
        LOOMVAL_DIRECT_RATE(k39pg)
        LOOMVAL_DIRECT_RATE(ca40ag)
        LOOMVAL_DIRECT_RATE(ca40ap)
        LOOMVAL_DIRECT_RATE(sc43pg)
        LOOMVAL_DIRECT_RATE(ti44ag)
        LOOMVAL_DIRECT_RATE(ti44ap)
        LOOMVAL_DIRECT_RATE(v47pg)
        LOOMVAL_DIRECT_RATE(cr48ag)
        LOOMVAL_DIRECT_RATE(cr48ap)
        LOOMVAL_DIRECT_RATE(mn51pg)
        LOOMVAL_DIRECT_RATE(fe52ag)
        LOOMVAL_DIRECT_RATE(fe52ap)
        LOOMVAL_DIRECT_RATE(co55pg)
        LOOMVAL_DIRECT_RATE(pp)
        LOOMVAL_DIRECT_RATE(png)
        LOOMVAL_DIRECT_RATE(dpg)
        LOOMVAL_DIRECT_RATE(he3ng)
        LOOMVAL_DIRECT_RATE(he3he3)
        LOOMVAL_DIRECT_RATE(he3he4)
        LOOMVAL_DIRECT_RATE(c12pg)
        LOOMVAL_DIRECT_RATE(n14pg)
        LOOMVAL_DIRECT_RATE(n15pg)
        LOOMVAL_DIRECT_RATE(n15pa)
        LOOMVAL_DIRECT_RATE(o16pg)
        LOOMVAL_DIRECT_RATE(n14ag)
        LOOMVAL_DIRECT_RATE(fe52ng)
        LOOMVAL_DIRECT_RATE(fe53ng)
        LOOMVAL_DIRECT_RATE(fe54pg)
    }
#undef LOOMVAL_DIRECT_RATE

    const Real T = st.temp;

    // PORT (aprox19): langanke (Ni56_to_Fe54) at this temperature. bden*ye
    // spans the 1e6 cutoff (2e6 * 0.5 hits it exactly), every table interval
    // in log10(bden*ye) and the clamps at 6 and 11.
    for (Real bden : {1.0e5, 1.0e6, 2.0e6, 3.0e6, 1.0e7, 2.0e7, 5.0e7, 2.0e8, 1.0e9,
                      2.0e10, 1.0e11, 2.0e11, 1.0e12}) {
        for (Real ye : {0.5, 0.45, 0.3}) {
            for (Real y56 : {0.0, 1.0e-3, 1.0 / 56.0}) { check_langanke(c, T, bden, ye, y56); }
        }
    }

    // PORT (aprox19): ecapnuc (P_to_N) at this temperature.
    for (Real eta : {-50.0, -5.0, -1.0e-3, 0.0, 1.0e-3, 3.0, 50.0, 700.0, 1.0e3}) { check_ecapnuc(c, T, eta); }

    // PORT (aprox19): langanke's constexpr tables vs the stock globals that
    // rates_init() filled (test_react's init calls it), once, bit for bit.
    static bool tables_checked = false;
    if (!tables_checked) {
        tables_checked = true;
        for (int k = 1; k <= 6; ++k) { loom_check_bits(c, "table.rv", ::rv(k), mpl::rv(k)); }
        for (int j = 1; j <= 14; ++j) { loom_check_bits(c, "table.tv", ::tv(j), mpl::tv(j)); }
        for (int n = 1; n <= 2; ++n) {
            for (int i = 1; i <= 6; ++i) {
                for (int j = 1; j <= 14; ++j) {
                    loom_check_bits(c, "table.datn", ::datn(n, i, j), mpl::datn(n, i, j));
                }
            }
        }
        for (int k = 1; k <= 4; ++k) {
            loom_check_bits(c, "table.rfdm", ::rfdm(k), mpl::rfdm(k));
            loom_check_bits(c, "table.rfd0", ::rfd0(k), mpl::rfd0(k));
            loom_check_bits(c, "table.rfd1", ::rfd1(k), mpl::rfd1(k));
            loom_check_bits(c, "table.rfd2", ::rfd2(k), mpl::rfd2(k));
        }
        for (int j = 1; j <= 12; ++j) {
            loom_check_bits(c, "table.tfdm", ::tfdm(j), mpl::tfdm(j));
            loom_check_bits(c, "table.tfd0", ::tfd0(j), mpl::tfd0(j));
            loom_check_bits(c, "table.tfd1", ::tfd1(j), mpl::tfd1(j));
            loom_check_bits(c, "table.tfd2", ::tfd2(j), mpl::tfd2(j));
        }
    }
}

} // namespace loomval_port

#endif
