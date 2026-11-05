// aprox19 network glue for the Loom burn kernel (see README.md): the
// hand-translated, network-specific functions of
// networks/aprox19/actual_network.H that the generic networks/rhs.H
// translation (network_rhs.H) calls, for this configuration (aprox19,
// SCREENING (screen5), STRANG, no NEUTRINOS, no NSE, HIP branches):
//   NetParams / net_rates_params    runtime parameters of the rhs/jac path
//   net_analytical_rate_r/_d<rate>  evaluate_analytical_rate (rates.H, plus
//                                   the two weak rates that read the state)
//   postprocess_rate_r/_d<rate>     postprocess_rate
//   ener_gener_rate<spec>           ener_gener_rate
// All network DATA (Species/Rates indices, rhs_data fields, intermediate
// indices, zion/aion/bion/mion, screen factors) is generated from the stock
// headers into network_data.H by
// .claude/skills/loom-burn-kernel/scripts/gen_network_data.py.
//
// The stock rhs_state_t is not materialized (network_rhs.H): rhs_state.rho /
// eta / y_e are burn_state's fields, rhs_state.y is y. rhs_state.tf.t9 is
// read by postprocess_rate, whose hook gets no tf; it is recomputed as
// burn_state.T * 1.0e-9, which is the same single multiply get_tfactors does
// (and the .val of the dual one: dual_mul_num(temp, 1.0e-9)).
#ifndef MPL_NETWORK_H
#define MPL_NETWORK_H

#include "support.H"
#include "dual.H"
#include "network_data.H"
#include "burn.H"
#include "rates.H"
#include "screen.H"

namespace mpl {

// Runtime parameters read on the rhs/jac path (README rule 8). The fields
// are the RatesParams ones (rates.H/screen.H read them) plus use_tables.
// net_rates_params() builds the RatesParams value the rate/screen calls take.
struct NetParams {
    int network_use_c12ag_deboer17;             // network_rp::use_c12ag_deboer17 (bool)
    int screening_enable_debye_huckel_skip;     // screening_rp::enable_debye_huckel_skip (bool)
    Real screening_debye_huckel_skip_threshold; // screening_rp::debye_huckel_skip_threshold
    int network_use_tables;                     // network_rp::use_tables: must be 0 (not ported)
};

MPL_DEVICE RatesParams net_rates_params(const NetParams& params)
{
    RatesParams rp{params.network_use_c12ag_deboer17,
                   params.screening_enable_debye_huckel_skip,
                   params.screening_debye_huckel_skip_threshold};
    return rp;
}

// ---------------------------------------------------------------------------
// networks/aprox19/actual_network.H:1329 evaluate_analytical_rate
// ---------------------------------------------------------------------------

// number_t = Real. P_to_N and Ni56_to_Fe54 read rhs_state.rho / eta / y_e / y
// (ecapnuc, langanke); every other rate reads only tf (rates.H).
template <int rate>
MPL_DEVICE rate_t<Real> net_analytical_rate_r(const burn_t& burn_state, const NetYArray1D& y,
                                              const tf_t<Real>& tf, const RatesParams& rp)
{
    if constexpr (rate == Rates::P_to_N) {
        // construct_rate zeroes rates before evaluate_analytical_rate
        rate_t<Real> rates{0.0, 0.0};
        if (tf.temp >= 1.0e6 && burn_state.rho >= 1.0e-9) {
            // Loom 0.1.0 rejects an uninitialized local bound to a Real&
            // out-parameter (LOWERING/059 "runtime value-reference arguments
            // require an automatic SSA object"; probe r25 in the skill kit), so
            // the outputs are zero-initialized (ecapnuc writes rpen/rnep
            // first; spenc/snepc are discarded). The input copies are not
            // required (probe r26) but harmless.
            const Real eta_in = burn_state.eta;
            const Real temp_in = tf.temp;
            Real tmp_fr = 0.0, tmp_rr = 0.0;
            Real tmp1 = 0.0, tmp2 = 0.0;
            ecapnuc(eta_in, temp_in,
                    tmp_fr, tmp_rr, tmp1, tmp2);
            rates.fr = tmp_fr;
            rates.rr = tmp_rr;
        }
        return rates;
    }
    else if constexpr (rate == Rates::Ni56_to_Fe54) {
        rate_t<Real> rates{0.0, 0.0};
        if (tf.temp >= 1.0e6 && burn_state.rho >= 1.0e-9) {
            // initialized out locals, as for ecapnuc (langanke writes both
            // outputs first)
            const Real temp_in = tf.temp;
            const Real rho_in = burn_state.rho;
            const Real y56_in = y(Species::Ni56);
            const Real ye_in = burn_state.y_e;
            Real tmp_fr = 0.0, tmp_rr = 0.0;
            langanke(temp_in, rho_in,
                     y56_in, ye_in, tmp_fr, tmp_rr);
            rates.fr = tmp_fr;

            // scale by 1/54 (see actual_network.H)
            rates.fr *= 1.0 / 54.0;
        }
        return rates;
    }
    else {
        return evaluate_analytical_rate_r<rate>(tf, rp);
    }
}

// number_t = autodiff::dual. The weak rates are Real-valued: assigning them
// to a dual gives a zero derivative (dual_const).
template <int rate>
MPL_DEVICE rate_t<Dual> net_analytical_rate_d(const burn_t& burn_state, const NetYArray1D& y,
                                              const tf_t<Dual>& tf, const RatesParams& rp)
{
    if constexpr (rate == Rates::P_to_N) {
        rate_t<Dual> rates{dual_const(0.0), dual_const(0.0)};
        if (tf.temp.val >= 1.0e6 && burn_state.rho >= 1.0e-9) {
            // Loom 0.1.0 rejects an uninitialized local bound to a Real&
            // out-parameter (LOWERING/059 "runtime value-reference arguments
            // require an automatic SSA object"; probe r25 in the skill kit), so
            // the outputs are zero-initialized (ecapnuc writes rpen/rnep
            // first; spenc/snepc are discarded). The input copies are not
            // required (probe r26) but harmless.
            const Real eta_in = burn_state.eta;
            const Real temp_in = tf.temp.val;
            Real tmp_fr = 0.0, tmp_rr = 0.0;
            Real tmp1 = 0.0, tmp2 = 0.0;
            ecapnuc(eta_in, temp_in,
                    tmp_fr, tmp_rr, tmp1, tmp2);
            rates.fr = dual_const(tmp_fr);
            rates.rr = dual_const(tmp_rr);
        }
        return rates;
    }
    else if constexpr (rate == Rates::Ni56_to_Fe54) {
        rate_t<Dual> rates{dual_const(0.0), dual_const(0.0)};
        if (tf.temp.val >= 1.0e6 && burn_state.rho >= 1.0e-9) {
            // initialized out locals, as for ecapnuc (langanke writes both
            // outputs first)
            const Real temp_in = tf.temp.val;
            const Real rho_in = burn_state.rho;
            const Real y56_in = y(Species::Ni56);
            const Real ye_in = burn_state.y_e;
            Real tmp_fr = 0.0, tmp_rr = 0.0;
            langanke(temp_in, rho_in,
                     y56_in, ye_in, tmp_fr, tmp_rr);
            rates.fr = dual_const(tmp_fr);

            // rates.fr *= 1.0_rt / 54.0_rt: compound assignment, full dual
            rates.fr = dual_mul(rates.fr, dual_const(1.0 / 54.0));
        }
        return rates;
    }
    else {
        return evaluate_analytical_rate_d<rate>(tf, rp);
    }
}

// ---------------------------------------------------------------------------
// networks/aprox19/actual_network.H:1505 postprocess_rate
// ---------------------------------------------------------------------------

MPL_DEVICE constexpr int net_is_alpha_chain_rate(int rate)
{
    return (rate == Rates::Mg24_He4_to_Si28 ||
            rate == Rates::Si28_He4_to_S32 ||
            rate == Rates::S32_He4_to_Ar36 ||
            rate == Rates::Ar36_He4_to_Ca40 ||
            rate == Rates::Ca40_He4_to_Ti44 ||
            rate == Rates::Ti44_He4_to_Cr48 ||
            rate == Rates::Cr48_He4_to_Fe52) ? 1 : 0;
}

// from networks/aprox19/actual_network.H:1505 postprocess_rate (number_t = Real)
// rates1 is written by the beta limits (N14_2H1_to_*, O16_2H1_to_N14_He4);
// it is the caller's per-rate copy, as in stock.
template <int rate>
MPL_DEVICE void postprocess_rate_r(const burn_t& burn_state, const NetYArray1D& y,
                                   rate_t<Real>* rates, rate_t<Real>* rates1,
                                   rate_t<Real>* rates2, rate_t<Real>* rates3)
{
    using namespace Species;

    // rhs_state.tf.t9 (see the header comment)
    const Real t9 = burn_state.T * 1.0e-9;

    if constexpr (rate == Rates::P_P_N_N_to_He4) {
        Real denom = rates2->rr * rates1->rr + y(N) * rates3->fr * rates1->rr + y(N) * y(P) * rates3->fr * rates2->fr;

        if (denom > 1.0e-50 && t9 > 1.5) {
            Real zz = 1.0 / denom;

            rates->rr = rates3->rr * rates2->rr * rates1->rr * zz;

            rates->fr = rates3->fr * rates2->fr * rates1->fr * zz;
        }
    }

    if constexpr (rate == Rates::H1_H1_H1_to_He3) {
        rates->fr = rates1->fr;
    }

    if constexpr (rate == Rates::He3_He4_H1_to_2He4) {
        Real xx = 0.896 / y(He4);
        rates->fr = math::min(rates->fr, xx);
    }

    if constexpr (net_is_alpha_chain_rate(rate) != 0) {
        Real rate_ir = 0.0e0;
        Real denom = rates1->rr + rates2->fr;

        if (denom > 1.0e-30) {
            Real zz = 1.0 / denom;
            rate_ir = rates1->rr * zz;
        }

        rates->fr += rates1->fr * (1.0 - rate_ir);
        rates->rr += rates2->rr * rate_ir;
    }

    if constexpr (rate == Rates::O16_O16_to_Si28_He4) {
        Real rate_ir = 0.0e0;
        Real denom = rates1->rr + rates2->fr;

        if (denom > 1.0e-30) {
            Real zz = 1.0 / denom;
            rate_ir = rates1->rr * zz;
        }

        rates->fr = 0.56 * rates3->fr + 0.34 * rates3->fr * rate_ir;
    }

    if constexpr (rate == Rates::O16_O16_to_S32) {
        Real rate_ir = 0.0e0;
        Real denom = rates1->rr + rates2->fr;

        if (denom > 1.0e-30) {
            Real zz = 1.0 / denom;
            rate_ir = rates1->rr * zz;
        }

        rates->fr = 0.1 * rates3->fr + 0.34 * rates3->fr * (1.0 - rate_ir);
    }

    if constexpr (rate == Rates::C12_2H1_to_N14) {
        rates->fr = rates1->fr;
    }

    if constexpr (rate == Rates::C12_O16_to_Si28) {
        rates->fr = rates1->fr;
        rates->rr = rates1->rr;
    }

    if constexpr (rate == Rates::N14_2H1_to_O16) {
        if (y(H1) > 1.0e-30) {
            Real xx = 5.68e-3 / (y(H1) * 1.57);
            rates1->fr = math::min(rates1->fr, xx);
        }

        Real tot = rates2->fr + rates3->fr;
        Real invtot = 1.0 / tot;

        rates->fr = rates1->fr * (rates2->fr * invtot);
    }

    if constexpr (rate == Rates::N14_2H1_to_C12_He4) {
        if (y(H1) > 1.0e-30) {
            Real xx = 5.68e-3 / (y(H1) * 1.57);
            rates1->fr = math::min(rates1->fr, xx);
        }

        Real tot = rates2->fr + rates3->fr;
        Real invtot = 1.0 / tot;

        rates->fr = rates1->fr * (rates3->fr * invtot);
    }

    if constexpr (rate == Rates::N14_He4_to_Ne20) {
        rates->fr = rates1->fr * 0.5;
    }

    if constexpr (rate == Rates::O16_2H1_to_N14_He4) {
        Real xx = 0.0105 / y(H1);
        rates1->fr = math::min(rates1->fr, xx);

        rates->fr = rates1->fr;
    }

    if constexpr (rate == Rates::Fe52_2N_to_Fe54) {
        Real denom = rates1->rr + y(N) * rates2->fr;

        if (denom > 1.0e-50 && t9 > 1.5) {
            Real zz = 1.0 / denom;

            rates->fr = rates1->fr * rates2->fr * zz;

            rates->rr = rates2->rr * rates1->rr * zz;
        }
    }

    if constexpr (rate == Rates::Fe54_2P_to_Ni56) {
        Real denom = rates1->rr + y(P) * (rates2->fr + rates3->rr);

        if (denom > 1.0e-50 && t9 > 1.5) {
            Real zz = 1.0 / denom;

            rates->fr = rates1->fr * rates2->fr * zz;

            rates->rr = rates2->rr * rates1->rr * zz;
        }
    }

    if constexpr (rate == Rates::Fe52_He4_to_Fe54_2P) {
        Real denom = rates2->rr + y(P) * (rates3->fr + rates1->rr);

        if (denom > 1.0e-50 && t9 > 1.5) {
            Real zz = 1.0 / denom;

            rates->fr = rates1->fr * rates2->rr * zz;

            rates->rr = rates2->fr * rates1->rr * zz;
        }
    }

    if constexpr (rate == Rates::Fe52_He4_to_Ni56) {
        Real denom = rates3->rr + y(P) * (rates2->fr + rates1->rr);

        if (denom > 1.0e-50 && t9 > 1.5) {
            Real zz = 1.0 / denom;

            Real fr_add = rates1->fr * rates2->fr * zz;
            rates->fr += y(P) * fr_add;

            Real rr_add = rates2->rr * rates1->rr * zz;
            rates->rr += y(P) * rr_add;
        }
    }
}

// from networks/aprox19/actual_network.H:1505 postprocess_rate (number_t = autodiff::dual)
// (dual.H section 3 trees noted per statement)
template <int rate>
MPL_DEVICE void postprocess_rate_d(const burn_t& burn_state, const NetYArray1D& y,
                                   rate_t<Dual>* rates, rate_t<Dual>* rates1,
                                   rate_t<Dual>* rates2, rate_t<Dual>* rates3)
{
    using namespace Species;

    // rhs_state.tf.t9.val (see the header comment)
    const Real t9 = burn_state.T * 1.0e-9;

    if constexpr (rate == Rates::P_P_N_N_to_He4) {
        // Add(Add(Mul(r2rr, r1rr), Mul(NDM(yN, r3fr), r1rr)), Mul(NDM(yN*yP, r3fr), r2fr))
        Dual denom = dual_add(dual_add(dual_mul(rates2->rr, rates1->rr),
                                       dual_mul(dual_mul_num(rates3->fr, y(N)), rates1->rr)),
                              dual_mul(dual_mul_num(rates3->fr, y(N) * y(P)), rates2->fr));

        if (denom.val > 1.0e-50 && t9 > 1.5) {
            Dual zz = dual_div(dual_const(1.0), denom);

            rates->rr = dual_mul(dual_mul(dual_mul(rates3->rr, rates2->rr), rates1->rr), zz);

            rates->fr = dual_mul(dual_mul(dual_mul(rates3->fr, rates2->fr), rates1->fr), zz);
        }
    }

    if constexpr (rate == Rates::H1_H1_H1_to_He3) {
        rates->fr = rates1->fr;
    }

    if constexpr (rate == Rates::He3_He4_H1_to_2He4) {
        Real xx = 0.896 / y(He4);
        rates->fr = dual_min(rates->fr, dual_const(xx));
    }

    if constexpr (net_is_alpha_chain_rate(rate) != 0) {
        Dual rate_ir = dual_const(0.0e0);
        Dual denom = dual_add(rates1->rr, rates2->fr);

        if (denom.val > 1.0e-30) {
            Dual zz = dual_div(dual_const(1.0), denom);
            rate_ir = dual_mul(rates1->rr, zz);
        }

        rates->fr = dual_add(rates->fr, dual_mul(rates1->fr, dual_sub(dual_const(1.0), rate_ir)));
        rates->rr = dual_add(rates->rr, dual_mul(rates2->rr, rate_ir));
    }

    if constexpr (rate == Rates::O16_O16_to_Si28_He4) {
        Dual rate_ir = dual_const(0.0e0);
        Dual denom = dual_add(rates1->rr, rates2->fr);

        if (denom.val > 1.0e-30) {
            Dual zz = dual_div(dual_const(1.0), denom);
            rate_ir = dual_mul(rates1->rr, zz);
        }

        // Add(NDM(0.56, r3fr), Mul(NDM(0.34, r3fr), rate_ir))
        rates->fr = dual_add(dual_mul_num(rates3->fr, 0.56),
                             dual_mul(dual_mul_num(rates3->fr, 0.34), rate_ir));
    }

    if constexpr (rate == Rates::O16_O16_to_S32) {
        Dual rate_ir = dual_const(0.0e0);
        Dual denom = dual_add(rates1->rr, rates2->fr);

        if (denom.val > 1.0e-30) {
            Dual zz = dual_div(dual_const(1.0), denom);
            rate_ir = dual_mul(rates1->rr, zz);
        }

        // Add(NDM(0.1, r3fr), Mul(NDM(0.34, r3fr), Add(1.0, Neg(rate_ir))))
        rates->fr = dual_add(dual_mul_num(rates3->fr, 0.1),
                             dual_mul(dual_mul_num(rates3->fr, 0.34),
                                      dual_sub(dual_const(1.0), rate_ir)));
    }

    if constexpr (rate == Rates::C12_2H1_to_N14) {
        rates->fr = rates1->fr;
    }

    if constexpr (rate == Rates::C12_O16_to_Si28) {
        rates->fr = rates1->fr;
        rates->rr = rates1->rr;
    }

    if constexpr (rate == Rates::N14_2H1_to_O16) {
        if (y(H1) > 1.0e-30) {
            Real xx = 5.68e-3 / (y(H1) * 1.57);
            rates1->fr = dual_min(rates1->fr, dual_const(xx));
        }

        Dual tot = dual_add(rates2->fr, rates3->fr);
        Dual invtot = dual_div(dual_const(1.0), tot);

        // Mul(r1fr, Mul(r2fr, invtot))
        rates->fr = dual_mul(rates1->fr, dual_mul(rates2->fr, invtot));
    }

    if constexpr (rate == Rates::N14_2H1_to_C12_He4) {
        if (y(H1) > 1.0e-30) {
            Real xx = 5.68e-3 / (y(H1) * 1.57);
            rates1->fr = dual_min(rates1->fr, dual_const(xx));
        }

        Dual tot = dual_add(rates2->fr, rates3->fr);
        Dual invtot = dual_div(dual_const(1.0), tot);

        rates->fr = dual_mul(rates1->fr, dual_mul(rates3->fr, invtot));
    }

    if constexpr (rate == Rates::N14_He4_to_Ne20) {
        // Mul(r1fr, 0.5)
        rates->fr = dual_mul_num(rates1->fr, 0.5);
    }

    if constexpr (rate == Rates::O16_2H1_to_N14_He4) {
        Real xx = 0.0105 / y(H1);
        rates1->fr = dual_min(rates1->fr, dual_const(xx));

        rates->fr = rates1->fr;
    }

    if constexpr (rate == Rates::Fe52_2N_to_Fe54) {
        // Add(r1rr, NDM(yN, r2fr))
        Dual denom = dual_add_ndm(rates1->rr, y(N), rates2->fr);

        if (denom.val > 1.0e-50 && t9 > 1.5) {
            Dual zz = dual_div(dual_const(1.0), denom);

            rates->fr = dual_mul(dual_mul(rates1->fr, rates2->fr), zz);

            rates->rr = dual_mul(dual_mul(rates2->rr, rates1->rr), zz);
        }
    }

    if constexpr (rate == Rates::Fe54_2P_to_Ni56) {
        // Add(r1rr, Mul(yP, Add(r2fr, r3rr)))
        Dual denom = dual_add(rates1->rr,
                              dual_mul(dual_const(y(P)), dual_add(rates2->fr, rates3->rr)));

        if (denom.val > 1.0e-50 && t9 > 1.5) {
            Dual zz = dual_div(dual_const(1.0), denom);

            rates->fr = dual_mul(dual_mul(rates1->fr, rates2->fr), zz);

            rates->rr = dual_mul(dual_mul(rates2->rr, rates1->rr), zz);
        }
    }

    if constexpr (rate == Rates::Fe52_He4_to_Fe54_2P) {
        Dual denom = dual_add(rates2->rr,
                              dual_mul(dual_const(y(P)), dual_add(rates3->fr, rates1->rr)));

        if (denom.val > 1.0e-50 && t9 > 1.5) {
            Dual zz = dual_div(dual_const(1.0), denom);

            rates->fr = dual_mul(dual_mul(rates1->fr, rates2->rr), zz);

            rates->rr = dual_mul(dual_mul(rates2->fr, rates1->rr), zz);
        }
    }

    if constexpr (rate == Rates::Fe52_He4_to_Ni56) {
        Dual denom = dual_add(rates3->rr,
                              dual_mul(dual_const(y(P)), dual_add(rates2->fr, rates1->rr)));

        if (denom.val > 1.0e-50 && t9 > 1.5) {
            Dual zz = dual_div(dual_const(1.0), denom);

            Dual fr_add = dual_mul(dual_mul(rates1->fr, rates2->fr), zz);
            // rates.fr += y(P) * fr_add: x = dual_add(x, EVAL(NDM(yP, fr_add)))
            rates->fr = dual_add(rates->fr, dual_mul_num(fr_add, y(P)));

            Dual rr_add = dual_mul(dual_mul(rates2->rr, rates1->rr), zz);
            rates->rr = dual_add(rates->rr, dual_mul_num(rr_add, y(P)));
        }
    }
}

// from networks/aprox19/actual_network.H:1765 ener_gener_rate<spec>
// (rhs_state is unused there): dydt * network::mion<spec>() * C::Legacy::enuc_conv2
template <int spec>
MPL_DEVICE Real ener_gener_rate(const burn_t& burn_state, const Real dydt)
{
    constexpr Real mion = net_mion(spec);
    constexpr Real enuc_conv2 = net_legacy_enuc_conv2;
    return dydt * mion * enuc_conv2;
}

} // namespace mpl

// generic networks/rhs.H machinery (uses the hooks above)
#include "network_rhs.H"

#endif
