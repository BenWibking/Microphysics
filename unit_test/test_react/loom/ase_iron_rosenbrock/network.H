// Network glue for the Loom burn kernel of a pynucastro network (see
// README.md): the parts of the network's actual_rhs.H that are the same for
// every pynucastro network, for this configuration (SCREENING, STRANG, no
// NEUTRINOS, no NSE_NET, HIP branches), around the generated bodies:
//   pyn_rates.H    rates, tables, partition functions (gen_pynucastro_net.py)
//   pyn_network.H  evaluate_screening_{r,d}, rhs_nuc, jac_nuc (same generator)
// and the dense LU that integration/VODE uses for a network without
// NEW_NETWORK_IMPLEMENTATION (util/linpack.H dgefa / dgesl, with
// integrator.linalg_do_pivoting).
//
// Entry points (called by vode.H):
//   actual_rhs(const burn_t&, const NetParams&, Array1D<Real, 1, neqs>& ydot)
//   actual_jac(const burn_t&, const NetParams&, JacNetArray2D& jac)
//   net_dgefa<allow_pivot>(JacNetArray2D& a, NetPivotArray& pivot, int& info)
//   net_dgesl<allow_pivot>(const JacNetArray2D& a, const NetPivotArray& pivot,
//                          Array1D<Real, 1, neqs>& b)
// MPL_NET_DENSE_LU tells vode.H to keep a pivot array and call these.
//
// What differs from a template instantiation, all value-identical:
//   * evaluate_rates<do_T_derivatives, T> is evaluate_rates_r (T = rate_t)
//     and evaluate_rates_d (T = rate_derivs_t);
//   * ener_gener_rate's constexpr_for lambda is a loop over the same species
//     in the same order (network::mion<n>() -> net_mion(n));
//   * actual_jac's jac_slice_2 lambda (jac.get(i, j)) is
//     ener_gener_rate_jac_col(jac, j, ...);
//   * uninitialized locals passed as Real& are zero-initialized (probe r25),
//     record locals get `{}` (probe d14).
#ifndef MPL_NETWORK_H
#define MPL_NETWORK_H

#include "support.H"
#include "dual.H"
#include "network_data.H"
#include "burn.H"
#include "pyn_rates.H"
#include "screen.H"

#if MPL_GEN_NEUTRINOS
#error "network.H: NEUTRINOS is defined, but neutrino cooling (sneut5) is not ported; regenerate network_data.H with USE_NEUTRINOS=FALSE"
#endif
#if MPL_GEN_NSE || MPL_GEN_NSE_NET
#error "network.H: NSE / NSE_NET builds are not ported"
#endif
#if !MPL_GEN_SCREENING
#error "network.H: built for SCREENING (evaluate_screening is always called)"
#endif

#define MPL_NET_DENSE_LU 1

namespace mpl {

// Runtime parameters read on the rhs/jac path (README rule 8): the
// screening_rp ones (bool -> int). net_screen_params() builds the
// ScreenParams value the screening calls take.
struct NetParams {
    int screening_enable_debye_huckel_skip;          // screening_rp::enable_debye_huckel_skip
    Real screening_debye_huckel_skip_threshold;      // screening_rp::debye_huckel_skip_threshold
    int screening_enable_chabrier1998_quantum_corr;  // screening_rp::enable_chabrier1998_quantum_corr
};

MPL_DEVICE ScreenParams net_screen_params(const NetParams& params)
{
    ScreenParams sp{params.screening_enable_debye_huckel_skip,
                    params.screening_debye_huckel_skip_threshold,
                    params.screening_enable_chabrier1998_quantum_corr};
    return sp;
}

// from integration/utils/jacobian_utilities.H:18 temperature_to_energy_jacobian
MPL_DEVICE Real temperature_to_energy_jacobian(const burn_t& state, const Real jac_T)
{
    Real jac_e = jac_T / state.cv;

    return jac_e;
}

// from actual_rhs.H ener_gener_rate (T = amrex::Array1D<amrex::Real, 1, neqs>)
MPL_DEVICE void ener_gener_rate(const Array1D<Real, 1, neqs>& dydt, Real& enuc)
{
    // Computes the instantaneous energy generation rate (from the nuclei)

    // This is basically e = m c**2

    enuc = 0.0;

    for (int n = 1; n < NumSpec + 1; ++n) {
        enuc += dydt(n) * net_mion(n);
    }

    enuc *= net_enuc_conv2;
}

// from actual_rhs.H ener_gener_rate (T = actual_jac's jac_slice_2 lambda:
// dydt(i) = jac.get(i, j))
MPL_DEVICE void ener_gener_rate_jac_col(const JacNetArray2D& jac, const int j, Real& enuc)
{
    enuc = 0.0;

    for (int n = 1; n < NumSpec + 1; ++n) {
        enuc += jac(n, j) * net_mion(n);
    }

    enuc *= net_enuc_conv2;
}

} // namespace mpl

#include "pyn_network.H"

namespace mpl {

// from actual_rhs.H evaluate_rates<do_T_derivatives = 0, rate_t>
MPL_DEVICE void evaluate_rates_r(const burn_t& state,
                                 const Array1D<Real, 1, NumSpec>& Y,
                                 const ScreenParams& sp,
                                 rate_t& rate_eval)
{
    using namespace Rates;

    // here Y is consistent with state.xn

    const Real rhoy = state.rho * state.y_e;

    const tf_t tfactors = evaluate_tfactors(state.T);

    // Precompute screening terms

    evaluate_screening_r(state, Y, sp, rate_eval);

    // Fill in different rates

    reaclib_rates::fill_rates_r(tfactors, rate_eval);

    tabular_weak_rates::fill_rates_r(state.T, state.rho, rhoy, Y, rate_eval);

    temp_tabular::fill_rates_r(tfactors, rate_eval);

    modified_rates::fill_rates_r(tfactors, rate_eval);

    branched_rates::fill_rates_r(tfactors, rate_eval);

    derived_rates::fill_rates_r(tfactors, rate_eval);

    approximate_rates::fill_rates_r(tfactors, state.rho, Y, rate_eval);
}

// from actual_rhs.H evaluate_rates<do_T_derivatives = 1, rate_derivs_t>
MPL_DEVICE void evaluate_rates_d(const burn_t& state,
                                 const Array1D<Real, 1, NumSpec>& Y,
                                 const ScreenParams& sp,
                                 rate_derivs_t& rate_eval)
{
    using namespace Rates;

    const Real rhoy = state.rho * state.y_e;

    const tf_t tfactors = evaluate_tfactors(state.T);

    evaluate_screening_d(state, Y, sp, rate_eval);

    reaclib_rates::fill_rates_d(tfactors, rate_eval);

    tabular_weak_rates::fill_rates_d(state.T, state.rho, rhoy, Y, rate_eval);

    temp_tabular::fill_rates_d(tfactors, rate_eval);

    modified_rates::fill_rates_d(tfactors, rate_eval);

    branched_rates::fill_rates_d(tfactors, rate_eval);

    derived_rates::fill_rates_d(tfactors, rate_eval);

    approximate_rates::fill_rates_d(tfactors, state.rho, Y, rate_eval);
}

// from actual_rhs.H actual_rhs (no NEUTRINOS: sneut = 0)
MPL_DEVICE void actual_rhs(const burn_t& state, const NetParams& np, Array1D<Real, 1, neqs>& ydot)
{
    const ScreenParams sp = net_screen_params(np);

    // Set molar abundances
    Array1D<Real, 1, NumSpec> Y{};
    for (int i = 1; i < NumSpec + 1; ++i) {
        Y(i) = state.xn[i-1] * aion_inv[i-1];
    }

    // build the rates

    rate_t rate_eval{};

    evaluate_rates_r(state, Y, sp, rate_eval);

    rhs_nuc(state, ydot, Y, rate_eval.screened_rates);

    // ion binding energy contributions

    Real enuc{};
    ener_gener_rate(ydot, enuc);

    // include any weak rate neutrino losses
    enuc += rate_eval.enuc_weak;

    // Get the thermal neutrino losses

    Real sneut{};

    // Append the energy equation (this is erg/g/s)

    ydot(net_ienuc) = enuc - sneut;
}

// from actual_rhs.H actual_jac (MatrixType = JacNetArray2D; no NEUTRINOS)
MPL_DEVICE void actual_jac(const burn_t& state, const NetParams& np, JacNetArray2D& jac)
{
    const ScreenParams sp = net_screen_params(np);

    // Set molar abundances
    Array1D<Real, 1, NumSpec> Y{};
    for (int i = 1; i < NumSpec + 1; ++i) {
        Y(i) = state.xn[i-1] * aion_inv[i-1];
    }

    rate_derivs_t rate_eval{};

    evaluate_rates_d(state, Y, sp, rate_eval);

    // region 1: J_ij = d(Ydot_i)/dY_j

    jac_nuc(state, jac, Y, rate_eval);

    // region 2: J_{ienuc,j} = d(eps_nuc)/dY_j = -N_A sum_i d(Ydot_i)/dY_j m_i c^2

    for (int j = 1; j < NumSpec + 1; ++j) {
        Real tmp{};
        ener_gener_rate_jac_col(jac, j, tmp);
        jac(net_ienuc, j) = tmp;
    }

    // Account for the weak rate neutrino losses
    for (int jcol = 1; jcol < NumSpec + 1; ++jcol) {
        jac(net_ienuc, jcol) += rate_eval.denuc_weak_dY(jcol) +
                                rate_eval.denuc_weak_dYe * net_zion(jcol);
    }

    // Account for the thermal neutrino losses

    Real dsneutdt{};

    // region 3: J_{i,ienuc} = d(Ydot_i)/dT for rows i = 1, ..., NumSpec

    Array1D<Real, 1, neqs> yderivs{};

    rhs_nuc(state, yderivs, Y, rate_eval.dscreened_rates_dT);

    for (int k = 1; k < NumSpec + 1; ++k) {
        jac(k, net_ienuc) = temperature_to_energy_jacobian(state, yderivs(k));
    }

    // region 4: J_{ienuc, ienuc} = d(edot)/de, first as d/dT

    Real jac_e_T{};

    ener_gener_rate(yderivs, jac_e_T);

    // Account for the weak rate neutrino losses
    jac_e_T += rate_eval.denuc_weak_dT;

    // and the thermal neutrino contribution

    jac_e_T -= dsneutdt;

    // finally, convert d(de/dt)/dT to d(de/dt)/de

    jac(net_ienuc, net_ienuc) = temperature_to_energy_jacobian(state, jac_e_T);
}

// ---------------------------------------------------------------------------
// util/linpack.H (num_eqs = neqs; RArray2D = JacNetArray2D, IArray1D = the
// dvode_t pivot array)
// ---------------------------------------------------------------------------

using NetPivotArray = Array1D<short, 1, neqs>;

// Loops are written with strict upper bounds (`j < n + 1`, not `j <= n`;
// value-identical): Loom proves those counted-loop indices in range (with
// `<=`, the back substitution fails SUBRANGE/026, as in network_rhs.H).

// from util/linpack.H:10 dgesl
template <bool allow_pivot>
MPL_DEVICE void net_dgesl(const JacNetArray2D& a, const NetPivotArray& pivot, Array1D<Real, 1, neqs>& b)
{
    constexpr int num_eqs = neqs;
    constexpr int nm1 = num_eqs - 1;

    // solve a * x = b
    // first solve l * y = b
    if constexpr (nm1 >= 1) {
        for (int k = 1; k < num_eqs; ++k) {

            Real t{};
            if constexpr (allow_pivot) {
                int l = pivot(k);
                MPL_ASSUME(1 <= l && l <= num_eqs);
                t = b(l);
                if (l != k) {
                    b(l) = b(k);
                    b(k) = t;
                }
            } else {
                t = b(k);
            }

            for (int j = k+1; j < num_eqs + 1; ++j) {
                b(j) += t * a(j,k);
            }
        }
    }

    // now solve u * x = y
    for (int kb = 1; kb < num_eqs + 1; ++kb) {

        const int k = num_eqs + 1 - kb;
        b(k) = b(k) / a(k,k);
        Real t = -b(k);
        for (int j = 1; j < k; ++j) {
            b(j) += t * a(j,k);
        }
    }
}

// from util/linpack.H:57 dgefa
template <bool allow_pivot>
MPL_DEVICE void net_dgefa(JacNetArray2D& a, NetPivotArray& pivot, int& info)
{
    // gaussian elimination with partial pivoting

    info = 0;
    constexpr int num_eqs = neqs;
    constexpr int nm1 = num_eqs - 1;

    Real t{};   // stock: uninitialized, written before every read

    if constexpr (nm1 >= 1) {

        for (int k = 1; k < num_eqs; ++k) {

            // find l = pivot index
            int l = k;

            if constexpr (allow_pivot) {
                Real dmax = math::abs(a(k,k));
                for (int i = k+1; i < num_eqs + 1; ++i) {
                    Real ai = math::abs(a(i, k));
                    if (ai > dmax) {
                        l = i;
                        dmax = ai;
                    }
                }

                pivot(k) = static_cast<short>(l);
            }
            MPL_ASSUME(1 <= l && l <= num_eqs);

            // zero pivot implies this column already triangularized
            if (a(l,k) != 0.0e0) {

                if constexpr (allow_pivot) {
                    // interchange if necessary
                    if (l != k) {
                        t = a(l,k);
                        a(l,k) = a(k,k);
                        a(k,k) = t;
                    }
                }

                // compute multipliers
                t = -1.0e0 / a(k,k);
                for (int j = k+1; j < num_eqs + 1; ++j) {
                    a(j,k) *= t;
                }

                // row elimination with column indexing
                for (int j = k+1; j < num_eqs + 1; ++j) {
                    t = a(l,j);

                    if constexpr (allow_pivot) {
                        if (l != k) {
                            a(l,j) = a(k,j);
                            a(k,j) = t;
                        }
                    }

                    for (int i = k+1; i < num_eqs + 1; ++i) {
                        a(i,j) += t * a(i,k);
                    }
                }

            } else {
                info = k;
            }

        }

    }

    if constexpr (allow_pivot) {
        pivot(num_eqs) = static_cast<short>(num_eqs);
    }

    if (a(num_eqs,num_eqs) == 0.0e0) {
        info = num_eqs;
    }
}

} // namespace mpl

#endif
