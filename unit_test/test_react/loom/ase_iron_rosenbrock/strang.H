// Strang integrator glue for the Loom burn kernel (see README.md), shared by
// the integrator modules: integration/integrator_type_strang.H,
// integration/integrator_type.H, integration/integrator_rhs_strang.H,
// integration/integrator_setup_strang.H and interfaces/burn_type.H
// normalize_abundances_burn, for STRANG, no NSE, no NONAKA_PLOT, HIP branches
// (AMREX_USE_GPU: no printing, no xn_in in state_backup_t).
//
// Extracted mechanically from the shipped vode.H (validated there for
// dvode_t): every function that takes the integrator state is a template on
// its type StrangIntT, as in the stock headers (`template <typename BurnT,
// typename IntegratorT>`), passed by pointer. StrangIntT needs the members
// y(1..neqs), t, tout, n_rhs, n_jac, n_step, rtol/atol_spec/enuc and
// jacobian_type. The including integrator module defines IntegratorParams
// (with at least the integrator_* fields read below and network_small_x)
// before including this header.
#ifndef MPL_STRANG_H
#define MPL_STRANG_H

#include "support.H"
#include "burn.H"
#include "eos.H"
#include "network.H"

namespace mpl {

// from interfaces/burn_type.H:338 normalize_abundances_burn
MPL_DEVICE void normalize_abundances_burn(burn_t* state, const IntegratorParams& ip)
{
    Real sum = 0.0;
    for (int n = 0; n < NumSpec; ++n) {
        state->xn[n] = math::max(ip.network_small_x, math::min(1.0, state->xn[n]));
        sum += state->xn[n];
    }
    for (int n = 0; n < NumSpec; ++n) {
        state->xn[n] /= sum;
    }
}

// ---------------------------------------------------------------------------
// integration/integrator_type_strang.H, integration/integrator_type.H
// ---------------------------------------------------------------------------

// from integration/integrator_type_strang.H:10 integrator_to_burn
MPL_DEVICE void integrator_to_burn(const StrangIntT* vs, burn_t* state, const IntegratorParams& ip)
{
    for (int n = 1; n <= NumSpec; ++n) {
        state->xn[n-1] = vs->y(n);
    }

    state->e = vs->y(net_ienuc);

    if (ip.integrator_scale_system) {
        state->e *= state->e_scale;
    }

    state->time = vs->t;
}

// from integration/integrator_type_strang.H:30 burn_to_integrator
MPL_DEVICE void burn_to_integrator(const burn_t& state, StrangIntT* vs, const IntegratorParams& ip)
{
    for (int n = 1; n <= NumSpec; ++n) {
        vs->y(n) = state.xn[n-1];
    }

    vs->y(net_ienuc) = state.e;

    if (ip.integrator_scale_system) {
        vs->y(net_ienuc) /= state.e_scale;
    }
}

// from integration/integrator_type_strang.H:47 clean_state
MPL_DEVICE void clean_state(const Real time, burn_t* state, StrangIntT* vs, const IntegratorParams& ip)
{
    (void) time;
    (void) state;

    if (ip.integrator_do_species_clip) {
        for (int n = 1; n <= NumSpec; ++n) {
            vs->y(n) = math::clamp(vs->y(n), ip.integrator_SMALL_X_SAFE, 1.0);
        }
    }

    if (ip.integrator_use_number_densities) {
        for (int n = 1; n <= NumSpec; ++n) {
            vs->y(n) = math::max(vs->y(n), ip.integrator_SMALL_X_SAFE);
        }
    }

    if (ip.integrator_renormalize_abundances) {
        Real sum = 0.0;

        for (int n = 1; n <= NumSpec; ++n) {
            sum += vs->y(n);
        }

        for (int n = 1; n <= NumSpec; ++n) {
            vs->y(n) /= sum;
        }
    }
}

// from integration/integrator_type.H:34 update_thermodynamics (no AUX_THERMO)
MPL_DEVICE void update_thermodynamics(burn_t* state, const StrangIntT* vs, const IntegratorParams& ip,
                                      const EosParams& ep, EosTables tab)
{
    for (int n = 1; n <= NumSpec; ++n) {
        state->xn[n-1] = vs->y(n);
    }
    state->e = vs->y(net_ienuc);

    if (ip.integrator_scale_system) {
        state->e *= state->e_scale;
    }

    if (ip.integrator_call_eos_in_rhs) {
        eos(eos_input_re, state, ep, tab);
    }

    if (state->T_fixed > 0.0) {
        state->T = state->T_fixed;
    }
}

// ---------------------------------------------------------------------------
// integration/integrator_rhs_strang.H
// ---------------------------------------------------------------------------

// from integration/integrator_rhs_strang.H:24 rhs (ydot(1..neqs);
// in_jacobian only feeds NONAKA_PLOT and is dropped)
MPL_DEVICE void rhs(const Real time, burn_t* state, StrangIntT* vs, Array1D<Real, 1, neqs>& ydot,
                    const IntegratorParams& ip, const EosParams& ep, const NetParams& np, EosTables tab)
{
    clean_state(time, state, vs, ip);

    update_thermodynamics(state, vs, ip, ep, tab);

    if (state->T <= ep.EOSData_mintemp || state->T >= ip.integrator_MAX_TEMP) {

        for (int n = 1; n <= neqs; ++n) {
            ydot(n) = 0.0;
        }

        return;
    }

    state->time = time;

    actual_rhs(*state, np, ydot);

    if (!ip.integrator_use_number_densities) {
        for (int n = 1; n <= NumSpec; ++n) {
            ydot(n) *= aion[n-1];
        }
    }

    if (ip.integrator_scale_system) {
        ydot(net_ienuc) /= state->e_scale;
    }

    if (!ip.integrator_integrate_energy) {
        ydot(net_ienuc) = 0.0;
    }

    if (ip.integrator_react_boost > 0.0) {
        for (int n = 1; n <= neqs; ++n) {
            ydot(n) *= ip.integrator_react_boost;
        }
    }
}

// from integration/integrator_rhs_strang.H:114 jac (pd(i, j))
MPL_DEVICE void jac(const Real time, burn_t* state, StrangIntT* vs, JacNetArray2D& pd,
                    const IntegratorParams& ip, const EosParams& ep, const NetParams& np, EosTables tab)
{
    (void) time;

    if (state->T <= ep.EOSData_mintemp || state->T >= ip.integrator_MAX_TEMP) {
        // pd.zero()
        for (int i = 1; i <= neqs; ++i) {
            for (int j = 1; j <= neqs; ++j) {
                pd(i, j) = 0.0;
            }
        }
        return;
    }

    integrator_to_burn(vs, state, ip);

    actual_jac(*state, np, pd);

    if (!ip.integrator_use_number_densities) {
        for (int j = 1; j <= NumSpec; ++j) {
            for (int i = 1; i <= neqs; ++i) {
                pd(j, i) *= aion[j-1];
                pd(i, j) *= aion_inv[j-1];
            }
        }
    }

    if (ip.integrator_correct_jacobian_for_const_e) {

        eos_re_extra_t eos_state{};
        eos_state.rho = state->rho;
        eos_state.T = state->T;
        eos_state.e = state->e;
        for (int n = 0; n < NumSpec; n++) {
            eos_state.xn[n] = state->xn[n];
        }

        eos(eos_input_re, &eos_state, ep, tab);

        eos_xderivs_t eos_xderivs = composition_derivatives(eos_state);

        for (int jcol = 1; jcol <= NumSpec; ++jcol) {
            for (int irow = 1; irow <= neqs; ++irow) {
                pd(irow, jcol) -= eos_xderivs.dedX[jcol-1] * pd(irow, net_ienuc);
            }
        }
    }

    if (ip.integrator_scale_system) {
        for (int j = 1; j <= neqs; ++j) {
            pd(net_ienuc, j) /= state->e_scale;
        }

        for (int i = 1; i <= neqs; ++i) {
            pd(i, net_ienuc) *= state->e_scale;
        }
    }

    if (ip.integrator_react_boost > 0.0) {
        // pd.mul(react_boost)
        for (int i = 1; i <= neqs; ++i) {
            for (int j = 1; j <= neqs; ++j) {
                pd(i, j) *= ip.integrator_react_boost;
            }
        }
    }

    if (!ip.integrator_integrate_energy) {
        for (int j = 1; j <= neqs; ++j) {
            pd(net_ienuc, j) = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// integration/integrator_setup_strang.H
// ---------------------------------------------------------------------------

// from integration/integrator_setup_strang.H:19 state_backup_t (AMREX_USE_GPU: no xn_in)
struct state_backup_t {
    Real T_in{};
    Real e_in{};
};

// from integration/integrator_setup_strang.H:27 integrator_setup
// (stock returns the value-initialized IntegratorT; here the caller passes a
// value-initialized integrator state)
MPL_DEVICE void integrator_setup(burn_t* state, Real dt, bool is_retry, StrangIntT* vs,
                                 const IntegratorParams& ip, const EosParams& ep, EosTables tab)
{
    if (!is_retry) {
        vs->atol_spec = ip.integrator_atol_spec;
        vs->atol_enuc = ip.integrator_atol_enuc;

        vs->rtol_spec = ip.integrator_rtol_spec;
        vs->rtol_enuc = ip.integrator_rtol_enuc;
    } else {
        vs->atol_spec = ip.integrator_retry_atol_spec > 0 ?
            ip.integrator_retry_atol_spec : ip.integrator_atol_spec;
        vs->atol_enuc = ip.integrator_retry_atol_enuc > 0 ?
            ip.integrator_retry_atol_enuc : ip.integrator_atol_enuc;

        vs->rtol_spec = ip.integrator_retry_rtol_spec > 0 ?
            ip.integrator_retry_rtol_spec : ip.integrator_rtol_spec;
        vs->rtol_enuc = ip.integrator_retry_rtol_enuc > 0 ?
            ip.integrator_retry_rtol_enuc : ip.integrator_rtol_enuc;
    }

    if (is_retry && ip.integrator_retry_swap_jacobian) {
        vs->jacobian_type = (ip.integrator_jacobian == 1) ? 2 : 1;
    } else {
        vs->jacobian_type = static_cast<short>(ip.integrator_jacobian);
    }

    state->success = true;

    vs->t = 0.0;
    vs->tout = dt;

    for (int n = 0; n < SVAR; ++n) {
        state->ydot_a[n] = 0;
    }

    eos(eos_input_rt, state, ep, tab);

    state->e_scale = state->e;

    if (ip.integrator_scale_system) {
        vs->atol_enuc /= state->e_scale;
    }

    burn_to_integrator(*state, vs, ip);
}

// from integration/integrator_setup_strang.H:101 integrator_backup
MPL_DEVICE state_backup_t integrator_backup(const burn_t& state)
{
    state_backup_t state_save{};

    state_save.T_in = state.T;
    state_save.e_in = state.e;

    return state_save;
}

// from integration/integrator_setup_strang.H:120 integrator_cleanup
// (no printing on GPU)
MPL_DEVICE void integrator_cleanup(StrangIntT* vs, burn_t* state, int istate,
                                   const state_backup_t& state_save, Real dt,
                                   const Real final_state_species_failure_tolerance_factor,
                                   const IntegratorParams& ip)
{
    (void) dt;

    integrator_to_burn(vs, state, ip);

    if (ip.integrator_subtract_internal_energy) {
        state->e -= state_save.e_in;
    }

    if (!ip.integrator_use_number_densities) {
        normalize_abundances_burn(state, ip);
    }

    state->n_rhs = vs->n_rhs;
    state->n_jac = vs->n_jac;
    state->n_step = vs->n_step;

    state->error_code = static_cast<short>(istate);

    if (istate != IERR_SUCCESS) {
        state->success = false;
    }

    const Real final_state_species_failure_tolerance =
        final_state_species_failure_tolerance_factor * ip.integrator_species_failure_tolerance;

    for (int n = 1; n <= NumSpec; ++n) {
        if (vs->y(n) < -final_state_species_failure_tolerance) {
            state->success = false;
        }

        if (!ip.integrator_use_number_densities) {
            if (vs->y(n) > 1.0 + ip.integrator_species_failure_tolerance) {
                state->success = false;
            }
        }
    }
}

} // namespace mpl

#endif
