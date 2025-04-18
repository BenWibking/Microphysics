#ifndef actual_integrator_H
#define actual_integrator_H

#include <AMReX_Algorithm.H>

#include <network.H>
#include <actual_network.H>
#ifdef NEW_NETWORK_IMPLEMENTATION
#include <rhs.H>
#else
#include <actual_rhs.H>
#endif
#include <burn_type.H>
#include <eos_type.H>
#include <eos.H>
#include <extern_parameters.H>
#include <fe_type.H>
#include <integrator_data.H>

template <typename IntT, typename BurnT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real calculate_dt (IntT& int_state, BurnT& state, amrex::Array1D<amrex::Real, 1, NumSpec>& spec_rhs, amrex::Real& ener_rhs)
{
    using namespace microphysics::forward_euler;

    // Our timestepping strategy is to prevent any quantity
    // from changing by more than a certain factor in any
    // timestep. We ignore this for species below atol_spec.

    amrex::Real dt = 1.0e200_rt;

    for (int n = 1; n <= NumSpec; ++n) {

        if (state.xn[n-1] >= integrator_rp::atol_spec) {

            amrex::Real target_dX;
            if (spec_rhs(n) > 0.0) {
                target_dX = (integrator_rp::maximum_timestep_change_factor - 1.0_rt) * state.xn[n-1];
            } else {
                target_dX = (1.0_rt - 1.0_rt / integrator_rp::maximum_timestep_change_factor) * state.xn[n-1];
            }

            amrex::Real dXdt = amrex::max(std::abs(spec_rhs(n)), 1.0e-30_rt);

            dt = amrex::min(dt, target_dX / dXdt);

        }

    }

    if (integrator_rp::integrate_energy) {

        amrex::Real target_de;
        if (ener_rhs > 0.0) {
            target_de = (integrator_rp::maximum_timestep_change_factor - 1.0_rt) * state.e;
        } else {
            target_de = (1.0_rt - 1.0_rt / integrator_rp::maximum_timestep_change_factor) * state.e;
        }

        amrex::Real dedt = amrex::max(std::abs(ener_rhs), 1.0e-30_rt);

        dt = amrex::min(dt, target_de / dedt);

    }

    dt = amrex::min(dt, integrator_rp::ode_max_dt);

    return dt;
}

template <typename IntT, typename BurnT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void clean_state (const amrex::Real time, IntT& int_state, BurnT& state)
{
    using namespace microphysics::forward_euler;

    // Renormalize the abundances.

    normalize_abundances_burn(state);

    // Evaluate the EOS to get T from e.

    if (integrator_rp::call_eos_in_rhs) {
        eos(eos_input_re, state);
    }

    // Ensure that the temperature always stays within reasonable limits.

    state.T = amrex::Clamp(state.T, EOSData::mintemp, integrator_rp::MAX_TEMP);

}

template <typename IntT, typename BurnT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void evaluate_rhs (IntT& int_state, BurnT& state, amrex::Array1D<amrex::Real, 1, NumSpec>& spec_rhs, amrex::Real& ener_rhs)
{
    using namespace microphysics::forward_euler;

    // Evaluate the RHS.

    amrex::Array1D<amrex::Real, 1, neqs> ydot;

    actual_rhs(state, ydot);

    int_state.n_rhs += 1;

    // Scale species terms by A (they come from the RHS in terms of Y, not X).

    for (int n = 1; n <= NumSpec; ++n) {
        spec_rhs(n) = aion[n-1] * ydot(n);
    }

    ener_rhs = ydot(net_ienuc);
}

template <typename BurnT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void initialize_state (BurnT& state)
{
    using namespace microphysics::forward_euler;

    // We assume that (rho, T) coming in are valid, do an EOS call
    // to fill the rest of the thermodynamic variables.

    eos(eos_input_rt, state);

    state.success = true;
    state.error_code = IERR_SUCCESS;

    // Initialize ydot to zero for Strang burn.

    for (int n = 0; n < SVAR; ++n) {
        state.ydot_a[n] = 0;
    }
}

template <typename IntT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void initialize_int_state (IntT& int_state)
{
    using namespace microphysics::forward_euler;

    // Set the tolerances.

    int_state.atol_spec = integrator_rp::atol_spec;  // mass fractions
    int_state.atol_enuc = integrator_rp::atol_enuc;  // energy generated

    int_state.rtol_spec = integrator_rp::rtol_spec;  // mass fractions
    int_state.rtol_enuc = integrator_rp::rtol_enuc;  // energy generated

    // Initialize the integration time.

    int_state.t = 0.0_rt;

    int_state.n_step = 0;
    int_state.n_rhs = 0;
}

template <typename BurnT>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void actual_integrator (BurnT& state, amrex::Real dt, bool is_retry=false)
{
    using namespace microphysics::forward_euler;

    constexpr int int_neqs = integrator_neqs<BurnT>();

    fe_t<int_neqs> fe;

    initialize_state(state);

    initialize_int_state(fe);

    fe.tout = dt;

    amrex::Real T_in = state.T;
    amrex::Real e_in = state.e;
    amrex::Real xn_in[NumSpec];
    for (int n = 0; n < NumSpec; ++n) {
        xn_in[n] = state.xn[n];
    }

    while (fe.t < (1.0_rt - timestep_safety_factor) * dt && fe.n_step < integrator_rp::ode_max_steps) {
        // Evaluate the RHS.

        amrex::Array1D<amrex::Real, 1, NumSpec> spec_rhs;
        amrex::Real ener_rhs;

        evaluate_rhs(fe, state, spec_rhs, ener_rhs);

       // Calculate the timestep.

        amrex::Real dt_sub = calculate_dt(fe, state, spec_rhs, ener_rhs);

        // Prevent the timestep from overshooting the final time.

        if (fe.t + dt_sub > dt) {
            dt_sub = dt - fe.t;
        }

        // Apply the update.

        for (int n = 1; n <= NumSpec; ++n) {
            fe.y(n) += spec_rhs(n) * dt_sub;
            state.xn[n-1] += spec_rhs(n) * dt_sub;
        }

        if (integrator_rp::integrate_energy) {
            fe.y(net_ienuc) += ener_rhs * dt_sub;
            state.e += ener_rhs * dt_sub;
        }

        fe.t += dt_sub;
        ++fe.n_step;

        clean_state(fe.t, fe, state);
    }

    if (fe.n_step >= integrator_rp::ode_max_steps) {
        state.success = false;
        state.error_code = IERR_TOO_MANY_STEPS;
    }

    state.time = fe.t;
    state.n_step = fe.n_step;
    state.n_rhs = fe.n_rhs;
    state.n_jac = 0;

#ifndef AMREX_USE_GPU
    if (integrator_rp::burner_verbose) {
        // Print out some integration statistics, if desired.
        std::cout <<  "integration summary: " << std::endl;
        std::cout <<  "dens: " << state.rho << " temp: " << state.T << std::endl;
        std::cout <<  "energy released: " << state.e - e_in << std::endl;
        std::cout <<  "number of steps taken: " << state.n_step << std::endl;
        std::cout <<  "number of f evaluations: " << state.n_rhs << std::endl;
    }
#endif

    // If we failed, print out the current state of the integration.

    if (!state.success) {
#ifndef AMREX_USE_GPU
        std::cout << "ERROR: integration failed in net" << std::endl;
        std::cout << "time = " << state.time << std::endl;
        std::cout << "dens = " << state.rho << std::endl;
        std::cout << "temp start = " << T_in << std::endl;
        std::cout << "xn start = ";
        for (int n = 0; n < NumSpec; ++n) {
            std::cout << xn_in[n] << " ";
        }
        std::cout << std::endl;
        std::cout << "temp current = " << state.T << std::endl;
        std::cout << "xn current = ";
        for (int n = 0; n < NumSpec; ++n) {
            std::cout << state.xn[n] << " ";
        }
        std::cout << std::endl;
        std::cout << "energy generated = " << state.e - e_in << std::endl;
#endif
    }

}

#endif
