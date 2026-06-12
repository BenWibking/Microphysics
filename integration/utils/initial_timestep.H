#ifndef INITIAL_TIMESTEP_H
#define INITIAL_TIMESTEP_H

#include <AMReX_REAL.H>
#include <AMReX_Algorithm.H>

#include <actual_network.H>
#ifdef NEW_NETWORK_IMPLEMENTATION
#include <rhs.H>
#else
#include <actual_rhs.H>
#endif
#include <burn_type.H>

template <typename BurnT, typename IntT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real initial_react_dt (BurnT& burn_state, IntT& int_state,
                              amrex::Array1D<amrex::Real, 1, INT_NEQS>& ydot,
                              int* n_rhs = nullptr)
{
    // this is a version of the initial timestep estimator from VODE
    // we come in with int_state..y() storing the initial solution

    // initial lower and upper bounds on the timestep

    constexpr int int_neqs = integrator_neqs<BurnT>();
    constexpr amrex::Real PT1 = 0.1_rt;

    const amrex::Real tdist = std::abs(int_state.tout - int_state.t);
    const amrex::Real tround = std::numeric_limits<amrex::Real>::epsilon() * amrex::max(std::abs(int_state.t), std::abs(int_state.tout));
    const amrex::Real posneg = int_state.tout >= int_state.t ? 1.0_rt : -1.0_rt;

    amrex::Real hL = 100.0_rt * tround;
    amrex::Real hU = PT1 * tdist;

    // initial guess for the itteraction

    amrex::Array1D<amrex::Real, 1, INT_NEQS> y_old;
    for (int n = 1; n <= int_neqs; ++n) {
        y_old(n) = int_state.y(n);
    }

    for (int n = 1; n <= int_neqs; ++n) {
       amrex::Real atol{};
#ifdef PHOTOCHEMISTRY
       if (n <= NumSpec) {
           atol = int_state.atol_spec;
       } else if (n == net_ienuc) {
           atol = int_state.atol_enuc;
       } else {
           atol = int_state.atol_rad_num;
       }
#else
       atol = n == net_ienuc ? int_state.atol_enuc : int_state.atol_spec;
#endif
       const amrex::Real dely = PT1 * std::abs(y_old(n)) + atol;
       const amrex::Real af = std::abs(ydot(n));
       if (af * hU > dely) {
           hU = dely / af;
       }
    }

    amrex::Real h = std::sqrt(hL * hU);
    amrex::Real h_old = 10.0_rt * h;

    // Iterate on ddydtt = (RHS(t + h, y + h * dydt) - dydt) / h

    amrex::Array1D<amrex::Real, 1, INT_NEQS> ewt;
    amrex::Array1D<amrex::Real, 1, INT_NEQS> ydot_temp;
    amrex::Array1D<amrex::Real, 1, INT_NEQS> ddydtt;

    for (int n = 1; n <= 4; n++) {

       h_old = h;

       // Get the error weighting -- this is similar to VODE's dewset
       // routine

       for (int ii = 1; ii <= int_neqs; ii++) {
           amrex::Real rtol{};
           amrex::Real atol{};
#ifdef PHOTOCHEMISTRY
           if (ii <= NumSpec) {
               rtol = int_state.rtol_spec;
               atol = int_state.atol_spec;
           } else if (ii == net_ienuc) {
               rtol = int_state.rtol_enuc;
               atol = int_state.atol_enuc;
           } else {
               rtol = int_state.rtol_rad_num;
               atol = int_state.atol_rad_num;
           }
#else
           rtol = ii == net_ienuc ? int_state.rtol_enuc : int_state.rtol_spec;
           atol = ii == net_ienuc ? int_state.atol_enuc : int_state.atol_spec;
#endif
           ewt(ii) = 1.0_rt / (rtol * std::abs(y_old(ii)) + atol);
       }

       // Construct the trial point.

       const amrex::Real h_signed = posneg * h;
       for (int ii = 1; ii <= int_neqs; ii++) {
           int_state.y(ii) = y_old(ii) + h_signed * ydot(ii);
       }

       // Call the RHS, then estimate the finite difference.

       rhs(int_state.t + h_signed, burn_state, int_state, ydot_temp);
       if (n_rhs != nullptr) {
           *n_rhs += 1;
       }

       for (int ii = 1; ii <= int_neqs; ii++) {
           ddydtt(ii) = (ydot_temp(ii) - ydot(ii)) / h_signed;
       }

       amrex::Real yddnorm = 0.0_rt;
       for (int ii = 1; ii <= int_neqs; ii++) {
           yddnorm += amrex::Math::powi<2>(ddydtt(ii) * ewt(ii));
       }
       yddnorm = std::sqrt(yddnorm / int_neqs);

       if (yddnorm*hU*hU > 2.0_rt) {
          h = std::sqrt(2.0_rt / yddnorm);
       } else {
          h = std::sqrt(h * hU);
       }

       if (h_old < 2.0_rt * h && h_old > 0.5_rt * h) {
           break;
       }

    }

    // Save the final timestep, with a bias factor.

    amrex::Real dt = h / 2.0_rt;
    dt = amrex::Clamp(dt, hL, hU);

    dt = amrex::min(dt, integrator_rp::ode_max_dt);

    // restore the old time solution
    for (int n = 1; n <= int_neqs; ++n) {
        int_state.y(n) = y_old(n);
    }

    return dt;
}

#endif
