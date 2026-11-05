// Host-side parameter setup for the Loom burn kernel of he-burn/ase-iron +
// helmholtz + VODE + chabrier1998 (loom/ase_iron_vode; see README.md). Kept
// in the kernel directory: unit_test/test_react/loom_params.H includes it
// in place of its own body when LOOM_KERNEL_DIR has one, so several ports
// can share one test_react checkout. Shared by the native validators and the
// HIP launcher, like loom_params.H.
#ifndef LOOM_PARAMS_PORT_H
#define LOOM_PARAMS_PORT_H

#include <extern_parameters.H>
#include <AMReX.H>
#include <loom_eos_params.H>  // fill_eos_params, fill_eos_tables

#include LOOM_KERNEL_PATH(burner.H)

// Runtime settings the kernel does not implement: abort with a clear message.
inline void loom_require_supported_settings()
{
    // ase-iron is a pynucastro network: network.use_tables and
    // network.use_c12ag_deboer17 are not read on its burn path (stock or
    // kernel), so nothing needs rejecting. Every screening_rp parameter the
    // chabrier1998 path reads is ported (Debye-Hueckel skip, quantum
    // correction), and so are integrator.jacobian = 1 / 2 and
    // integrator.linalg_do_pivoting = 0 / 1.
}

// Network runtime parameters: every mpl::NetParams field.
inline void fill_net_params(mpl::NetParams* p)
{
    p->screening_enable_debye_huckel_skip = screening_rp::enable_debye_huckel_skip ? 1 : 0;
    p->screening_debye_huckel_skip_threshold = screening_rp::debye_huckel_skip_threshold;
    p->screening_enable_chabrier1998_quantum_corr = screening_rp::enable_chabrier1998_quantum_corr ? 1 : 0;
}

// Integrator runtime parameters (+ network_rp::small_x): the VODE ones plus
// linalg_do_pivoting (dense LU, MPL_NET_DENSE_LU).
inline void fill_integrator_params(mpl::IntegratorParams* p)
{
    p->integrator_rtol_spec = integrator_rp::rtol_spec;
    p->integrator_rtol_enuc = integrator_rp::rtol_enuc;
    p->integrator_atol_spec = integrator_rp::atol_spec;
    p->integrator_atol_enuc = integrator_rp::atol_enuc;
    p->integrator_retry_rtol_spec = integrator_rp::retry_rtol_spec;
    p->integrator_retry_rtol_enuc = integrator_rp::retry_rtol_enuc;
    p->integrator_retry_atol_spec = integrator_rp::retry_atol_spec;
    p->integrator_retry_atol_enuc = integrator_rp::retry_atol_enuc;
    p->integrator_jacobian = integrator_rp::jacobian;
    p->integrator_use_burn_retry = integrator_rp::use_burn_retry ? 1 : 0;
    p->integrator_retry_swap_jacobian = integrator_rp::retry_swap_jacobian ? 1 : 0;
    p->integrator_scale_system = integrator_rp::scale_system ? 1 : 0;
    p->integrator_call_eos_in_rhs = integrator_rp::call_eos_in_rhs ? 1 : 0;
    p->integrator_integrate_energy = integrator_rp::integrate_energy ? 1 : 0;
    p->integrator_use_number_densities = integrator_rp::use_number_densities ? 1 : 0;
    p->integrator_subtract_internal_energy = integrator_rp::subtract_internal_energy ? 1 : 0;
    p->integrator_do_species_clip = integrator_rp::do_species_clip ? 1 : 0;
    p->integrator_renormalize_abundances = integrator_rp::renormalize_abundances ? 1 : 0;
    p->integrator_do_corrector_validation = integrator_rp::do_corrector_validation ? 1 : 0;
    p->integrator_correct_jacobian_for_const_e = integrator_rp::correct_jacobian_for_const_e ? 1 : 0;
    p->integrator_ode_max_steps = integrator_rp::ode_max_steps;
    p->integrator_ode_max_dt = integrator_rp::ode_max_dt;
    p->integrator_SMALL_X_SAFE = integrator_rp::SMALL_X_SAFE;
    p->integrator_MAX_TEMP = integrator_rp::MAX_TEMP;
    p->integrator_react_boost = integrator_rp::react_boost;
    p->integrator_species_failure_tolerance = integrator_rp::species_failure_tolerance;
    p->integrator_X_reject_buffer = integrator_rp::X_reject_buffer;
    p->network_small_x = network_rp::small_x;
    p->integrator_linalg_do_pivoting = integrator_rp::linalg_do_pivoting ? 1 : 0;
}

// Everything the burn reads. Aborts on unsupported settings.
inline void fill_burn_params(mpl::BurnParams* p)
{
    loom_require_supported_settings();
    fill_eos_params(&p->eos);
    fill_net_params(&p->net);
    fill_integrator_params(&p->integ);
}

#endif
