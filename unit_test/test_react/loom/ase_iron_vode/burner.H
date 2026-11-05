// Burner entry point for the Loom burn kernel (see README.md):
// interfaces/burner.H (no NSE), integration/integrator.H,
// integration/VODE/actual_integrator.H, for any rhs.H-framework network + helmholtz + VODE,
// STRANG, HIP branches (see vode.H).
//
// Entry point:
//   MPL_DEVICE void burner(burn_t* state, Real dt, const BurnParams& params, EosTables tab)
// mirrors stock burner(state, dt). BurnParams aggregates the scalar-only
// per-module params structs (read through const references); the helmholtz
// tables stay a by-value EosTables (README rule 8). burn_t is passed by
// pointer because it is written (a mutable struct reference that writes
// scalar fields does not import, probe r15).
#ifndef MPL_BURNER_H
#define MPL_BURNER_H

#include "support.H"
#include "burn.H"
#include "eos.H"
#include "network.H"
#include "vode.H"

namespace mpl {

// All runtime parameters of the burn (scalars only).
struct BurnParams {
    EosParams eos;
    NetParams net;
    IntegratorParams integ;
};

// from integration/VODE/actual_integrator.H:14 actual_integrator
MPL_DEVICE void actual_integrator(burn_t* state, Real dt, bool is_retry,
                                  const IntegratorParams& ip, const EosParams& ep,
                                  const NetParams& np, EosTables tab)
{
    dvode_t vode_state{};
    integrator_setup(state, dt, is_retry, &vode_state, ip, ep, tab);
    state_backup_t state_save = integrator_backup(*state);

    int istate = dvode(state, &vode_state, ip, ep, np, tab);

    integrator_cleanup(&vode_state, state, istate, state_save, dt,
                       vode_final_state_species_failure_tolerance_factor, ip);
}

// from integration/integrator.H:33 integrator + :11 integrator_wrapper
// Stock: integrator_wrapper<use_burn_retry>: actual_integrator(is_retry =
// false); with retry enabled and a failed first attempt, restore the saved
// state and run actual_integrator(is_retry = true) once more. Here both
// attempts go through ONE actual_integrator call site in a two-pass loop (same
// calls, arguments and states in the same order; the saved copy is unused
// without retry). Loom inlines everything (no noinline): the stock shape
// inlines VODE three times, and for a network with a large RHS (ase-iron:
// 6 RHS sites per VODE copy) that overflowed Loom's per-region block count
// (RESOURCE_EXHAUSTED, region block count exceeds UINT16_MAX).
MPL_DEVICE void integrator(burn_t* state, Real dt,
                           const IntegratorParams& ip, const EosParams& ep,
                           const NetParams& np, EosTables tab)
{
    const bool enable_retry = ip.integrator_use_burn_retry != 0;
    const burn_t old_state = *state;

    for (int attempt = 0; attempt < 2; ++attempt) {
        if (attempt == 1) {
            if (!enable_retry || state->success) {
                break;
            }
            *state = old_state;
        }
        const bool is_retry = (attempt == 1);
        actual_integrator(state, dt, is_retry, ip, ep, np, tab);
    }
}

// from interfaces/burner.H:25 burner (no NSE: integrator(state, dt))
// The params sub-structs are copied to locals once, so everything below
// reads locals rather than the kernel-argument buffer.
MPL_DEVICE void burner(burn_t* state, Real dt, const BurnParams& params, EosTables tab)
{
    const EosParams ep = params.eos;
    const NetParams np = params.net;
    const IntegratorParams ip = params.integ;

    integrator(state, dt, ip, ep, np, tab);
}

} // namespace mpl

#endif
