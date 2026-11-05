// Burner entry point for the Loom burn kernel (see README.md):
// interfaces/burner.H (no NSE), integration/integrator.H,
// integration/Rosenbrock/actual_integrator.H, for a network + helmholtz +
// Rosenbrock (Rodas5P), STRANG, HIP branches (see rosenbrock.H).
//
// Entry point:
//   MPL_DEVICE void burner(burn_t* state, Real dt, const BurnParams& params, EosTables tab)
// mirrors stock burner(state, dt). BurnParams aggregates the scalar-only
// per-module params structs; the helmholtz tables stay a by-value EosTables.
// burn_t is passed by pointer because it is written.
#ifndef MPL_BURNER_H
#define MPL_BURNER_H

#include "support.H"
#include "burn.H"
#include "eos.H"
#include "network.H"
#include "rosenbrock.H"

namespace mpl {

// All runtime parameters of the burn (scalars only).
struct BurnParams {
    EosParams eos;
    NetParams net;
    IntegratorParams integ;
};

// from integration/Rosenbrock/actual_integrator.H:16 actual_integrator
// (integrator_cleanup's final_state_species_failure_tolerance_factor takes its
// default, 1.0)
MPL_DEVICE void actual_integrator(burn_t* state, Real dt, bool is_retry,
                                  const IntegratorParams& ip, const EosParams& ep,
                                  const NetParams& np, EosTables tab)
{
    rosenbrock_t rosenbrock_state{};
    integrator_setup(state, dt, is_retry, &rosenbrock_state, ip, ep, tab);
    state_backup_t state_save = integrator_backup(*state);

    int istate = rosenbrock_integrator(state, &rosenbrock_state, ip, ep, np, tab);

    integrator_cleanup(&rosenbrock_state, state, istate, state_save, dt,
                       1.0, ip);
}

// from integration/integrator.H:33 integrator + :11 integrator_wrapper
// Stock: integrator_wrapper<use_burn_retry>: actual_integrator(is_retry =
// false); with retry enabled and a failed first attempt, restore the saved
// state and run actual_integrator(is_retry = true) once more. Here both
// attempts go through ONE actual_integrator call site in a two-pass loop (same
// calls, arguments and states in the same order; the saved copy is unused
// without retry). Loom inlines everything (no noinline): the stock shape
// inlines the integrator three times, and for a network with a large RHS
// (ase-iron) the VODE port overflowed Loom's 16-bit block count
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

// One burner attempt (actual_integrator with is_retry), for the kernel root,
// which does the retry itself: it rebuilds the zone's initial burn_t from the
// global input arrays for the second attempt instead of keeping a copy live
// across the inlined integrator (that backup, ~120 doubles, made Loom's
// register allocation fail: BACKEND/005 spill-traffic-register-exhausted,
// peak 557-950 VGPRs; without it the burn compiles at 256). The params are
// copied to locals as in burner().
MPL_DEVICE void burner_attempt(burn_t* state, Real dt, bool is_retry,
                               const BurnParams& params, EosTables tab)
{
    const EosParams ep = params.eos;
    const NetParams np = params.net;
    const IntegratorParams ip = params.integ;
    actual_integrator(state, dt, is_retry, ip, ep, np, tab);
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
