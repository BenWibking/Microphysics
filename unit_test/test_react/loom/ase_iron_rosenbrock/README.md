# Loom burn kernel: he-burn/ase-iron + helmholtz + Rosenbrock (Rodas5P) + chabrier1998 (neutrinos off)

Translation of the Microphysics Strang burn for one configuration into the C++
subset that Loom's `loom-import-cxx` accepts (Loom 0.1.0), following the
`loom-burn-kernel` skill (`.claude/skills/loom-burn-kernel/`). Agent notes:
`agent-notes/08-loom-ase-iron-rosenbrock.md` (the network translation was
done in `07-loom-ase-iron-vode`, whose VODE burn does not fit Loom's 16-bit
block count).

Configuration (fixed):
`NETWORK_DIR=he-burn/ase-iron EOS_DIR=helmholtz INTEGRATOR_DIR=Rosenbrock SCREEN_METHOD=chabrier1998 USE_NEUTRINOS=FALSE`,
`integrator.rosenbrock_tableau=0` (Rodas5P; the host launcher rejects 1,
ROS2S), Strang, no NSE, `XTRA_CXXFLAGS=-ffp-contract=off`, inputs
`inputs_ase_iron` (rho 1e7-1e9, T 3e9-7e9, Ni56 + He4, tmax 1e-6). The kernel
follows the HIP build's code path (validate the stock CPU path with
`integrator.use_jacobian_caching=0`).

## Files

| File | Contents | Source |
|---|---|---|
| `support.H`, `f64_math*.H`, `dual.H`, `eos.H`, `burn_kernel.cpp` | platform layer (+ `exp10`), OCML recipes, POD autodiff, helmholtz, kernel root | skill `assets/` |
| `network_data.H`, `burn.H` | generated (`gen_network_data.py`, pynucastro mode) | generator |
| `pyn_rates.H`, `pyn_network.H` | generated (`gen_pynucastro_net.py`) | generator |
| `screen.H`, `network.H` | chabrier1998 + fast_atan (branch-free); network glue + util/linpack.H LU | `../ase_iron_vode/` (fast_atan rewritten here) |
| `strang.H` | Strang glue (rhs, jac, clean_state, setup, cleanup, ...) on `StrangIntT*` | extracted from the shipped `vode.H` |
| `rosenbrock.H` | `rosenbrock_t`, Rodas5P tables, `rosenbrock_integrator` and its helpers | hand-translated |
| `burner.H` | burner, retry (one `actual_integrator` call site) | `../ase_iron_vode/` |
| `loom_params_port.H` | host fill_*; rejects `integrator.rosenbrock_tableau != 0` | hand-written |
| `abi_layout.H` | `gen_abi_layout.py` | generator |
| `validate_*.H` | dual, EOS, rates, network (from `../ase_iron_vode/`), integrator (Rosenbrock), kernel | |

## Value-identical rewrites (beyond `../ase_iron_vode/README.md`)

- Rodas5P only; its constexpr coefficient functions are namespace-scope
  constexpr tables (printed from the stock functions; same under x87).
- Stage vectors ak1..ak8 are rows of one `Array2D ak(i, n)` (stock
  `stage()` / `stage_vector()` return references through if-chains);
  `rhs()` into ak1 and `solve()` on a stage go through a row copy.
- `std::isnan` / `std::isinf` -> `v != v` / `abs(v) == inf`.
- Strang glue on a type alias, not a template: Loom 0.1.0 rejects
  `eos_re_extra_t eos_state{};` inside a function template instantiation
  (PARSE/036; probe t09 vs t10 in the agent notes).
- burner: one `actual_integrator` call site (two-pass loop).
- fast_atan / fast_atan_1 branch-free (ternary selects in the stock order).
- Partition functions: compile-time species, no memoization, no redundant
  find_index (`gen_pynucastro_net.py`, `get_partition_function_indexed`).
- Burn retry in the kernel root: each attempt rebuilds the zone state from the
  global inputs and writes its outputs (a `burn_t` live across the inlined
  integrator exhausts registers, BACKEND/005).
- Rosenbrock loops single-exit, with flags (TARGET/034); `MPL_ASSUME` on the
  stage indices (SUBRANGE/026).

## Performance rewrites (value-identical)

From the 2026-10-30 campaign (agent notes, "Performance campaign"); each one
is bit-exact against stock HIP on all 4096 zones.

- Screening CSE (`gen_pynucastro_net.py --screen-cse`): chabrier1998
  `helmholtz_F` once per distinct `Gamma_e z^(5/3)` (28 of 117 per call);
  `actual_log_screen_pre_r/_d` take the precomputed values.
- Partition-function precompute (`--pf-precompute`): the 30 species' values
  once per `fill_rates`, in `pf_cache.data`.
- Stage solve in place on the `ak` rows (`rb_dgesl_stage`, `rb_solve_stage`),
  with no row copy.

The pynucastro files are generated with
`gen_pynucastro_net.py --net networks/he-burn/ase-iron --emit-dir <dir> --screen-cse --pf-precompute`.

| Kernel | Compile | VGPR / scratch | Loom / stock HIP ms |
|---|---|---|---|
| port before the campaign | 424 s | 256 / 50,480 B | 60,000 / 32,300 = 1.86 |
| this tree (md5 deed71e3) | 367 s | 256 / 51,640 B | 38,600 / 32,850 = 1.18 |

Timings on moth (gfx90a MI210, ROCm 10.1), `inputs_ase_iron` at full tmax,
stock and Loom in the same process. Stock hipcc uses 512 registers (VGPR +
AGPR) and 22 KB scratch; Loom 0.1.0 is capped at 256 VGPRs with no AGPR
spill space. The candidates tried and rejected are in the agent notes.

## Validation

`agent-notes/08-loom-ase-iron-rosenbrock/scripts/val.sh <set> "<-DLOOM_VALIDATE_* flags>"`
(builds `loomvalrb_<set>` with INTEGRATOR_DIR=Rosenbrock, runs it with
`inputs_ase_iron integrator.use_jacobian_caching=0 unit_test.loom_validate=1`).
