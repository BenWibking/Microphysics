# Loom burn kernel: he-burn/ase-iron + helmholtz + VODE + chabrier1998 (neutrinos off)

Translation of the Microphysics Strang burn for one configuration into the C++
subset that Loom's `loom-import-cxx` accepts (Loom 0.1.0), following the
`loom-burn-kernel` skill (`.claude/skills/loom-burn-kernel/`). The first port
of a pynucastro-generated network. Agent notes:
`agent-notes/07-loom-ase-iron-vode.md`.

Configuration (fixed):
`NETWORK_DIR=he-burn/ase-iron EOS_DIR=helmholtz INTEGRATOR_DIR=VODE SCREEN_METHOD=chabrier1998 USE_NEUTRINOS=FALSE`,
Strang, no NSE, `XTRA_CXXFLAGS=-ffp-contract=off`, inputs `inputs_ase_iron`
(iron-group grid: rho 1e7-1e9, T 3e9-7e9, Ni56 + He4, tmax 1e-6). The kernel
follows the **HIP** build's code path: no `ALLOW_JACOBIAN_CACHING` (validate
the stock CPU path with `integrator.use_jacobian_caching=0`), `AMREX_USE_GPU`
branches taken, no host printing.

## Files

| File | Contents | Source |
|---|---|---|
| `support.H`, `f64_math*.H`, `dual.H` | platform layer, OCML f64 recipes (+ `exp10`, from `__ocml_exp10_f64`), POD autodiff | skill `assets/kernel/` + `exp10` |
| `eos.H`, `burner.H`, `burn_kernel.cpp` | helmholtz, burner, kernel root | skill `assets/` (as is) |
| `vode.H` | VODE + Strang glue; with `MPL_NET_DENSE_LU` (set by `network.H`) the util/linpack.H dense LU with `integrator.linalg_do_pivoting` and a pivot array | skill `assets/modules/` + dense-LU hook |
| `network_data.H`, `burn.H` | **generated** (`scripts/gen_network_data.py`, pynucastro mode), checked against the stock headers | generator |
| `pyn_rates.H`, `pyn_network.H` | **generated** (`scripts/gen_pynucastro_net.py`): tfactors, rate_t / rate_derivs_t, partition-function and weak-rate tables, every rate function as `_r` / `_d`, fill_rates per namespace; evaluate_screening_{r,d}, rhs_nuc, jac_nuc | generator |
| `screen.H` | chabrier1998 screening, plasma state, Debye-Hueckel, `fast_atan` (Real and dual) | hand-translated (+ shipped screen5 module's plasma state / Debye-Hueckel) |
| `network.H` | `NetParams`, evaluate_rates_{r,d}, ener_gener_rate, actual_rhs, actual_jac, util/linpack.H dgefa / dgesl | hand-translated |
| `loom_params_port.H` | host `fill_*_params`, `loom_require_supported_settings()`; `../../loom_params.H` includes it when present | hand-written |
| `abi_layout.H` | host-measured `sizeof`/`offsetof` of every `*Params` record | `scripts/gen_abi_layout.py` |
| `validate_*.H` | native bit-exact validators: dual, EOS, integrator, kernel (skill templates, integrator adapted for the pivot array and ase-iron hard zones); rates and network rewritten for the pynucastro interface | |
| `tests/import/` | smoke-compile roots: `smoke_rates`, `smoke_net`, `smoke_burn`, `smoke_wave1` (EOS) | |

Regenerate (from the Microphysics root):

```bash
uv run --no-project .claude/skills/loom-burn-kernel/scripts/gen_network_data.py \
  --out <scratch> --emit-dir unit_test/test_react/loom/ase_iron_vode \
  --support-dir unit_test/test_react/loom/ase_iron_vode \
  -- COMP=llvm CXX=clang++ CC=clang NETWORK_DIR=he-burn/ase-iron EOS_DIR=helmholtz \
     INTEGRATOR_DIR=VODE SCREEN_METHOD=chabrier1998 USE_NEUTRINOS=FALSE
uv run --no-project .claude/skills/loom-burn-kernel/scripts/gen_pynucastro_net.py \
  --net networks/he-burn/ase-iron --emit-dir unit_test/test_react/loom/ase_iron_vode
python3 .claude/skills/loom-burn-kernel/scripts/gen_abi_layout.py \
  --kernel-dir unit_test/test_react/loom/ase_iron_vode
```

## Translation decisions (all value-identical; covered by the validators)

- Template rate functions (`T` = `rate_t` / `rate_derivs_t`) are instantiated
  by the generator: `if constexpr (std::is_same_v<T, rate_derivs_t>)` blocks
  kept or dropped, calls renamed to the `_r` / `_d` copy.
- Tables (`inline AMREX_GPU_DEVICE` Array1D / Array3D) are namespace-scope
  `constexpr` arrays read through `template <int ID>` accessors: a
  namespace-scope constexpr object passed by reference does not import
  (probes t01-t03), the id accessor does (t07, t08).
- `pf_cache_t`'s default constructor is `pf_cache_init()` (records must be
  aggregates, probe t04).
- Uninitialized `Real` locals passed as `Real&` out-arguments are
  zero-initialized (probe r25); record locals get `{}` (d14).
- Counted loops use strict upper bounds (`i < n + 1`): Loom proves the index in
  range only then (SUBRANGE/026 in dgesl's back substitution otherwise).
- `const auto& screened_rates = rate_eval.screened_rates` (a reference to a
  field, r10) is replaced by direct field accesses; MathArray2D `set`/`add`
  by `jac(i, j) = / +=`; the `constexpr_for` and `jac_slice_2` lambdas by
  loops in the same order.
- `amrex::Math::exp10` is `std::pow(10.0, x)` on the host (AMReX's host
  branch) and a transcription of ROCm 10.1's `__ocml_exp10_f64` on the
  device; the chabrier1998 / fast_atan gcem constants are exact hex literals
  of the stock compile-time values (the same with an x87 long double PI).

## Validation

```bash
agent-notes/07-loom-ase-iron-vode/scripts/val.sh <set> "<-DLOOM_VALIDATE_... flags>"
# builds loomvalai_<set> (TMP_BUILD_DIR=tmp_valai_<set>) and runs it from a scratch dir with
# inputs_ase_iron integrator.use_jacobian_caching=0 unit_test.loom_validate=1
```

Results are recorded in the agent note.
