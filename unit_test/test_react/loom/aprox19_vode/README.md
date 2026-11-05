# Loom burn kernel: aprox19 + helmholtz + VODE (neutrinos off)

Translation of the Microphysics Strang burn for one configuration into the C++
subset that Loom's `loom-import-cxx` accepts (Loom 0.1.0), following the
`loom-burn-kernel` skill (`.claude/skills/loom-burn-kernel/`). Agent notes:
`agent-notes/06-loom-aprox19-vode.md`.

Configuration (fixed):
`NETWORK_DIR=aprox19 EOS_DIR=helmholtz INTEGRATOR_DIR=VODE USE_NEUTRINOS=FALSE`,
screening `screen5`, Strang, no NSE, `XTRA_CXXFLAGS=-ffp-contract=off`, inputs
`inputs_aprox19`. The kernel follows the **HIP** build's code path: no
`ALLOW_JACOBIAN_CACHING` (validate the stock CPU path with
`integrator.use_jacobian_caching=0`), `AMREX_USE_GPU` branches taken, no host
printing. `network.use_tables` is not ported (the host launcher rejects it).

## Files

| File | Contents | Source |
|---|---|---|
| `support.H`, `f64_math*.H`, `dual.H` | platform layer, OCML f64 recipes, POD autodiff | skill `assets/kernel/` (as is) |
| `eos.H`, `screen.H`, `network_rhs.H`, `vode.H`, `burner.H` | helmholtz, screen5, generic `networks/rhs.H`, VODE, burner | skill `assets/modules/` (as is) |
| `burn_kernel.cpp` | kernel root `mpl::burn_zones` | skill `assets/root/` (as is) |
| `network_data.H`, `burn.H` | **generated** (`scripts/gen_network_data.py`, checked against the stock headers); do not edit | generator |
| `rates.H` | tfactors, the 46 `rates/aprox_rates.H` rates aprox19 reaches (`_r` Real, `_d` dual), `ecapnuc`, `langanke` and its tables (namespace-scope `constexpr`), `evaluate_analytical_rate_{r,d}` | hand-translated |
| `network.H` | aprox19 glue from `actual_network.H`: `NetParams`, `net_analytical_rate_*` (P_to_N / Ni56_to_Fe54 read the state: ecapnuc, langanke), `postprocess_rate_*`, `ener_gener_rate` | hand-translated |
| `validate_*.H` | native bit-exact validators (skill templates); `validate_port.H` holds the aprox19 hooks | skill `assets/validators/` + PORT lines |
| `abi_layout.H` | host-measured `sizeof`/`offsetof` of every `*Params` record | `scripts/gen_abi_layout.py` |
| `tests/import/` | smoke-compile roots for `scripts/loom_smoke.sh` | skill `assets/smoke/` |
| `../../loom_params.H` | host `fill_*_params`, `loom_require_supported_settings()` | skill template, PORT lines filled |

Regenerate the network data (from the Microphysics root):

```bash
uv run --no-project .claude/skills/loom-burn-kernel/scripts/gen_network_data.py \
  --out <scratch> --emit-dir unit_test/test_react/loom/aprox19_vode \
  --support-dir unit_test/test_react/loom/aprox19_vode \
  -- COMP=llvm CXX=clang++ CC=clang NETWORK_DIR=aprox19 EOS_DIR=helmholtz \
     INTEGRATOR_DIR=VODE USE_NEUTRINOS=FALSE
```

## Translation rules

`.claude/skills/loom-burn-kernel/assets/prompts/_common.md` ("Hard rules")
and `references/loom-frontend-rules.md` are binding; `dual.H`'s header is the
only way to write dual arithmetic. In short: statement for statement, HIP
branches, `mpl::math` only, runtime parameters through scalar params structs,
`MPL_ASSUME` for runtime-bounded private-array indices, lint clean.

Port-specific decisions:
- `evaluate_analytical_rate_{r,d}` (rates.H, state-free) returns zeros for
  `P_to_N` and `Ni56_to_Fe54`, like the stock call with a zero `rhs_state`;
  `network.H`'s `net_analytical_rate_{r,d}` evaluates them with
  `burn_state.rho / eta / y_e` and `y`.
- `postprocess_rate` reads `rhs_state.tf.t9`; the hook gets no `tf`, so
  `network.H` recomputes `burn_state.T * 1.0e-9` (the same single multiply
  `get_tfactors` performs, bit-identical).

## Validation

```bash
cd unit_test/test_react
make -j8 COMP=gnu CXX=g++-16 CC=gcc-16 NETWORK_DIR=aprox19 EOS_DIR=helmholtz INTEGRATOR_DIR=VODE \
  USE_NEUTRINOS=FALSE XTRA_CXXFLAGS=-ffp-contract=off USE_LOOM_VALIDATE=TRUE \
  LOOM_KERNEL_DIR=loom/aprox19_vode TMP_BUILD_DIR=tmp_val_<set> EBASE=loomval_<set> \
  XTRA_CPPFLAGS="-DLOOM_VALIDATE_DUAL -DLOOM_VALIDATE_EOS ..."
# run from a scratch directory:
<path>/loomval_<set>3d.gnu.ex inputs_aprox19 n_cell=16 unit_test.loom_validate=1 integrator.use_jacobian_caching=0
```

(`agent-notes/06-loom-aprox19-vode/scripts/val.sh` wraps this.) Results are
recorded in the agent note.
