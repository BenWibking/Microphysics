// Rosenbrock integrator for the Loom burn kernel (see README.md):
// integration/Rosenbrock/rosenbrock_type.H, rosenbrock_tableau.H (Rodas5P
// only) and rosenbrock_integrator.H, for STRANG, no NSE, HIP branches, with
// the Strang glue from strang.H and the dense LU (util/linpack.H) from
// network.H (MPL_NET_DENSE_LU).
//
// Translation, statement for statement except (all value-identical):
//  * Rodas5P only: integrator.rosenbrock_tableau must be 0 (the host
//    launcher rejects 1, ROS2S), so rosenbrock_integrator<BurnT, Tableau> is
//    instantiated once. Its constexpr coefficient functions C::a(i, j) etc.
//    are namespace-scope constexpr tables rb_a[i][j] (values printed from the
//    stock functions with %.17g; the same with an x87 long double _rt), read
//    with the stage loops' runtime indices.
//  * The stage vectors ak1..ak8 are rows of one Array2D ak(i, n): the stock
//    stage(rstate, i, n) / stage_vector(rstate, i) return references through
//    an if-chain. rhs() writes ak1 through a local copy, solve() works on a
//    copy of the row (rhs writes every component; dgesl reads and writes only
//    b), so the values are the same.
//  * std::isnan(v) / std::isinf(v) -> v != v / abs(v) == inf;
//    std::numeric_limits<Real>::epsilon() -> UROUND; amrex/std math ->
//    mpl::math; record locals get {} (probe d14).
//  * The in_jacobian argument of rhs only feeds NONAKA_PLOT and is dropped.
#ifndef MPL_ROSENBROCK_H
#define MPL_ROSENBROCK_H

#include "support.H"
#include "burn.H"
#include "eos.H"
#include "network.H"

#if !MPL_NET_DENSE_LU
#error "rosenbrock.H needs network.H's util/linpack.H LU (MPL_NET_DENSE_LU)"
#endif

namespace mpl {

// ---------------------------------------------------------------------------
// runtime parameters (README rule 8)
// ---------------------------------------------------------------------------

// integrator_rp::* read on the Rosenbrock burn path, plus network_rp::small_x
// (normalize_abundances_burn). bool parameters are stored as int (0/1).
struct IntegratorParams {
    Real integrator_rtol_spec;
    Real integrator_rtol_enuc;
    Real integrator_atol_spec;
    Real integrator_atol_enuc;
    Real integrator_retry_rtol_spec;
    Real integrator_retry_rtol_enuc;
    Real integrator_retry_atol_spec;
    Real integrator_retry_atol_enuc;
    int integrator_jacobian;
    int integrator_use_burn_retry;
    int integrator_retry_swap_jacobian;
    int integrator_scale_system;
    int integrator_call_eos_in_rhs;
    int integrator_integrate_energy;
    int integrator_use_number_densities;
    int integrator_subtract_internal_energy;
    int integrator_do_species_clip;
    int integrator_renormalize_abundances;
    int integrator_correct_jacobian_for_const_e;
    int integrator_ode_max_steps;
    Real integrator_ode_max_dt;
    Real integrator_SMALL_X_SAFE;
    Real integrator_MAX_TEMP;
    Real integrator_react_boost;
    Real integrator_species_failure_tolerance;
    Real integrator_X_reject_buffer;
    Real network_small_x;
    int integrator_linalg_do_pivoting;
    int integrator_rosenbrock_tableau;   // must be 0 (Rodas5P; checked on the host)
    Real integrator_h211b_b;
    Real integrator_h211b_k;
    Real integrator_h211b_fac_min;
    Real integrator_h211b_fac_max;
    Real integrator_h211b_reduction_fac;
};

// ---------------------------------------------------------------------------
// integration/integrator_data.H, Rosenbrock constants
// ---------------------------------------------------------------------------

// enum integrator_errors
constexpr int IERR_SUCCESS = 1;
constexpr int IERR_BAD_INPUTS = -1;
constexpr int IERR_DT_UNDERFLOW = -2;
constexpr int IERR_SPRAD_CONVERGENCE = -3;
constexpr int IERR_TOO_MANY_STEPS = -4;
constexpr int IERR_TOO_MUCH_ACCURACY_REQUESTED = -5;
constexpr int IERR_CORRECTOR_CONVERGENCE = -6;
constexpr int IERR_LU_DECOMPOSITION_ERROR = -7;
constexpr int IERR_BAD_STATE_IN_CORRECTOR = -8;

// std::numeric_limits<double>::epsilon()
constexpr Real UROUND = 2.220446049250313080847263336181640625e-16;

// rosenbrock_type.H timestep_safety_factor
constexpr Real timestep_safety_factor = 1.0e-4;

// rosenbrock_tableau.H rodas5p_tableau: stages, gamma, ct(i), b(i), e(i),
// a(i, j), c(i, j) for i, j = 0..8 (index 0 unused)
constexpr int rb_stages = 8;
constexpr Real rb_gamma = 0.21193756319429014;
constexpr Real rb_ct[9] = {0.0, 0.0, 0.63581268958287041, 0.4095798393397535, 0.97693067250607157, 0.42884036095586642, 1.0, 1.0, 1.0};
constexpr Real rb_b[9] = {1.0, -7.5028463993061214, 2.5618461448039191, -11.627539656261098, -0.18268767659942256, 0.030198172008377946, 1.0, 1.0, 1.0};
constexpr Real rb_e[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
constexpr Real rb_a[9][9] = {
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 2.8493943797479391, 0.45842242204463923, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, -6.9540285098091008, 2.489845061869568, -10.358996098473584, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 2.8029986275628964, 0.5072464736228206, -0.3988312541770524, -0.047211872304046408, 0.0, 0.0, 0.0, 0.0},
    {0.0, -7.5028463993061214, 2.5618461448039191, -11.627539656261098, -0.18268767659942256, 0.030198172008377946, 0.0, 0.0, 0.0},
    {0.0, -7.5028463993061214, 2.5618461448039191, -11.627539656261098, -0.18268767659942256, 0.030198172008377946, 1.0, 0.0, 0.0},
    {0.0, -7.5028463993061214, 2.5618461448039191, -11.627539656261098, -0.18268767659942256, 0.030198172008377946, 1.0, 1.0, 0.0},
};
constexpr Real rb_c[9][9] = {
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, -14.155112264123755, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, -17.97296035885952, -2.8596932954512941, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 147.12150275711716, -1.41221402718213, 71.68940251302358, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 165.43517024871676, -0.45928234564911258, 42.909383369586031, -5.9619867215733056, 0.0, 0.0, 0.0, 0.0},
    {0.0, 24.854864614690072, -3.0009227002832186, 47.493111002076802, 5.5814197821558125, -0.66106918252494706, 0.0, 0.0, 0.0},
    {0.0, 30.912732140285989, -3.1208243349937974, 77.799546460708925, 34.286460282947829, -19.097331116725623, -28.087943162872662, 0.0, 0.0},
    {0.0, 37.802771233905631, -3.2571969029072276, 112.26918849496327, 66.934723124404698, -40.066189370910017, -54.667802628779683, -9.4886165230962707, 0.0},
};

// from integration/Rosenbrock/rosenbrock_type.H:24 rosenbrock_t<int_neqs = neqs>
// ak(i, n) = stock ak<i>(n), i = 1..8. Passed by pointer, as dvode_t in vode.H.
struct rosenbrock_t {
    Real t{};
    Real tout{};
    Real dt{};

    int n_step{};
    int n_rhs{};
    int n_jac{};

    Real atol_spec{};
    Real rtol_spec{};

    Real atol_enuc{};
    Real rtol_enuc{};

    Array1D<Real, 1, neqs> y{};
    Array1D<Real, 1, neqs> ynew{};
    Array2D<Real, 1, 8, 1, neqs> ak{};
    Array1D<Real, 1, neqs> work{};

    JacNetArray2D jac{};
    NetPivotArray pivot{};

    short jacobian_type{};
};

// the integrator state type of strang.H's glue
using StrangIntT = rosenbrock_t;

} // namespace mpl

#include "strang.H"

namespace mpl {

// ---------------------------------------------------------------------------
// integration/Rosenbrock/rosenbrock_integrator.H namespace rosenbrock
// ---------------------------------------------------------------------------

// from rosenbrock_integrator.H:130 rtol_for / :137 atol_for
MPL_DEVICE Real rb_rtol_for(const rosenbrock_t* rs, const int n)
{
    return n == net_ienuc ? rs->rtol_enuc : rs->rtol_spec;
}

MPL_DEVICE Real rb_atol_for(const rosenbrock_t* rs, const int n)
{
    return n == net_ienuc ? rs->atol_enuc : rs->atol_spec;
}

// from rosenbrock_integrator.H:144 error_norm
MPL_DEVICE Real rb_error_norm(const rosenbrock_t* rs)
{
    Real err = 0.0;
    for (int n = 1; n < neqs + 1; ++n) {
        const Real sk = rb_atol_for(rs, n) +
            rb_rtol_for(rs, n) * math::max(math::abs(rs->y(n)), math::abs(rs->ynew(n)));
        const Real term = rs->work(n) / sk;
        err += term * term;
    }
    return math::sqrt(err / static_cast<Real>(neqs));
}

// from rosenbrock_integrator.H:158 species_change_valid (STRANG). The early
// `return false` in the loop is a flag (pure check, same result): Loom's
// branch lowering rejects divergent loops with several exits.
MPL_DEVICE bool rb_species_change_valid(const rosenbrock_t* rs, const IntegratorParams& ip)
{
    constexpr Real increase_change_factor = 4.0;
    constexpr Real decrease_change_factor = 0.25;

    bool valid = true;
    for (int i = 1; i < NumSpec + 1; ++i) {
        if (math::abs(rs->y(i)) > ip.integrator_X_reject_buffer * rs->atol_spec &&
            math::abs(rs->ynew(i)) > ip.integrator_X_reject_buffer * rs->atol_spec &&
            (math::abs(rs->ynew(i)) > increase_change_factor * math::abs(rs->y(i)) ||
             math::abs(rs->ynew(i)) < decrease_change_factor * math::abs(rs->y(i)))) {
            valid = false;
        }
    }

    return valid;
}

// std::isnan / std::isinf
MPL_DEVICE bool rb_isnan(const Real v) { return v != v; }
MPL_DEVICE bool rb_isinf(const Real v)
{
    const Real infinity = __builtin_bit_cast(Real, 0x7ff0000000000000ULL);
    return math::abs(v) == infinity;
}

// from rosenbrock_integrator.H:180 matrix_is_finite (early returns -> flag,
// pure check, same result)
MPL_DEVICE bool rb_matrix_is_finite(const JacNetArray2D& matrix)
{
    bool finite = true;
    for (int j = 1; j < neqs + 1; ++j) {
        for (int i = 1; i < neqs + 1; ++i) {
            const Real value = matrix(i, j);
            if (rb_isnan(value) || rb_isinf(value)) {
                finite = false;
            }
        }
    }

    return finite;
}

// from rosenbrock_integrator.H:196 valid_integrator_state (STRANG); reads the
// runtime parameter atol_spec, not rstate's (possibly retry) tolerance, as
// stock. Early returns -> flag (pure check, same result).
MPL_DEVICE bool rb_valid_integrator_state(const Real time, const burn_t& state,
                                          const Array1D<Real, 1, neqs>& y,
                                          const IntegratorParams& ip)
{
    (void) time;
    (void) state;

    bool valid = true;

    for (int n = 1; n < neqs + 1; ++n) {
        const Real yn = y(n);
        if (rb_isnan(yn) || rb_isinf(yn)) {
            valid = false;
        }
    }

    for (int n = 1; n < NumSpec + 1; ++n) {
        if (y(n) < -ip.integrator_atol_spec) {
            valid = false;
        }

        if (!ip.integrator_use_number_densities && y(n) > 1.0) {
            valid = false;
        }
    }

    if (y(net_ienuc) <= 0.0) {
        valid = false;
    }

    return valid;
}

// from rosenbrock_integrator.H:240 evaluate_jacobian
MPL_DEVICE void rb_evaluate_jacobian(burn_t* state, rosenbrock_t* rs, const Real time,
                                     const IntegratorParams& ip, const EosParams& ep,
                                     const NetParams& np, EosTables tab)
{
    {
        Array1D<Real, 1, neqs> ak1{};
        rhs(time, state, rs, ak1, ip, ep, np, tab);
        for (int n = 1; n < neqs + 1; ++n) {
            rs->ak(1, n) = ak1(n);
        }
    }
    rs->n_rhs += 1;

    // stock: `return` once the analytic Jacobian is finite; here a flag
    bool done = false;
    if (rs->jacobian_type == 1) {
        jac(time, state, rs, rs->jac, ip, ep, np, tab);
        rs->n_jac += 1;

        if (rb_matrix_is_finite(rs->jac)) {
            done = true;
        }
    }

    if (done) {
        return;
    }

    Array1D<Real, 1, neqs> ewt{};
    Array1D<Real, 1, neqs> ybase{};

    // rhs() may clean or renormalize all components of rstate.y.
    for (int i = 1; i < neqs + 1; ++i) {
        ybase(i) = rs->y(i);
    }

    Real fac = 0.0;
    for (int i = 1; i < neqs + 1; ++i) {
        ewt(i) = 1.0 / (rb_rtol_for(rs, i) * math::abs(rs->y(i)) + rb_atol_for(rs, i));
        fac += (rs->ak(1, i) * ewt(i)) * (rs->ak(1, i) * ewt(i));
    }
    fac = math::sqrt(fac / static_cast<Real>(neqs));

    Real R0 = 1000.0 * math::abs(rs->dt) *
        UROUND * static_cast<Real>(neqs) * fac;
    if (R0 == 0.0) {
        R0 = 1.0;
    }

    for (int j = 1; j < neqs + 1; ++j) {
        const Real yj = ybase(j);
        const Real R = math::max(math::sqrt(UROUND) * math::abs(yj), R0 / ewt(j));
        for (int n = 1; n < neqs + 1; ++n) {
            rs->y(n) = ybase(n);
        }
        rs->y(j) += R;

        rhs(time, state, rs, rs->work, ip, ep, np, tab);

        fac = 1.0 / R;
        for (int i = 1; i < neqs + 1; ++i) {
            rs->jac(i, j) = (rs->work(i) - rs->ak(1, i)) * fac;
        }

        for (int n = 1; n < neqs + 1; ++n) {
            rs->y(n) = ybase(n);
        }
    }

    rs->n_rhs += neqs;
    rs->n_jac += 1;
}

// from rosenbrock_integrator.H:300 rhs_at
MPL_DEVICE void rb_rhs_at(const Real time, burn_t* state, rosenbrock_t* rs,
                          const Array1D<Real, 1, neqs>& y, Array1D<Real, 1, neqs>& ydot,
                          const IntegratorParams& ip, const EosParams& ep,
                          const NetParams& np, EosTables tab)
{
    Array1D<Real, 1, neqs> ysave{};

    for (int n = 1; n < neqs + 1; ++n) {
        ysave(n) = rs->y(n);
        rs->y(n) = y(n);
    }

    rhs(time, state, rs, ydot, ip, ep, np, tab);

    for (int n = 1; n < neqs + 1; ++n) {
        rs->y(n) = ysave(n);
    }
}

// from rosenbrock_integrator.H:322 decompose
MPL_DEVICE int rb_decompose(rosenbrock_t* rs, const Real fac, const IntegratorParams& ip)
{
    for (int j = 1; j < neqs + 1; ++j) {
        for (int i = 1; i < neqs + 1; ++i) {
            rs->jac(i, j) = -rs->jac(i, j);
        }
        rs->jac(j, j) += fac;
    }

    int ierr_linpack = 0;
    if (ip.integrator_linalg_do_pivoting == 1) {
        constexpr bool allow_pivot{true};
        net_dgefa<allow_pivot>(rs->jac, rs->pivot, ierr_linpack);
    } else {
        constexpr bool allow_pivot{false};
        net_dgefa<allow_pivot>(rs->jac, rs->pivot, ierr_linpack);
    }

    return ierr_linpack;
}

// util/linpack.H dgesl (num_eqs = neqs) on stage row ak(istage, .) in place:
// network.H net_dgesl's statements with b(j) -> rs->ak(istage, j) (same
// arithmetic, same order; no copy of the row through a local array)
template <bool allow_pivot>
MPL_DEVICE void rb_dgesl_stage(rosenbrock_t* rs, const int istage)
{
    MPL_ASSUME(1 <= istage && istage <= 8);
    constexpr int num_eqs = neqs;
    constexpr int nm1 = num_eqs - 1;

    // solve a * x = b
    // first solve l * y = b
    if constexpr (nm1 >= 1) {
        for (int k = 1; k < num_eqs; ++k) {

            Real t{};
            if constexpr (allow_pivot) {
                int l = rs->pivot(k);
                MPL_ASSUME(1 <= l && l <= num_eqs);
                t = rs->ak(istage, l);
                if (l != k) {
                    rs->ak(istage, l) = rs->ak(istage, k);
                    rs->ak(istage, k) = t;
                }
            } else {
                t = rs->ak(istage, k);
            }

            for (int j = k+1; j < num_eqs + 1; ++j) {
                rs->ak(istage, j) += t * rs->jac(j,k);
            }
        }
    }

    // now solve u * x = y
    for (int kb = 1; kb < num_eqs + 1; ++kb) {

        const int k = num_eqs + 1 - kb;
        rs->ak(istage, k) = rs->ak(istage, k) / rs->jac(k,k);
        Real t = -rs->ak(istage, k);
        for (int j = 1; j < k; ++j) {
            rs->ak(istage, j) += t * rs->jac(j,k);
        }
    }
}

// from rosenbrock_integrator.H:345 solve, on stage vector ak<istage>
// (stage_vector(rstate, istage)), in place
MPL_DEVICE void rb_solve_stage(rosenbrock_t* rs, const int istage, const IntegratorParams& ip)
{
    if (ip.integrator_linalg_do_pivoting == 1) {
        rb_dgesl_stage<true>(rs, istage);
    } else {
        rb_dgesl_stage<false>(rs, istage);
    }
}

// from rosenbrock_integrator.H:361 rosenbrock_integrator<BurnT, Tableau = rodas5p_tableau>
MPL_DEVICE int rosenbrock_integrator(burn_t* state, rosenbrock_t* rs,
                                     const IntegratorParams& ip, const EosParams& ep,
                                     const NetParams& np, EosTables tab)
{
    constexpr int stages = rb_stages;

    if (rs->tout == rs->t) {
        return IERR_SUCCESS;
    }

    // stock: `return` from inside the loop; a flag (no side effects in the loop)
    bool too_much_accuracy = false;
    for (int n = 1; n < neqs + 1; ++n) {
        if (rb_atol_for(rs, n) <= 0.0 ||
            rb_rtol_for(rs, n) <= 10.0 * UROUND) {
            too_much_accuracy = true;
        }
    }
    if (too_much_accuracy) {
        return IERR_TOO_MUCH_ACCURACY_REQUESTED;
    }

    rs->n_rhs = 0;
    rs->n_jac = 0;
    rs->n_step = 0;

    rs->dt = rs->tout - rs->t;

    const Real posneg = rs->tout >= rs->t ? 1.0 : -1.0;
    const Real hmax = math::min(ip.integrator_ode_max_dt,
                                math::abs(rs->tout - rs->t));
    Real h = math::min(math::abs(rs->dt), hmax) * posneg;
    if (math::abs(h) <= 10.0 * UROUND) {
        h = 1.e-6 * posneg;
    }

    bool reject = false;
    bool last = false;
    int nsing = 0;
    int n_reject = 0;
    Real facold = 1.0;
    Real errold = 1.0;
    Real hopt = h;
    Real x = rs->t;
    Array1D<Real, 1, neqs> rhs_tmp{};

    // main evolution loop. Single-exit form of the stock loops (Loom's branch
    // lowering rejects divergent loops with several exits or continuations,
    // TARGET/034 divergent_loop_direct_exits / single_continuation): every
    // `return X` sets istate = X (both loops then end, nothing else runs);
    // the inner loop's `continue`s are the else-branches below; its `break`
    // (step accepted) sets step_done; the stage loop's `break` becomes the
    // valid_stage_state guard on the remaining stages. Same statements in the
    // same order.

    int istate = 0;   // 0: integrating

    while (istate == 0) {

        if (rs->n_step > ip.integrator_ode_max_steps) {
            rs->t = x;
            rs->dt = h;
            istate = IERR_TOO_MANY_STEPS;
        } else if (0.1 * math::abs(h) <= math::abs(x) * UROUND) {
            rs->t = x;
            rs->dt = h;
            istate = IERR_DT_UNDERFLOW;
        } else if (last) {
            rs->t = x;
            rs->dt = hopt;
            istate = IERR_SUCCESS;
        } else {

            hopt = h;
            if ((x + h * (1.0 + timestep_safety_factor) - rs->tout) * posneg >= 0.0) {
                h = rs->tout - x;
                last = true;
            }

            // take a step -- this loop will keep trying until a step is
            // successful or a catastrophic error is encountered.

            bool step_done = false;
            while (!step_done && istate == 0) {
                rs->dt = h;

                if (0.1 * math::abs(h) <= math::abs(x) * UROUND) {
                    rs->t = x;
                    rs->dt = h;
                    istate = IERR_DT_UNDERFLOW;
                } else {

                    const Real fac = 1.0 / (h * rb_gamma);
                    rb_evaluate_jacobian(state, rs, x, ip, ep, np, tab);
                    int ierr_linpack = rb_decompose(rs, fac, ip);

                    if (ierr_linpack != 0) {
                        nsing += 1;
                        if (nsing >= 5) {
                            rs->t = x;
                            rs->dt = h;
                            istate = IERR_LU_DECOMPOSITION_ERROR;
                        } else {
                            h *= 0.5;
                            reject = true;
                            last = false;
                            // stock: continue
                        }
                    } else {

                        rb_solve_stage(rs, 1, ip);

                        bool valid_stage_state = true;
                        for (int istage = 2; istage < stages + 1; ++istage) {
                            // bounds for Loom (SUBRANGE/026)
                            MPL_ASSUME(2 <= istage && istage <= 8);
                            if (valid_stage_state) {
                                for (int n = 1; n < neqs + 1; ++n) {
                                    rs->ynew(n) = rs->y(n);
                                    rs->ak(istage, n) = 0.0;

                                    for (int jstage = 1; jstage < istage; ++jstage) {
                                        MPL_ASSUME(1 <= jstage && jstage <= 7);
                                        const Real akj = rs->ak(jstage, n);
                                        rs->ynew(n) += rb_a[istage][jstage] * akj;
                                        rs->ak(istage, n) +=
                                            (rb_c[istage][jstage] / h) * akj;
                                    }
                                }

                                // intermediate stage time
                                Real _time = x + rb_ct[istage] * h;

                                if (!rb_valid_integrator_state(_time, *state, rs->ynew, ip)) {
                                    h *= 0.25;
                                    reject = true;
                                    n_reject += 1;
                                    last = false;
                                    valid_stage_state = false;
                                    // stock: break
                                } else {
                                    rb_rhs_at(x + rb_ct[istage] * h, state, rs, rs->ynew, rhs_tmp, ip, ep, np, tab);
                                    rs->n_rhs += 1;

                                    for (int n = 1; n < neqs + 1; ++n) {
                                        rs->ak(istage, n) += rhs_tmp(n);
                                    }
                                    rb_solve_stage(rs, istage, ip);
                                }
                            }
                        }

                        // stock: if (!valid_stage_state) continue;
                        if (valid_stage_state) {

                            for (int n = 1; n < neqs + 1; ++n) {
                                Real y_update = 0.0;
                                Real err = 0.0;

                                for (int istage = 1; istage < stages + 1; ++istage) {
                                    MPL_ASSUME(1 <= istage && istage <= 8);
                                    const Real aki = rs->ak(istage, n);
                                    y_update += rb_b[istage] * aki;
                                    err += rb_e[istage] * aki;
                                }

                                rs->ynew(n) = rs->y(n) + y_update;
                                rs->work(n) = err;
                            }

                            rs->n_step += 1;

                            Real _time = x + h;
                            const bool valid_state = rb_valid_integrator_state(_time, *state, rs->ynew, ip);
                            if (!valid_state) {
                                reject = true;
                                n_reject += 1;
                                last = false;
                                h *= 0.25;
                                // stock: continue
                            } else {

                                const bool valid_update = rb_species_change_valid(rs, ip);
                                constexpr Real err_min = 1.e-10;
                                Real err = rb_error_norm(rs);
                                Real controller_fac = math::clamp(
                                    math::pow(1.0 / math::max(err, err_min),
                                              1.0 / (ip.integrator_h211b_b * ip.integrator_h211b_k)) *
                                    math::pow(1.0 / math::max(errold, err_min),
                                              1.0 / (ip.integrator_h211b_b * ip.integrator_h211b_k)) *
                                    math::pow(facold, -1.0 / ip.integrator_h211b_b),
                                    ip.integrator_h211b_fac_min, ip.integrator_h211b_fac_max);
                                facold = controller_fac;
                                errold = err;

                                Real hnew = h * controller_fac;

                                if (err <= 1.0 && valid_update) {
                                    for (int n = 1; n < neqs + 1; ++n) {
                                        rs->y(n) = rs->ynew(n);
                                    }
                                    x += h;

                                    if (math::abs(hnew) > hmax) {
                                        hnew = posneg * hmax;
                                    }
                                    if (reject) {
                                        hnew = posneg * math::min(math::abs(hnew), math::abs(h));
                                    }
                                    reject = false;
                                    n_reject = 0;
                                    h = hnew;
                                    step_done = true;   // stock: break
                                } else {

                                    // if we made it here, then we've failed
                                    // cut the timestep in prep for trying again

                                    reject = true;
                                    n_reject += 1;
                                    last = false;
                                    if (n_reject >= 2) {
                                        hnew *= ip.integrator_h211b_reduction_fac;
                                    }
                                    if (valid_update) {
                                        h = posneg * math::min(math::abs(hnew),
                                                               ip.integrator_h211b_reduction_fac * math::abs(h));
                                    } else {
                                        h *= 0.25;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return istate;
}

} // namespace mpl

#endif
