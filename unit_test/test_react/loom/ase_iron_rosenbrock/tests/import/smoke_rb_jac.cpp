// Loom smoke: Rosenbrock evaluate_jacobian (analytic or numerical, two RHS
// sites) + decompose + one solve, on a burn_t / rosenbrock_t set up as
// integrator_setup does.
#include "../../burner.H"

namespace mpl {

MPL_KERNEL void smoke_rb_jac(const BurnParams* params,
                             const Real* f, const Real* dpdf, const Real* ef, const Real* xf,
                             const Real* d, const Real* t, const Real* dt_sav, const Real* dt2_sav,
                             const Real* dti_sav, const Real* dt2i_sav, const Real* dd_sav,
                             const Real* dd2_sav, const Real* ddi_sav, const Real* dd2i_sav,
                             Real* out, const Real* in, Real dt, int n)
{
    EosTables tab{f, dpdf, ef, xf, d, t, dt_sav, dt2_sav, dti_sav, dt2i_sav,
                  dd_sav, dd2_sav, ddi_sav, dd2i_sav};
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const EosParams ep = params->eos;
        const NetParams np = params->net;
        const IntegratorParams ip = params->integ;
        burn_t s{};
        s.rho = in[2 * idx];
        s.T = in[2 * idx + 1];
        for (int k = 0; k < NumSpec; ++k) { s.xn[k] = 1.0 / NumSpec; }
        s.T_fixed = -1.0;
        rosenbrock_t rs{};
        integrator_setup(&s, dt, false, &rs, ip, ep, tab);
        rs.dt = dt;
        rb_evaluate_jacobian(&s, &rs, 0.0, ip, ep, np, tab);
        const int ierr = rb_decompose(&rs, 1.0 / (dt * rb_gamma), ip);
        rb_solve_stage(&rs, 1, ip);
        out[idx] = rs.ak(1, 1) + rs.ak(1, neqs) + static_cast<Real>(ierr + rs.n_rhs);
    }
}

} // namespace mpl
