// Loom smoke: one Rosenbrock stage (rhs_at + solve) and the step checks.
#include "../../burner.H"

namespace mpl {

MPL_KERNEL void smoke_rb_stage(const BurnParams* params,
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
        for (int k = 1; k < neqs + 1; ++k) { rs.ynew(k) = rs.y(k); }
        Array1D<Real, 1, neqs> rhs_tmp{};
        rb_rhs_at(dt, &s, &rs, rs.ynew, rhs_tmp, ip, ep, np, tab);
        for (int k = 1; k < neqs + 1; ++k) { rs.ak(2, k) = rhs_tmp(k); }
        rb_solve_stage(&rs, 2, ip);
        const bool v = rb_valid_integrator_state(dt, s, rs.ynew, ip) && rb_species_change_valid(&rs, ip);
        out[idx] = rs.ak(2, 1) + rb_error_norm(&rs) + (v ? 1.0 : 0.0);
    }
}

} // namespace mpl
