// Loom import/compile smoke test for network.H: actual_rhs, actual_jac and
// the dense LU (net_dgefa / net_dgesl, pivoting on and off).
#include "../../burn.H"
#include "../../network.H"

namespace mpl {

MPL_KERNEL void smoke_net(const NetParams* params, Real* out, const Real* in, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        burn_t s{};
        s.T = in[2 * idx];
        s.rho = in[2 * idx + 1];
        s.y_e = 0.5;
        s.cv = 1.0e8;
        for (int k = 0; k < NumSpec; ++k) {
            s.xn[k] = 1.0 / NumSpec;
        }
        const NetParams np = *params;
        Array1D<Real, 1, neqs> ydot{};
        actual_rhs(s, np, ydot);
        JacNetArray2D jac{};
        actual_jac(s, np, jac);
        NetPivotArray pivot{};
        int info{};
        if (s.T > 3.0e9) {
            net_dgefa<true>(jac, pivot, info);
            net_dgesl<true>(jac, pivot, ydot);
        } else {
            net_dgefa<false>(jac, pivot, info);
            net_dgesl<false>(jac, pivot, ydot);
        }
        Real acc = static_cast<Real>(info);
        for (int k = 1; k <= neqs; ++k) {
            acc += ydot(k);
        }
        out[idx] = acc;
    }
}

} // namespace mpl
