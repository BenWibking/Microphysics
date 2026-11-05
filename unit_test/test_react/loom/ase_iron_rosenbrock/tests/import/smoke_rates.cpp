// Loom import/compile smoke test for the pynucastro rates (pyn_rates.H) and
// screening (screen.H): evaluate_rates_r / evaluate_rates_d of network.H at
// one (rho, T) per lane with a fixed composition, every rate and screen
// factor reduced into one output.
#include "../../burn.H"
#include "../../network.H"

namespace mpl {

MPL_KERNEL void smoke_rates(const NetParams* params, Real* out, const Real* in, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        burn_t s{};
        s.T = in[2 * idx];
        s.rho = in[2 * idx + 1];
        s.y_e = 0.5;
        Array1D<Real, 1, NumSpec> Y{};
        for (int k = 1; k <= NumSpec; ++k) {
            Y(k) = 1.0 / (30.0 * k);
        }
        const NetParams np = *params;
        const ScreenParams sp = net_screen_params(np);
        rate_t r{};
        evaluate_rates_r(s, Y, sp, r);
        rate_derivs_t d{};
        evaluate_rates_d(s, Y, sp, d);
        Real acc = r.enuc_weak + d.enuc_weak + d.denuc_weak_dT;
        for (int k = 1; k <= Rates::NumRates; ++k) {
            acc += r.screened_rates(k) + d.screened_rates(k) + d.dscreened_rates_dT(k);
        }
        out[idx] = acc;
    }
}

} // namespace mpl
