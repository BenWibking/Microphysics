#ifndef LINPACK_H
#define LINPACK_H

#include <AMReX_REAL.H>
#include <AMReX_Array.H>

#include <ArrayUtilities.H>

template <int num_eqs, bool allow_pivot>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dgesl (const RArray2D& a, const IArray1D& pivot, RArray1D& b)
{

    constexpr int nm1 = num_eqs - 1;

    // solve a * x = b
    // first solve l * y = b
    if constexpr (nm1 >= 1) {
        for (int k = 1; k <= nm1; ++k) {

            amrex::Real t{};
            if constexpr (allow_pivot) {
                int l = pivot(k);
                t = b(l);
                if (l != k) {
                    b(l) = b(k);
                    b(k) = t;
                }
            } else {
                t = b(k);
            }

            AMREX_UNROLL_LOOP(NET_LOOP_UNROLL_LEN)
            for (int j = k+1; j <= num_eqs; ++j) {
                b(j) += t * a(j,k);
            }
        }
    }

    // now solve u * x = y
    for (int kb = 1; kb <= num_eqs; ++kb) {

        const int k = num_eqs + 1 - kb;
        b(k) = b(k) / a(k,k);
        amrex::Real t = -b(k);
        AMREX_UNROLL_LOOP(NET_LOOP_UNROLL_LEN)
        for (int j = 1; j <= k-1; ++j) {
            b(j) += t * a(j,k);
        }
    }

}



template <int num_eqs, bool allow_pivot>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dgefa (RArray2D& a, IArray1D& pivot, int& info)
{

    // dgefa factors a matrix by gaussian elimination.
    // a is returned in the form a = l * u where
    // l is a product of permutation and unit lower
    // triangular matrices and u is upper triangular.

    // gaussian elimination with partial pivoting

    info = 0;
    constexpr int nm1 = num_eqs - 1;

    amrex::Real t;

    if constexpr (nm1 >= 1) {

        for (int k = 1; k <= nm1; ++k) {

            // find l = pivot index
            int l = k;

            if constexpr (allow_pivot) {
                amrex::Real dmax = std::abs(a(k,k));
                AMREX_UNROLL_LOOP(NET_LOOP_UNROLL_LEN)
                for (int i = k+1; i <= num_eqs; ++i) {
                    amrex::Real ai = std::abs(a(i, k));
                    if (ai > dmax) {
                        l = i;
                        dmax = ai;
                    }
                }

                pivot(k) = static_cast<short>(l);
            }

            // zero pivot implies this column already triangularized
            if (a(l,k) != 0.0e0_rt) {

                if constexpr (allow_pivot) {
                    // interchange if necessary
                    if (l != k) {
                        t = a(l,k);
                        a(l,k) = a(k,k);
                        a(k,k) = t;
                    }
                }

                // compute multipliers
                t = -1.0e0_rt / a(k,k);
                AMREX_UNROLL_LOOP(NET_LOOP_UNROLL_LEN)
                for (int j = k+1; j <= num_eqs; ++j) {
                    a(j,k) *= t;
                }

                // row elimination with column indexing
                for (int j = k+1; j <= num_eqs; ++j) {
                    t = a(l,j);

                    if constexpr (allow_pivot) {
                        if (l != k) {
                            a(l,j) = a(k,j);
                            a(k,j) = t;
                        }
                    }

                    AMREX_UNROLL_LOOP(NET_LOOP_UNROLL_LEN)
                    for (int i = k+1; i <= num_eqs; ++i) {
                        a(i,j) += t * a(i,k);
                    }
                }

            } else {
                info = k;
            }

        }

    }

    if constexpr (allow_pivot) {
        pivot(num_eqs) = static_cast<short>(num_eqs);
    }

    if (a(num_eqs,num_eqs) == 0.0e0_rt) {
        info = num_eqs;
    }

}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dgesl5_nopivot (const RArray2D& a, RArray1D& b)
{
    amrex::Real t = b(1);
    b(2) += t * a.arr[1];
    b(3) += t * a.arr[2];
    b(4) += t * a.arr[3];
    b(5) += t * a.arr[4];

    t = b(2);
    b(3) += t * a.arr[7];
    b(4) += t * a.arr[8];
    b(5) += t * a.arr[9];

    t = b(3);
    b(4) += t * a.arr[13];
    b(5) += t * a.arr[14];

    t = b(4);
    b(5) += t * a.arr[19];

    b(5) /= a.arr[24];
    t = -b(5);
    b(1) += t * a.arr[20];
    b(2) += t * a.arr[21];
    b(3) += t * a.arr[22];
    b(4) += t * a.arr[23];

    b(4) /= a.arr[18];
    t = -b(4);
    b(1) += t * a.arr[15];
    b(2) += t * a.arr[16];
    b(3) += t * a.arr[17];

    b(3) /= a.arr[12];
    t = -b(3);
    b(1) += t * a.arr[10];
    b(2) += t * a.arr[11];

    b(2) /= a.arr[6];
    t = -b(2);
    b(1) += t * a.arr[5];

    b(1) /= a.arr[0];
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void dgefa5_nopivot (RArray2D& a)
{
    amrex::Real multiplier = -1.0e0_rt / a.arr[0];
    a.arr[1] *= multiplier;
    a.arr[2] *= multiplier;
    a.arr[3] *= multiplier;
    a.arr[4] *= multiplier;

    amrex::Real t = a.arr[5];
    a.arr[6] += t * a.arr[1];
    a.arr[7] += t * a.arr[2];
    a.arr[8] += t * a.arr[3];
    a.arr[9] += t * a.arr[4];

    t = a.arr[10];
    a.arr[11] += t * a.arr[1];
    a.arr[12] += t * a.arr[2];
    a.arr[13] += t * a.arr[3];
    a.arr[14] += t * a.arr[4];

    t = a.arr[15];
    a.arr[16] += t * a.arr[1];
    a.arr[17] += t * a.arr[2];
    a.arr[18] += t * a.arr[3];
    a.arr[19] += t * a.arr[4];

    t = a.arr[20];
    a.arr[21] += t * a.arr[1];
    a.arr[22] += t * a.arr[2];
    a.arr[23] += t * a.arr[3];
    a.arr[24] += t * a.arr[4];

    multiplier = -1.0e0_rt / a.arr[6];
    a.arr[7] *= multiplier;
    a.arr[8] *= multiplier;
    a.arr[9] *= multiplier;

    t = a.arr[11];
    a.arr[12] += t * a.arr[7];
    a.arr[13] += t * a.arr[8];
    a.arr[14] += t * a.arr[9];

    t = a.arr[16];
    a.arr[17] += t * a.arr[7];
    a.arr[18] += t * a.arr[8];
    a.arr[19] += t * a.arr[9];

    t = a.arr[21];
    a.arr[22] += t * a.arr[7];
    a.arr[23] += t * a.arr[8];
    a.arr[24] += t * a.arr[9];

    multiplier = -1.0e0_rt / a.arr[12];
    a.arr[13] *= multiplier;
    a.arr[14] *= multiplier;

    t = a.arr[17];
    a.arr[18] += t * a.arr[13];
    a.arr[19] += t * a.arr[14];

    t = a.arr[22];
    a.arr[23] += t * a.arr[13];
    a.arr[24] += t * a.arr[14];

    multiplier = -1.0e0_rt / a.arr[18];
    a.arr[19] *= multiplier;

    t = a.arr[23];
    a.arr[24] += t * a.arr[19];
}

#endif
