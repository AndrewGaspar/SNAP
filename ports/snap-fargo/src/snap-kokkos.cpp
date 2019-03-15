// Task Definitions Include
#include <snap-defs.hpp>

// STL Includes
#include <algorithm>
#include <cmath>
#include <iostream>

// Third Party Includes
#include <Kokkos_Core.hpp>

// Imports
using Kokkos::ALL;
using Kokkos::deep_copy;
using Kokkos::subview;

void snap::inner_convergence(std::int32_t nx,
                             std::int32_t ny,
                             std::int32_t nz,
                             std::int32_t ng,
                             std::int32_t ng_per_thread,
                             std::int32_t inno,
                             double tolr,
                             Kokkos::View<std::int32_t const *, Kokkos::LayoutLeft> grp_act,
                             Kokkos::View<double const ****, Kokkos::LayoutLeft> flux0,
                             Kokkos::View<double ****, Kokkos::LayoutLeft> flux0pi,
                             Kokkos::View<std::int32_t *, Kokkos::LayoutLeft> iits,
                             Kokkos::View<double *, Kokkos::LayoutLeft> dfmxi) {
    Kokkos::View<double ****, Kokkos::LayoutLeft> df(
        "snap::inner_convergence::df", nx, ny, nz, ng_per_thread);

    Kokkos::parallel_for(ng_per_thread, KOKKOS_LAMBDA(std::int32_t n) {
        auto const g = grp_act(n) - 1;
        if (g == -1) return;

        iits(g) = inno;

        auto const df_n = subview(df, ALL(), ALL(), ALL(), n);
        auto const flux0pi_g = subview(flux0pi, ALL(), ALL(), ALL(), g);
        auto const flux0_g = subview(flux0, ALL(), ALL(), ALL(), g);

        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    if (std::abs(flux0pi_g(i, j, k)) < tolr) {
                        flux0pi_g(i, j, k) = 1.0;
                        df_n(i, j, k) = 0.0;
                    } else {
                        df_n(i, j, k) = 1.0;
                    }

                    df_n(i, j, k) = std::abs(flux0_g(i, j, k) / flux0pi_g(i, j, k) - df_n(i, j, k));
                }
            }
        }

        double maxval = df_n(0, 0, 0);
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    maxval = std::max(maxval, df_n(i, j, k));
                }
            }
        }
        dfmxi(g) = maxval;
    });
}