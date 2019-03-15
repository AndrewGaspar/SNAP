// Task Definitions Include
#include <snap-defs.hpp>

// STL Includes
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

// Third Party Includes
#include <Kokkos_Core.hpp>

// Imports
using Kokkos::ALL;
using Kokkos::deep_copy;
using Kokkos::subview;

void snap::outer_convergence(std::int32_t nx,
                             std::int32_t ny,
                             std::int32_t nz,
                             std::int32_t ng,
                             std::int32_t ng_per_thread,
                             double tolr,
                             ::Kokkos::View<std::int32_t const *, Kokkos::LayoutLeft> grp_act,
                             ::Kokkos::View<double const ****, Kokkos::LayoutLeft> flux0,
                             ::Kokkos::View<double ****, Kokkos::LayoutLeft> flux0po,
                             ::Kokkos::View<double *, Kokkos::LayoutStride> dfmxo) {
    Kokkos::View<double ****, Kokkos::LayoutLeft> df(
        "snap::outer_convergence::df", nx, ny, nz, ng_per_thread);

    Kokkos::parallel_reduce(ng_per_thread,
                            KOKKOS_LAMBDA(std::int32_t n, double &dfmxo) {
                                auto const df_n = subview(df, ALL(), ALL(), ALL(), n);

                                auto const g = grp_act(n) - 1;
                                if (g == -1) {
                                    deep_copy(df_n, -1);
                                    return;
                                }

                                auto const flux0po_g = subview(flux0po, ALL(), ALL(), ALL(), g);
                                auto const flux0_g = subview(flux0, ALL(), ALL(), ALL(), g);

                                for (int k = 0; k < nz; k++) {
                                    for (int j = 0; j < ny; j++) {
                                        for (int i = 0; i < nx; i++) {
                                            if (std::abs(flux0po_g(i, j, k)) < tolr) {
                                                flux0po_g(i, j, k) = 1.0;
                                                df_n(i, j, k) = 0.0;
                                            } else {
                                                df_n(i, j, k) = 1.0;
                                            }

                                            df_n(i, j, k) =
                                                std::abs(flux0_g(i, j, k) / flux0po_g(i, j, k) -
                                                         df_n(i, j, k));
                                        }
                                    }
                                }

                                dfmxo = df_n(0, 0, 0);

                                for (int k = 0; k < nz; k++) {
                                    for (int j = 0; j < ny; j++) {
                                        for (int i = 0; i < nx; i++) {
                                            dfmxo = std::max(dfmxo, df_n(i, j, k));
                                        }
                                    }
                                }
                            },
                            Kokkos::Max<double>(dfmxo(0)));
}