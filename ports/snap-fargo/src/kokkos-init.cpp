#include <Kokkos_Core.hpp>

extern "C" void kokkos_init() {
    Kokkos::InitArguments args;
    // 8 (CPU) threads per NUMA region
    args.num_threads = 8;

    Kokkos::initialize(args);
}

extern "C" void kokkos_final() { Kokkos::finalize(); }