#include "WallRatchetGPU.cuh"

// SphereWall template
template cudaError_t compute_wall_ratchet<SphereWall>(Scalar4*,
                                                          Scalar*,
                                                          const unsigned int*,
                                                          const Scalar4*,
                                                          const int3*,
                                                          const BoxDim&,
                                                          const SphereWall&,
                                                          Scalar,
                                                          Scalar,
                                                          Scalar,
                                                          int,
                                                          unsigned int,
                                                          unsigned int,
                                                          unsigned int);
