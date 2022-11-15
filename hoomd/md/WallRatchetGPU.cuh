#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include "WallData.h"

#include <stdio.h>
#include <cuda_runtime.h>

using namespace std;

//! Kernel driver to compute wall ratchet on the GPU
template<class T>
cudaError_t compute_wall_ratchet(Scalar4* forces,
                                   Scalar* virials,
                                   const unsigned int* group,
                                   const Scalar4* positions,
                                   const int3* images,
                                   const BoxDim& box,
                                   const T& wall,
                                   Scalar k,
                                   Scalar chi,
                                   Scalar alpha,
                                   int nratchets,
                                   unsigned int N,
                                   unsigned int virial_pitch,
                                   unsigned int block_size);

// only compile kernels in NVCC
#ifdef NVCC
namespace kernel
{
//! Kernel to compute wall harmonic on the GPU
/*!
 * \param forces Forces on particles
 * \param virials Virial per particle
 * \param group Indexes of particles in the group
 * \param positions Particle positions
 * \param images Particle images
 * \param box Global simulation box
 * \param wall WallData object defining surface
 * \param k Spring constant
 * \param N Number of particles
 * \param virial_pitch Pitch of the virial
 *
 * \tparam T WallData object type
 *
 * One thread per particle. Computes the harmonic potential with spring force \a k
 * based on the distance of the particle in the \a group from the surface defined by
 * the \a wall.
 */
template<class T>
__global__ void compute_wall_ratchet(Scalar4* forces,
                                       Scalar* virials,
                                       const unsigned int* group,
                                       const Scalar4* positions,
                                       const int3* images,
                                       const BoxDim box,
                                       const T wall,
                                       Scalar k,
                                       Scalar chi,
                                       Scalar alpha,
                                       int nratchets,
                                       unsigned int N,
                                       unsigned int virial_pitch)
    {
    // one thread per particle
    const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    const unsigned int pidx = group[idx];

    // unwrapped particle coordinate
    const Scalar4 pos = positions[pidx];
    const int3 image = images[pidx];
    const Scalar3 r = make_scalar3(pos.x, pos.y, pos.z);

    // vector to point from surface
    bool inside;
    vec3<Scalar> dr = vecPtToWall(wall, vec3<Scalar>(box.shift(r, image)), inside);

    // signed distance between the particle and the wall
    const Scalar distToWall = distWall(wall, vec3<Scalar>(box.shift(r, image)));

    Scalar3 f = make_scalar3(0.0, 0.0, 0.0);
    Scalar energy = 0;
    Scalar virial[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    if (distToWall < 0)
    {
        // Angle per ratchet
        Scalar phi = 2.0*M_PI/nratchets;

        // Polar angle
        Scalar theta = atan2(pos.y,pos.x);
        Scalar theta1 = (theta < 0) ? (theta + (2*M_PI)) : theta;
        theta1 = fmod(theta1, phi); 

        // Angle upto which force decreases
        Scalar phi1 = alpha*phi;

        Scalar f_theta = 0.0; 
        // Angle-dependent force 
        if (theta1 < (phi1))
        {
          f_theta = -chi*theta1;
        }
        else
        {
          Scalar chi2 = chi*phi1 / (phi-phi1);
          f_theta = chi2*theta1 - (chi+chi2)*phi1;
        }

        // printf("dtw: %f, x: %f, y: %f, z: %f, drx: %f, dry: %f, drz: %f, theta1: %f, f_theta: %f\n", distToWall,pos.x,pos.y,pos.z,dr.x,dr.y,dr.z,theta1,f_theta);
        // distance between particle and origin
        const Scalar rdist = distToOrigin(wall, vec3<Scalar>(box.shift(r, image)));

        // Potential energy U = (1/2)*k*dr**2 + (1/2)*chi*theta**2
        // polar components of force
        f_theta = f_theta/rdist;
        Scalar f_r = - k*sqrt(dr.x*dr.x+dr.y*dr.y+dr.z*dr.z);

        // harmonic force points along the point-to-wall vector (cancellation of minus signs)
        f = make_scalar3(cos(theta)*f_r - sin(theta)*f_theta, sin(theta)*f_r + cos(theta)*f_theta, 0.0);

        // squared-distance gives energy: irrelavant here!!
        energy = Scalar(0.5)*k*dot(dr,dr);

        // virial is dyadic product of force with position
        Scalar virial[6];
        virial[0] = f.x * r.x;
        virial[1] = f.x * r.y;
        virial[2] = f.x * r.z;
        virial[3] = f.y * r.y;
        virial[4] = f.y * r.z;
        virial[5] = f.z * r.z;
    }

    forces[pidx] = make_scalar4(f.x, f.y, f.z, energy);
    for (unsigned int j=0; j < 6; ++j)
        virials[virial_pitch*j+pidx] = virial[j];
    }
} // end namespace kernel

/*!
 * \param forces Forces on particles
 * \param virials Virial per particle
 * \param group Indexes of particles in the group
 * \param positions Particle positions
 * \param images Particle images
 * \param box Global simulation box
 * \param wall WallData object defining surface
 * \param k Spring constant
 * \param N Number of particles
 * \param virial_pitch Pitch of the virial
 * \param block_size Number of threads per block
 *
 * \tparam T WallData object type
 *
 * \sa kernel::compute_wall_harmonic
 */
template<class T>
cudaError_t compute_wall_ratchet(Scalar4* forces,
                                   Scalar* virials,
                                   const unsigned int* group,
                                   const Scalar4* positions,
                                   const int3* images,
                                   const BoxDim& box,
                                   const T& wall,
                                   Scalar k,
                                   Scalar chi,
                                   Scalar alpha,
                                   int nratchets,
                                   unsigned int N,
                                   unsigned int virial_pitch,
                                   unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)kernel::compute_wall_ratchet<T>);
        max_block_size = attr.maxThreadsPerBlock;
        }

    const unsigned int run_block_size = min(block_size, max_block_size);
    const unsigned int num_blocks = (N+run_block_size-1)/run_block_size;

    kernel::compute_wall_ratchet<<<num_blocks,run_block_size>>>
        (forces, virials, group, positions, images, box, wall, k, chi, alpha, nratchets, N, virial_pitch);

    return cudaSuccess;
    }

#endif
