// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "TwoStepABDMobilityField.h"

#ifndef __TWO_STEP_ABD_MOBILITYFIELD_GPU_H__
#define __TWO_STEP_ABD_MOBILITYFIELD_GPU_H__

/*! \file TwoStepABDMobilityFieldDGPU.h
    \brief Declares the TwoStepABDMobilityFieldGPU class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

//! Implements Brownian dynamics on the GPU
/*! GPU accelerated version of TwoStepABDMobilityField

    \ingroup updaters
*/
class PYBIND11_EXPORT TwoStepABDMobilityFieldGPU : public TwoStepABDMobilityField
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepABDMobilityFieldGPU(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<ParticleGroup> group,
                     std::shared_ptr<Variant> T,
                     unsigned int seed,
                     bool use_lambda,
                     Scalar lambda,
                     bool noiseless_t,
                     bool noiseless_r,
                     Scalar rcon,
                     Scalar alpha,
                     Scalar v0,
                     bool td_field,
                     bool rd_field);

        virtual ~TwoStepABDMobilityFieldGPU() {};

        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

    protected:
        unsigned int m_block_size;               //!< block size
    };

//! Exports the TwoStepABDMobilityFieldGPU class to python
void export_TwoStepABDMobilityFieldGPU(pybind11::module& m);

#endif // #ifndef __TWO_STEP_ABD_MOBILITYFIELD_GPU_H__
