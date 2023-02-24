// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "TwoStepBDChannel.h"

#ifndef __TWO_STEP_BDCHANNEL_GPU_H__
#define __TWO_STEP_BDCHANNEL_GPU_H__

/*! \file TwoStepBDChannelGPU.h
    \brief Declares the TwoStepBDChannelGPU class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

//! Implements Brownian dynamics on the GPU
/*! GPU accelerated version of TwoStepBDChannel

    \ingroup updaters
*/
class PYBIND11_EXPORT TwoStepBDChannelGPU : public TwoStepBDChannel
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepBDChannelGPU(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<ParticleGroup> group,
                     std::shared_ptr<Variant> T,
                     unsigned int seed,
                     bool use_lambda,
                     Scalar lambda,
                     Scalar fconst,
                     Scalar y0,
                     Scalar width,
                     bool noiseless_t,
                     bool noiseless_r);

        virtual ~TwoStepBDChannelGPU() {};

        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

    protected:
        unsigned int m_block_size;               //!< block size
    };

//! Exports the TwoStepBDChannelGPU class to python
void export_TwoStepBDChannelGPU(pybind11::module& m);

#endif // #ifndef __TWO_STEP_BDCHANNEL_GPU_H__
