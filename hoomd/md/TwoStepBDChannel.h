// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "TwoStepLangevinBase.h"

#ifndef __TWO_STEP_BDCHANNEL_H__
#define __TWO_STEP_BDCHANNEL_H__

/*! \file TwoStepLangevin.h
    \brief Declares the TwoStepLangevin class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

//! Integrates part of the system forward in two steps with Brownian dynamics
/*! Implements Brownian dynamics.

    Brownian dynamics modifies the Langevin equation by setting the acceleration term to 0 and assuming terminal
    velocity.

    \ingroup updaters
*/
class PYBIND11_EXPORT TwoStepBDChannel : public TwoStepLangevinBase
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepBDChannel(std::shared_ptr<SystemDefinition> sysdef,
                    std::shared_ptr<ParticleGroup> group,
                    std::shared_ptr<Variant> T,
                    unsigned int seed,
                    bool use_lambda,
                    Scalar lambda,
                    Scalar fconst,
                    Scalar y0,
                    Scalar width,
                    int geometry,
                    bool noiseless_t,
                    bool noiseless_r
                    );

        virtual ~TwoStepBDChannel();

        //! Performs the second step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

    protected:
        Scalar m_fconst;
        Scalar m_y0;
        Scalar m_width;
        int m_geometry;
        bool m_noiseless_t;
        bool m_noiseless_r;
    };

//! Exports the TwoStepLangevin class to python
void export_TwoStepBDChannel(pybind11::module& m);

#endif // #ifndef __TWO_STEP_BDCHANNEL_H__
