// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "TwoStepLangevinBase.h"

#ifndef __TWO_STEP_ABD_MOBILITYFIELD_H__
#define __TWO_STEP_ABD_MOBILITYFIELD_H__

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
class PYBIND11_EXPORT TwoStepABDMobilityField : public TwoStepLangevinBase
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepABDMobilityField(std::shared_ptr<SystemDefinition> sysdef,
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
                    bool rd_field
                    );

        virtual ~TwoStepABDMobilityField();

        //! Performs the second step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

    protected:
        bool m_noiseless_t;
        bool m_noiseless_r;
        Scalar m_rcon;
        Scalar m_alpha;
        Scalar m_v0;
        bool m_td_field;
        bool m_rd_field;
    };

//! Exports the TwoStepLangevin class to python
void export_TwoStepABDMobilityField(pybind11::module& m);

#endif // #ifndef __TWO_STEP_ABD_MOBILITYFIELD_H__
