// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander
// This file is created by Ashreya using LJ implementation as reference

#ifndef __PAIR_EVALUATOR_WHDFfinite_H__
#define __PAIR_EVALUATOR_WHDFfinite_H__

#ifndef NVCC
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairWHDFfinite.h
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for evaluating the WHDF (finite) pair potential. The potential operates between a lower and upper distance cutoff.
//! For reference, see "The Lennard-Jones potential: when (not) to use it."
//
class EvaluatorPairWHDFfinite
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        typedef Scalar3 param_type;

        //! Constructs the pair potential evaluator
        /*! \param _rsq Squared distance between the particles
            \param _rcutsq Squared distance at which the potential goes to 0
            \param _ronsq Squared distance at which the potential begins to operate
            \param _params Per type pair parameters of this potential
        */
        DEVICE EvaluatorPairWHDFfinite(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
            : rsq(_rsq), rcutsq(_rcutsq), epsilon(_params.x), sigma(_params.y), ron(_params.z)
            {
            }

        //! LJ doesn't use diameter
        DEVICE static bool needsDiameter() { return false; }
        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj) { }

        //! LJ doesn't use charge
        DEVICE static bool needsCharge() { return false; }
        //! Accept the optional diameter values
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        DEVICE void setCharge(Scalar qi, Scalar qj) { }

        //! Evaluate the force and energy
        /*! \param force_divr Output parameter to write the computed force divided by r.
            \param pair_eng Output parameter to write the computed pair energy
            \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the cutoff
            \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are performed
                  in PotentialPair.

            \return True if they are evaluated or false if they are not because we are beyond the cutoff
        */
        DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
            {
            // compute the force divided by r in force_divr
            Scalar ronsq = ron*ron;
            if (ronsq < rsq && rsq <= rcutsq)
                {
                Scalar r2inv = Scalar(1.0)/rsq;
                Scalar r4inv = r2inv * r2inv;
                Scalar sigmasq = sigma*sigma;
                Scalar rcutsq_divsigmasq = rcutsq/sigmasq;
                Scalar alpha = 2*rcutsq_divsigmasq*(Scalar(3.0)/(Scalar(2.0)*(rcutsq_divsigmasq-1)))*(Scalar(3.0)/(Scalar(2.0)*(rcutsq_divsigmasq-1)))*(Scalar(3.0)/(Scalar(2.0)*(rcutsq_divsigmasq-1)));
                force_divr = 2*epsilon*alpha*(rcutsq*r2inv - Scalar(1.0))*(Scalar(2.0)*rcutsq*r4inv*(sigmasq*r2inv - Scalar(1.0))+ sigmasq*r4inv*(rcutsq*r2inv - Scalar(1.0) ));  

                pair_eng = epsilon*alpha*(sigmasq*r2inv-1) *(rcutsq*r2inv-1)*(rcutsq*r2inv-1);

                if (energy_shift)
                    {
                    Scalar rcut2inv = Scalar(1.0)/rcutsq;
                    pair_eng -= epsilon*alpha*(sigmasq*rcut2inv-1) *(rcutsq*rcut2inv-1)*(rcutsq*rcut2inv-1);
                    }
                return true;
                }
            else
                return false;
            }

        #ifndef NVCC
        //! Get the name of this potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("whdffinite");
            }

        std::string getShapeSpec() const
            {
            throw std::runtime_error("Shape definition not supported for this pair potential.");
            }
        #endif

    protected:
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
        Scalar epsilon;     //!< epsilon parameter extracted from the params passed to the constructor
        Scalar sigma;     //!< sigma parameter extracted from the params passed to the constructor
        Scalar ron;     //!< ron parameter extracted from the params passed to the constructor
    };


#endif // __PAIR_EVALUATOR_WHDFfinite_H__
