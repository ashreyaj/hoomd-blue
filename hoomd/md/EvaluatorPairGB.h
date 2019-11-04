// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#ifndef __EVALUATOR_PAIR_GB_H__
#define __EVALUATOR_PAIR_GB_H__

#ifndef NVCC
#include <string>
#endif

#define HOOMD_GB_MIN(i,j) ((i > j) ? j : i)
#define HOOMD_GB_MAX(i,j) ((i > j) ? i : j)

#include "hoomd/VectorMath.h"
using namespace hoomd;
using namespace std;

/*! \file EvaluatorPairGB.h
    \brief Defines a an evaluator class for the Gay-Berne potential
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
//! HOSTDEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

struct pair_gb_params
    {
    Scalar epsilon;   //! The energy scale.
    Scalar lperp;     //! The semiaxis length perpendicular to the particle orientation.
    Scalar lpar;      //! The semiaxis length parallel to the particle orientation.

    //! Load dynamic data members into shared memory and increase pointer
    /*! \param ptr Pointer to load data to (will be incremented)
        \param available_bytes Size of remaining shared memory allocation
     */
    HOSTDEVICE void load_shared(char *& ptr, unsigned int &available_bytes) const
        {
        // No-op for this struct since it contains no arrays.
        }
    };


/*!
 * Gay-Berne potential as formulated by Allen and Germano,
 * with shape-independent energy parameter, for identical uniaxial particles.
 */

class EvaluatorPairGB
    {
    public:
        typedef pair_gb_params param_type;

        //! Constructs the pair potential evaluator
        /*! \param _dr Displacement vector between particle centers of mass
            \param _rcutsq Squared distance at which the potential goes to 0
            \param _q_i Quaternion of i^th particle
            \param _q_j Quaternion of j^th particle
            \param _params Per type pair parameters of this potential
        */
        HOSTDEVICE EvaluatorPairGB(const Scalar3& _dr,
                               const Scalar4& _qi,
                               const Scalar4& _qj,
                               const Scalar _rcutsq,
                               const param_type& _params)
            : dr(_dr),rcutsq(_rcutsq),qi(_qi),qj(_qj),
              params(_params)
            {
            }

        //! uses diameter
        HOSTDEVICE static bool needsDiameter()
            {
            return false;
            }

        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        HOSTDEVICE void setDiameter(Scalar di, Scalar dj){}

        //! whether pair potential requires charges
        HOSTDEVICE static bool needsCharge( )
            {
            return false;
            }

        //! Accept the optional diameter values
        //! This function is pure virtual
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        HOSTDEVICE void setCharge(Scalar qi, Scalar qj){}

        //! Evaluate the force and energy
        /*! \param force Output parameter to write the computed force.
            \param pair_eng Output parameter to write the computed pair energy.
            \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the cutoff.
            \param torque_i The torque exerted on the i^th particle.
            \param torque_j The torque exerted on the j^th particle.
            \return True if they are evaluated or false if they are not because we are beyond the cutoff.
        */
        HOSTDEVICE  bool
        evaluate(Scalar3& force, Scalar& pair_eng, bool energy_shift, Scalar3& torque_i, Scalar3& torque_j)
            {
            
            // Set parameters
            Scalar mu = Scalar(1.0);
            Scalar nu = Scalar(2.0);
            Scalar rgb_sig0 = params.lperp;
            Scalar rgb_eps0 = params.epsilon;
            Scalar k1 = params.lpar/params.lperp;
            Scalar k1sq = k1*k1;
            Scalar k2 = Scalar(5.0);
            
            // Derived parameters
            Scalar chi = (k1sq - Scalar(1.0)) / (k1sq + Scalar(1.0));
            Scalar xhi = (pow(k2, (Scalar(1.0)/mu)) - Scalar(1.0)) / (pow(k2, (Scalar(1.0)/mu)) + Scalar(1.0));
            
            Scalar rsq = dot(dr,dr);
            Scalar r = fast::sqrt(rsq);
            vec3<Scalar> sij = dr/r;

    	    // obtain orientations of particles
    	    Scalar4 oa = quat_to_scalar4(qi);
    	    Scalar half_a_angle = atan2(oa.w,oa.x);
    	    Scalar a_angle = Scalar(2.0)*half_a_angle;
    	    vec3<Scalar> u1 = vec3<Scalar>(fast::cos(a_angle),fast::sin(a_angle),Scalar(0.0));
    
    	    Scalar4 ob = quat_to_scalar4(qj);
    	    Scalar half_b_angle = atan2(ob.w,ob.x);
    	    Scalar b_angle = Scalar(2.0)*half_b_angle;
    	    vec3<Scalar> u2 = vec3<Scalar>(fast::cos(b_angle),fast::sin(b_angle),Scalar(0.0));
    	    
    	    Scalar ci	= dot(sij,u1);
    	    Scalar cj	= dot(sij,u2);
	        Scalar cij	= dot(u1,u2);
	        Scalar cp	= ci + cj;
	        Scalar cm	= ci - cj;
	        
	        // Sigma formula
            Scalar cpchi = cp / (Scalar(1.0) + chi*cij);
            Scalar cmchi = cm / (Scalar(1.0) - chi*cij);
            Scalar sigma = rgb_sig0 / sqrt(Scalar(1.0) - Scalar(0.5)*chi*(cp*cpchi+cm*cmchi));

            // Orientation dependant cutoff (See Bott et al PRE 98, 2018)
            Scalar CUTOFFCONSTANT = Scalar(1.12246204830937298); // 2^(1/6)
    	    Scalar rcut  = (CUTOFFCONSTANT - Scalar(1.0)) * rgb_sig0 + sigma;
    	    
    	    // Declare variables for later use
    	    Scalar Fx = Scalar(0.0);
    	    Scalar Fy = Scalar(0.0);
    	    Scalar Gx1 = Scalar(0.0);
    	    Scalar Gy1 = Scalar(0.0);
    	    Scalar Gx2 = Scalar(0.0);
    	    Scalar Gy2 = Scalar(0.0);
    	    
    	    if (r < rcut && rgb_eps0 != Scalar(0.0))
                {
                    // Epsilon formula
                    Scalar eps1    = Scalar(1.0) / sqrt(Scalar(1.0) - (chi*cij*chi*cij));
                    Scalar cpxhi   = cp / (Scalar(1.0) + xhi*cij);
                    Scalar cmxhi   = cm / (Scalar(1.0) - xhi*cij);
                    Scalar eps2    = Scalar(1.0) - Scalar(0.5)*xhi*(cp*cpxhi + cm*cmxhi);
                    Scalar epsilon = rgb_eps0 * pow(eps1,nu) * pow(eps2,mu);
                    
                    // Potential at rij
                    Scalar rho      = (r - sigma + rgb_sig0) / rgb_sig0;
                    Scalar rinv     = Scalar(1.0)/rho;
                    Scalar rho3     = rinv * rinv * rinv;
                    Scalar rho6     = rho3 * rho3;
                    Scalar rho12    = rho6 * rho6;
                    Scalar rhoterm  = Scalar(4.0)*(rho12 - rho6);
                    Scalar drhoterm = -Scalar(24.0) * (Scalar(2.0) * rho12 - rho6) / rho;
                    Scalar pot = epsilon * rhoterm;
                    
                    // Derivatives of sigma
                    Scalar sig_prefac = Scalar(0.5)*chi*sigma*sigma*sigma;
                    Scalar dsig_dci  = sig_prefac*(cpchi+cmchi);
                    Scalar dsig_dcj  = sig_prefac*(cpchi-cmchi);
                    sig_prefac = sig_prefac*(Scalar(0.5)*chi);
                    Scalar dsig_dcij = -sig_prefac*(cpchi*cpchi-cmchi*cmchi);

                    // Derivatives of epsilon
                    Scalar eps_prefac = -mu*xhi*pow(eps1,nu)*pow(eps2,(mu-Scalar(1.0)));
                    Scalar deps_dci  = eps_prefac*(cpxhi+cmxhi);
                    Scalar deps_dcj  = eps_prefac*(cpxhi-cmxhi);
                    eps_prefac = eps_prefac*(Scalar(0.5)*xhi);
                    Scalar deps_dcij = -eps_prefac*(cpxhi*cpxhi-cmxhi*cmxhi);
                    deps_dcij = deps_dcij + nu*chi*chi*(pow(eps1,(nu+Scalar(2.0))))*(pow(eps2,mu))*cij;
    
                    // Derivatives of potential
                    Scalar dpot_drij   = epsilon * drhoterm;
                    Scalar dpot_dci    = rhoterm * deps_dci  - epsilon * drhoterm * dsig_dci;
                    Scalar dpot_dcj    = rhoterm * deps_dcj  - epsilon * drhoterm * dsig_dcj;
                    Scalar dpot_dcij   = rhoterm * deps_dcij - epsilon * drhoterm * dsig_dcij;
                    
                    // Components of directors
                	Scalar u1x	= u1.x;
                	Scalar u1y	= u1.y;
                	Scalar u2x	= u2.x;
                	Scalar u2y	= u2.y;
                    
                    // Forces
                    Fx  = - dpot_drij*sij.x - dpot_dci*(u1x-ci*sij.x)/r - dpot_dcj*(u2x-cj*sij.x)/r;
                    Fy  = - dpot_drij*sij.y - dpot_dci*(u1y-ci*sij.y)/r - dpot_dcj*(u2y-cj*sij.y)/r;
                    
                    // Directional derivatives
                    Gx1  = dpot_dci*sij.x + dpot_dcij*u2x;
                    Gy1  = dpot_dci*sij.y + dpot_dcij*u2y;
                    
                    Gx2  = dpot_dcj*sij.x + dpot_dcij*u1x;
                    Gy2  = dpot_dcj*sij.y + dpot_dcij*u1y;

                if (energy_shift)
                    {
                        // Potential at cutoff
                        rho      = (rcut - sigma + rgb_sig0) / rgb_sig0;
                        rinv     = Scalar(1.0)/rho;
                        rho3     = rinv * rinv * rinv;
                        rho6     = rho3 * rho3;
                        rho12    = rho6 * rho6;
                        Scalar cutterm  = Scalar(4.0)*(rho12 - rho6);
                        Scalar dcutterm = -Scalar(24.0) * (Scalar(2.0) * rho12 - rho6) / rho;
                        pot = pot - epsilon * cutterm;
                        
                        // Derivatives of potential at cutoff
                        dpot_drij   = epsilon * dcutterm;
                        dpot_dci    = cutterm * deps_dci - epsilon * dcutterm * dsig_dci;
                        dpot_dcj    = cutterm * deps_dcj - epsilon * dcutterm * dsig_dcj;
                        dpot_dcij   = cutterm * deps_dcij - epsilon * dcutterm * dsig_dcij;
                    
                        // Adjusting forces accounting for cutoff terms
                        Fx = Fx + dpot_dci*(u1x-ci*sij.x)/r + dpot_dcj*(u2x-cj*sij.x)/r;
                        Fy = Fy + dpot_dci*(u1y-ci*sij.y)/r + dpot_dcj*(u2y-cj*sij.y)/r;
                    
                        // Adjusting torques accounting for cutoff terms
                        Gx1 = Gx1 - (dpot_dci*sij.x + dpot_dcij*u2x);
                        Gy1 = Gy1 - (dpot_dci*sij.y + dpot_dcij*u2y);
                        
                        Gx2 = Gx2 - (dpot_dcj*sij.x + dpot_dcij*u1x);
                        Gy2 = Gy2 - (dpot_dcj*sij.y + dpot_dcij*u1y);
                    }
                }
            else
                return false;

            // compute vector force and torque
            // Force
            vec3<Scalar> F = vec3<Scalar>(Fx,Fy,Scalar(0.0));
            force = vec_to_scalar3(F);
            
            // Torque
            vec3<Scalar> G1 = vec3<Scalar>(Gx1,Gy1,Scalar(0.0));
            vec3<Scalar> G2 = vec3<Scalar>(Gx2,Gy2,Scalar(0.0));
            torque_i = vec_to_scalar3(cross(u1, G1));
            torque_j = vec_to_scalar3(cross(u2, G2));

            return true;
            }

        #ifndef NVCC
        //! Get the name of the potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return "gb";
            }

        std::string getShapeSpec() const
            {
            std::ostringstream shapedef;
            shapedef << "{\"type\": \"Ellipsoid\", \"a\": " << params.lperp <<
                        ", \"b\": " << params.lperp <<
                        ", \"c\": " << params.lpar <<
                        "}";
            return shapedef.str();
            }
        #endif

    protected:
        vec3<Scalar> dr;   //!< Stored dr from the constructor
        Scalar rcutsq;     //!< Stored rcutsq from the constructor
        quat<Scalar> qi;   //!< Orientation quaternion for particle i
        quat<Scalar> qj;   //!< Orientation quaternion for particle j
        const param_type &params;  //!< The pair potential parameters
    };


#undef HOOMD_GB_MIN
#undef HOOMD_GB_MAX
#endif // __EVALUATOR_PAIR_GB_H__
