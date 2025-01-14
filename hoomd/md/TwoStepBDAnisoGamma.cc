// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "TwoStepBDAnisoGamma.h"
#include "hoomd/VectorMath.h"
#include "QuaternionMath.h"
#include "hoomd/HOOMDMath.h"

#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"
using namespace hoomd;


#ifdef ENABLE_MPI
#include "hoomd/HOOMDMPI.h"
#endif

namespace py = pybind11;
using namespace std;

/*! \file TwoStepBDAnisoGamma.h
    \brief Contains code for the TwoStepBDAnisoGamma class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param T Temperature set point as a function of time
    \param seed Random seed to use in generating random numbers
    \param use_lambda If true, gamma=lambda*diameter, otherwise use a per-type gamma via setGamma()
    \param lambda Scale factor to convert diameter to gamma
    \param noiseless_t If set true, there will be no translational noise (random force)
    \param noiseless_r If set true, there will be no rotational noise (random torque)
*/
TwoStepBDAnisoGamma::TwoStepBDAnisoGamma(std::shared_ptr<SystemDefinition> sysdef,
                           std::shared_ptr<ParticleGroup> group,
                           std::shared_ptr<Variant> T,
                           unsigned int seed,
                           bool use_lambda,
                           Scalar lambda,
                           bool noiseless_t,
                           bool noiseless_r
                           )
  : TwoStepLangevinBase(sysdef, group, T, seed, use_lambda, lambda),
    m_noiseless_t(noiseless_t), m_noiseless_r(noiseless_r)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepBDAnisoGamma" << endl;
    }

TwoStepBDAnisoGamma::~TwoStepBDAnisoGamma()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepBDAnisoGamma" << endl;
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1

    The integration method here is from the book "The Langevin and Generalised Langevin Approach to the Dynamics of
    Atomic, Polymeric and Colloidal Systems", chapter 6.
*/
void TwoStepBDAnisoGamma::integrateStepOne(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    // profile this step
    if (m_prof)
        m_prof->push("BD step 1");

    // grab some initial variables
    const Scalar currentTemp = m_T->getValue(timestep);
    const unsigned int D = Scalar(m_sysdef->getNDimensions());

    const GlobalArray< Scalar4 >& net_force = m_pdata->getNetForce();
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);

    ArrayHandle<Scalar3> h_gamma_r(m_gamma_r, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_torque(m_pdata->getNetTorqueArray(), access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::host, access_mode::read);

    const BoxDim& box = m_pdata->getBox();

    // perform the first half step
    // r(t+deltaT) = r(t) + (Fc(t) + Fr)*deltaT/gamma
    // v(t+deltaT) = random distribution consistent with T
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        unsigned int ptag = h_tag.data[j];

        // Initialize the RNG
        RandomGenerator rng(RNGIdentifier::TwoStepBDAnisoGamma, m_seed, ptag, timestep);

        // compute the random force
        UniformDistribution<Scalar> uniform(Scalar(-1), Scalar(1));
        Scalar rpar = uniform(rng);
        Scalar rperp = uniform(rng);

        Scalar gamma;
        if (m_use_lambda)
            gamma = m_lambda*h_diameter.data[j];
        else
            {
            unsigned int type = __scalar_as_int(h_pos.data[j].w);
            gamma = h_gamma.data[type];
            }


        // obtain rotation matrices (space->body)
	Scalar qx = h_orientation.data[j].x;
	Scalar qy = h_orientation.data[j].y;
	Scalar qz = h_orientation.data[j].z;
	Scalar qw = h_orientation.data[j].w;
	Scalar half_theta = atan2(qw,qx);
	Scalar theta = 2*half_theta;
	Scalar or_x = fast::cos(theta);
	Scalar or_y = fast::sin(theta);
	Scalar or_z = Scalar(0.0);
	Scalar orp_x = -or_y; // orientation perpendicular to axis in 2D. e_perp = R.e_par where R is the rotation matrix
	Scalar orp_y = or_x;
	Scalar orp_z = Scalar(0.0);


	// In this hack, the value of gamma is actually the value of gamma_par/gamma_perp
	Scalar gamma_perp = Scalar(1.0);
	Scalar gamma_par = gamma * gamma_perp;

    // compute the bd force (the extra factor of 3 is because <rx^2> is 1/3 in the uniform -1,1 distribution
    // it is not the dimensionality of the system
   	Scalar coeff_perp = fast::sqrt(Scalar(3.0)*Scalar(2.0)*currentTemp/(gamma_perp * m_deltaT));
    Scalar coeff_par = fast::sqrt(Scalar(3.0)*Scalar(2.0)*currentTemp/(gamma_par * m_deltaT));
    if (m_noiseless_t)
	{
        coeff_par = Scalar(0.0);
        coeff_perp = Scalar(0.0);
	}
    Scalar vr_x = rpar*coeff_par*or_x + rperp*coeff_perp*orp_x;
    Scalar vr_y = rpar*coeff_par*or_y + rperp*coeff_perp*orp_y;
   	Scalar vr_z = rpar*coeff_par*or_z + rperp*coeff_perp*orp_z;

    if (D < 3)
	{
        vr_z = Scalar(0.0);
	}

	// Convert force from lab frame to body frame so that we can apply the mobility matrix known.
	Scalar fxl = h_net_force.data[j].x;
	Scalar fyl = h_net_force.data[j].y;
	Scalar fzl = h_net_force.data[j].z;

	Scalar fxb = (fxl*or_x) + (fyl*or_y);
	Scalar fyb = -(fxl*or_y) + (fyl*or_x);
	Scalar fzb = Scalar(0.0);

	// Compute velocity in body frame
	Scalar vxb = fxb / gamma_par;
	Scalar vyb = fyb / gamma_perp;
	Scalar vzb = Scalar(0.0);

	// Convert velocity to lab frame
	Scalar vxl = vxb*or_x - vyb*or_y;
	Scalar vyl = vxb*or_y + vyb*or_x;
	Scalar vzl = Scalar(0.0);
        // update position
	h_pos.data[j].x += (vxl + vr_x)*m_deltaT;
	h_pos.data[j].y += (vyl + vr_y)*m_deltaT;
	h_pos.data[j].z += (vzl + vr_z)*m_deltaT;

        // particles may have been moved slightly outside the box by the above steps, wrap them back into place
        box.wrap(h_pos.data[j], h_image.data[j]);

        // draw a new random velocity for particle j
        Scalar mass =  h_vel.data[j].w;
        Scalar sigma = fast::sqrt(currentTemp/mass);
        NormalDistribution<Scalar> normal(sigma);
        h_vel.data[j].x = vxl;
        h_vel.data[j].y = vyl;
        if (D > 2)
            {
            h_vel.data[j].z = normal(rng);
            }
        else
            {
            h_vel.data[j].z = 0;
            }

        // rotational random force and orientation quaternion updates
        if (m_aniso)
            {
            unsigned int type_r = __scalar_as_int(h_pos.data[j].w);
            Scalar3 gamma_r = h_gamma_r.data[type_r];
            if (gamma_r.x > 0 || gamma_r.y > 0 || gamma_r.z > 0)
                {
                vec3<Scalar> p_vec;
                quat<Scalar> q(h_orientation.data[j]);
                vec3<Scalar> t(h_torque.data[j]);
                vec3<Scalar> I(h_inertia.data[j]);

            	// obtain rotation matrices (space->body)
	        rotmat3<Scalar> rotA(conj(q));

                bool x_zero, y_zero, z_zero;
                x_zero = (I.x < EPSILON); y_zero = (I.y < EPSILON); z_zero = (I.z < EPSILON);

	        // compute the bd force (the extra factor of 3 is because <rx^2> is 1/3 in the uniform -1,1 distribution
	        // it is not the dimensionality of the system

                Scalar3 sigma_r = make_scalar3(fast::sqrt(Scalar(3.0)*Scalar(2.0)*gamma_r.x*currentTemp/m_deltaT),
                                               fast::sqrt(Scalar(3.0)*Scalar(2.0)*gamma_r.y*currentTemp/m_deltaT),
                                               fast::sqrt(Scalar(3.0)*Scalar(2.0)*gamma_r.z*currentTemp/m_deltaT));
                if (m_noiseless_r)
                    sigma_r = make_scalar3(0,0,0);

                vec3<Scalar> bf_torque;
                bf_torque.x = (sigma_r.x)*uniform(rng);
                bf_torque.y = (sigma_r.y)*uniform(rng);
                bf_torque.z = (sigma_r.z)*uniform(rng);

                if (x_zero) bf_torque.x = 0;
                if (y_zero) bf_torque.y = 0;
                if (z_zero) bf_torque.z = 0;

                // use the damping by gamma_r and rotate back to lab frame
                // Notes For the Future: take special care when have anisotropic gamma_r
                // if aniso gamma_r, first rotate the torque into particle frame and divide the different gamma_r
                // and then rotate the "angular velocity" back to lab frame and integrate
                bf_torque = rotate(q, bf_torque);
                if (D < 3)
                    {
                    bf_torque.x = 0;
                    bf_torque.y = 0;

                    Scalar qx = h_orientation.data[j].x;
                    Scalar qy = h_orientation.data[j].y;
                    Scalar qz = h_orientation.data[j].z;
                    Scalar qw = h_orientation.data[j].w;
                    Scalar half_theta = atan2(qw,qx);
                    Scalar theta = 2*half_theta;
		    // Orientation of particle before applying torque
                    Scalar ori_x = fast::cos(theta);
                    Scalar ori_y = fast::sin(theta);
                    Scalar ori_z = Scalar(0.0);
		    // Orientation of particle after applying torque
		    Scalar ort_x = ori_x + (t.z*ori_y/gamma_r.x)*m_deltaT;
		    Scalar ort_y = ori_y - (t.z*ori_x/gamma_r.x)*m_deltaT;
		    Scalar ort_z = Scalar(0.0);
		    // Obtain angle made by particle with x-axis and then apply rotational noise to get the final orientation
		    // Then, convert to a quaternion and update particle property
		    Scalar ang_tor = atan2(ort_y,ort_x);
		    Scalar ang_final = atan2(ort_y,ort_x) + (bf_torque.z/gamma_r.x)*m_deltaT;
		    h_orientation.data[j] = make_scalar4(cos(0.5*ang_final), 0.0, 0.0, sin(0.5*ang_final));
                    }

                // do the integration for quaternion
//                q += Scalar(0.5) * m_deltaT * ((t + bf_torque) / vec3<Scalar>(gamma_r)) * q ;
//                q = q * (Scalar(1.0) / slow::sqrt(norm2(q)));
//                h_orientation.data[j] = quat_to_scalar4(q);

                // draw a new random ang_mom for particle j in body frame
                p_vec.x = NormalDistribution<Scalar>(fast::sqrt(currentTemp * I.x))(rng);
                p_vec.y = NormalDistribution<Scalar>(fast::sqrt(currentTemp * I.y))(rng);
                p_vec.z = NormalDistribution<Scalar>(fast::sqrt(currentTemp * I.z))(rng);
                if (x_zero) p_vec.x = 0;
                if (y_zero) p_vec.y = 0;
                if (z_zero) p_vec.z = 0;

                // !! Note this isn't well-behaving in 2D,
                // !! because may have effective non-zero ang_mom in x,y

                // store ang_mom quaternion
                quat<Scalar> p = Scalar(2.0) * q * p_vec;
                h_angmom.data[j] = quat_to_scalar4(p);
                }
            }
        }

    // done profiling
    if (m_prof)
        m_prof->pop();
    }


/*! \param timestep Current time step
*/
void TwoStepBDAnisoGamma::integrateStepTwo(unsigned int timestep)
    {
    // there is no step 2 in Brownian dynamics.
    }

void export_TwoStepBDAnisoGamma(py::module& m)
    {
    py::class_<TwoStepBDAnisoGamma, std::shared_ptr<TwoStepBDAnisoGamma> >(m, "TwoStepBDAnisoGamma", py::base<TwoStepLangevinBase>())
    .def(py::init< std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<Variant>,
                            unsigned int,
                            bool,
                            Scalar,
                            bool,
                            bool>())
        ;
    }
