// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "TwoStepABDMobilityField.h"
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

/*! \file TwoStepABDMobilityField.h
    \brief Contains code for the TwoStepABDMobilityField class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param T Temperature set point as a function of time
    \param seed Random seed to use in generating random numbers
    \param use_lambda If true, gamma=lambda*diameter, otherwise use a per-type gamma via setGamma()
    \param lambda Scale factor to convert diameter to gamma
    \param noiseless_t If set true, there will be no translational noise (random force)
    \param noiseless_r If set true, there will be no rotational noise (random torque)
    \param rcon Radius of circular confinement centered at the origin
    \param alpha Decay length of mobility
    \param v0 propulsion speed 
    \param td_field If set true, the translational mobility will depend on the distance from the wall
    \param rd_field If set true, the rotational mobility will depend on the distance from the wall
*/
TwoStepABDMobilityField::TwoStepABDMobilityField(std::shared_ptr<SystemDefinition> sysdef,
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
                           )
  : TwoStepLangevinBase(sysdef, group, T, seed, use_lambda, lambda),
    m_noiseless_t(noiseless_t), m_noiseless_r(noiseless_r), m_rcon(rcon), m_alpha(alpha), m_v0(v0), m_td_field(td_field), m_rd_field(rd_field)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepABDMobilityField" << endl;
    }

TwoStepABDMobilityField::~TwoStepABDMobilityField()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepABDMobilityField" << endl;
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1

    The integration method here is from the book "The Langevin and Generalised Langevin Approach to the Dynamics of
    Atomic, Polymeric and Colloidal Systems", chapter 6.
*/
void TwoStepABDMobilityField::integrateStepOne(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    // profile this step
    if (m_prof)
        m_prof->push("ABDMobilityField step 1");

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

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        unsigned int ptag = h_tag.data[j];

        // Initialize the RNG
        RandomGenerator rng(RNGIdentifier::TwoStepABDMobilityField, m_seed, ptag, timestep);

        // compute the random force
        UniformDistribution<Scalar> uniform(Scalar(-1), Scalar(1));
        NormalDistribution<Scalar> stdnormal(Scalar(1.0));
        Scalar rand = stdnormal(rng);

        Scalar gamma;
        if (m_use_lambda)
            gamma = m_lambda*h_diameter.data[j];
        else
            {
            unsigned int type = __scalar_as_int(h_pos.data[j].w);
            gamma = h_gamma.data[type];
            }
        if (D > 2)
        {
		    // compute the random force
		    Scalar randx = uniform(rng);
		    Scalar randy = uniform(rng);
		    Scalar randz = uniform(rng);

		    // compute the bd force (the extra factor of 3 is because <rx^2> is 1/3 in the uniform -1,1 distribution
		    // it is not the dimensionality of the system
		    Scalar coeff = fast::sqrt(Scalar(3.0)*Scalar(2.0)*gamma*currentTemp/m_deltaT);
		    if (m_noiseless_t)
		        coeff = Scalar(0.0);
		    Scalar Fr_x = randx*coeff;
		    Scalar Fr_y = randy*coeff;
		    Scalar Fr_z = randz*coeff;

		    // update position
		    h_pos.data[j].x += (h_net_force.data[j].x + Fr_x) * m_deltaT / gamma;
		    h_pos.data[j].y += (h_net_force.data[j].y + Fr_y) * m_deltaT / gamma;
		    h_pos.data[j].z += (h_net_force.data[j].z + Fr_z) * m_deltaT / gamma;

		    // particles may have been moved slightly outside the box by the above steps, wrap them back into place
		    box.wrap(h_pos.data[j], h_image.data[j]);

		    // draw a new random velocity for particle j
            Scalar mass =  h_vel.data[j].w;
            Scalar sigma = fast::sqrt(currentTemp/mass);
		    NormalDistribution<Scalar> normal(sigma);
		    h_vel.data[j].x = normal(rng);
		    h_vel.data[j].y = normal(rng);
		    h_vel.data[j].z = normal(rng);

        }

		// 2D brownian integrator with position dependent mobility
		else
		{
            // get orientation
            Scalar qx = h_orientation.data[j].x;
            Scalar qw = h_orientation.data[j].w;
            Scalar half_theta = atan2(qw,qx);
            Scalar theta = 2*half_theta;

            // obtain position dependent mobility
            Scalar mu0 = Scalar(1.0)/gamma;
            Scalar mu = mu0;
            Scalar dmu_dh = Scalar(0.0); 
            Scalar dh_dx = Scalar(0.0); 
            Scalar dh_dy = Scalar(0.0); 

 	        // obtain position of particle
   			Scalar rx = h_pos.data[j].x;
    		Scalar ry = h_pos.data[j].y;
            Scalar r2 = (rx*rx) + (ry*ry);
            Scalar r = fast::sqrt(r2);
            Scalar h = fast::sqrt( (m_rcon*m_rcon) - (r+0.5)*(r+0.5) ); // particle diameter is taken to be unity

		    if (m_td_field)
			{
                dh_dx = - rx*(r+0.5) / (h*r);
                dh_dy = - ry*(r+0.5) / (h*r);
                mu = mu0*tanh(h/m_alpha);
                dmu_dh = mu0 * ( 1 - tanh(h/m_alpha)*tanh(h/m_alpha) )/m_alpha;
			}
             
		    // compute the bd force 
		    NormalDistribution<Scalar> stdnormal(Scalar(1.0));
            Scalar randx = stdnormal(rng);
            Scalar randy = stdnormal(rng);
		   	Scalar coeff_rand = fast::sqrt(Scalar(2.0)*currentTemp*mu/m_deltaT);
		    if (m_noiseless_t)
			{
		        coeff_rand = Scalar(0.0);
			}

			// get deterministic part of speed 
			Scalar vxl = m_v0*fast::cos(theta) + mu*h_net_force.data[j].x + currentTemp*dmu_dh*dh_dx;
			Scalar vyl = m_v0*fast::sin(theta) + mu*h_net_force.data[j].y + currentTemp*dmu_dh*dh_dy;

		    // update position
			h_pos.data[j].x += (vxl + coeff_rand*randx)*m_deltaT;
			h_pos.data[j].y += (vyl + coeff_rand*randy)*m_deltaT;
			h_pos.data[j].z += Scalar(0.0);

		    // particles may have been moved slightly outside the box by the above steps, wrap them back into place
		    box.wrap(h_pos.data[j], h_image.data[j]);

		    // draw a new random velocity for particle j
            Scalar mass =  h_vel.data[j].w;
            Scalar sigma = fast::sqrt(currentTemp/mass);
		    NormalDistribution<Scalar> normal(sigma);
		    h_vel.data[j].x = vxl; 
		    h_vel.data[j].y = vyl; 
		    h_vel.data[j].z = Scalar(0.0); 
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

	            // compute the bd torque 

                Scalar3 sigma_r = make_scalar3(fast::sqrt(Scalar(2.0)*gamma_r.x*currentTemp/m_deltaT),
                                               fast::sqrt(Scalar(2.0)*gamma_r.y*currentTemp/m_deltaT),
                                               fast::sqrt(Scalar(2.0)*gamma_r.z*currentTemp/m_deltaT));
                if (m_noiseless_r)
                    sigma_r = make_scalar3(0,0,0);

                vec3<Scalar> bf_torque;
                bf_torque.x = (sigma_r.x)*stdnormal(rng);
                bf_torque.y = (sigma_r.y)*stdnormal(rng);
                bf_torque.z = (sigma_r.z)*stdnormal(rng);

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

                    Scalar mu_r0 = Scalar(1.0)/gamma_r.z;
                    Scalar mu_r = mu_r0; 

                    if (m_rd_field)
                    {
 	                    // obtain position of particle
               			Scalar rx = h_pos.data[j].x;
    		            Scalar ry = h_pos.data[j].y;
                        Scalar r2 = (rx*rx) + (ry*ry);
                        Scalar r = fast::sqrt(r2);
                        Scalar h = fast::sqrt( (m_rcon*m_rcon) - (r+0.5)*(r+0.5) ); // particle diameter is taken to be unity
                        mu_r = mu_r0*tanh(h/m_alpha);
                        bf_torque.z = fast::sqrt(Scalar(2.0)*currentTemp/(m_deltaT*mu_r))*stdnormal(rng);
                    }

                    Scalar qx = h_orientation.data[j].x;
                    Scalar qw = h_orientation.data[j].w;
                    Scalar half_theta = atan2(qw,qx);
                    Scalar theta = 2*half_theta;
        		    // Orientation of particle before applying torque
                    Scalar ori_x = fast::cos(theta);
                    Scalar ori_y = fast::sin(theta);
		            // Orientation of particle after applying torque
        		    Scalar ort_x = ori_x + (t.z*ori_y*mu_r)*m_deltaT;
        		    Scalar ort_y = ori_y - (t.z*ori_x*mu_r)*m_deltaT;
		            // Obtain angle made by particle with x-axis and then apply rotational noise to get the final orientation
        		    // Then, convert to a quaternion and update particle property
        		    Scalar ang_final = atan2(ort_y,ort_x) + (bf_torque.z*mu_r)*m_deltaT;
		            h_orientation.data[j] = make_scalar4(cos(0.5*ang_final), 0.0, 0.0, sin(0.5*ang_final));
                    }

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
void TwoStepABDMobilityField::integrateStepTwo(unsigned int timestep)
    {
    // there is no step 2 in Brownian dynamics.
    }

void export_TwoStepABDMobilityField(py::module& m)
    {
    py::class_<TwoStepABDMobilityField, std::shared_ptr<TwoStepABDMobilityField> >(m, "TwoStepABDMobilityField", py::base<TwoStepLangevinBase>())
    .def(py::init< std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<Variant>,
                            unsigned int,
                            bool,
                            Scalar,
                            bool,
                            bool,
                            Scalar,
                            Scalar,
                            Scalar,
                            bool,
                            bool>())
        ;
    }
