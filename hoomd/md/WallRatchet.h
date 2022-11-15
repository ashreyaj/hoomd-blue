#ifndef _WALL_RATCHET_H_
#define _WALL_RATCHET_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/ForceCompute.h"
#include "hoomd/ParticleGroup.h"
#include "WallData.h"
#include "hoomd/HOOMDMath.h"
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

//! Applies a harmonic force relative to a WallData object for a group of particles
/*
 * Particles are confined to a surface defined by a WallData object using a harmonic potential that is a
 * function of the distance \a d from this surface:
 *
 * \f[ V(d) = \frac{k}{2} d^2 \f]
 *
 */
template<class T>
class PYBIND11_EXPORT WallRatchet : public ForceCompute
    {
    public:
        //! Constructs the compute
        /*!
         * \param sysdef HOOMD system definition.
         * \param group Particle group to compute on.
         * \param wall Constraining wall.
         * \param k Force constant.
         */
        WallRatchet(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<ParticleGroup> group,
                             std::shared_ptr<T> wall,
                             Scalar k,
                             Scalar chi,
                             Scalar alpha,
                             int nratchets)
            : ForceCompute(sysdef), m_group(group), m_wall(wall), m_k(k), m_chi(chi), m_al(alpha), m_nr(nratchets)
            {
            m_exec_conf->msg->notice(5) << "Constructing WallRatchet[T = " << typeid(T).name() << "]" << std::endl;
            }

        //! Destructor
        virtual ~WallRatchet()
            {
            m_exec_conf->msg->notice(5) << "Destroying WallRatchet[T = " << typeid(T).name() << "]" << std::endl;
            }

        //! Get the force constant
        Scalar getForceConstant() const
            {
            return m_k;
            }

        //! Get the angular force constant
        Scalar getAngularForceConstant() const
            {
            return m_chi;
            }

        //! Get the asymmetry
        Scalar getAsymmetry() const
            {
            return m_al;
            }

        //! Get the number of ratchets
        int getNumRatchets() const
            {
            return m_nr;
            }

        //! Set the force constant
        void setForceConstant(Scalar k)
            {
            m_k = k;
            }

        //! Set the force constant for the angular potential
        void setAngularForceConstant(Scalar chi)
            {
            m_chi = chi;
            }

        //! Set the asymmetry of the potential
        void setAsymmetry(Scalar alpha)
            {
            m_al = alpha;
            }

        //! Set the number of ratchets
        void setNumRatchets(int nratchets)
            {
            m_nr = nratchets;
            }

        //! Get the wall object for this class
        std::shared_ptr<T> getWall() const
            {
            return m_wall;
            }

        //! Set the wall object for this class
        void setWall(std::shared_ptr<T> wall)
            {
            m_wall = wall;
            }

    protected:
        std::shared_ptr<ParticleGroup> m_group; //!< Group to apply forces to
        std::shared_ptr<T> m_wall;              //!< WallData object for restraint
        Scalar m_k;                             //!< Spring constant
        Scalar m_chi;                           //!< Force constant of angular potential
        Scalar m_al;                            //!< Asymmetry parameter
        int m_nr;                               //!< Force constant of angular potential

        //! Compute the forces
        /*!
         * \param timestep Current timestep
         *
         * Harmonic+ratchet forces are computed on all particles in group based on their distance from the surface and the angular coordinate.
         */
        virtual void computeForces(unsigned int timestep)
            {
            ArrayHandle<unsigned int> h_group(m_group->getIndexArray(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
            ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::read);
            const BoxDim box = m_pdata->getGlobalBox();

            ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

            // zero the forces and virial before calling
            memset((void*)h_force.data, 0, sizeof(Scalar4)*m_force.getNumElements());
            memset((void*)h_virial.data, 0, sizeof(Scalar)*m_virial.getNumElements());

            const T wall = *m_wall;

            for (unsigned int idx=0; idx < m_group->getNumMembers(); ++idx)
                {
                const unsigned int pidx = h_group.data[idx];

                // unwrapped particle coordinate
                const Scalar4 pos = h_pos.data[pidx];
                const int3 image = h_image.data[pidx];
                const Scalar3 r = make_scalar3(pos.x, pos.y, pos.z);

                // vector to point from surface (inside is required but not used by this potential)
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
                  Scalar phi = 2.0*M_PI/m_nr;

                  // Polar angle
                  Scalar theta = atan2(pos.y,pos.x);
                  Scalar theta1 = (theta < 0) ? (theta + (2.0*M_PI)) : theta;
                  theta1 = std::fmod(theta1, phi); 
                   
                  // Angle upto which force decreases
                  Scalar phi1 = m_al*phi;

                  Scalar f_theta = 0.0; 
                  // Angle-dependent force 
                  if (theta1 < (phi1))
                  {
                    f_theta = -m_chi*theta1;
                  }
                  else
                  {
                    Scalar chi2 = m_chi*phi1 / (phi-phi1);
                    f_theta = chi2*theta1 - (m_chi+chi2)*phi1;
                  }

                  // harmonic force points along the point-to-wall vector (cancellation of minus signs)
                  f = make_scalar3(m_k*dr.x - std::sin(theta)*f_theta, m_k*dr.y + std::cos(theta)*f_theta, 0.0);

                  // squared-distance gives energy due to harmonic potential
                  energy = Scalar(0.5)*m_k*dot(dr,dr);

                  // virial is dyadic product of force with position (in this box)
                  virial[0] = f.x * r.x;
                  virial[1] = f.x * r.y;
                  virial[2] = f.x * r.z;
                  virial[3] = f.y * r.y;
                  virial[4] = f.y * r.z;
                  virial[5] = f.z * r.z;
                }

                h_force.data[pidx] = make_scalar4(f.x, f.y, f.z, energy);
                for (unsigned int j=0; j < 6; ++j)
                    h_virial.data[m_virial_pitch*j+pidx] = virial[j];
                }
            }
    };

//! Exports the WallRatchet to python
/*!
 * \param m Python module to export to.
 * \param name Name for the potential.
 */
template<class T>
void export_WallRatchet(pybind11::module& m, const std::string& name)
    {
    namespace py = pybind11;
    py::class_<WallRatchet<T>,std::shared_ptr<WallRatchet<T>>>(m,name.c_str(),py::base<ForceCompute>())
    .def(py::init<std::shared_ptr<SystemDefinition>,std::shared_ptr<ParticleGroup>,std::shared_ptr<T>,Scalar,Scalar,Scalar,int>())
    .def("getForceConstant", &WallRatchet<T>::getForceConstant)
    .def("setForceConstant", &WallRatchet<T>::setForceConstant)
    .def("getAngularForceConstant", &WallRatchet<T>::getAngularForceConstant)
    .def("setAngularForceConstant", &WallRatchet<T>::setAngularForceConstant)
    .def("getAsymmetry", &WallRatchet<T>::getAsymmetry)
    .def("setAsymmetry", &WallRatchet<T>::setAsymmetry)
    .def("getNumRatchets", &WallRatchet<T>::getNumRatchets)
    .def("setNumRatchets", &WallRatchet<T>::setNumRatchets)
    .def("getWall", &WallRatchet<T>::getWall)
    .def("setWall", &WallRatchet<T>::setWall)
    ;
    }

//! Exports the SphereWall to python
/*!
 * \param m Python module to export to.
 */
void export_SphereWall(pybind11::module& m);

#endif // _WALL_RATCHET_H_
