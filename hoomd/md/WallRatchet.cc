#include "WallRatchet.h"

template class WallRatchet<SphereWall>;

void export_SphereWall(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<SphereWall, std::shared_ptr<SphereWall>>(m, "_SphereWall")
    .def(py::init<Scalar,Scalar3,bool>())
    .def_readwrite("radius", &SphereWall::r);
    }
