r"""
Ratchet wall
"""

import numpy
from hoomd import _hoomd
from hoomd.md import _md
import hoomd
from hoomd.md import force

class sphere(force._force):
    def __init__(self, group, radius, origin, k, chi, alpha, nratchets):
        hoomd.util.print_status_line()

        # initialize the base class
        force._force.__init__(self)

        # create the c++ mirror class
        if hoomd.context.exec_conf.isCUDAEnabled():
            _cpp = _md.SphereWallRatchetGPU
        else:
            _cpp = _md.SphereWallRatchet

        # process the parameters
        self._radius = radius
        self._origin = _hoomd.make_scalar3(origin[0],origin[1],origin[2])

        self.cpp_force = _cpp(hoomd.context.current.system_definition,
                              group.cpp_group,
                              _md._SphereWall(self._radius, self._origin, True),
                              k, chi, alpha, nratchets)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

    def set_params(self, radius=None, origin=None, k=None, chi=None, alpha=None, nratchets=None):
        hoomd.util.print_status_line()
        self.check_initialization()

        if radius is not None:
            self._radius = radius
            self.cpp_force.getWall().radius = self._radius

        if origin is not None:
            self._origin = _hoomd.make_scalar3(origin[0],origin[1],origin[2])
            self.cpp_force.setWall(_md._SphereWall(self._radius, self._origin, True))

        if k is not None:
            self.cpp_force.setForceConstant(k)

        if chi is not None:
            self.cpp_force.setAngularForceConstant(chi)

        if alpha is not None:
            self.cpp_force.setAsymmetry(chi)

        if nratchets is not None:
            self.cpp_force.setNumRatchets(nratchets)

    def update_coeffs(self):
        pass
