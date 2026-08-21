import numpy as np
import pandas as pd
import abc
import os

import dolfin as df
import dolfin_adjoint as dfa

from .. import utils
from .. import FEM_utils

class Setup(abc.ABC):

    def __init__(self, options, dim, FEM_options = dict()):
        """
        Abstract base class for finite-element test-case setups.

        A setup defines the geometry, mesh, function space, loading, boundary
        conditions, and optional reaction-force measurements associated with a
        mechanical test case.

        Derived classes must implement the initialization logic, the naming
        conventions used for output folders and load identifiers, and the
        problem-specific finite-element setup.

        Parameters
        ----------
        options : mapping
            Test-case-specific options, such as geometric parameters or mesh
            settings.
        dim : int
            Spatial dimension of the problem.
        FEM_options : mapping, optional
            Options controlling the finite-element discretization, such as the
            polynomial order. Default is an empty dictionary.

        Notes
        -----
        Subclasses implementing reaction-force measurements are expected to define
        ``self.my_ds`` as a tagged boundary integration measure and
        ``self.reaction_force_tags`` as the list of boundary tags on which the
        reaction forces are evaluated.
        """

        self.options = options
        self.FEM_options = FEM_options
        self.dim = dim
        
        self.body_force = np.zeros(self.dim)
        self.has_reaction_forces = False

        self._initialization()

    @abc.abstractmethod
    def _initialization(self):
        """
        Perform test-case-specific initialization.

        This method is called automatically by the base-class constructor.
        Derived classes should use it to initialize test-case metadata and
        configuration, such as labels, number of loads, geometry parameters,
        and whether reaction forces are available.
        """
        pass

    @abc.abstractmethod
    def get_folder_base(self):
        """
        Return the test-case-specific base folder identifier.

        The returned string is used to distinguish geometries or discretization
        configurations when constructing paths for meshes and simulation
        outputs.
        """
        pass

    @abc.abstractmethod
    def get_load_suffix(self, load):
        """
        Return a string identifier associated with a load value, used to identify 
        the load case in output paths or files.
        """
        pass

    @abc.abstractmethod
    def setup(self, load, dolfin_adjoint = False):
        """
        Build the finite-element problem for a given load case.

        This method must initialize the mesh, displacement function space,
        boundary conditions, and external loading required by the elasticity
        solver.

        In particular, derived classes are expected to define at least
        ``self.mesh``, ``self.V``, ``self.get_BC``, and ``self.get_load``.

        Parameters
        ----------
        load : scalar
            Load parameter defining the current mechanical test.
        dolfin_adjoint : bool, optional
            If True, construct differentiable FEniCS objects compatible with
            ``dolfin_adjoint`` where applicable. Default is False.

        Notes
        -----
        ``self.get_BC`` should return a pair ``(bc, bc_homo)``, where ``bc``
        contains the Dirichlet boundary conditions for the displacement and
        ``bc_homo`` contains their homogeneous counterparts for Newton
        increments.

        ``self.get_load`` should be callable as
        ``get_load(v, u, F, coeff=1.0)`` and return the external virtual-work
        contribution.

        Setups supporting reaction-force computation should additionally define
        ``self.my_ds`` and ``self.reaction_force_tags``.
        """
        pass

    def solve(self, material, load, compute_reaction_forces = True, dolfin_adjoint = False, do_export = False, do_export_pvd = False, use_hashed_path = False, **kwargs):
        """
        Set up and solve the elasticity problem for a given material and load.

        The test-case-specific finite-element problem is first constructed by
        calling :meth:`setup`. The resulting problem is then solved through
        ``FEM_utils.solve_elasticity``. Reaction forces may optionally be
        computed and selected results may be exported to disk.

        Parameters
        ----------
        material : MaterialLaw
            Constitutive material model used in the elasticity problem.
        load : scalar
            Load parameter defining the mechanical test.
        compute_reaction_forces : bool, optional
            If True, compute reaction forces when supported by the setup.
            Default is True.
        dolfin_adjoint : bool, optional
            If True, use the dolfin-adjoint-compatible formulation where
            applicable. Default is False.
        do_export : bool, optional
            If True, export available solution data, degrees of freedom, and
            reaction forces to CSV files. Default is False.
        do_export_pvd : bool, optional
            If True and ``do_export`` is enabled, export finite-element fields
            to PVD files. Default is False.
        use_hashed_path : bool, optional
            If True, use hashed output paths when constructing the data
            directory. Default is False.
        **kwargs
            Additional keyword arguments forwarded to
            ``FEM_utils.solve_elasticity``.

        Returns
        -------
        results : dict
            Dictionary returned by ``FEM_utils.solve_elasticity``. Depending
            on the requested options, it may additionally contain the entry
            ``'reaction_forces'``.
        """

        solution_file = None
        if do_export:
            folder_base = utils.get_path_data(self, material, load, use_hashed_path = use_hashed_path)
            if not os.path.exists(folder_base):
                os.makedirs(folder_base)
            data_file            = folder_base + '/data.csv'
            dof_file             = folder_base + '/dof.csv'
            reaction_forces_file = folder_base + '/reaction_forces.csv'
            if do_export_pvd:
                solution_file        = folder_base + '/solution'

        self.setup(load, dolfin_adjoint = dolfin_adjoint)

        results = FEM_utils.solve_elasticity(self.mesh, self.V, material, self.get_load, self.get_BC, 
                                                 dolfin_adjoint = dolfin_adjoint, pvd_file = solution_file, **kwargs)

        if compute_reaction_forces:
            results['reaction_forces'] = self.get_reaction_forces(results['F'], material, dolfin_adjoint = dolfin_adjoint)

        if do_export:
            if 'data' in results:
                pd.DataFrame(results['data']).to_csv(data_file, index = False)
                print('exported %s' % data_file)
            if 'dof' in results:
                pd.DataFrame({'dof': results['dof']}).to_csv(dof_file, index = False)
                print('exported %s' % dof_file)
            if self.has_reaction_forces and compute_reaction_forces:
                pd.DataFrame({'forces': results['reaction_forces']}).to_csv(reaction_forces_file, index = False)
                print('exported %s' % reaction_forces_file)

        return results

    def get_reaction_forces(self, F, material, dolfin_adjoint = False):
        """
        Compute normal reaction forces on selected boundary portions.

        For each boundary tag listed in ``self.reaction_force_tags``, this
        method integrates the normal component of the first
        Piola--Kirchhoff traction, over the corresponding boundary portion.

        Parameters
        ----------
        F : ufl.core.expr.Expr
            Deformation gradient associated with the current solution.
        material : MaterialLaw
            Constitutive material model used to compute the first
            Piola--Kirchhoff stress tensor.
        dolfin_adjoint : bool, optional
            If True, use dolfin-adjoint-compatible assembly and material
            coefficients. Default is False.

        Returns
        -------
        list
            Reaction-force values, one for each entry of
            ``self.reaction_force_tags``. If reaction-force computation is not
            supported by the setup, an empty list is returned.
        """

        if not self.has_reaction_forces:
            return []
        if dolfin_adjoint:
            dfx = dfa
            format = 'fenics-adjoint'
        else:
            dfx = df
            format = 'fenics'
        normal = df.FacetNormal(self.mesh)
        P = material.piola(F, self.dim, format = format)
        return [dfx.assemble(df.dot(df.dot(P, normal), normal) * self.my_ds(tag)) for tag in self.reaction_force_tags]

factory = dict()

def register(setup, setup_class):
    """
    Register a test-case setup class in the setup factory.
    """
    factory[setup] = setup_class