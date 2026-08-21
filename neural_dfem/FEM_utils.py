import numpy as np
import dolfin as df
import dolfin_adjoint as dfa

from . import utils

def compute_invariants(F, material_dim, space_dim):
    """
    Compute the principal invariants associated with the deformation gradient.

    For a three-dimensional material law applied to a two-dimensional
    problem, the invariants are completed according to the plane-strain
    assumption.

    Parameters
    ----------
    F : ufl.core.expr.Expr
        Deformation gradient.
    material_dim : int
        Dimension of the constitutive law. Must be either 2 or 3.
    space_dim : int
        Dimension of the computational domain. Must be either 2 or 3.

    Returns
    -------
    I1 : ufl.core.expr.Expr
        First invariant of the right Cauchy--Green tensor.
    I2 : ufl.core.expr.Expr or None
        Second invariant of the right Cauchy--Green tensor.
        Returns ``None`` for two-dimensional material laws.
    J : ufl.core.expr.Expr
        Determinant of the deformation gradient, ``det(F)``.
    """
    C  = F.T*F # Right Cauchy-Green tensor
    I1 = df.tr(C)
    J  = df.det(F)
    if material_dim == 3:
        if space_dim == 2:
            I1, I2 = utils.plane_strain_invariants(I1, J)
        else:
            I2 = 0.5*(df.tr(C)**2 - df.tr(C*C))
    else:
        I2 = None

    return I1, I2, J

def solve_elasticity(mesh, V, material, load, get_BC,
                     init_dof = None,
                     energy_autodiff = True,
                     dolfin_adjoint = False,
                     builtin_newton = False,
                     do_ramp = True,
                     ramp_value_start = 0.0,
                     ramp_incr_start = 1.0,
                     pvd_file = None, return_dof = False, return_data_table = False):
    """
    Solve a linear or nonlinear elasticity problem.

    Parameters
    ----------
    mesh : dolfin.Mesh
        Computational mesh.
    V : dolfin.FunctionSpace
        Function space for the displacement field.
    material : object
        Material model. The object is expected to provide the attributes
        ``dim`` and ``linear``. For nonlinear problems it must additionally
        provide either ``law(I1, I2, J, format=...)`` or
        ``piola(F, space_dim, format=...)``, depending on the value of
        ``energy_autodiff``.
    load : callable
        Function defining the external virtual work. It is called as
        ``load(v, d, F)`` or, during load ramping,
        ``load(v, d, F, coeff=ramp_val)``.
    get_BC : callable
        Function returning the Dirichlet boundary conditions. For nonlinear
        ramped problems it is called as ``get_BC(ramp_val)`` and must return
        ``(bc, bc_homo)``, where ``bc`` contains the boundary conditions for
        the solution and ``bc_homo`` their homogeneous counterparts used for
        Newton increments.
    init_dof : array_like, optional
        Initial values of the displacement degrees of freedom.
    energy_autodiff : bool, optional
        If True, compute the internal virtual work by differentiating the
        strain-energy functional with respect to the displacement. If False,
        use the first Piola--Kirchhoff stress returned by ``material.piola``.
        Default is True.
    dolfin_adjoint : bool, optional
        If True, use ``dolfin_adjoint`` objects and solvers where applicable.
        Default is False.
    builtin_newton : bool, optional
        If True, use the nonlinear solver provided by FEniCS instead of the
        custom Newton implementation. Default is False.
    do_ramp : bool, optional
        If True, solve nonlinear problems by progressively ramping the load
        from ``ramp_value_start`` to one. Default is True.
    ramp_value_start : float, optional
        Initial value of the load-ramping parameter. Default is 0.0.
    ramp_incr_start : float, optional
        Initial increment of the load-ramping parameter. Default is 1.0.
    pvd_file : str or None, optional
        Prefix used to write the displacement and invariant fields to PVD
        files. If None, no files are written.
    return_dof : bool, optional
        If True, include the displacement degrees of freedom in the returned
        dictionary. Default is False.
    return_data_table : bool, optional
        If True, evaluate the displacement and deformation invariants at mesh
        vertices and include them in the returned dictionary. Default is False.

    Returns
    -------
    results : dict
        Dictionary containing at least:

        ``'d'``
            Computed displacement field.
        ``'F'``
            Deformation gradient associated with the computed displacement.

        Depending on the optional arguments, the dictionary may also contain:

        ``'dof'``
            Array of displacement degrees of freedom.
        ``'data'``
            Dictionary containing displacement components and deformation
            invariants evaluated at mesh vertices.

    Notes
    -----
    If ``space_dim == 2`` and ``material.dim == 3``, the problem is treated
    as plane strain when computing the constitutive invariants.

    """

    if dolfin_adjoint:
        format = 'fenics-adjoint'
        dfx = dfa
    else:
        format = 'fenics'
        dfx = df

    space_dim = mesh.topology().dim()
    assert space_dim    in [2,3], "Only 2D and 3D problems are supported."
    assert material.dim in [2,3], "Only 2D and 3D material laws are supported."
    assert material.dim >= space_dim, "Material dimension must be greater than or equal to space dimension."

    if space_dim == 2 and material.dim == 3: print("Plane strain problem.")

    d = dfx.Function(V, name = 'displacement')
    v = df.TestFunction(V)
    
    if init_dof is not None: d.vector().set_local(init_dof)

    I = df.Identity(space_dim) # Identity tensor
    F = I + df.grad(d)   # Deformation gradient

    if material.linear:
        bc, _ = get_BC()
        u = df.TrialFunction(V)
        df.solve(material.bilinear_form(u, v) == load(v, u, F), d, bc)
    else:
        if energy_autodiff:
            I1, I2, J = compute_invariants(F, material.dim, space_dim)
            internal_energy_variation = df.derivative(material.law(I1, I2, J, format = format)*df.dx, d, v)
        else:
            internal_energy_variation = df.inner(material.piola(F, space_dim, format = format), df.grad(v)) * df.dx
        
        if do_ramp:
            residual = lambda ramp_val: internal_energy_variation - load(v, d, F, coeff = ramp_val)
            Newton_ramp(residual, d, get_BC = get_BC, dolfin_adjoint = dolfin_adjoint, builtin = builtin_newton, 
                        ramp_value_start = ramp_value_start, ramp_incr_start = ramp_incr_start)
        else:
            residual = internal_energy_variation - load(v, d, F)
            bc, _ = get_BC(1.0)
            dfx.solve(residual == 0, d, bc, solver_parameters={ "newton_solver": { "maximum_iterations": 20 }})

    results = {'d': d, 'F': F}

    if return_dof:
        results['dof'] = d.vector().get_local()

    if pvd_file is not None or return_data_table:
        I1, I2, J = compute_invariants(F, material.dim, space_dim)
        S = df.FunctionSpace(mesh, 'Lagrange', 1)
        I1 = df.project(I1, S)
        J  = df.project(J , S)
        I1.rename("I1", "I1")
        J.rename("J", "J")
        if material.dim == 3:
            I2  = df.project(I2 , S)
            I2.rename("I2", "I2")

    if pvd_file is not None:
        df.File(pvd_file + "_d.pvd") << d
        df.File(pvd_file + "_I1.pvd") << I1
        df.File(pvd_file + "_J.pvd") << J
        if material.dim == 3:
            df.File(pvd_file + "_I2.pvd") << I2

    if return_data_table:
        axes = ['x', 'y', 'z']
        data = dict()

        coord = mesh.coordinates()

        if space_dim == 2:
            d_vec  = np.array([d (x,y) for x,y in zip(coord[:,0], coord[:,1])])
            I1_vec = np.array([I1(x,y) for x,y in zip(coord[:,0], coord[:,1])])
            J_vec  = np.array([J (x,y) for x,y in zip(coord[:,0], coord[:,1])])
            if material.dim == 3:
                I2_vec = np.array([I2(x,y) for x,y in zip(coord[:,0], coord[:,1])])
        else:
            d_vec  = np.array([d (x,y,z) for x,y,z in zip(coord[:,0], coord[:,1], coord[:,2])])
            I1_vec = np.array([I1(x,y,z) for x,y,z in zip(coord[:,0], coord[:,1], coord[:,2])])
            I2_vec = np.array([I2(x,y,z) for x,y,z in zip(coord[:,0], coord[:,1], coord[:,2])])
            J_vec  = np.array([J (x,y,z) for x,y,z in zip(coord[:,0], coord[:,1], coord[:,2])])

        for ax in range(space_dim):
            data['d' + axes[ax]] = d_vec[:,ax]

        data['I1'] = I1_vec
        data['J'] = J_vec
        if material.dim == 3:
            data['I2'] = I2_vec

        results['data'] = data

    return results

def Newton(residual, d, bc = None, bc_homo = None,
           dolfin_adjoint = False,
           builtin = False,
           max_iter = 40,
           tol_res = 1e-10,
           res_max = 1e10,
           tol_incr = 1e-8,
           verbose = True):
    """
    Solve a nonlinear variational problem using Newton's method.

    Parameters
    ----------
    residual : ufl.Form
        Residual form of the nonlinear variational problem: ``residual(d; v) = 0``.
    d : dolfin.Function
        Current solution. The function is updated in place during the Newton
        iterations.
    bc : dolfin.DirichletBC or sequence of dolfin.DirichletBC, optional
        Dirichlet boundary conditions imposed on the solution. These are
        applied to ``d`` before starting the Newton iterations.
    bc_homo : dolfin.DirichletBC or sequence of dolfin.DirichletBC, optional
        Homogeneous Dirichlet boundary conditions imposed on the Newton
        correction. They are applied when assembling the residual and
        Jacobian.
    dolfin_adjoint : bool, optional
        If True, use ``dolfin_adjoint`` assembly and solve operations where
        applicable. Default is False.
    builtin : bool, optional
        If True, delegate the nonlinear solve to the built-in FEniCS solver.
        If False, use the custom Newton implementation. Default is False.
    max_iter : int, optional
        Maximum number of Newton iterations. Default is 40.
    tol_res : float, optional
        Absolute tolerance on the Euclidean norm of the assembled residual.
        Default is 1e-10.
    res_max : float, optional
        Maximum admissible residual norm. The solve is considered failed if
        this value is exceeded. Default is 1e10.
    tol_incr : float, optional
        Tolerance on the relative Newton increment,
        ``||delta_d|| / ||d||``. Default is 1e-8.
    verbose : bool, optional
        If True, print convergence information at each Newton iteration.
        Default is True.

    Returns
    -------
    status : int
        Zero if the nonlinear solve converged, negative if it failed.
    message : str
        Empty string on successful convergence, otherwise a short description
        of the failure.

    """

    dfx = dfa if dolfin_adjoint else df

    if builtin:
        try:
            dfx.solve(residual == 0, d, bc)
            return (0, "")
        except Exception as ex:
            return (-1, str(ex))

    else:
        d_trial = df.TrialFunction(d.function_space())
        d_incr = dfx.Function(d.function_space())
        jacobian = df.derivative(residual, d, d_trial)

        if bc is None:
            bc = []
        elif not (isinstance(bc, list) or isinstance(bc, tuple)):
            bc = [bc]
        if bc_homo is None:
            bc_homo = []
        elif not (isinstance(bc_homo, list) or isinstance(bc_homo, tuple)):
            bc_homo = [bc_homo]

        for bc_i in bc:
            bc_i.apply(d.vector())

        res = incr = 0.
        iNewt = 0
        while iNewt < max_iter:
            r = dfx.assemble(residual)
            for bc_i in bc_homo:
                bc_i.apply(r)
            res = r.norm('l2')
            if verbose: print('  Newton iteration %3d --- res = %1.3e' % (iNewt+1, res), end = '')
            if res < tol_res:
                if verbose: print('')
                return (0, '')
            if res > res_max or np.isnan(res):
                if verbose: print('')
                return (-1, 'max_res')
            J = dfx.assemble(jacobian)
            for bc_i in bc_homo:
                bc_i.apply(J, r)
            try:
                dfx.solve(J, d_incr.vector(), -r)
            except Exception as e:
                if verbose: 
                    print('')
                    print(e)
                return (-1, 'solver')
            d.assign(d + d_incr)
            iNewt += 1
            sol = d.vector().norm('l2')
            incr = d_incr.vector().norm('l2')
            if verbose: print(' --- incr/sol = %1.3e' % (incr/sol))
            if incr/sol < tol_incr:
                return (0, '')
        return (-1, "max iter")

def Newton_ramp(residual, d, get_BC = lambda: (None, None),
                dolfin_adjoint = False,
                builtin = False,
                ramp_value_start = 0.0,
                ramp_incr_start = 1.0, ramp_incr_max = 1.0, ramp_incr_min = 1e-16,
                ramp_success_mult = 1.2, ramp_failure_mult = 0.5):

    """
    Solve a nonlinear problem using adaptive load ramping and Newton's method.

    A scalar ramp parameter is progressively increased from
    ``ramp_value_start`` to one. At each ramp value, the corresponding
    nonlinear problem is solved with :func:`Newton`.

    If Newton converges, the ramp increment is increased by
    ``ramp_success_mult``. If Newton fails, the previous converged solution is
    restored and the increment is reduced by ``ramp_failure_mult``. The
    procedure stops successfully when a ramp value of one is reached.

    The last converged displacement field is stored before each new ramp
    attempt. If Newton fails, this state is restored before retrying with a
    smaller load increment.

    Parameters
    ----------
    residual : callable
        Function of the ramp parameter returning the nonlinear residual form,
        i.e. ``residual(ramp_val)``.
    d : dolfin.Function
        Solution field. It is updated in place and used as the initial guess
        for successive ramp steps.
    get_BC : callable, optional
        Function returning ``(bc, bc_homo)`` for a given ramp value,
        ``get_BC(ramp_val)``. ``bc`` contains the boundary conditions for the
        solution and ``bc_homo`` their homogeneous counterparts for the Newton
        correction.
    dolfin_adjoint : bool, optional
        If True, use ``dolfin_adjoint`` functionality where applicable.
        Default is False.
    builtin : bool, optional
        If True, use the built-in FEniCS nonlinear solver within each ramp
        step. Otherwise use the custom Newton solver. Default is False.
    ramp_value_start : float, optional
        Initial value of the ramp parameter. Default is 0.0.
    ramp_incr_start : float, optional
        Initial ramp increment. Default is 1.0.
    ramp_incr_max : float, optional
        Maximum allowed ramp increment. Default is 1.0.
    ramp_incr_min : float, optional
        Minimum allowed ramp increment. An exception is raised if the
        increment falls below this value. Default is 1e-16.
    ramp_success_mult : float, optional
        Factor by which the ramp increment is multiplied after a successful
        Newton solve. Default is 1.2.
    ramp_failure_mult : float, optional
        Factor by which the ramp increment is multiplied after a failed
        Newton solve. Default is 0.5.

    Raises
    ------
    Exception
        If the ramp increment falls below ``ramp_incr_min`` before the full
        load is reached.

    """
        
    dfx = dfa if dolfin_adjoint else df
    
    ramp_val = ramp_value_start
    ramp_incr = min(ramp_incr_start, ramp_incr_max)
    ramp_iter = 1

    d_last = d.vector().get_local()

    while ramp_val < 1:
        ramp_incr = min([ramp_incr, ramp_incr_max, 1. - ramp_val])
        if ramp_incr < ramp_incr_min:
            print("  Ramp increment too small, stopping.")
            raise Exception("Ramp increment too small, stopping.")
        ramp_val += ramp_incr
        print("==========================================================================" )
        print("Ramp iteration %3d, ramp_val = %1.6f --> %1.6f, ramp_incr = %1.2e"
                    % (ramp_iter, ramp_val - ramp_incr, ramp_val, ramp_incr))
        print("==========================================================================" )
        bc, bc_homo = get_BC(ramp_val)
        (ret, msg) = Newton(residual(ramp_val), d, bc = bc, bc_homo = bc_homo, verbose = True, builtin = builtin, dolfin_adjoint = dolfin_adjoint)
        if ret < 0:
            print("  Newton failed! " + msg)
            d.vector().set_local(d_last)
            d.vector().apply("insert")
            ramp_val -= ramp_incr
            ramp_incr *= ramp_failure_mult
        else:
            ramp_incr *= ramp_success_mult
            d_last = d.vector().get_local()
            ramp_iter += 1
            print("  Newton converged! " + msg)