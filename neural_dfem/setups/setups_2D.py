import numpy as np
import os
import dolfin as df
import dolfin_adjoint as dfa

from .setup import Setup, register

class PlateHoles(Setup):

    def _initialization(self):

        self.label = 'plate_holes'

        assert self.dim == 2, "PlateHoles test case is only implemented for 2D problems."

        self.num_loads = 1

        self.n_holes = self.options['n_holes']

    def get_folder_base(self):
        folder = 'ref%1.2f' % self.options['ref']
        if self.n_holes >= 1: folder += '_%1.5f-%1.5f-%1.5f' % (self.options['x1'], self.options['y1'], self.options['R1'])
        if self.n_holes >= 2: folder += '_%1.5f-%1.5f-%1.5f' % (self.options['x2'], self.options['y2'], self.options['R2'])
        if self.n_holes >= 3: folder += '_%1.5f-%1.5f-%1.5f' % (self.options['x3'], self.options['y3'], self.options['R3'])
        return folder

    def get_load_suffix(self, load):
        return 'force_%1.5f' % load

    def generate_mesh(self):
        if not PlateHoles.check_params(self.options, verbose = True):
            raise Exception("Invalid parameters for PlateHoles test case.")

        path_mesh = "data/mesh/plate_holes/%s" % self.get_folder_base()
        os.makedirs(path_mesh, exist_ok=True)

        params = self.options
                
        header_geo = ''  
        if params['n_holes'] > 0:
            header_geo += 'x1 = ' + str(params['x1']) + ';\n'
            header_geo += 'y1 = ' + str(params['y1']) + ';\n'
            header_geo += 'R1 = ' + str(params['R1']) + ';\n'
        if params['n_holes'] > 1:
            header_geo += 'x2 = ' + str(params['x2']) + ';\n'
            header_geo += 'y2 = ' + str(params['y2']) + ';\n'
            header_geo += 'R2 = ' + str(params['R2']) + ';\n'
        if params['n_holes'] > 2:
            header_geo += 'x3 = ' + str(params['x3']) + ';\n'
            header_geo += 'y3 = ' + str(params['y3']) + ';\n'
            header_geo += 'R3 = ' + str(params['R3']) + ';\n'

        header_geo += 'ref1 = ' + str(params['ref']) + ';\n'
        header_geo += 'ref2 = ' + str(params['ref']) + ';\n'

        if params['n_holes'] == 1:
            base_path = 'mesh/plate_1holes.geo'
        elif params['n_holes'] == 2:
            base_path = 'mesh/plate_2holes.geo'
        elif params['n_holes'] == 3:
            base_path = 'mesh/plate_3holes.geo'
        else:
            raise Exception("not implemented!")
        
        with open(base_path, "r") as f:
            body_geo = f.read()

        param_geo = header_geo + body_geo
        mesh_path_full = os.path.join(path_mesh, 'plate_holes')
        with open(mesh_path_full + '.geo', 'w') as f:
            f.write(param_geo)
            pass

        os.system('gmsh -2 %s.geo -format msh2 -o %s.msh' % (mesh_path_full, mesh_path_full))
        os.system('dolfin-convert %s.msh %s.xml' % (mesh_path_full, mesh_path_full))

    def setup(self, load, dolfin_adjoint = False):

        dfx = dfa if dolfin_adjoint else df

        mesh_path_full = os.path.join("data/mesh/plate_holes", self.get_folder_base(), 'plate_holes')
        if not os.path.exists(mesh_path_full + '.xml'):
            print('Mesh not found, generating mesh...')
            self.generate_mesh()
            print(f'Generated mesh {mesh_path_full}!')
            
        self.mesh = dfx.Mesh(mesh_path_full + '.xml') 
        self.mesh_original = self.mesh

        facet_domains = df.MeshFunction("size_t", self.mesh, mesh_path_full + '_facet_region.xml')
        my_ds = df.ds(subdomain_data=facet_domains)
        normal = df.FacetNormal(self.mesh)
        tag_bot   = 10
        tag_right = 11
        tag_top   = 12
        tag_left  = 13

        k_BC_leftright = 1e-2
        k_BC_topbot    = 1e-2

        FEM_order = self.FEM_options.get('FEM_order', 1)
        self.V = df.VectorFunctionSpace(self.mesh, 'Lagrange', FEM_order)

        self.get_BC = lambda coeff = 1.0: (None, None)

        def get_load(v, u, F, coeff = 1.0):
            bc = - df.dot(v, normal) * dfx.Constant(load * coeff) * my_ds(tag_right) \
                 - df.dot(v, normal) * dfx.Constant(load * coeff) * my_ds(tag_left ) \
                 + df.dot(u, normal) * df.dot(v, normal) * dfx.Constant(k_BC_leftright) * my_ds(tag_left ) \
                 + df.dot(u, normal) * df.dot(v, normal) * dfx.Constant(k_BC_leftright) * my_ds(tag_right) \
                 + df.dot(u, normal) * df.dot(v, normal) * dfx.Constant(k_BC_topbot)    * my_ds(tag_top  ) \
                 + df.dot(u, normal) * df.dot(v, normal) * dfx.Constant(k_BC_topbot)    * my_ds(tag_bot  )
            return -bc

        self.get_load = get_load

    @staticmethod
    def check_params(params, margin = 0.1, verbose = False):
        # margin: safety margin among holes and boundaries

        ok = True
        errors = []

        if (params['n_holes'] > 0 and params['x1'] < -1 + params['R1'] + margin): errors.append('low  x1'); ok = False
        if (params['n_holes'] > 0 and params['x1'] > +1 - params['R1'] - margin): errors.append('high x1'); ok = False
        if (params['n_holes'] > 0 and params['y1'] < -1 + params['R1'] + margin): errors.append('low  y1'); ok = False
        if (params['n_holes'] > 0 and params['y1'] > +1 - params['R1'] - margin): errors.append('high y1'); ok = False
        if (params['n_holes'] > 1 and params['x2'] < -1 + params['R2'] + margin): errors.append('low  x2'); ok = False
        if (params['n_holes'] > 1 and params['x2'] > +1 - params['R2'] - margin): errors.append('high x2'); ok = False
        if (params['n_holes'] > 1 and params['y2'] < -1 + params['R2'] + margin): errors.append('low  y2'); ok = False
        if (params['n_holes'] > 1 and params['y2'] > +1 - params['R2'] - margin): errors.append('high y2'); ok = False
        if (params['n_holes'] > 2 and params['x3'] < -1 + params['R3'] + margin): errors.append('low  x3'); ok = False
        if (params['n_holes'] > 2 and params['x3'] > +1 - params['R3'] - margin): errors.append('high x3'); ok = False
        if (params['n_holes'] > 2 and params['y3'] < -1 + params['R3'] + margin): errors.append('low  y3'); ok = False
        if (params['n_holes'] > 2 and params['y3'] > +1 - params['R3'] - margin): errors.append('high y3'); ok = False

        if (params['n_holes'] > 1 and ((params['x1'] - params['x2'])**2 + (params['y1'] - params['y2'])**2) < (params['R1'] + params['R2'] + margin)**2): errors.append('collapse P1-P2'); ok = False
        if (params['n_holes'] > 2 and ((params['x1'] - params['x3'])**2 + (params['y1'] - params['y3'])**2) < (params['R1'] + params['R3'] + margin)**2): errors.append('collapse P1-P3'); ok = False
        if (params['n_holes'] > 2 and ((params['x2'] - params['x3'])**2 + (params['y2'] - params['y3'])**2) < (params['R2'] + params['R3'] + margin)**2): errors.append('collapse P2-P3'); ok = False

        if verbose and not ok:
            print('PlateHoles parameters check failed:')
            for error in errors:
                print(error)

        return ok

    @staticmethod
    def get_random_params(tot_samples, ref = 0.03, seed = 0, margin = 0.1, Rmin = 0.1, Rmax = 0.3, n_holes_min = 1, n_holes_max = 3):
        """
        Generate random parameters for the PlateHoles test case.

        tot_samples: total number of samples to generate
        ref: reference length for the mesh size
        seed: random seed for reproducibility
        margin: safety margin among holes and boundaries
        Rmin: minimum holes radius
        Rmax: maximum holes radius
        n_holes_min: minimum number of holes
        n_holes_max: maximum number of holes
        """
        
        test_case_options = list()
        np.random.seed(seed)
        while len(test_case_options) < tot_samples:

            params = {        
                'x1'      : np.random.uniform(low = -1 + Rmin + margin, high = 1 - Rmin - margin),
                'y1'      : np.random.uniform(low = -1 + Rmin + margin, high = 1 - Rmin - margin),
                'R1'      : np.random.uniform(low = Rmin, high = Rmax), 
                'x2'      : np.random.uniform(low = -1 + Rmin + margin, high = 1 - Rmin - margin),
                'y2'      : np.random.uniform(low = -1 + Rmin + margin, high = 1 - Rmin - margin),
                'R2'      : np.random.uniform(low = Rmin, high = Rmax), 
                'x3'      : np.random.uniform(low = -1 + Rmin + margin, high = 1 - Rmin - margin),
                'y3'      : np.random.uniform(low = -1 + Rmin + margin, high = 1 - Rmin - margin),
                'R3'      : np.random.uniform(low = Rmin, high = Rmax), 
                'n_holes' : np.random.randint(low = n_holes_min, high = n_holes_max+1),
                'ref'     : ref,
            }

            if not PlateHoles.check_params(params, margin = margin): continue
            test_case_options.append(params)
        
        return test_case_options

class SquareHoleCorner(Setup):

    def _initialization(self):

        self.label = 'square_hole_corner'

        assert self.dim == 2, "SquareHoleCorner test case is only implemented for 2D problems."

        self.num_loads = 1
        self.has_reaction_forces = True

    def get_folder_base(self):
        return 'ref0.0312'

    def get_load_suffix(self, load):
        return 'delta_%1.5f' % load

    def setup(self, load, dolfin_adjoint = False):

        dfx = dfa if dolfin_adjoint else df

        mesh_path_full = 'data/mesh/square_hole_corner/square_hole_corner'

        self.mesh = dfx.Mesh(mesh_path_full + '.xml') 
        self.mesh_original = self.mesh

        facet_domains = df.MeshFunction("size_t", self.mesh, mesh_path_full + '_facet_region.xml')
        self.my_ds = df.ds(subdomain_data=facet_domains)
        self.reaction_force_tags = [1,2,3,4]  # bottom, right, top, left
        
        tol = 1e-14
        boundary_left  = lambda x, on_boundary: on_boundary and x[0] < tol
        boundary_bot   = lambda x, on_boundary: on_boundary and x[1] < tol
        boundary_right = lambda x, on_boundary: on_boundary and x[0] > 1.0 - tol
        boundary_top   = lambda x, on_boundary: on_boundary and x[1] > 1.0 - tol

        FEM_order = self.FEM_options.get('FEM_order', 1)
        self.V = df.VectorFunctionSpace(self.mesh, 'Lagrange', FEM_order)

        self.get_BC = lambda coeff = 1.0: \
            ([dfx.DirichletBC(self.V.sub(0), dfx.Constant(0.0           ), boundary_left),
              dfx.DirichletBC(self.V.sub(1), dfx.Constant(0.0           ), boundary_bot  ),
              dfx.DirichletBC(self.V.sub(0), dfx.Constant(coeff*load*0.5), boundary_right),
              dfx.DirichletBC(self.V.sub(1), dfx.Constant(coeff*load    ), boundary_top  )],
             [dfx.DirichletBC(self.V.sub(0), dfx.Constant(0.0           ), boundary_left),
              dfx.DirichletBC(self.V.sub(1), dfx.Constant(0.0           ), boundary_bot  ),
              dfx.DirichletBC(self.V.sub(0), dfx.Constant(0.0           ), boundary_right),
              dfx.DirichletBC(self.V.sub(1), dfx.Constant(0.0           ), boundary_top  )])

        self.get_load = lambda v, u, F, coeff = 1.0: 0.0

class SquareHoles(Setup):

    def _initialization(self):

        self.label = 'square_holes'
        self.holes_type = self.options['holes_type']

        assert self.dim == 2, "SquareHoles test case is only implemented for 2D problems."

        self.num_loads = 1
        self.has_reaction_forces = True

    def get_folder_base(self):
        if self.holes_type == '2ell':
            return '2ell_ref0.0152'

    def get_load_suffix(self, load):
        return 'delta_%1.5f' % load

    def setup(self, load, dolfin_adjoint = False):

        dfx = dfa if dolfin_adjoint else df

        if self.holes_type == '2ell':
            mesh_path_full = 'data/mesh/square_holes/square_holes_2ell'

        self.mesh = dfx.Mesh(mesh_path_full + '.xml') 
        self.mesh_original = self.mesh

        facet_domains = df.MeshFunction("size_t", self.mesh, mesh_path_full + '_facet_region.xml')
        self.my_ds = df.ds(subdomain_data=facet_domains)
        self.reaction_force_tags = [3]  # top
        
        tol = 1e-14
        boundary_top   = lambda x, on_boundary: on_boundary and x[1] > 1.0 - tol
        boundary_bot   = lambda x, on_boundary: on_boundary and x[1] < tol

        FEM_order = self.FEM_options.get('FEM_order', 1)
        self.V = df.VectorFunctionSpace(self.mesh, 'Lagrange', FEM_order)

        self.get_BC = lambda coeff = 1.0: \
            ([dfx.DirichletBC(self.V.sub(0), dfx.Constant(0.0       ), boundary_bot),
              dfx.DirichletBC(self.V.sub(1), dfx.Constant(0.0       ), boundary_bot),
              dfx.DirichletBC(self.V.sub(0), dfx.Constant(0.0       ), boundary_top),
              dfx.DirichletBC(self.V.sub(1), dfx.Constant(coeff*load), boundary_top)],
             [dfx.DirichletBC(self.V.sub(0), dfx.Constant(0.0       ), boundary_bot),
              dfx.DirichletBC(self.V.sub(1), dfx.Constant(0.0       ), boundary_bot),
              dfx.DirichletBC(self.V.sub(0), dfx.Constant(0.0       ), boundary_top),
              dfx.DirichletBC(self.V.sub(1), dfx.Constant(0.0       ), boundary_top)])

        self.get_load = lambda v, u, F, coeff = 1.0: 0.0

register('plate_holes', PlateHoles)
register('square_hole_corner', SquareHoleCorner)
register('square_holes', SquareHoles)