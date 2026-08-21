import numpy as np
import dolfin as df
import ufl
import dolfin_adjoint as dfa

from .setup import Setup, register

class CubeHole(Setup):

    def _initialization(self):

        self.label = 'cube_hole'
        self.version = self.options['version']
        self.torsion_rate = self.options['torsion_rate']

        assert self.dim == 3, "CubeHole test case is only implemented for 3D problems."

        self.num_loads = 1
        self.has_reaction_forces = True

    def get_folder_base(self):
        return f'{self.version}_tors{self.torsion_rate}'

    def get_load_suffix(self, load):
        return f'delta_{load:1.5f}'

    def setup(self, load, dolfin_adjoint = False):

        dfx = dfa if dolfin_adjoint else df

        mesh_path_full = f'data/mesh/cube_hole/cube_hole_{self.version}'
        xdmf_meshfile    = f"{mesh_path_full}.xdmf"
        xdmf_meshfile_bm = f"{mesh_path_full}_bm.xdmf"

        self.mesh = dfx.Mesh()
        with df.XDMFFile(xdmf_meshfile) as infile:
            infile.read(self.mesh)
        mvc = df.MeshValueCollection("size_t", self.mesh, 2) 
        with df.XDMFFile(xdmf_meshfile_bm) as infile:
            infile.read(mvc, "tags")
        facet_domains = df.cpp.mesh.MeshFunctionSizet(self.mesh, mvc)

        self.mesh_original = self.mesh

        x_min, x_max = -0.5, 0.5
        self.my_ds = df.ds(subdomain_data=facet_domains)
        self.reaction_force_tags = [5] # x = x_min
        
        tol = 1e-10
        boundary_x_max   = lambda x, on_boundary: on_boundary and x[0] > x_max - tol
        boundary_x_min   = lambda x, on_boundary: on_boundary and x[0] < x_min + tol

        FEM_order = self.FEM_options.get('FEM_order', 1)
        self.V = df.VectorFunctionSpace(self.mesh, 'Lagrange', FEM_order)

        u_D = lambda coeff: df.Expression(("L", "(cos(theta)-1.0)*x[1]-sin(theta)*x[2]", "sin(theta)*x[1]+(cos(theta)-1.0)*x[2]"), 
                                    L = coeff*load, theta = 2*np.pi*coeff*load*self.torsion_rate, degree = 4)

        self.get_BC = lambda coeff = 1.0: \
            ([dfx.DirichletBC(self.V, dfx.Constant((       0.0, 0.0, 0.0)), boundary_x_min),
              dfx.DirichletBC(self.V, u_D(coeff)                          , boundary_x_max)],
             [dfx.DirichletBC(self.V, dfx.Constant((       0.0, 0.0, 0.0)), boundary_x_min),
              dfx.DirichletBC(self.V, dfx.Constant((       0.0, 0.0, 0.0)), boundary_x_max)])

        self.get_load = lambda v, u, F, coeff = 1.0: 0.0

class Clip(Setup):

    def _initialization(self):

        self.label = 'clip'
        self.h = self.options['h']
        self.eulerian_load = self.options['eulerian_load']
        self.K1 = 0.01
        self.K2 = 0.1

        assert self.dim == 3, "Clip test case is only implemented for 3D problems."

        self.num_loads = 1
        self.has_reaction_forces = False

    def get_folder_base(self):
        return f"h{self.h:1.2f}_K{self.K1:1.3f}_K{self.K2:1.3f}{'_EL' if self.eulerian_load else ''}"

    def get_load_suffix(self, load):
        return f'delta_{load:1.5f}'

    def setup(self, load, dolfin_adjoint = False):

        dfx = dfa if dolfin_adjoint else df

        mesh_path_full = f'data/mesh/clip/clip_h{self.h:1.2f}'
        xdmf_meshfile    = f"{mesh_path_full}.xdmf"
        xdmf_meshfile_bm = f"{mesh_path_full}_bm.xdmf"

        self.mesh = dfx.Mesh()
        with df.XDMFFile(xdmf_meshfile) as infile:
            infile.read(self.mesh)
        mvc = df.MeshValueCollection("size_t", self.mesh, 2) 
        with df.XDMFFile(xdmf_meshfile_bm) as infile:
            infile.read(mvc, "tags")
        facet_domains = df.cpp.mesh.MeshFunctionSizet(self.mesh, mvc)

        self.mesh_original = self.mesh
        self.my_ds = df.ds(subdomain_data=facet_domains)

        bottom_tag = 10
        front_tag = 12
        inner_tag = 15
        lateral_tag = [13,14,16,17]
        hole_tag = [2,5,8]

        FEM_order = self.FEM_options.get('FEM_order', 1)
        self.V = df.VectorFunctionSpace(self.mesh, 'Lagrange', FEM_order)

        bcs = list()
        bcs.append(dfx.DirichletBC(self.V.sub(2), dfx.Constant(0.0), facet_domains, bottom_tag))
        self.get_BC = lambda coeff = 1.0: (bcs, bcs)

        normal = df.FacetNormal(self.mesh)
        def get_load(v, u, F, coeff = 1.0):
            if self.eulerian_load:
                bc = df.dot(v, ufl.cofac(F)*normal) * dfx.Constant(load * coeff) * self.my_ds(inner_tag)
            else:
                bc = df.dot(v,              normal) * dfx.Constant(load * coeff) * self.my_ds(inner_tag)
                
            # front-ward force on the inner surface
            bc += df.dot(v,dfx.Constant((0.0, -1.0, 0.0))) * dfx.Constant(0.2 * load * coeff) * self.my_ds(inner_tag)

            for tag in lateral_tag + [front_tag]:
                bc += df.dot(u, normal) * df.dot(v, normal) * dfx.Constant(self.K1) * self.my_ds(tag)
            for tag in hole_tag:
                bc += df.dot(u, normal) * df.dot(v, normal) * dfx.Constant(self.K2) * self.my_ds(tag)
            return -bc

        self.get_load = get_load

class CubeCornerHole(Setup):

    def _initialization(self):

        self.label = 'cube_corner_hole'
        self.h = self.options['h']

        assert self.dim == 3, "Cube corner hole test case is only implemented for 3D problems."

        self.num_loads = 1
        self.has_reaction_forces = True

    def get_folder_base(self):
        return f'h{self.h:1.2f}'

    def get_load_suffix(self, load):
        return f'delta_{load:1.5f}'

    def setup(self, load, dolfin_adjoint = False):

        dfx = dfa if dolfin_adjoint else df

        mesh_path_full = f'data/mesh/cube_corner_hole/cube_corner_hole_h{self.h:1.2f}'
        xdmf_meshfile    = f"{mesh_path_full}.xdmf"
        xdmf_meshfile_bm = f"{mesh_path_full}_bm.xdmf"

        self.mesh = dfx.Mesh()
        with df.XDMFFile(xdmf_meshfile) as infile:
            infile.read(self.mesh)
        mvc = df.MeshValueCollection("size_t", self.mesh, 2) 
        with df.XDMFFile(xdmf_meshfile_bm) as infile:
            infile.read(mvc, "tags")
        facet_domains = df.cpp.mesh.MeshFunctionSizet(self.mesh, mvc)

        self.mesh_original = self.mesh
        self.my_ds = df.ds(subdomain_data=facet_domains)

        x_min_tag, x_max_tag, y_min_tag, y_max_tag, z_min_tag, z_max_tag = 1, 2, 3, 4, 5, 6
        cavity_tag = 7
        domain_tag = 1

        FEM_order = self.FEM_options.get('FEM_order', 1)
        self.V = df.VectorFunctionSpace(self.mesh, 'Lagrange', FEM_order)
        self.reaction_force_tags = [x_max_tag, y_max_tag, z_max_tag]

        self.get_BC = lambda coeff = 1.0: ([
            dfx.DirichletBC(self.V.sub(0), dfx.Constant(0.0), facet_domains, x_min_tag),
            dfx.DirichletBC(self.V.sub(1), dfx.Constant(0.0), facet_domains, y_min_tag),
            dfx.DirichletBC(self.V.sub(2), dfx.Constant(0.0), facet_domains, z_min_tag),
            dfx.DirichletBC(self.V.sub(0), dfx.Constant(coeff*load*1.0 ), facet_domains, x_max_tag),
            dfx.DirichletBC(self.V.sub(1), dfx.Constant(coeff*load*0.5 ), facet_domains, y_max_tag),
            dfx.DirichletBC(self.V.sub(2), dfx.Constant(coeff*load*0.25), facet_domains, z_max_tag),
        ], [
            dfx.DirichletBC(self.V.sub(0), dfx.Constant(0.0), facet_domains, x_min_tag),
            dfx.DirichletBC(self.V.sub(1), dfx.Constant(0.0), facet_domains, y_min_tag),
            dfx.DirichletBC(self.V.sub(2), dfx.Constant(0.0), facet_domains, z_min_tag),
            dfx.DirichletBC(self.V.sub(0), dfx.Constant(0.0), facet_domains, x_max_tag),
            dfx.DirichletBC(self.V.sub(1), dfx.Constant(0.0), facet_domains, y_max_tag),
            dfx.DirichletBC(self.V.sub(2), dfx.Constant(0.0), facet_domains, z_max_tag),
        ])

        self.get_load = lambda v, u, F, coeff = 1.0: 0.0

register('cube_hole', CubeHole)
register('cube_corner_hole', CubeCornerHole)
register('clip', Clip)