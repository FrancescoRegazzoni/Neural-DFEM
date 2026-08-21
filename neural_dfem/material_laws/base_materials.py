import dolfin as df

from .material_law import ParametricMaterialLaw, register

class MooneyRivlin(ParametricMaterialLaw):
    def __init__(self, params, **kwargs):
        self.name             = 'MooneyRivlin'
        self.dim              = 3
        self.coefficient_tags = ['C1', 'C2', 'K']
        self.option_tags      = []
        self.base_name        = f'mooney-rivlin'        
        super().__init__(params, **kwargs)      

    def law_invariants_parametric(self, I1, I2, J, c, format = 'fenics', return_var = 'W'):
        if return_var == 'W':
            return c['C1'] * (I1 - 3 - 2*df.ln(J)) + c['C2'] * (I2 - 3 - 4*df.ln(J)) + c['K'] * 0.5*(J-1)*df.ln(J)
        if return_var == 'dWdI1':
            return c['C1']
        elif return_var == 'dWdI2':
            return c['C2']
        elif return_var == 'dWdJ':
            return - c['C1'] * (2.0 / J) - c['C2'] * (4.0 / J) + 0.5 * c['K'] * ((J-1)/J + df.ln(J))

class Isihara(ParametricMaterialLaw):
    def __init__(self, params, **kwargs):
        self.name             = 'Isihara'
        self.dim              = 3
        self.coefficient_tags = ['C1', 'C2', 'C3', 'K']
        self.option_tags      = []
        self.base_name        = f'isihara'        
        super().__init__(params, **kwargs)      

    def law_invariants_iso_parametric(self, I1_iso, I2_iso, J, c, format = 'fenics', return_var = 'W'):
        if return_var == 'W':
            return c['C1'] * (I1_iso - 3) + c['C2'] * (I2_iso - 3) + c['C3'] * (I1_iso - 3)**2 + c['K'] * (J - 1)**2
        if return_var == 'dWdI1':
            return c['C1'] + 2 * c['C3'] * (I1_iso - 3)
        elif return_var == 'dWdI2':
            return c['C2']
        elif return_var == 'dWdJ':
            return 2 * c['K'] * (J - 1)

class Fung(ParametricMaterialLaw):
    def __init__(self, params, **kwargs):
        self.name             = 'Fung'
        self.dim              = 3
        self.coefficient_tags = ['C', 'b', 'K']
        self.option_tags      = []
        self.base_name        = f'fu'        
        super().__init__(params, **kwargs)

    def law_invariants_iso_parametric(self, I1_iso, I2_iso, J, c, format = 'fenics', return_var = 'W'):
        if return_var == 'W':
            return 0.5*c['C']/c['b'] * (df.exp(c['b'] * (I1_iso - 3)) - 1) + c['K'] * 0.25 * ((J-1)**2 + df.ln(J)**2)
        if return_var == 'dWdI1':
            return 0.5 * c['C'] * df.exp(c['b'] * (I1_iso - 3))
        elif return_var == 'dWdI2':
            return 0.0
        elif return_var == 'dWdJ':
            return c['K'] * 0.5 * ((J - 1) + df.ln(J) / J)
        elif return_var == 'd2WdJ2':
            return c['K'] * 0.5 * (1 - df.ln(J) / J**2)
       
register('MooneyRivlin', MooneyRivlin)
register('Isihara', Isihara)
register('Fung', Fung)