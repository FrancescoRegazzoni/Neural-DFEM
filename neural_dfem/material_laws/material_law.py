import numpy as np
import json
import abc
import pathlib
import dolfin as df
import ufl
import dolfin_adjoint as dfa

from .. import FEM_utils

class MaterialLaw(abc.ABC):
    """
    Base class for constitutive material laws.

    A material law can be defined either in terms of the standard invariants
    ``(I1, I2, J)`` through a ``law_invariants`` method, or in terms of the
    isochoric invariants ``(I1_iso, I2_iso, J)`` through a
    ``law_invariants_iso`` method.

    Derived classes are expected to define the material dimension ``dim`` and
    provide one of these invariant-based constitutive interfaces.
    """

    def __init__(self):
        pass
    
    def law(self, I1, I2, J, format = 'fenics', return_var = 'W'):
        """
        Evaluate the strain-energy density or one of its derivatives.

        If the constitutive model is expressed in terms of the standard
        invariants, this method directly delegates to ``law_invariants``.
        If instead the model is expressed in terms of isochoric invariants,
        the invariants are transformed according to

        ``I1_iso = J**(-2/dim) * I1``

        and, in three dimensions,

        ``I2_iso = J**(-4/dim) * I2``.

        Derivatives with respect to the standard invariants are obtained from
        the isochoric formulation using the chain rule.

        Parameters
        ----------
        I1 : scalar-like
            First invariant of the right Cauchy--Green tensor.
        I2 : scalar-like or None
            Second invariant of the right Cauchy--Green tensor. It is required
            for three-dimensional material laws and may be None in two
            dimensions.
        J : scalar-like
            Determinant of the deformation gradient.
        format : {'fenics', 'fenics-adjoint'}, optional
            Backend used to represent material parameters and symbolic
            expressions. Default is 'fenics'.
        return_var : {'W', 'dWdI1', 'dWdI2', 'dWdJ'}, optional
            Quantity to return: the strain-energy density or one of its
            derivatives with respect to the standard invariants.
            Default is 'W'.

        Returns
        -------
        scalar-like
            Requested strain-energy quantity.

        """

        if hasattr(self, 'law_invariants'):
            result = self.law_invariants(I1, I2, J, format = format, return_var = return_var)
        elif hasattr(self, 'law_invariants_iso'):
            I1_iso = J**(-2/self.dim) * I1
            I2_iso = J**(-4/self.dim) * I2 if self.dim == 3 else None
            
            if return_var == 'W':
                result = self.law_invariants_iso(I1_iso, I2_iso, J, format = format, return_var = 'W')
            elif return_var == 'dWdI1':
                result = J**(-2/self.dim) * self.law_invariants_iso(I1_iso, I2_iso, J, format = format, return_var = 'dWdI1')
            elif return_var == 'dWdI2':
                result = J**(-4/self.dim) * self.law_invariants_iso(I1_iso, I2_iso, J, format = format, return_var = 'dWdI2')
            elif return_var == 'dWdJ':
                dWdI1iso = self.law_invariants_iso(I1_iso, I2_iso, J, format = format, return_var = 'dWdI1')
                dWdJ     = self.law_invariants_iso(I1_iso, I2_iso, J, format = format, return_var = 'dWdJ')
                result   = dWdJ - 2/self.dim * J**(-2/self.dim-1) * I1 * dWdI1iso
                if self.dim == 3:
                    dWdI2iso = self.law_invariants_iso(I1_iso, I2_iso, J, format = format, return_var = 'dWdI2')
                    result += - 4/self.dim * J**(-4/self.dim-1) * I2 * dWdI2iso
        else:
            raise Exception('Either law_invariants or law_invariants_iso must be implemented.')

        return result

    def save_from_params(self, params, filename):
        """
        Save a material model defined by the supplied parameters.

        This base implementation does not provide serialization and always
        returns False. Derived classes may override it.

        Parameters
        ----------
        params : array_like or mapping
            Material parameters to be saved.
        filename : str or pathlib.Path
            Destination JSON file.

        Returns
        -------
        bool
            True if the model definition is written successfully.
        """
        print('exporting not implemented for this material model')
        return False

    def piola(self, F, space_dim, format = 'fenics'):
        """
        Compute the first Piola--Kirchhoff stress tensor.

        The stress is obtained from the derivatives of the strain-energy
        density with respect to the invariants of the right Cauchy--Green
        tensor. For a hyperelastic material,

        ``P = dW/dF``.

        Parameters
        ----------
        F : ufl.core.expr.Expr
            Deformation gradient.
        space_dim : int
            Dimension of the computational domain.
        format : {'fenics', 'fenics-adjoint'}, optional
            Backend used when evaluating the constitutive law.
            Default is 'fenics'.

        Returns
        -------
        ufl.core.expr.Expr
            First Piola--Kirchhoff stress tensor.

        Notes
        -----
        The deformation invariants are computed using ``self.dim`` as the
        material dimension and ``space_dim`` as the spatial dimension. Hence,
        a three-dimensional material law may also be used in a two-dimensional
        plane-strain setting.
        """

        I1, I2, J = FEM_utils.compute_invariants(F, self.dim, space_dim)
        dWdI1 = self.law(I1, I2, J, format = format, return_var = 'dWdI1')
        dWdJ  = self.law(I1, I2, J, format = format, return_var = 'dWdJ')
        P   = dWdI1 * 2 * F + dWdJ * ufl.cofac(F)
        if self.dim == 3:
            dWdI2 = self.law(I1, I2, J, format = format, return_var = 'dWdI2')
            P += dWdI2 * (2*I1*F - 2*F*F.T*F)
        return P
    
class ParametricMaterialLaw(MaterialLaw):
    """
    Base class for constitutive laws depending on a finite set of parameters.

    Material coefficients and model options are initialized from a parameter
    dictionary. Constitutive laws may be implemented through either
    ``law_invariants_parametric`` or ``law_invariants_iso_parametric``;
    the corresponding non-parametric interface is created automatically.

    Derived classes are expected to define, at least, ``coefficient_tags``,
    ``option_tags``, ``base_name``, and ``name``.
    """
        
    def __init__(self, params, path_label = None):
        """
        Initialize a parametric material model.

        Parameters
        ----------
        params : mapping
            Dictionary containing the material coefficients and model options.
            The required keys are determined by ``coefficient_tags`` and
            ``option_tags``.
        path_label : str or None, optional
            Label used when organizing model paths or outputs. If None, a
            descriptive label is generated from the material coefficients.
        """

        self.initialize_coefficients(params)
        self.initialize_options(params)
        self.linear = False
        self.descr_label = self.base_name + '_' + '_'.join(['%s_%1.3f' % (key, self.coefficients_values[key]) for key in self.coefficient_tags])
        self.path_label = self.descr_label if path_label == None else path_label

        if hasattr(self, 'law_invariants_parametric'):
            self.law_invariants = self._law_invariants
        if hasattr(self, 'law_invariants_iso_parametric'):
            self.law_invariants_iso = self._law_invariants_iso

    def _law_invariants(self, I1, I2, J, format = 'fenics', return_var = 'W'):
        c = self.get_coefficients(format = format)
        return self.law_invariants_parametric(I1, I2, J, c, format = format, return_var = return_var)

    def _law_invariants_iso(self, I1_iso, I2_iso, J, format = 'fenics', return_var = 'W'):
        c = self.get_coefficients(format = format)
        return self.law_invariants_iso_parametric(I1_iso, I2_iso, J, c, format = format, return_var = return_var)

    def initialize_coefficients(self, params):
        self.coefficients_values = {}
        for key in self.coefficient_tags:
            self.coefficients_values[key] = params[key]

    def initialize_options(self, params):
        self.options_values = {}
        for key in self.option_tags:
            self.options_values[key] = params[key]

    def initialize_control(self):
        self.coefficients_dfa = {}
        self.controls = []
        for key in self.coefficient_tags:
            self.coefficients_dfa[key] = dfa.Constant(self.coefficients_values[key])
            self.controls.append(dfa.Control(self.coefficients_dfa[key]))
    
    def get_coefficients(self, format = 'fenics'):
        if format == 'fenics':
            return {key: df.Constant(self.coefficients_values[key]) for key in self.coefficient_tags}
        elif format == 'fenics-adjoint':
            return self.coefficients_dfa
        else:
            raise Exception('Unknown format %s' % format)
    
    def save_from_params(self, params, filename):
        outdict = dict()
        outdict['model'] = self.name
        outdict['params'] = dict()
        for (i, key) in enumerate(self.coefficient_tags):
            outdict['params'][key] = params[i]
        with open(filename, 'w') as outfile:
            json.dump(outdict, outfile, indent = 2)
        return True
        
factory = dict()

def register(elastic_model, model_class):
    """
    Register a material model class in the material-law factory.
    """
    factory[elastic_model] = model_class

def get_from_factory(elastic_model, params, **kwargs):
    """
    Instantiate a registered material model.
    """
    return factory[elastic_model](params = params, **kwargs)

def get(model, modifications = None):
    """
    Construct a material model from a definition or a JSON file.

    The model can be specified directly by a dictionary containing the model
    identifier and its parameters, or by the path to a JSON file with the same
    information. Selected parameters can optionally be overridden before the
    material model is instantiated.

    Parameters
    ----------
    model : dict or str
        Material-model specification. A dictionary must contain the keys
        ``'model'`` and ``'params'``. Otherwise, the argument is interpreted
        as the path to a JSON model-definition file.
    modifications : mapping or None, optional
        Parameter values that override those provided by the original model
        definition. Default is None.

    Returns
    -------
    MaterialLaw
        Instantiated material model.

    Notes
    -----
    For models stored below ``data/models/_/``, a relative path label is
    automatically inferred from the model file location.
    """
    
    def apply_modifications(params):
        if modifications is not None:
            for key, value in modifications.items():
                params[key] = value
        return params

    if isinstance(model, dict):
        return get_from_factory(model['model'], apply_modifications(model['params']))
    else:
        filepath = model
        with open(filepath, 'r') as infile:
            definition = json.load(infile)

        path_label = None
        if 'data/models' in filepath and (('/_/' in filepath)):
            path_label = str(pathlib.Path(filepath).relative_to("data/models").with_suffix('').parent)

        return factory[definition['model']](params = apply_modifications(definition['params']), path_label = path_label)