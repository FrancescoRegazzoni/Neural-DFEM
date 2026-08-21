import numpy as np
import json
import hashlib
import dolfin as df
import dolfin_adjoint as dfa

from .material_law import MaterialLaw, register
from .. import utils
    
class HyperelasticNN(MaterialLaw):
    def __init__(self, params, path_label = None, verbose = False):
        self.name = 'HyperelasticNN'
        self.linear = False
        
        self.dim                = params['dim']
        self.num_neurons        = params['num_neurons']
        self.activation         = params['activation']
        self.stiff_factor       = params['stiff_factor']
        self.iso_invariants     = params['iso_invariants']
        self.pos_transformation = params['pos_transformation']
        self.bulk_type          = params['bulk_type']
        self.skip_connections   = params['skip_connections']

        has_params_values = 'params' in params.keys()

        if has_params_values:
            if path_label == None:
                raise Exception('Path label should be provided if params are set')
            self.path_label = path_label
            self.descr_label = self.path_label.replace('/', '_')
        else:
            if not path_label == None:
                raise Exception('Path label should not be provided if params are not set')

            serialized = json.dumps(params, sort_keys=True).encode('utf-8')
            self.path_label = 'hyperelasticNN-random/' + hashlib.md5(serialized).hexdigest()
            self.descr_label = f"HNN-{'-'.join(str(k) for k in params['num_neurons'])}neur"

            if params['activation'] == 'softplus':
                self.descr_label += '-sp'
            elif params['activation'] == 'squared-softplus':
                self.descr_label += '-sqsp'
            elif params['activation'] == 'exp':
                self.descr_label += '-ex'
            else:
                raise Exception(f"not implemented: {params['activation']}")

            if params['pos_transformation'] == 'softplus':
                self.descr_label += '-sp'
            elif params['pos_transformation'] == 'square':
                self.descr_label += '-sq'
            elif params['pos_transformation'] == 'modulus':
                self.descr_label += '-md'
            else:
                raise Exception(f"not implemented: {params['pos_transformation']}")

            if params['iso_invariants']:
                self.descr_label += '-iso'
                
            self.descr_label += '-sk' if params['skip_connections'] else '-nsk'

            self.descr_label += f'-stiff{params["stiff_factor"]}-inifact{params["init_factor"]}'
            self.descr_label += f'-{params["bulk_type"]}'
            if not params["bulk_type"] == 'none':
                self.descr_label += f'-init{params["init_bulk"]}'
            self.descr_label += f'-seed{params["seed"]}'

        if self.pos_transformation == 'square':
            self.df_pos = lambda x: x*x
        elif self.pos_transformation == 'softplus':
            self.df_pos = lambda x: df.ln(df.exp(x) + 1.0)
        elif self.pos_transformation == 'modulus':
            self.df_pos = lambda x: abs(x)
        else:
            raise Exception('Unknown positivity transformation')
        
        if self.activation == 'softplus':
            self.df_act = lambda x: df.ln(df.exp(x) + 1.0) - np.log(2.0)
            self.df_dact = lambda x: df.exp(x) / (df.exp(x) + 1)
        elif self.activation == 'squared-softplus':
            self.df_act = lambda x: df.ln(df.exp(x) + 1.0)**2 - np.log(2.0)**2
            self.df_dact = lambda x: 2 * df.ln(df.exp(x) + 1.0) * df.exp(x) / (df.exp(x) + 1)
        elif self.activation == 'exp':
            self.df_act = lambda x: df.exp(x) - 1.0
            self.df_dact = lambda x: df.exp(x)
        else:
            raise Exception('Unknown activation')
        
        if self.bulk_type == 'linear-log':
            self.df_bulk_function  = lambda J: 0.5*(J-1)*df.ln(J)
            self.df_dbulk_function = lambda J: 0.5*(J-1)/J + 0.5*df.ln(J)
        elif self.bulk_type == 'none':
            pass
        else:
            raise Exception('Unknown bulk type')

        if self.iso_invariants:
            self.law_invariants_iso = self.law_generic
        else:
            self.law_invariants = self.law_generic

        self.sum_factor = self.stiff_factor / self.num_neurons[-1]

        self.num_neurons_skip = self.num_neurons if self.skip_connections else [self.num_neurons[0]]
        
        if has_params_values:
            self.Wext = np.array(params['params']['Wext'])[None,:]
            self.WI   = [np.array(W)[None,:] for W in params['params']['WI']]
            if self.dim == 3:
                self.WI2  = [np.array(W)[None,:] for W in params['params']['WI2']]
            self.WJ   = [np.array(W)[None,:] for W in params['params']['WJ']]
            self.b    = [np.array(b)[None,:] for b in params['params']['b']]
            self.W    = [np.array(W) for W in params['params']['W']]
            if not self.bulk_type == 'none':
                self.bulk = params['params']['bulk']
        else:
            init_factor = params['init_factor']
            init_bulk = params['init_bulk']
            np.random.seed(params['seed'])
            self.Wext = np.random.randn(1,self.num_neurons[-1]) * init_factor
            self.WI   = [np.random.randn(1,n) * init_factor for n in self.num_neurons_skip]
            if self.dim == 3:
                self.WI2  = [np.random.randn(1,n) * init_factor for n in self.num_neurons_skip]
            self.WJ   = [np.random.randn(1,n) * init_factor for n in self.num_neurons_skip]
            self.b    = [np.zeros((1,n)) * init_factor for n in self.num_neurons]
            self.W    = [np.random.randn(self.num_neurons[i], self.num_neurons[i+1]) * init_factor for i in range(len(self.num_neurons)-1)]
            
            if not self.bulk_type == 'none':
                self.bulk = init_bulk

        self.initialized_ufl = False

        if verbose:
            print(f'Wext: {self.Wext}')
            print(f'WI:   {self.WI}')
            if self.dim == 3:
                print(f'WI2:  {self.WI2}')
            print(f'WJ:   {self.WJ}')
            print(f'b:    {self.b}')
            print(f'W:    {self.W}')
            if not self.bulk_type == 'none':
                print(f'bulk: {self.bulk}')

    def initialize_ufl(self):
        self.ufl_Wext = [dfa.Constant(v) for v in self.Wext.tolist()[0]]
        self.ufl_WI   = [[dfa.Constant(v) for v in W.tolist()[0]] for W in self.WI]
        if self.dim == 3:
            self.ufl_WI2  = [[dfa.Constant(v) for v in W.tolist()[0]] for W in self.WI2]
        self.ufl_WJ   = [[dfa.Constant(v) for v in W.tolist()[0]] for W in self.WJ]
        self.ufl_b    = [[dfa.Constant(v) for v in W.tolist()[0]] for W in self.b]
        self.ufl_W = [[[dfa.Constant(v) for v in inner] for inner in middle] for middle in self.W]
        if not self.bulk_type == 'none':
            self.ufl_bulk = dfa.Constant(self.bulk)
        
        self.initialized_ufl = True

    def initialize_control(self):
        self.initialize_ufl()

        def nested_list_to_control(v):
            if isinstance(v, list):
                return [nested_list_to_control(x) for x in v]
            else:
                return dfa.Control(v)

        self.controls = list()
        self.controls += nested_list_to_control(self.ufl_Wext)
        self.controls += list(utils.flatten(nested_list_to_control(self.ufl_WI)))
        if self.dim == 3:
            self.controls += list(utils.flatten(nested_list_to_control(self.ufl_WI2)))
        self.controls += list(utils.flatten(nested_list_to_control(self.ufl_WJ)))
        self.controls += list(utils.flatten(nested_list_to_control(self.ufl_b)))
        self.controls += list(utils.flatten(nested_list_to_control(self.ufl_W)))
        if not self.bulk_type == 'none':
            self.controls += [dfa.Control(self.ufl_bulk)]
        
        self.starting_value = list()
        self.starting_value += list(utils.flatten(self.Wext))
        self.starting_value += list(utils.flatten(self.WI))
        if self.dim == 3:
            self.starting_value += list(utils.flatten(self.WI2))
        self.starting_value += list(utils.flatten(self.WJ))
        self.starting_value += list(utils.flatten(self.b))
        self.starting_value += list(utils.flatten(self.W))
        if not self.bulk_type == 'none':
            self.starting_value += [self.bulk]

    def df_law_generic(self, inputs, return_var = 'W', format = 'fenics'):     

        dfx = dfa if format == 'fenics-adjoint' else df

        if not self.initialized_ufl: self.initialize_ufl()

        I = inputs[0]
        if self.dim == 3:
            I2 = inputs[1]
        J = inputs[self.dim - 1]

        Wext = self.ufl_Wext
        WI   = self.ufl_WI  
        if self.dim == 3:
            WI2  = self.ufl_WI2  
        WJ   = self.ufl_WJ  
        b    = self.ufl_b
        W    = self.ufl_W

        for lay in range(len(self.num_neurons)):
            z_list = list()
            z_Id_list = list()
            if not self.iso_invariants: dzdI1_Id_list = list()
            if not self.iso_invariants and self.dim == 3: dzdI2_Id_list = list()
            dzdJ_Id_list  = list()
            if return_var == 'dWdI1': dzdI1_list = list()
            if return_var == 'dWdI2' and self.dim == 3: dzdI2_list = list()
            if return_var == 'dWdJ': dzdJ_list = list()

            for i in range(self.num_neurons[lay]):  
                y    = b[lay][i]
                y_Id = b[lay][i]
                
                if lay == 0 or self.skip_connections:
                    WI_pos = self.df_pos(WI[lay][i])
                    y += WI_pos*(I-self.dim) + WJ[lay][i]*(J - 1)

                    if not self.iso_invariants: dydI1_Id = WI_pos
                    dydJ_Id  = WJ[lay][i]
                    if return_var == 'dWdI1': dydI1 = WI_pos
                    if return_var == 'dWdJ': dydJ = WJ[lay][i]
                    if self.dim == 3:
                        WI2_pos = self.df_pos(WI2[lay][i])
                        if self.iso_invariants:
                            y += WI2_pos*(I2**1.5-3.0**1.5)
                        else:
                            y += WI2_pos*(I2-3.0)
                        if not self.iso_invariants:
                            dydI2_Id = WI2_pos
                        if return_var == 'dWdI2': 
                            if self.iso_invariants:
                                dydI2 = WI2_pos*1.5*(I2**0.5)
                            else:
                                dydI2 = WI2_pos
                else:
                    if not self.iso_invariants:                   dydI1_Id = 0.0
                    if not self.iso_invariants and self.dim == 3: dydI2_Id = 0.0
                    if True:                                      dydJ_Id  = 0.0
                    if return_var == 'dWdI1':                     dydI1    = 0.0
                    if return_var == 'dWdI2' and self.dim == 3:   dydI2    = 0.0
                    if return_var == 'dWdJ':                      dydJ     = 0.0

                if lay > 0:
                    for j in range(self.num_neurons[lay-1]):
                        W_pos_norm = self.df_pos(W[lay-1][j][i]) / self.num_neurons[lay-1]
                        if True:                                      y        += W_pos_norm * z_list_last[j]
                        if True:                                      y_Id     += W_pos_norm * z_Id_list_last[j]
                        if not self.iso_invariants:                   dydI1_Id += W_pos_norm * dzdI1_Id_list_last[j]
                        if not self.iso_invariants and self.dim == 3: dydI2_Id += W_pos_norm * dzdI2_Id_list_last[j]
                        if True:                                      dydJ_Id  += W_pos_norm * dzdJ_Id_list_last[j]
                        if return_var == 'dWdI1':                     dydI1    += W_pos_norm * dzdI1_list_last[j]
                        if return_var == 'dWdI2' and self.dim == 3:   dydI2    += W_pos_norm * dzdI2_list_last[j]
                        if return_var == 'dWdJ':                      dydJ     += W_pos_norm * dzdJ_list_last[j]

                if True:                                      z_list.append(       self.df_act(y))
                if True:                                      z_Id_list.append(    self.df_act(y_Id))
                if not self.iso_invariants:                   dzdI1_Id_list.append(self.df_dact(y_Id) * dydI1_Id)
                if not self.iso_invariants and self.dim == 3: dzdI2_Id_list.append(self.df_dact(y_Id) * dydI2_Id)
                if True:                                      dzdJ_Id_list.append( self.df_dact(y_Id) * dydJ_Id)
                if return_var == 'dWdI1':                     dzdI1_list.append(   self.df_dact(y)    * dydI1)
                if return_var == 'dWdI2' and self.dim == 3:   dzdI2_list.append(   self.df_dact(y)    * dydI2)
                if return_var == 'dWdJ':                      dzdJ_list.append(    self.df_dact(y)    * dydJ)

            if True:                                      z_list_last        = z_list
            if True:                                      z_Id_list_last     = z_Id_list
            if not self.iso_invariants:                   dzdI1_Id_list_last = dzdI1_Id_list
            if not self.iso_invariants and self.dim == 3: dzdI2_Id_list_last = dzdI2_Id_list
            if True:                                      dzdJ_Id_list_last  = dzdJ_Id_list
            if return_var == 'dWdI1':                     dzdI1_list_last    = dzdI1_list
            if return_var == 'dWdI2' and self.dim == 3:   dzdI2_list_last    = dzdI2_list
            if return_var == 'dWdJ':                      dzdJ_list_last     = dzdJ_list

        Wext_pos = [self.df_pos(w) for w in Wext]
        def accumulate_last_layer(v_list):
            return sum([Wext_pos[i] * v_list[i] for i in range(self.num_neurons[-1])]) * dfx.Constant(self.sum_factor)
        if   return_var == 'W'    : ret = accumulate_last_layer(z_list) - accumulate_last_layer(z_Id_list)
        elif return_var == 'dWdI1': ret = accumulate_last_layer(dzdI1_list)
        elif return_var == 'dWdI2': ret = accumulate_last_layer(dzdI2_list)
        elif return_var == 'dWdJ' : ret = accumulate_last_layer(dzdJ_list)

        if return_var in ['W', 'dWdJ']:
            dret_dJ  = accumulate_last_layer(dzdJ_Id_list)
            if self.iso_invariants:
                omega = -dret_dJ
            else:
                dret_dI1 = accumulate_last_layer(dzdI1_Id_list)
                omega = -2*dret_dI1 - dret_dJ
                if self.dim == 3:
                    dret_dI2 = accumulate_last_layer(dzdI2_Id_list)
                    omega += -4*dret_dI2
            if   return_var == 'W'   : ret += omega * (J-1)
            elif return_var == 'dWdJ': ret += omega

        if not self.bulk_type == 'none':
            bulk_pos = self.df_pos(self.ufl_bulk) * dfx.Constant(self.stiff_factor)
            if   return_var == 'W'   : ret += bulk_pos*self.df_bulk_function(J)
            elif return_var == 'dWdJ': ret += bulk_pos*self.df_dbulk_function(J)
        
        return ret
    
    def law_generic(self, I1, I2, J, format = 'fenics', return_var = 'W'):
        invariants = [I1,J] if self.dim == 2 else [I1,I2,J]
        return self.df_law_generic(invariants, return_var = return_var, format = format)

    def save_from_params(self, params, filename):
        outdict = dict()
        outdict['model'] = 'HyperelasticNN'
        outdict['params'] = dict()
        outdict['params']['dim'] = self.dim
        outdict['params']['num_neurons'] = self.num_neurons
        outdict['params']['skip_connections'] = self.skip_connections
        outdict['params']['activation'] = self.activation
        outdict['params']['pos_transformation'] = self.pos_transformation
        outdict['params']['iso_invariants'] = self.iso_invariants
        outdict['params']['bulk_type'] = self.bulk_type
        outdict['params']['stiff_factor'] = self.stiff_factor
        outdict['params']['params'] = dict()

        idx = 0
        def get_params_chunk(size):
            nonlocal idx
            chunk = params[idx:idx+size]
            idx += size
            return chunk
        outdict['params']['params']['Wext'] = get_params_chunk(self.num_neurons[-1])
        outdict['params']['params']['WI'  ] = [get_params_chunk(n) for n in self.num_neurons_skip]
        if self.dim == 3:
            outdict['params']['params']['WI2' ] = [get_params_chunk(n) for n in self.num_neurons_skip]
        outdict['params']['params']['WJ'  ] = [get_params_chunk(n) for n in self.num_neurons_skip]
        outdict['params']['params']['b'   ] = [get_params_chunk(n) for n in self.num_neurons]
        outdict['params']['params']['W'   ] = [np.array(get_params_chunk(self.num_neurons[i]*self.num_neurons[i+1])).reshape((self.num_neurons[i], self.num_neurons[i+1])).tolist() 
                                                for i in range(len(self.num_neurons)-1)]
        if not self.bulk_type == 'none':   
            outdict['params']['params']['bulk'] = get_params_chunk(1)[0]
        with open(filename, 'w') as outfile:
            json.dump(outdict, outfile, indent = 2)
        return True
    
register('HyperelasticNN', HyperelasticNN)