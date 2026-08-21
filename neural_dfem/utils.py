import numpy as np
import pandas as pd
import json
import os
import time
import hashlib

plane_strain_invariants = lambda I1, J: (I1 + 1.0, I1 + J**2)

def get_path_data(test_case, material, load, use_hashed_path = False):
    if use_hashed_path:
        material_path = '_hashed/' + hashlib.md5(material.path_label.encode('utf-8')).hexdigest()
    else:
        material_path = material.path_label

    base_path = 'data/data/'
    return os.path.join(base_path, material_path, test_case.label, test_case.get_folder_base(), test_case.get_load_suffix(load))

class data_loader:
    def __init__(self, test_case, material):
        self.test_case = test_case
        self.material = material
        
        self.dof_last = None
        self.load_last = 0

    def get_data(self, load, 
                 return_dof = False, return_data_table = False, return_reaction_forces = False, raise_exceptions = True,
                 do_export = True, do_export_pvd = False, use_hashed_path = False,
                 solution_continuation = False, do_force_recompute = False, only_load = False, verbosity = 1, **kwargs):

        folder_base = get_path_data(self.test_case, self.material, load, use_hashed_path = use_hashed_path)
        data_file            = folder_base + '/data.csv'
        dof_file             = folder_base + '/dof.csv'
        reaction_forces_file = folder_base + '/reaction_forces.csv'
        
        if only_load and do_force_recompute:
            raise Exception("Incompatible options: only_load and do_force_recompute")

        if not do_force_recompute:
            if (not return_data_table      or os.path.isfile(data_file           )) and \
               (not return_dof             or os.path.isfile(dof_file            )) and \
               (not return_reaction_forces or os.path.isfile(reaction_forces_file) or not self.test_case.has_reaction_forces):
                if verbosity > 0:
                    print('Loading existing data from %s...' % folder_base)
                results = dict()
                if return_data_table:
                    results['data'] = pd.read_csv(data_file)
                if return_dof:
                    results['dof'] = pd.read_csv(dof_file)['dof'].to_numpy()
                if return_reaction_forces and self.test_case.has_reaction_forces:
                    results['reaction_forces'] = pd.read_csv(reaction_forces_file)['forces'].to_numpy()
                return results
        
        if only_load:
            raise Exception("Non existing data: " + folder_base)

        if solution_continuation and self.dof_last is not None:
            ramp_value_start = self.load_last / load
            init_dof = self.dof_last
        else:
            ramp_value_start = 0.0
            init_dof = None
        try:
            results = self.test_case.solve(self.material, load, 
                                        ramp_value_start = ramp_value_start, init_dof = init_dof,
                                        return_dof = return_dof, return_data_table = return_data_table, compute_reaction_forces = return_reaction_forces,
                                        do_export = do_export, do_export_pvd = do_export_pvd, use_hashed_path = use_hashed_path, **kwargs)
        except Exception as e:
            if raise_exceptions:
                raise e
            else:
                print(f"Exception during FEM solve: {e}")
                return None
        self.dof_last = results['dof']
        self.load_last = load
        return results

def add_choice(parser, name, default, help_str="", shortcut=None):

    dest = name
    flag = f"--{name.replace('_','-')}"
    no_flag = f"--no-{name.replace('_','-')}"
    
    kwargs_flag = dict(
        action="store_true", # if not default else "store_false",
        default=default,
        dest=dest,
        help=help_str + f" (default: {default})"
    )
    if shortcut:
        parser.add_argument(shortcut, flag, **kwargs_flag)
    else:
        parser.add_argument(flag, **kwargs_flag)
    
    parser.add_argument(
        no_flag, 
        action="store_false", # if not default else "store_true",
        dest=dest, 
    )

def load_json_list(files):
    output = list()
    for file in files:
        with open(file, 'r') as f:
            output.append(json.load(f))
    return output

def flatten(nested_list):
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten(item)
        elif isinstance(item, np.ndarray):
            yield from flatten(item.tolist())
        else:
            yield item

def retry_on_exception(func, max_retries = 10, delay = 1.0, verbose = True):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if verbose:
                print(f"Exception caught ({attempt+1}/{max_retries}):")
                print(e)
            if attempt < max_retries - 1:
                if verbose:
                    print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                if verbose:
                    print(f"No more attempts left, raising exception.")
                raise e
