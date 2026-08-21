#!/usr/bin/env python3
import argparse

from neural_dfem import material_laws
from neural_dfem import setups
from neural_dfem import utils

parser = argparse.ArgumentParser()

parser.add_argument("-t", "--testcases", nargs = '+', required = True, help = "List of test cases JSON files.")
parser.add_argument("-m", "--models"   , nargs = '+', required = True, help = "List of material models JSON files.")
utils.add_choice(parser, "recompute"   , False, "Force recomputation"         , "-f")
utils.add_choice(parser, "export"      , True , "Enable export"               , "-e")
utils.add_choice(parser, "export_pvd"  , False, "Enable PVD export"           , "-ep")
utils.add_choice(parser, "hashed_path" , False, "Enable hashed paths"         , "-hp")
utils.add_choice(parser, "autodiff"    , True , "Enable energy autodiff"      , "-a")
utils.add_choice(parser, "continuation", True , "Enable solution continuation", "-c")

args = parser.parse_args()

do_force_recompute    = args.recompute
do_export             = args.export
do_export_pvd         = args.export_pvd
use_hashed_path       = args.hashed_path
energy_autodiff       = args.autodiff
solution_continuation = args.continuation
test_cases_list       = utils.load_json_list(args.testcases)

FEM_options = {'FEM_order': 1}

for model in args.models:
    print('################################')
    print(model)
    print('################################')
    law = material_laws.get(model)

    for test_case_list_curr in test_cases_list:
        test_case = test_case_list_curr['test_case']
        test_case_options = test_case_list_curr['test_case_options']
        loads = test_case_list_curr['loads']
        space_dim =  test_case_list_curr['dim']
        if not type(test_case_options) == list: test_case_options = [test_case_options]

        for options in test_case_options:
            test_case_curr = setups.factory[test_case](options, space_dim, FEM_options = FEM_options)
            data_loader = utils.data_loader(test_case_curr, law)
            for load in loads:
                print('********************************')
                print('load: %s' % load)
                data_loader.get_data(load, energy_autodiff = energy_autodiff,
                                     return_dof = True, return_data_table = True, return_reaction_forces = True,
                                     do_export = do_export, do_export_pvd = do_export_pvd, use_hashed_path = use_hashed_path,
                                     solution_continuation = solution_continuation,
                                     do_force_recompute = do_force_recompute)
            del test_case_curr