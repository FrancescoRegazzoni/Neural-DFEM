#!/usr/bin/env python3
#%%
import time
import datetime
import numpy as np
import pandas as pd
import json
import functools
import operator
import argparse
import dolfin as df
import dolfin_adjoint as dfa
import os

from neural_dfem import material_laws
from neural_dfem import setups
from neural_dfem import utils

parser = argparse.ArgumentParser()

utils.add_choice(parser, "export", True , "Enable export", "-e")
parser.add_argument("-l", "--custom-label", default = '', help = "Custom label.")

parser.add_argument("-t", "--testcases", nargs = '+', required = True, help = "List of test cases JSON files.")
parser.add_argument("-d", "--testcases-label", required = True, help = "Label describing the test cases.")
parser.add_argument("-S", "--seed-noise", type=int, default = 0, help = "Seed used for generating noise.")
parser.add_argument("--noise", type=float,  default = 0, help = "Noise magnitude.")
utils.add_choice(parser, "boundary_observation", False , "Toggle boundary observation", "-b")

parser.add_argument("-M", "--model-target", required = True, help = "Target material model.")
parser.add_argument("-m", "--model-candidate", required = True, help = "Candidate material model.")

parser.add_argument("-o", "--optimizer", default = 'BFGS', help = "Optimizer (e.g. L-BFGS-B, BFGS, Newton-CG).")
parser.add_argument("--maxiter", type=int, default=100000, help="Maximum number of optimizer iterations.")

utils.add_choice(parser, "force_recompute_init", False , "Force recompute initialization."  , "-f")

utils.add_choice(parser, "hashed_path", False, "Enable hashed paths", "-hp")

args = parser.parse_args()

do_export     = args.export
add_timestamp = False
custom_label  = args.custom_label
hashed_path = args.hashed_path

seed_noise             = args.seed_noise
noise_magnitude        = args.noise
full_body_observation  = not args.boundary_observation
reaction_forces_weight = 1.0

model_target = args.model_target
model_candidate = args.model_candidate

###### Training method
native_optimizer  = False
method            = args.optimizer
optimizer_options = {"maxiter": args.maxiter, "gtol": 1e-10, "xrtol": 1e-6}

early_stopping           = True
early_stopping_patience  = 50
early_stopping_tolerance = 1e-4

###### Initialization
initialization          = 'warmstart' # 'zero', 'warmstart'
do_force_recompute_init = args.force_recompute_init
do_ramp                 = False
ramp_incr_start         = 0.5
solution_continuation   = True

####### Training test cases
test_cases_list = list()

test_case_label = f"{args.testcases_label}"
test_cases_list = list()
for test_case_file in args.testcases:
    with open(test_case_file, 'r') as f:
        test_cases_list.append(json.load(f))

#####################################################

t_0 = time.time()

material_law = material_laws.get(model_target)

data_d = list()
data_R = list()
if seed_noise is not None: np.random.seed(seed_noise)

for test_case_list_curr in test_cases_list:
    test_case = test_case_list_curr['test_case']
    test_case_options = test_case_list_curr['test_case_options']
    loads = test_case_list_curr['loads']
    if not type(test_case_options) == list: test_case_options = [test_case_options]
    for options in test_case_options:
        test_case_curr = setups.factory[test_case](options, test_case_list_curr['dim'])
        for load in loads:
            data_d_file = utils.get_path_data(test_case_curr, material_law, load, use_hashed_path = hashed_path) + '/dof.csv'
            print('loading %s...' % data_d_file, end = '')
            data_d_curr = pd.read_csv(data_d_file)['dof'].to_numpy()
            if noise_magnitude > 0:
                data_d_curr += noise_magnitude*np.random.randn(len(data_d_curr))
            data_d.append(data_d_curr)
            print('done!')
            if test_case_curr.has_reaction_forces:
                data_R_file = utils.get_path_data(test_case_curr, material_law, load, use_hashed_path = hashed_path) + '/reaction_forces.csv'
                print('loading %s...' % data_R_file, end = '')
                data_R_curr = pd.read_csv(data_R_file)['forces'].to_numpy()
                data_R.append(data_R_curr)
                print('done!')
            else:
                data_R.append(None)
print('')

candidate_law = material_laws.get(model_candidate)
candidate_law.initialize_control()

assert material_law.dim == candidate_law.dim, f"Dimension mismatch"

if do_export:
    model_label = f"{test_case_label}_{'fb-obs' if full_body_observation else 'bd-obs'}_"
    model_label += f"ns-{noise_magnitude:.1e}_{candidate_law.descr_label}{custom_label}"
    out_folder = 'data/models/' + material_law.path_label + '/_/adjoint_' + model_label
    if add_timestamp:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        out_folder = out_folder + '_%s' % timestamp

    print(f'OUTPUT FOLDER: {out_folder}')
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    out_filename =  out_folder + '/energy.json'
    history_filename =  out_folder + '/history.json'

# tracing 

J_d, N_d = list(), list()
J_R, N_R = list(), list()
i_sample = 0

for test_case_list_curr in test_cases_list:
    test_case = test_case_list_curr['test_case']
    test_case_options = test_case_list_curr['test_case_options']
    loads = test_case_list_curr['loads']
    if not type(test_case_options) == list: test_case_options = [test_case_options]
    for options in test_case_options:
        test_case_curr = setups.factory[test_case](options, test_case_list_curr['dim'])
        if initialization == 'warmstart': 
            init_data_loader = utils.data_loader(test_case_curr, candidate_law)
        for i_load, load in enumerate(loads):
            print('--------------------------------------')
            print(f'Tracing test case {test_case_curr.get_folder_base()} load {load}...')
            init_dof = None
            if initialization == 'warmstart':
                results = utils.retry_on_exception(lambda: 
                    init_data_loader.get_data(load, return_dof = True, do_export = True, solution_continuation = solution_continuation, do_force_recompute = do_force_recompute_init),
                    max_retries = 10, delay = 60)

                init_dof = results['dof']

            results = utils.retry_on_exception(lambda: test_case_curr.solve(candidate_law, load,
                                        init_dof = init_dof, dolfin_adjoint = True, builtin_newton = False,
                                        do_ramp = do_ramp, ramp_incr_start = ramp_incr_start),
                                        max_retries = 10, delay = 60)
            d = results['d']

            d_obs = dfa.Function(test_case_curr.V, name = 'displacement_exact')
            if not len(data_d[i_sample]) == len(d_obs.vector().get_local()):
                raise Exception('Dimension mismatch: %d != %d' % (len(data_d[i_sample]), len(d_obs.vector().get_local())))
            d_obs.vector().set_local(data_d[i_sample])

            differential = df.dx if full_body_observation else df.ds
            J_d.append(dfa.assemble((df.inner(d - d_obs, d - d_obs)) * differential))
            N_d.append(dfa.assemble((df.inner(    d_obs,     d_obs)) * differential))

            if test_case_curr.has_reaction_forces:
                for i_R, R in enumerate(results['reaction_forces']):
                    J_R.append((R-data_R[i_sample][i_R])**2)
                    N_R.append((  data_R[i_sample][i_R])**2)

            i_sample += 1

J = sum(J_d)/float(sum(N_d))
if len(J_R) > 0: J += reaction_forces_weight * sum(J_R)/float(sum(N_R))

print('--------------------------------------')
print('end tracing')
print('--------------------------------------')

deshape = lambda params: [param.values()[0] for param in params]

J_value = None
m_value = None
g_value = None
J_values = list()
m_values = list()
g_values = list()
t_values = list()
iteration_number = 0
first_time = True

def export_model(parameters):
    if do_export:
        if candidate_law.save_from_params(parameters, out_filename):
            print('saved to %s' % out_filename)

def dump_history():
    history = {'J': J_values, 'parameters_norm': m_values, 'gradient_norm': g_values, 'times': t_values}
    if do_export:
        with open(history_filename, 'w') as outfile:
            json.dump(history, outfile, indent = 2)
            print('saved to %s' % history_filename)

def derivative_cb_post(j, dj, m):
    global first_time, iteration_number, J_value, m_value, g_value
    J_value = j
    m_value = np.linalg.norm(deshape(m))
    g_value = np.linalg.norm(deshape(dj))
    print('================ derivative_cb_post')
    print(f'Current objective function = {j:1.6e}, norm(m) = {m_value:1.6e}, norm(grad) = {g_value:1.6e}')
    if first_time and iteration_number == 0:
        first_time = False
        J_values.append(J_value)
        m_values.append(m_value)
        g_values.append(g_value)
        t_values.append(time.time() - t_opt_0)
        dump_history()
    return dj

def iter_cb(m):
    global iteration_number
    iteration_number += 1
    elapsed_time = time.time() - t_opt_0
    print(f'================ iteration {iteration_number} - {elapsed_time:.2f} s - avg time per iter {elapsed_time / iteration_number:.2f} s')
    J_values.append(J_value)
    m_values.append(m_value)
    g_values.append(g_value)
    t_values.append(elapsed_time)
    dump_history()
    export_model(list(m))
    print(f'Objective function = {J_value:1.6e}')
    print("m = ", m)

Jhat = dfa.ReducedFunctional(J, candidate_law.controls, derivative_cb_post = derivative_cb_post)
bounds = None

export_model(deshape(candidate_law.controls))

print('--------------------------------------')
print('start optimization')
print('--------------------------------------')

t_opt_0 = time.time()
if native_optimizer:
    params_opt = dfa.minimize(Jhat, bounds = bounds, method = method, tol=1e-10, options = optimizer_options, callback = iter_cb)
else:
    import pyadjoint
    Jhat_np = pyadjoint.reduced_functional_numpy.ReducedFunctionalNumPy(Jhat)

    m = [p.tape_value() for p in Jhat_np.controls]
    m_global = Jhat_np.obj_to_array(m)
    def J_safe(m):
        if early_stopping and iteration_number >= early_stopping_patience + 1:
            improvement = (J_values[-early_stopping_patience-1] - J_values[-1]) / J_values[-early_stopping_patience-1]
            print(f'Relative improvement = {improvement}')
            if improvement < early_stopping_tolerance:
                print('Optimization early stopping!')
                return +np.inf
        try:
            return Jhat_np.__call__(m)
        except Exception as e:
            print(f'Error evaluating J: {e}')
            return 1e100
            
    def dJ_safe(m):
        # Errors are probably due to concurrent compilations (e.g. on a cluster).
        # In this case, we try again for 10 times after waiting for 1 minute. If this does not work, we clean energy.json and we quit execution.
        try:
            return utils.retry_on_exception(lambda: Jhat_np.derivative(m), max_retries = 10, delay = 60)
        except Exception as e:
            print(f'Error evaluating dJ. Quitting execution.')
            if do_export and os.path.isfile(out_filename):
                os.remove(out_filename)
                print(f'Removed {out_filename}')
            raise e

    import scipy.optimize
    opt_res = scipy.optimize.minimize(J_safe, m_global, method = method, bounds = bounds,
                                    jac = dJ_safe,  hessp = Jhat_np.hessian, tol=1e-10, options = optimizer_options, callback = iter_cb)
    print('Optimization result:')
    print(opt_res)
    params_opt = Jhat_np.set_controls(np.array(opt_res["x"]))

t_opt_1 = time.time()

export_model(deshape(params_opt))
print('Final values: ' + str(deshape(params_opt)))

print(f'Time for setup       : {t_opt_0 - t_0    :.2f} s')
print(f'Time for optimization: {t_opt_1 - t_opt_0:.2f} s')
print(f'Time total           : {t_opt_1 - t_0    :.2f} s')