# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import os


def evaluation(lmd, mu, kpa, bta):
    # print('>>> [lmd, mu, kpa, bta] = [%.6f, %.6f, %.6f, %.6f]' % (lmd, mu, kpa, bta))
    print('=====================Mesh Generation======================')
    # TPMS11_Mod.mesh_generation(lmd, mu, kpa, bta, plot='N', len_pct=1.5)  # len_pct=1.35
    print('=====================Pre-Processing=======================')
    os.system('abaqus cae noGUI=TPMS11\\TPMS11_PrePrc.py')
    print('========================Reading===========================')
    with open('result.txt', 'r') as file:
        line = file.readline().strip()
    # print('==================Solving & monitoring====================')
    # os.system('abaqus cae noGUI=TPMS11\\TPMS11_PostPrc.py')
    # print('=====================Post-Processing======================')
    return float(line)


# Define the search space
space = [Real(0.0, 0.2, name='lmd'),
         Real(0.0, 0.2, name='mu'),
         Real(0.0, 0.1, name='kpa'),
         Real(0, 0.5, name='bta')]

'''
# Case1
space = [Real(0.0, 1.0, name='lmd'),
         Real(0.0, 1.0, name='mu'),
         Real(0.0, 0.5, name='kpa'),
         Real(-0.5, 0.5, name='bta')]'''

# Dummy objective function for the sake of optimization (will not be used for actual evaluation)
@use_named_args(space)
def objective(**params):
    return 0  # Dummy return value, as we are not using this function for actual evaluation


# Load initial guesses from a CSV file
def load_initial_guesses(filename=''):
    df = pd.read_csv(filename)
    x_ini = df.iloc[:, :4].values.tolist()
    y_ini = df.iloc[:, 4].tolist()
    return x_ini, y_ini


def bo_sampling(filename, acq_func='LCB'):
    n_calls = 10
    # Load initial guesses
    x0, y0 = load_initial_guesses(filename)
    last_sample_evaluated = True
    if np.isnan(y0[-1]):
        x0, y0 = x0[:-1], y0[:-1]
        last_sample_evaluated = False

    # x0=x0, y0=y0,
    # Perform Bayesian optimization
    res = gp_minimize(func=objective, dimensions=space, n_calls=n_calls,
                      n_initial_points=10, initial_point_generator='random',
                      x0=x0, y0=y0, acq_func=acq_func, acq_optimizer='sampling', random_state=0)

    print(res)
    next_sample = res.x_iters[-n_calls]
    next_sample_rounded = [round(x, 4) for x in next_sample]
    print("Next sampling point:")
    print(", ".join(map(str, next_sample_rounded)))

    # Record the new sample point
    if last_sample_evaluated:
        df = pd.read_csv(filename)
        df.loc[len(df)] = next_sample_rounded + [np.nan, np.nan]
        df.to_csv(filename, index=False)


def whole_process(cmd):
    # sampling
    if (cmd // 100) % 10 == 1:
        bo_sampling(acq_func='LCB')
    elif (cmd // 100) % 10 == 2:
        bo_sampling(acq_func='EI')
    elif (cmd // 100) % 10 == 3:
        bo_sampling(acq_func='PI')

    # modeling
    if (cmd // 10) % 10 == 1:
        import TPMS11_Mod
    elif (cmd // 10) % 10 == 2:
        import TPMS11_Mod_obj2inp

    # simulation
    if cmd % 10 == 0:
        pass
        # os.system('abaqus cae script=TPMS11_PrePrc.py -- 0')
    elif cmd % 10 == 1:
        os.system('abaqus cae script=TPMS11_PrePrc.py -- 1')
    elif cmd % 10 == 2:
        os.system('abaqus cae noGUI=TPMS11_PostPrc.py')
    elif cmd % 10 == 3:
        os.system('abaqus cae script=TPMS11_PostPrc_after.py')
        # os.system('abaqus cae noGUI=TPMS11\\TPMS11_PostPrc_after.py')

if __name__ == '__main__'
    whole_process(10)
    # 211 => Sampling +　Modeling + TPMS11_PrePrc
    # 11 => Modeling + TPMS11_PrePrc + TPMS11_PostPrc
    # 21 => Update INP file + TPMS11_PrePrc + TPMS11_PostPrc
    # 0 => TPMS11_PrePrc
    # 1 => TPMS11_PrePrc + TPMS11_PostPrc
    # 2 => TPMS11_PostPrc
    # 3 => TPMS11_PostPrc_after
