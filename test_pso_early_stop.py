import sys
import time
import pandas as pd
# import booleanNetwork module from ./src
sys.path.append('./src')
from BNMPy import PBN, BMatrix
from BNMPy.steady_state import SteadyStateCalculator
from Optimizer.experiment_data import ExperimentData
from Optimizer.parameter_optimizer import ParameterOptimizer

# load the network to be optimized
# pbn = BMatrix.load_pbn_from_file('../input_files/Trairatphisan2014_case3.txt')
string = """
TGFa = 1
TNFa = 1
Raf = TGFa, 1
PI3K = TGFa, 1
Akt = PI3K
C8 = TNFa
NFkB = PI3K, 0.5
NFkB = TNFa, 0.5
ERK = Raf, 0.5
ERK = NFkB, 0.5
"""
pbn = BMatrix.load_pbn_from_string(string)

# Configure optimizer
config = {
    'pso_params': {
        'n_particles': 40,  # 20*number of parameters to optimize
        'iters': 1000,
        'options': {
            'c1': 0.5, # Cognitive parameter
            'c2': 0.3, # Social parameter
            'w': 0.9 # Inertia weight
        }
    },
    'steady_state': {
        'method': 'tsmc',
            'tsmc_params': {
                'epsilon': 0.001, # range of transition probability [Default=0.001]
                'r': 0.01, # range of accuracy (most sensitive) [Default=0.025]
                's': 0.95, # probability to acquire defined accuracy [Default=0.95]
                'p_mir': 0.001, # perturbation in Miranda's & Parga's scheme [Default=0.001, 0.1%]
                'initial_nsteps': 100, # initial simulation steps [Recommended at 100 steps for n < 100]
                'max_iterations': 5000 # maximum convergence iterations [Default=5000]
            },
    },
    'early_stopping_params': {
        'early_stopping': True,  # Enable/disable early stopping
        'early_stopping_mode': 'success_threshold',  # 'success_threshold' or 'ftol'
        'success_threshold': 0.00014984,  # SSE threshold for success_threshold mode
        'ftol': -1,  # Function tolerance for ftol mode (-1 = use method defaults)
        'ftol_iter': 10,  # Iterations to check for ftol mode
    },
    'max_try': 2  # Maximum number of attempts if optimization fails
}


# Initialize optimizer
optimizer = ParameterOptimizer(pbn, './data/Trairatphisan2014_case3.csv', config=config, nodes_to_optimize=['NFkB','ERK'], verbose=True)

# Run optimization
start_time = time.time()
result = optimizer.optimize(method='particle_swarm')
end_time = time.time()
print(f"Optimization took {end_time - start_time:.2f} seconds")
