import multiprocessing as mp
import torch
import time
import dotenv
import sys, os
import numpy as np
import train

from config import get_config
from equation import get_equation

def train_parallel_point(args):
    X_init, eq, cfg, debug_no_z, show_plot, use_cuda = args
    bsde = get_equation(eq, cfg.dim, cfg.total_time, cfg.num_time_interval)

    bsde.use_cuda = use_cuda

    if hasattr(bsde, 'setXInit'):
        bsde.setXInit(X_init)

    debug = False

    file_prefix = str(X_init)
    if len(file_prefix) > 20:
        file_prefix = str(X_init[0:5])

    train.train(cfg, bsde, file_prefix, debug_no_z=debug_no_z, show_plot=show_plot, debug=debug, use_cuda=use_cuda)
    print('finished', X_init)

if __name__ == '__main__':
    print('Start')
    print('---')

    dotenv.load_dotenv()

    eq = os.getenv('example')
    print('solve ', eq)
    print('---')

    cfg, eq = get_config(eq)

    min = 0
    max = 4
    amount = 15
    linspace = np.linspace(min, max, amount)
    pi = 0.5

    if cfg.dim == 100:
        pi = 0.01

    points = [
        [p, *np.repeat(pi, cfg.dim-1)] for p in linspace
    ]

    if eq.startswith('Laplace') or eq.startswith('NonequidistantLaplace'):
        X_init = np.repeat(0, cfg.dim)
        min = -.5
        max = .5
        if cfg.dim == 100:
            min = -.05
            max = .05

        linspace = np.linspace(min, max, amount)
        points = [
            np.repeat(p, cfg.dim) for p in linspace
        ]
    elif eq.startswith('NonequidistantQuadraticZ'):
        X_init = np.repeat(0, cfg.dim)
        min = -1
        max = 1
        if cfg.dim == 100:
            min = -.1
            max = .1

        linspace = np.linspace(min, max, amount)
        points = [
            np.repeat(p, cfg.dim) for p in linspace
        ]

    print(points)

    show_plot = False
    debug_no_z = False
    use_cuda = False

    compute_parallel = True

    spare_processes = 1
    print('spare_processes', spare_processes)

    if eq.startswith('Laplace') or eq.startswith('NonequidistantLaplace'):
        debug_no_z = True

    args = [eq, cfg, debug_no_z, show_plot, use_cuda]

    time_started = time.time()

    if compute_parallel:
        processes = mp.cpu_count()-spare_processes
        if processes <= spare_processes:
            processes = 1

        print('start parallel run on devices: ', processes)
        with mp.Pool(processes=processes) as pool:
            results = pool.map(train_parallel_point, [
                [p, *args] for p in points
            ])

        print(results)

    else:
        for p in points:
            results = train_parallel_point(
                [p, *args]
            )

            print(results)

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    duration = (time.time() - time_started)
    print('duration', duration)
    print('device', device)
    duration_rel = duration/len(points)
    print('final time on {:s}: {:f}, per point: {:f}'.format(str(device), duration, duration_rel))

    if debug_no_z:
        print('DEBUG')
        print('no Z')