import time
from solver import FeedForwardModel
import logging
import torch.optim as optim
import numpy as np
import os, sys
import torch
import json
import time
import multiprocessing as mp

from config import get_config
from equation import get_equation

time_started = time.time()

def train(config, bsde, file_prefix = '', debug_no_z=False, show_plot=False, debug=False, use_cuda=False, time_stamp=None):
    if debug:
        logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)-6s %(message)s')
    else:
        logging.basicConfig(level=logging.INFO,
                            format='%(levelname)-6s %(message)s')

    if bsde.y_init:
        logging.info('Y0_true: %.4e' % bsde.y_init)

    # build and train
    net = FeedForwardModel(config,bsde, debug_no_z, use_cuda)

    if use_cuda:
        net.cuda()

    eq = bsde.__class__.__name__
    if eq.startswith('Laplace') or eq.startswith('AdaptiveLaplace'):
        lr = 0.05
        optimizer = optim.Adam(net.parameters(), lr=lr)
    elif eq.startswith('AdaptiveQuadraticZ'):
        lr = 0.1
        optimizer = optim.Adam(net.parameters(), lr=lr)
    else:
        lr = 0.1
        optimizer = optim.Adam(net.parameters(), lr=lr)

    start_time = time.time()
    training_history = []

    dw_valid, x_valid = bsde.sample(config.valid_size)
    if use_cuda:
        dw_valid = dw_valid.cuda()
        x_valid = x_valid.cuda()

    if show_plot:
        bsde.plot_samples(dw_valid, x_valid)
        sys.exit(0)

    all_y = []
    all_z = []

    for step in range(config.num_iterations + 1):
        if step % config.logging_frequency == 0:
            net.eval()

            if use_cuda:
                loss, init, output = net(x_valid.cuda(), dw_valid.cuda())
            else:
                loss, init, output = net(x_valid, dw_valid)

            z = output['z'][0].mean()

            all_y.append([float(y) for y in output['y']])
            all_z.append([float(z) for z in output['z']])

            elapsed_time = time.time() - start_time
            training_history.append([step, loss, init.item(), z, elapsed_time])
            if config.verbose:
                skip = 1
                times = [i for i in range(bsde.num_time_interval)]

                logging.debug("points: %s" % ([f for f in times[::skip]]))
                logging.debug("y: %s" % (["{:.4f}".format(float(f)) for f in output['y'][::skip]]))
                logging.debug("z: %s" % (["{:.2f}".format(float(f)) for f in output['z'][::skip]]))
                logging.debug("f: %s" % (["{:.2f}".format(float(f)) for f in output['f'][::skip]]))
                logging.debug("xi: %s" % (["{:.2f}".format(float(f)) for f in output['xi']]))
                logging.debug("tau: %s" % (["{:.2f}".format(float(f)) for f in output['tau']]))

                logging.info("step: %5u,    loss: %.4e,   Y0: %.4e,  elapsed time %3u" % (
                    step, loss, init.item(), elapsed_time))

        dw_train, x_train = bsde.sample(config.batch_size)
        optimizer.zero_grad()
        net.train()
        if use_cuda:
            loss, _, _ = net(x_train.cuda(), dw_train.cuda())
        else:
            loss, _, _ = net(x_train, dw_train)
        loss.backward()

        optimizer.step()

    training_history = np.array(training_history)

    if bsde.y_init:
        logging.info('relative error of Y0: %s',
                     '{:.2%}'.format(
                         abs(bsde.y_init - training_history[-1, 2]) / bsde.y_init))

    params = {
        'N_T': config.num_time_interval,
        'T': config.total_time,
        'batch_size': config.batch_size,
        'valid_size': config.valid_size,
        'run': time_started,
    }
    param_string = ','.join(['{0}={1}'.format(k, params[k]) for k in params])


    folder = 'results/'+bsde.__class__.__name__+'-d='+str(config.dim)+'/'

    # create folder if not existing
    if not os.path.isdir(folder):
        os.makedirs(folder)

    filename = folder + file_prefix + param_string + '_' + bsde.__class__.__name__

    np.savetxt('{}_training_history.csv'.format(filename),
                training_history,
                fmt=['%d', '%.5e', '%.5e', '%.5e', '%d'],
                delimiter=",",
                header="step,loss_function,y0,z0,elapsed_time",
                comments='')

    filename = folder + file_prefix + param_string + '_process_' + bsde.__class__.__name__

    np.savetxt('{}_all_y.csv'.format(filename),
                np.array(all_y),
                delimiter=",",
                comments='')
    np.savetxt('{}_all_z.csv'.format(filename),
                np.array(all_z),
                delimiter=",",
                comments='')

    return bsde.y_init

if __name__ == '__main__':
    main_started = time.time()

    print('Start')
    print('---')

    eq = "AdaptiveTimestepsInsurance"
    cfg, eq = get_config(eq)

    show_plot = False
    debug_no_z = False

    p = 0
    pi = 0.1
    x = 1
    X_init = [x, *np.repeat(pi, cfg.dim-1)]

    if eq.startswith('Laplace') or eq.startswith('AdaptiveLaplace'):
        X_init = np.repeat(p, cfg.dim)
        debug_no_z = True

    if eq.startswith('AdaptiveQuadraticZ'):
        X_init = np.repeat(p, cfg.dim)

    print('---', X_init)
    bsde = get_equation(eq, cfg.dim, cfg.total_time, cfg.num_time_interval)

    if hasattr(bsde, 'setXInit'):
        bsde.setXInit(X_init)

    file_prefix = str(X_init)
    if len(file_prefix) > 20:
        file_prefix = str(X_init[0:5])

    use_cuda = False
    debug = True

    train(cfg, bsde, file_prefix, debug_no_z=debug_no_z, show_plot=show_plot, debug=debug, use_cuda=use_cuda)
    print('finished', X_init)

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    duration = (time.time() - main_started)
    print('final time on {:s}: {:f}'.format(str(device), duration))