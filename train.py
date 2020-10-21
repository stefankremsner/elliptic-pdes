import time
from solver import ForwardModel
import logging
import torch.optim as optim
import numpy as np
import os, sys
import torch
import dotenv
import json
import time
import multiprocessing as mp

time_started = time.time()

def train(config, bsde, file_prefix = '', debug_no_z=False, show_plot=False, debug=False, use_cuda=False, time_stamp=None):
    #if time_stamp is not None:
        #time_started = time_stamp
    if debug:
        logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)-6s %(message)s')
    else:
        logging.basicConfig(level=logging.INFO,
                            format='%(levelname)-6s %(message)s')

    if bsde.y_init:
        logging.info('Y0_true: %.4e' % bsde.y_init)

    # build and train
    net = ForwardModel(config,bsde, debug_no_z, use_cuda)

    if use_cuda:
        net.cuda()

    eq = bsde.__class__.__name__
    if eq.startswith('Laplace') or eq.startswith('NonequidistantLaplace'):
        lr = 0.05
        optimizer = optim.Adam(net.parameters(), lr=lr)
    elif eq.startswith('NonequidistantQuadraticZ'):
        lr = 0.1
        optimizer = optim.Adam(net.parameters(), lr=lr)
    else:
        lr = 0.05
        optimizer = optim.Adam(net.parameters(), lr=lr)

    start_time = time.time()
    # to save iteration results
    training_history = []

    dw_valid, x_valid = bsde.sample(config.validation_size)
    if use_cuda:
        dw_valid = dw_valid.cuda()
        x_valid = x_valid.cuda()

    if show_plot:
        bsde.plot_samples(dw_valid, x_valid)
        sys.exit(0)

    all_y = []
    all_z = []

    # begin sgd iteration
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
            xi = float(output['xi'][0])
            y_tau = float(all_y[-1][-1])
            abs_diff = xi-y_tau

            elapsed_time = time.time() - start_time
            training_history.append([step, loss, init.item(), z, elapsed_time])
            if config.verbose:
                skip = 1

                logging.debug("xi: %s" % (["{:.2f}".format(float(f)) for f in output['xi']]))
                logging.debug("tau: %s" % (["{:.2f}".format(float(f)) for f in output['tau']]))
                logging.debug("mean x[0]: %s" % (["{:.2f}".format(float(f)) for f in output['x']]))
                logging.debug("mean y: %s" % (["{:.4f}".format(float(f)) for f in output['y'][::skip]]))
                logging.debug("mean z: %s" % (["{:.2f}".format(float(f)) for f in output['z'][::skip]]))
                logging.debug("")
                logging.debug("f: %s" % (["{:.2f}".format(float(f)) for f in output['f'][::skip]]))
                logging.debug("f_dt: %s" % (["{:.2f}".format(float(f)) for f in output['f_dt'][::skip]]))

                logging.info("step: %5u,    loss: %.4e,   Y0: %.4e, (abs: %.4e) ,  elapsed time %3u" % (
                    step, loss, init.item(), abs_diff, elapsed_time))

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
        'validation_size': config.validation_size,
        'run': time_started,
    }
    param_string = ','.join(['{0}={1}'.format(k, params[k]) for k in params])


    folder = 'results/'+bsde.__class__.__name__+'-d='+str(config.dim)+'/'

    # create folder if not existent
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

    from config import get_config
    from equation import get_equation

    print('Start')
    print('---')

    dotenv.load_dotenv()

    eq = os.getenv('example')
    cfg, eq = get_config(eq)
    print('solve ', eq)
    print('---')

    show_plot = True
    show_plot = False
    debug_no_z = False

    if debug_no_z:
        print('DEBUG')
        print('no Z')

    p = 0
    pi = 0.5
    x = 0.5
    X_init = [x, *np.repeat(pi, cfg.dim-1)]

    if eq.startswith('Laplace') or eq.startswith('NonequidistantLaplace'):
        p = 0.05
        X_init = np.repeat(p, cfg.dim)
        debug_no_z = True

    if eq.startswith('NonequidistantQuadraticZ'):
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

    # set real value
    real = bsde.exact_solution(X_init)
    print('true value', real)

    train(cfg, bsde, file_prefix, debug_no_z=debug_no_z, show_plot=show_plot, debug=debug, use_cuda=use_cuda)
    print('finished', X_init)

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    duration = (time.time() - main_started)
    print('final time on {:s}: {:f}'.format(str(device), duration))