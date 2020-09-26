import numpy as np
import argparse
import sys

# define the scripts input parameter args
def readFromCmdLine(cfg, args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command for BSDE solver setting.")
    parser.add_argument("-E", "--example", help="BSDE example to solve")
    parser.add_argument("-D", "--dim", type=int, help="Number dimension.")
    parser.add_argument("-N", "--num_time_interval", type=int, help="Number of time steps.")
    parser.add_argument("-T", "--total_time", type=float, help="Final time T.")
    parser.add_argument("-BS", "--batch_size", type=int, help="Batch size.")
    parser.add_argument("-VS", "--valid_size", type=int, help="Validation size.")
    parser.add_argument("-I", "--num_iterations", type=int, help="Number of iterations.")
    options = parser.parse_args(args)

    args_dict = vars(parser.parse_args())

    # populate args into config
    for item in args_dict:
        if args_dict[item] is not None:
            setattr(cfg, item, args_dict[item])

    return options

class Config(object):
    batch_size = 128
    valid_size = 256
    num_time_interval = 200
    num_iterations = 500
    logging_frequency = 10
    verbose = True
    y_init_range = [0, 1]

class LaplaceOnBallConfig(Config):
    dim = 100
    y_init_range = [0, 1.5]
    num_hiddens = [dim, dim]
    num_iterations = 100

    # d = 2
    total_time = 5
    num_time_interval = 500
    valid_size = 64
    batch_size = 256

    # comment out for dim = 2
    # d = 100
    total_time = 0.01
    num_time_interval = 100
    batch_size = 64
    valid_size = 256


class LaplaceOnSmallerBallConfig(LaplaceOnBallConfig):
    # d = 2
    total_time = 1
    y_init_range = [0, 1]

    # comment out for dim = 2
    # d = 100
    total_time = 0.005
    y_init_range = [0, 0.005]
    

class AdaptiveLaplaceOnSmallerBallConfig(LaplaceOnSmallerBallConfig):
    pass

class InsuranceConfig(Config):
    # d = 2 and d = 100
    total_time = 5
    num_time_interval = 100
    dim = 100
    num_iterations = 300
    num_hiddens = [dim, 1]
    y_init_range = [0, 0.5]
    batch_size = 64
    valid_size = 128

class AdaptiveTimestepsInsuranceConfig(InsuranceConfig):
    pass

class AdaptiveQuadraticZConfig(LaplaceOnSmallerBallConfig):
    dim = 100
    num_hiddens = [dim, dim]
    valid_size = 256
    batch_size = 64
    y_init_range = [-1, 0]
    num_iterations = 300

    # d = 2
    total_time = 5
    num_time_interval = 500

    # comment out for dim = 2
    # d = 100
    total_time = 0.1
    num_time_interval = 50

def get_config_no_args(name):
    try:
        cfg = globals()[name+'Config']
        #readFromCmdLine(cfg)
        return cfg
    except KeyError:
        raise KeyError("Config for the required problem not found.")

def get_config(name):
    try:
        cfg = globals()[name+'Config']
        readFromCmdLine(cfg)

        # if other input name found
        if hasattr(cfg, 'example') and cfg.example is not None and cfg.example != name:
            name = cfg.example
            cfg = globals()[name+'Config']
            readFromCmdLine(cfg)

        return cfg, name
    except KeyError:
        raise KeyError("Config for the required problem not found.")
