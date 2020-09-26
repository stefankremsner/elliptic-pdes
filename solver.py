import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from equation import AdaptiveTimestepsEquation
from torch.nn import Parameter
import sys

TH_DTYPE = torch.float32

MOMENTUM = 0.99
EPSILON = 1e-6
DELTA_CLIP = 50.0

class Dense(nn.Module):

    def __init__(self, cin, cout, batch_norm=True, activate=True):
        super(Dense, self).__init__()
        self.cout = cout
        self.linear = nn.Linear(cin, cout)
        self.activate = activate
        if batch_norm:
            self.bn = nn.BatchNorm1d(cout,eps=EPSILON, momentum=MOMENTUM)
        else:
            self.bn = None
        nn.init.normal_(self.linear.weight,std=5.0/np.sqrt(cin+cout))

    def forward(self,x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        #if self.activate:
            #x = torch.relu(x)
        if self.activate:
            x = torch.tanh(x)
        return x


class Subnetwork(nn.Module):

    def __init__(self, config, dw_dim = None):
        super(Subnetwork, self).__init__()

        self.dw_dim = config.dim
        if dw_dim is not None:
            self.dw_dim = dw_dim

        self._config = config
        self.bn = nn.BatchNorm1d(config.dim,eps=EPSILON, momentum=MOMENTUM)
        self.layers = [Dense(config.num_hiddens[i-1], config.num_hiddens[i]) for i in range(1, len(config.num_hiddens)-1)]
        self.layers += [Dense(config.num_hiddens[-2], config.num_hiddens[-1], activate=False)]
        self.layers = nn.Sequential(*self.layers)

    def forward(self,x):
        x = self.bn(x)
        x = self.layers(x)
        return x

class FeedForwardModel(nn.Module):
    """The fully connected neural network model."""
    def __init__(self, config, bsde, debug_no_z=False, use_cuda=True):
        super(FeedForwardModel, self).__init__()

        self.debug_no_z = debug_no_z
        self.use_cuda = use_cuda

        self._config = config
        self._bsde = bsde
        self._bsde.use_cuda = use_cuda

        self._dim = bsde.dim
        self._num_time_interval = bsde.num_time_interval
        self._total_time = bsde.total_time

        if self.use_cuda:
            self._y_init = Parameter(torch.Tensor([1])).cuda()
        else:
            self._y_init = Parameter(torch.Tensor([1]))

        dw_dim = None
        if hasattr(self._bsde, 'dw_dim'):
            dw_dim = self._bsde.dw_dim

        self._y_init.data.uniform_(self._config.y_init_range[0], self._config.y_init_range[1])


        self.subnetwork_threshold = 10
        number_of_networks = self._num_time_interval-1 if self._num_time_interval < self.subnetwork_threshold else self.subnetwork_threshold
        self._subnetworkList = nn.ModuleList([Subnetwork(config, dw_dim) for _ in range(number_of_networks)])


    def forward(self, x, dw):
        no_z = self.debug_no_z

        # speed up if lower dimension due to degenerate brownian motion
        dw_dim = self._bsde.dim
        if hasattr(self._bsde, 'dw_dim') and self._bsde.dw_dim < self._bsde.dim:
            dw_dim = self._bsde.dw_dim

        use_adaptive_steps = True
        if not isinstance(self._bsde, AdaptiveTimestepsEquation):
            use_adaptive_steps = False
            time_stamp = np.arange(0, self._bsde.num_time_interval) * self._bsde.delta_t

        shape = [1, dw_dim]
        if self.use_cuda:
            z_init = torch.zeros(shape).uniform_(.1, 1).to(TH_DTYPE).cuda()
        else:
            z_init = torch.zeros(shape).uniform_(.05, .05).to(TH_DTYPE)

        if self.use_cuda:
            all_one_vec = torch.ones((dw.shape[0], 1), dtype=TH_DTYPE).cuda()
        else:
            all_one_vec = torch.ones((dw.shape[0], 1), dtype=TH_DTYPE)

        y = torch.mul(all_one_vec, self._y_init)

        if no_z:
            z = torch.zeros_like(all_one_vec)
        else:
            z = torch.mul(all_one_vec, z_init)
            
        m = {
            'f': [],
            'x': [],
            'dw': [],
            'y': [],
            'z': [],
            'xi': [],
            'tau': [],
        }

        m['y'].append(y.mean())        
        m['z'].append(z.mean())        

        for t_index in range(0, self._num_time_interval-1):
            if use_adaptive_steps:
                t = self._bsde.timesteps(t_index)
                delta_t = self._bsde.adaptive_delta_t(t_index)
            else:
                delta_t = self._bsde.delta_t
                t = t_index * delta_t

            f_increment = self._bsde.indicator_at_process(
                self._bsde.f_th(t, x[:, :, t_index], y, z),
                x[:, :, t_index], t
            )

            # calculate stopping time by summing up 1 - indicator
            tau = 1.0 - self._bsde.indicator_at_process(
                1.0,
                x[:, :, t_index], t
            )

            m['tau'].append(tau.mean())        

            y = y - delta_t * f_increment

            m['f'].append(f_increment.mean())        

            # include Z network
            if not no_z:
                if dw_dim == 1:
                    dw_t = dw[:, t_index]

                    z_increment = z * dw_t

                    stopped_z_increment = self._bsde.indicator_at_process(
                    z_increment, x[:, :, t_index], t
                    )
                else:
                    z_increment = torch.sum(z * dw[:, :, t_index], dim=1, keepdim=True)
                    stopped_z_increment = self._bsde.indicator_at_process(
                        z_increment, x[:, :, t_index], t
                    )

                y = y + stopped_z_increment

                m['z'].append(z_increment.mean())        

                # only use a limited number of networks 
                if t_index >= self.subnetwork_threshold:
                    z = torch.zeros_like(all_one_vec)
                else:
                    z = self._subnetworkList[t_index](x[:, :, t_index + 1]) / self._dim

            m['y'].append(y.mean())        

        xi = self._bsde.g_th(self._total_time, x[:, :, -1])
        m['xi'].append(torch.mean(xi))        
        delta = y - xi

        # use mse loss
        loss = torch.mean(delta**2)

        return loss, self._y_init, m



