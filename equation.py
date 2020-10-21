import numpy as np
import torch
import copy
from scipy.stats import multivariate_normal as normal
import sys
import matplotlib.pyplot as plt
import insurancevalues

class Equation(object):
    def __init__(self, dim, total_time, num_time_interval, X_init = 0):
        self._dim = dim
        self._total_time = total_time
        self._num_time_interval = num_time_interval
        self._delta_t = (self._total_time + 0.0) / self._num_time_interval
        self._sqrt_delta_t = np.sqrt(self._delta_t)
        self._y_init = None
        self.setXInit(X_init)

    def setXInit(self, X_init):
        self._x_init = X_init * np.ones(self._dim)

    def sample(self, num_sample):
        """Sample forward SDE."""
        raise NotImplementedError

    def f_th(self, t, x, y, z):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_th(self, t, x):
        """Terminal condition of the PDE."""
        raise NotImplementedError

    def sigma_on_gradient(self, t, x, y, z):
        return z

    @property
    def y_init(self):
        return self._y_init

    @property
    def dim(self):
        return self._dim

    @property
    def num_time_interval(self):
        return self._num_time_interval

    @property
    def total_time(self):
        return self._total_time

    @property
    def delta_t(self):
        return self._delta_t

    def plot_samples(self, dw, x):
        timeaxis = np.linspace(0, self._total_time, self._num_time_interval+1)
        if isinstance(self, NonequidistantTimestepsEquation):
            timeaxis = self.Nonequidistant_timesteps

        plot_folder = 'plots'

        norms = np.linalg.norm(x, axis=1)
        for s in range(x.shape[0]):
            plt.plot(timeaxis, norms[s, :], '-')
        plt.xlabel('t')
        plt.ylabel('|X|')
        plt.savefig('{:s}/{:s}-d={:d}-norm.png'.format(plot_folder, self.__class__.__name__, self._dim))
        plt.show()

        for d in [0, self._dim-1]:
            for s in range(x.shape[0]):
                path = x[s, d, :]
                plt.plot(timeaxis, path, '-')
            plt.savefig('{:s}/{:s}-d={:d}-cord={:d}-norm.png'.format(plot_folder, self.__class__.__name__, self._dim, d))
            plt.show()


def get_equation(name, dim, total_time, num_time_interval):
    try:
        return globals()[name](dim, total_time, num_time_interval)
    except KeyError:
        raise KeyError("Equation for the required problem not found.")

class HJB(Equation):
    def __init__(self, dim, total_time, num_time_interval):
        super(HJB, self).__init__(dim, total_time, num_time_interval)
        self._x_init = np.zeros(self._dim)
        self._sigma = np.sqrt(2.0)
        self._lambda = 1.0

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample,
                                     self._dim,
                                     self._num_time_interval]) * self._sqrt_delta_t
        x_sample = np.zeros([num_sample, self._dim, self._num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self._dim]) * self._x_init
        for i in range(self._num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self._sigma * dw_sample[:, :, i]

        return torch.FloatTensor(dw_sample), torch.FloatTensor(x_sample)

    def f_th(self, t, x, y, z):
        return -self._lambda * torch.sum(torch.square(z), 1, keepdims=True)

    def g_th(self, t, x):
        return torch.log((1 + torch.sum(torch.square(x), 1, keepdims=True)) / 2)

    def indicator_at_process(self, process, x, t, dim=1):
        return process

class LaplaceOnBall(Equation):
    def __init__(self, dim, total_time, num_time_interval, b = -0.75, r = 1):
        super(LaplaceOnBall, self).__init__(dim, total_time, num_time_interval)
        self._x_init = np.zeros(self._dim)
        self._sigma = np.sqrt(2)
        self.b = b
        self._ball_radius = r

    def exact_solution_axis(self):
        factor = 1
        if self._dim == 100:
            return [-self._ball_radius/self._dim - 0.06, self._ball_radius/self._dim + 0.06]
        return [-factor * self._ball_radius, factor * self._ball_radius]

    def exact_solution(self, x):
        x_vector = x*np.ones(self._dim)
        f = (self._ball_radius**2 - np.linalg.norm(x_vector)**2)
        self._y_init = -self.b/(self._sigma**2 * self._dim) * f * (f>0)
        return self._y_init
    
    # yields multiplication with the indicator: process * 1_{\tau}(X_t), meaning outside of \tau(\omega) we get 0
    def indicator_at_process(self, process, x, t, dim=1):
        norm = torch.norm(x, dim=dim)

        norm = torch.unsqueeze(norm, 1)

        # subtract eps in order to make < work with floats
        eps = 0.000001
        simple_mask = norm < self._ball_radius-eps

        #return torch.mul(simple_mask.cuda(), process)
        return torch.mul(simple_mask, process)

    def sample(self, num_sample):
        # rewrite using torch normal
        dw_sample = normal.rvs(size=[num_sample,
                                     self._dim,
                                     self._num_time_interval]) * self._sqrt_delta_t
        x_sample = np.zeros([num_sample, self._dim, self._num_time_interval + 1])

        x_sample[:, :, 0] = np.ones([num_sample, self._dim]) * self._x_init

        eps = 10**-8

        tau = np.zeros([num_sample, 1])
        for i in range(self._num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self._sigma * dw_sample[:, :, i]

            norm = np.linalg.norm(x_sample[:,:,i+1], axis=1)          
            last_norm = np.linalg.norm(x_sample[:,:,i], axis=1)          

            for sample in range(num_sample):
                if last_norm[sample] >= self._ball_radius:
                    # if solution already jumped outside of ball with radius before
                    # then just keep the same value for x
                    x_sample[sample, :, i + 1] = x_sample[sample, :, i]
                elif norm[sample] >= self._ball_radius:
                    tau[sample] = i
                    # for the first jump exceeding the radius, project it onto the border
                    # tf.clip_by_norm(x_sample, self._ball_radius, axes=1)
                    x_sample[sample, :, i + 1] = x_sample[sample, :, i+1]*self._ball_radius/(norm[sample]-eps)

        return torch.FloatTensor(dw_sample), torch.FloatTensor(x_sample)

    def sigma_on_gradient(self, t, x, y, z):
        return self._sigma * z

    def f_th(self, t, x, y, z):
        f = -self.b * torch.ones_like(y)
        return f

    def g_th(self, t, x):
        # return 0*x
        return 0 * torch.sum(x, dim=1, keepdim=True)

class LaplaceOnSmallerBall(LaplaceOnBall):
    def __init__(self, dim, total_time, num_time_interval, b = -0.75, r = 0.5):
        super(LaplaceOnSmallerBall, self).__init__(dim, total_time, num_time_interval, b, r)

class NonequidistantTimestepsEquation(Equation):
    def __init__(self, dim, total_time, num_time_interval):
        super(NonequidistantTimestepsEquation, self).__init__(dim, total_time, num_time_interval)
        self.Nonequidistant_timesteps = self.polynomial_time_instants()

    def polynomial_time_instants(self):
        polynomial_time_exponent = 4
        # create quadratic polynom values in [0, 1] to have small time instants
        # in the beginning
        unit_times = np.power(np.linspace(0, 1, self.num_time_interval+1), polynomial_time_exponent)
        # then scale it to the total time horizon
        times = unit_times * self.total_time

        return times

    def timesteps(self, i):
        return self.Nonequidistant_timesteps[i]

    def Nonequidistant_delta_t(self, i):
        return self.timesteps(i+1) - self.timesteps(i)

class Insurance(Equation):
    def __init__(self, dim, total_time, num_time_interval):
        super(Insurance, self).__init__(dim, total_time, num_time_interval)
        self.K = 1.8
        self.delta = 0.5
        self.X_max = 4

        self.rho = 1

        self.mu_states = np.linspace(1, 2, self._dim)

        self.q_states = np.zeros((self._dim, self._dim))
        for i in range(self._dim):
            for j in range(self._dim):
                if i == j and i % 2 == 0:
                    self.q_states[i][j] = -0.5
                if i == j and i % 2 == 1:
                    self.q_states[i][j] = -0.25
                if i == j+1 and i % 2 == 0:
                    self.q_states[i][j] = 0.25
                if i == j+1 and j+1 >= 3:
                    self.q_states[i][j] = 0.25
                else:
                    self.q_states[i][j] = 0

    def exact_solution_axis(self):
        return [0, 4]

    def exact_solution(self, x):
        if self._dim == 100:
            return np.nan
        return insurancevalues.solution(x)

    def drift_coefficient(self, t, x):
        n = x.shape[0]
        drift = np.zeros((n, self._dim))

        # coordinate d is treated as 0 here
        mu_d = self.mu_states[0]
        drift[:, 0] = self.mu_states[0]
        for i in range(1, self._dim):
            mu_i = self.mu_states[i]
            drift[:, 0] += (mu_i-mu_d)*x[:,i]

        for i in range(1, self._dim):
            q_di = self.q_states[0, i]

            s = q_di
            for j in range(1, self._dim):
                q_ji = self.q_states[j, i]
                s += (q_ji - q_di)*x[:,j] 

            drift[:, i] = s

        return drift

    # form as vector, not matrix
    def diffusion_coefficient(self, t, x):
        n = x.shape[0]
        diff = np.zeros((n, self._dim))
        diff[:, 0] = self.rho

        # coordinate d is treated as 0 here
        mu_d = self.mu_states[0]
        for i in range(1, self._dim):
            mu_i = self.mu_states[i]

            s = mu_i - mu_d
            for j in range(1, self._dim):
                mu_j = self.mu_states[j]
                s += (mu_j - mu_d)*x[:,j] 

            diffusion = x[:,i]*s/self.rho

            diff[:, i] = diffusion

        return diff
        
    def indicator_at_process(self, process, x, t, dim=1):
        Z = torch.unsqueeze(x[:,0], 1)
        simple_mask_below = (Z <= 0) 
        simple_mask_top = (Z >= self.X_max)

        # yields a logical or (bool + bool bytewise)
        simple_mask = (simple_mask_below + simple_mask_top) == 0

        return simple_mask * process

    def sigma_on_gradient(self, t, x, y, z):
        sig = torch.FloatTensor(self.diffusion_coefficient(t, x))
        sig_dim_z = torch.unsqueeze(sig[:, 0], dim=1)

        if self.dw_dim == 1:
            return z * sig_dim_z

        return torch.mul(z_single, sig_dim_z)
        

    def f_th(self, t, x, y, z):
        V_x = z[:,0:1]
        ind = V_x <= 1

        f = -self.delta * y + self.K*(1-V_x)*ind
        return self.indicator_at_process(f, x, t)

    def g_th(self, t, x):
        X = torch.unsqueeze(x[:,0], 1)
        return (X >= self.X_max) * self.K/self.delta

class NonequidistantTimestepsInsurance(Insurance, NonequidistantTimestepsEquation):
    def __init__(self, dim, total_time, num_time_interval):
        super(NonequidistantTimestepsInsurance, self).__init__(dim, total_time, num_time_interval)
        super(Insurance, self).__init__(dim, total_time, num_time_interval)
        self.dw_dim = 1

    def sample(self, num_sample):
        delta_t = np.diff(self.Nonequidistant_timesteps)
        sqrt_delta_t = np.sqrt(delta_t)

        gauss_sample = normal.rvs(size=[num_sample,
                                    self.dw_dim,
                                    self._num_time_interval])

        dw_sample = np.zeros_like(gauss_sample)

        for i in range(len(sqrt_delta_t)):
            sdt = sqrt_delta_t[i]
            if self.dw_dim == 1:
                dw_sample[:, i] = gauss_sample[:, i] * sdt
            else:
                dw_sample[:, :, i] = gauss_sample[:, :, i] * sdt

        x_sample = np.zeros([num_sample, self._dim, self._num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self._dim]) * self._x_init

        n = x_sample[0]

        dw_sample_all = np.tile(dw_sample, (self._dim,1,1))
        dw_sample_all = np.moveaxis(dw_sample_all, 0, 1)

        stop_process = True

        for i in range(self._num_time_interval):
            t = self.timesteps(i)
            delta_t = self.Nonequidistant_delta_t(i)

            x_last = x_sample[:, :, i]
            drift_coeff = self.drift_coefficient(t, x_last) 
            diffusion_coeff = self.diffusion_coefficient(t, x_last)
            if self.dw_dim == 1:
                m = np.multiply(diffusion_coeff, dw_sample_all[:, :, i])
                x_sample[:, :, i + 1] = x_sample[:, :, i] + drift_coeff * delta_t + np.multiply(diffusion_coeff, dw_sample_all[:, :, i])
            else:
                raise Exception('not implemented')
                
            if stop_process:
                Z = x_sample[:,0,i+1]
                Z_last = x_sample[:,0,i]
                eps = 0.001

                # vectorize
                for sample in range(num_sample):
                    if Z_last[sample] <= 0+eps or Z_last[sample] >= self.X_max:
                        x_sample[sample, :, i + 1] = x_sample[sample, :, i]
                    elif Z[sample] <= 0+eps or Z[sample] >= self.X_max:
                        x_sample[sample, :, i + 1] = x_sample[sample, :, i+1]
                        if i > 0 and Z[sample] <= 0+eps:
                            x_sample[sample, 0, i + 1] = 0
                        else:
                            x_sample[sample, 0, i + 1] = self.X_max

        return torch.FloatTensor(dw_sample), torch.FloatTensor(x_sample)
class NonequidistantLaplaceOnSmallerBall(LaplaceOnSmallerBall, NonequidistantTimestepsEquation):
    def __init__(self, dim, total_time, num_time_interval, b = -0.75, r = 0.5):
        super(NonequidistantTimestepsEquation, self).__init__(dim, total_time, num_time_interval)
        super(NonequidistantLaplaceOnSmallerBall, self).__init__(dim, total_time, num_time_interval)
        self._x_init = np.zeros(self._dim)
        self._sigma = np.sqrt(2)
        self.b = b
        self._ball_radius = r
        self.dw_dim = dim
    
    def sample(self, num_sample):
        delta_t = np.diff(self.Nonequidistant_timesteps)
        sqrt_delta_t = np.sqrt(delta_t)

        gauss_sample = normal.rvs(size=[num_sample,
                                    self.dw_dim,
                                    self._num_time_interval])

        dw_sample = np.zeros_like(gauss_sample)

        for i in range(len(sqrt_delta_t)):
            sdt = sqrt_delta_t[i]
            if self.dw_dim == 1:
                dw_sample[:, i] = gauss_sample[:, i] * sdt
            else:
                dw_sample[:, :, i] = gauss_sample[:, :, i] * sdt

        x_sample = np.zeros([num_sample, self._dim, self._num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self._dim]) * self._x_init

        n = x_sample[0]

        eps = 10**-8
        stop_process = True

        for i in range(self._num_time_interval):
            t = self.timesteps(i)
            delta_t = self.Nonequidistant_delta_t(i)

            x_last = x_sample[:, :, i]
            drift_coeff = 0
            diffusion_coeff = self._sigma
            
            gain = drift_coeff * delta_t + np.multiply(diffusion_coeff, dw_sample[:, :, i])
            x_sample[:, :, i + 1] = x_sample[:, :, i] + gain

            if stop_process:
                eps = 0.000001
                norm = np.linalg.norm(x_sample[:,:,i+1], axis=1)          
                last_norm = np.linalg.norm(x_sample[:,:,i], axis=1)          

                for d in range(self._dim):
                    x_sample[:, d, i + 1] = np.where(
                        last_norm >= self._ball_radius,
                        x_sample[:, d, i],
                        np.where(
                            norm >= self._ball_radius,
                                x_sample[:, d, i+1]*self._ball_radius/(norm-eps),
                                x_sample[:, d, i + 1]
                            ),
                    )

        return torch.FloatTensor(dw_sample), torch.FloatTensor(x_sample)

class NonequidistantQuadraticZ(NonequidistantLaplaceOnSmallerBall):
    def __init__(self, dim, total_time, num_time_interval, b = -0.75, r = 1):
        super(NonequidistantQuadraticZ, self).__init__(dim, total_time, num_time_interval, b=b, r=r)

    def exact_solution_axis(self):
        if self._dim == 100:
            return [-0.11, 0.11]
        return [-1, 1]

    def exact_solution(self, x):
        f = np.log((np.sum(x**2)+1)) - np.log(self._dim)
        self._y_init = f
        return self._y_init

    def f_th(self, t, x, y, z):
        pre_factor = 1

        exp_factor = 1
        f = pre_factor*(torch.sum(z**2, dim=1, keepdim=True) - 2*torch.exp(-exp_factor*y))

        return self.indicator_at_process(f, x, t)

    def g_th(self, t, x):
        return torch.log((1 + torch.sum(torch.square(x), 1, keepdims=True)) / self._dim)