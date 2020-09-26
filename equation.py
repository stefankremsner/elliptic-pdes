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
        raise NotImplementedError

    def f_th(self, t, x, y, z):
        raise NotImplementedError

    def g_th(self, t, x):
        raise NotImplementedError

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
        if isinstance(self, AdaptiveTimestepsEquation):
            timeaxis = self.adaptive_timesteps

        plot_folder = 'plots'

        norms = np.linalg.norm(x, axis=1)
        for s in range(x.shape[0]):
            plt.plot(timeaxis, norms[s, :], '-')
        plt.xlabel('t')
        plt.ylabel('|X|')
        plt.savefig('{:s}/{:s}-d={:d}-norm.png'.format(plot_folder, self.__class__.__name__, self._dim))
        plt.show()

        start_norm = np.mean(np.linalg.norm(x[:, :, 0], axis=1))
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
    def __init__(self, dim, total_time, num_time_interval, b = 0.75, r = 1):
        super(LaplaceOnBall, self).__init__(dim, total_time, num_time_interval)
        self._x_init = np.zeros(self._dim)
        self._sigma = 1
        self.b = b
        self._ball_radius = r

    def exact_solution_axis(self):
        factor = 1
        if self._dim == 100:
            factor = 0.2
        return [-factor * self._ball_radius, factor * self._ball_radius]

    def exact_solution(self, x):
        x_vector = x*np.ones(self._dim)
        f = (self._ball_radius**2 - np.linalg.norm(x_vector)**2)
        self._y_init = self.b/(self._dim) * f * (f>0)
        return self._y_init
    
    # yields multiplication with the indicator: process * 1_{\tau}(X_t), meaning outside of \tau(\omega) we get 0
    def indicator_at_process(self, process, x, t, dim=1):
        norm = torch.norm(x, dim=dim)

        dim = self._dim

        simple_mask = norm < self._ball_radius

        return torch.mul(simple_mask, process)

    def sample(self, num_sample):
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

    def f_th(self, t, x, y, z):
        f = self.b * torch.ones_like(y)
        return self.indicator_at_process(f, x, t)

    def g_th(self, t, x):
        return 0 * torch.sum(x, dim=1, keepdim=True)

class LaplaceOnSmallerBall(LaplaceOnBall):
    def __init__(self, dim, total_time, num_time_interval, b = 0.75, r = 0.5):
        super(LaplaceOnSmallerBall, self).__init__(dim, total_time, num_time_interval, b, r)

class AdaptiveTimestepsEquation(Equation):
    def __init__(self, dim, total_time, num_time_interval, X_init = 0):
        super(AdaptiveTimestepsEquation, self).__init__(dim, total_time, num_time_interval)
        self.adaptive_timesteps = self.polynomial_time_instants()

    def polynomial_time_instants(self):
        polynomial_time_exponent = 4
        # create quadratic polynom values in [0, 1] to have small time instants
        # in the beginning
        unit_times = np.power(np.linspace(0, 1, self.num_time_interval+1), polynomial_time_exponent)
        # then scale it to the total time horizon
        times = unit_times * self.total_time

        return times

    def timesteps(self, i):
        return self.adaptive_timesteps[i]

    def adaptive_delta_t(self, i):
        return self.timesteps(i+1) - self.timesteps(i)

class Insurance(Equation):
    def __init__(self, dim, total_time, num_time_interval, X_init = 0.5):
        super(Insurance, self).__init__(dim, total_time, num_time_interval)
        self.setXInit(X_init)
        self.K = 0.2
        self.delta = 0.5
        self.X_max = 5

        self.w_sigma = 1

        self.mu_states = np.linspace(2, 1, self._dim)

        q_values = []
        for i in range(self._dim):
            v = 0.25
            if i % 2 == 1:
                v = 0.5
            q_values.append(v)

        self.q_states = np.zeros((self._dim, self._dim))
        for i in range(self._dim):
            for j in range(self._dim):
                q = q_values[i]
                if i == j:
                    self.q_states[i][j] = -q
                if i == j+1:
                    self.q_states[i][j] = q
                if i == 0 and j == 1:
                    self.q_states[i][j] = q

    def exact_solution_axis(self):
        return [0, 4]

    def exact_solution(self, x):
        if self._dim == 100:
            return np.nan
        return insurancevalues.solution(x)

    def calculate_nu(self, t, x):
        M = self._dim

        # create nu
        mu_M = self.mu_states[M-1]
        nu = mu_M 
        for j in range(1,M):
            mu_j = self.mu_states[j-1]
            nu = nu + (mu_j - mu_M)*x[:,j]

        return nu
        
    def drift_coefficient(self, t, x):
        M = self._dim
        nu = self.calculate_nu(t, x)

        coeffs = np.zeros_like(x)
        coeffs[:, 0] = nu

        for i in range(1, self._dim):
            q_Mi = self.q_states[M-1,i-1]
            drift = q_Mi
            for j in range(0,M-1):
                qji = -self.q_states[i-1, j]
                drift = drift + (qji - q_Mi)*x[:,j+1]

            coeffs[:, i] = drift

        return coeffs

    def diffusion_coefficient(self, t, x):
        coeff_first = np.array(self.w_sigma)

        n = x.shape[0]
        coeffs = np.zeros((n, self._dim))
        coeffs[:, 0] = np.repeat(coeff_first, n)

        nu = self.calculate_nu(t, x)

        for i in range(1, self._dim):
            pi = x[:,i]
            mu_i = self.mu_states[i]
            diffusion = pi * (mu_i - nu)/self.w_sigma

            coeffs[:, i] = diffusion

        return coeffs
        
    def indicator_at_process(self, process, x, t, dim=1):
        Z = x[:,0]
        simple_mask_below = (Z <= 0) 
        simple_mask_top = (Z >= self.X_max)

        # yields a logical or (bool + bool bytewise)
        simple_mask = (simple_mask_below + simple_mask_top) == 0

        return simple_mask * process

    def f_th(self, t, x, y, z):
        # first component of z is the gradient V_x 
        V_x = z[:,0]
        ind = V_x <= 1
        f = self.delta * y - self.K*(1-V_x)*ind
        
        return self.indicator_at_process(f, x, t)

    def g_th(self, t, x):
        X = x[:,0]
        return (X >= self.X_max) * self.K/self.delta


class AdaptiveTimestepsInsurance(Insurance, AdaptiveTimestepsEquation):
    def __init__(self, dim, total_time, num_time_interval, X_init = 0.5):
        super(AdaptiveTimestepsInsurance, self).__init__(dim, total_time, num_time_interval)
        super(Insurance, self).__init__(dim, total_time, num_time_interval)
        self.dw_dim = 1

    def sample(self, num_sample):
        delta_t = np.diff(self.adaptive_timesteps)
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
            delta_t = self.adaptive_delta_t(i)

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
class AdaptiveLaplaceOnSmallerBall(LaplaceOnSmallerBall, AdaptiveTimestepsEquation):
    def __init__(self, dim, total_time, num_time_interval, b = 0.75, r = 0.5):
        super(AdaptiveTimestepsEquation, self).__init__(dim, total_time, num_time_interval)
        super(AdaptiveLaplaceOnSmallerBall, self).__init__(dim, total_time, num_time_interval)
        self._x_init = np.zeros(self._dim)
        self._sigma = 1
        self.b = b
        self._ball_radius = r
        self.dw_dim = dim
    
    def sample(self, num_sample):
        delta_t = np.diff(self.adaptive_timesteps)
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
            delta_t = self.adaptive_delta_t(i)

            x_last = x_sample[:, :, i]
            drift_coeff = 0
            diffusion_coeff = 1
            
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

class AdaptiveQuadraticZ(AdaptiveLaplaceOnSmallerBall):
    def __init__(self, dim, total_time, num_time_interval, b = 0.75, r = 1):
        super(AdaptiveQuadraticZ, self).__init__(dim, total_time, num_time_interval, b=b, r=r)

    def exact_solution_axis(self):
        if self._dim == 100:
            return [-0.1, 0.1]
        return [-1, 1]

    def exact_solution(self, x):
        f = np.log((np.sum(x**2)+1)) - np.log(2)
        self._y_init = f * (f<0)
        return self._y_init

    def f_th(self, t, x, y, z):
        pre_factor = 1/2
        f = pre_factor * (torch.sum(z**2, dim=1, keepdim=True) - 2*self._dim/(torch.sum(x**2, axis=1)+1))
        return self.indicator_at_process(f, x, t)