import numpy as np
import Optimizer

class AdamRsgf(Optimizer.Optimizer):    
    
    def _optimize_one_step(self, objective_fun, theta):
        theta = np.array(theta)
        self._thetas += [theta]
        # Evaulate objective function
        objective_value = objective_fun(theta)
        self._objective_values += [objective_value]
        
        # Increment optimization iterations by 1.
        self._t += 1
        if self._t == 1:
            # initialze the moving average of gradient and squared gradient to zero
            self._m_list = [np.zeros_like(theta)]
            self._v_list = [np.zeros_like(theta)]
        
        # Current perturbation size
        ct = self._c/self._t**self._c_decay
        self._c_list += [ct]
        
        # Compute averaged objective values with perturbation
        cumulated_gradient = 0
        for _ in range(self._num_repeat):
            # Sample a random perturbation mu from Gaussian distribution 
            # with mean 0 and identity covariance matrix.
            mu = np.random.normal(size=theta.shape)
            theta_plus = theta + ct * mu
            self._thetas_plus += [theta_plus]
            
            objective_value_plus = objective_fun(theta_plus)
            self._objective_values_plus += [objective_value_plus]
            
            # Estimate the gradient
            gradient = (objective_value_plus - objective_value) * mu/ct
            self._g_ls += [gradient]
            cumulated_gradient += gradient
        
        gradient = cumulated_gradient/float(self._num_repeat)
        
        # Moving average of gradient and squared gradient.
        beta = self._beta/self._t ** self._beta_decay
        gamma = self._gamma/self._t ** self._gamma_decay
        m = beta * self._m_list[-1] + (1. - beta) * np.array(gradient)
        v = gamma * self._v_list[-1] + (1. - gamma) * np.array(gradient)**2
        self._m_list += [m]
        self._v_list += [v]
        
        # Normalized gradient and squared gradient.
        m_normalization_factor = beta * self._m_normalization_factors_list[-1] + (1 - beta)
        v_normalization_factor = gamma * self._v_normalization_factors_list[-1] + (1 - gamma)
        m_hat = m/m_normalization_factor
        v_hat = v/v_normalization_factor
        self._m_normalization_factors_list += [m_normalization_factor]
        self._v_normalization_factors_list += [v_normalization_factor]

        # Effective learning rate.
        at = self._a/(self._t + self._A) ** self._a_decay 
        at = at/(np.sqrt(v_hat)+ self._delta)
        self._a_list += [at]
        
        # Gradient descent to the next theta value.
        step = at * m_hat
        theta = theta - step
        return theta
    

   