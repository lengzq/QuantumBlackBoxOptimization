import numpy as np

class Optimizer():
    
    def __init__(self, options):
        self._init_algorithm_params(options)
        self._init_optimization_params()
    
    def _init_algorithm_params(self, options):
        '''
        Learning rate is defined as a_t = a/(t + A)^a_decay
        Effective learning rate is defined as a_t/(sqrt(hat{v_t}) + delta)
        Perturbation size is defined as c_t = c/t^c_decay
        '''    
        # Initial learning rate.
        self._a = options['a']         
        # Learning rate smoothing parameter.
        self._A = options['A']   
        # Decay rate of learning rate.
        self._a_decay = options['a decay'] 
        
        # Iniitial perturbation size.
        self._c = options['c']    
        # Decay rate of perturbation size.
        self._c_decay = options['c decay'] 
        
        # Discount factor for gradient moving average.
        self._gamma = options['gamma']   
        # Decay rate for gamma.
        self._gamma_decay = options['gamma decay']   

        # Discount factor for squared gradient moving average.
        self._beta = options['beta']
        # Decay rate for beta.
        self._beta_decay = options['beta decay']
        
        # delta value for preventing exploding effective learning rate.
        self._delta = options['delta']
        
        # Repeat gradient estimation _num_repeat times and use the 
        # averaged value as gradient. 
        self._num_repeat = options['num repeat']
        assert isinstance(self._num_repeat, int), 'num repeat must be integer'
        assert self._num_repeat>=1, 'num repeat must be greater than 1'
        
    def _init_optimization_params(self):
        # Current number of iterations. 
        self._t = 0
        # List of parameters need to be optimized.
        self._thetas = []      
        # list of objective function values f(thetas).
        self._objective_values = []
        # List of thetas + pertubation values
        self._thetas_plus = []  
        # list of objective function values f(thetas+pertubation).
        self._objective_values_plus = []
        # List of thetas - pertubation values.
        self._thetas_minus = [] 
        # list of objective function values f(thetas-pertubation).
        self._objective_values_minus = []
        # List of learning rates.
        self._a_list = []
        # List of perturbation sizes.
        self._c_list = []
        # List of gradient values.
        self._g_ls = []
        # List of gradient moving averages, 
        # where m_t = beta_t * m_{t-1}  + (1 - beta_t) g_t.
        self._m_list = []
        # List of normalization factors for gradient moving averages.
        self._m_normalization_factors_list = [0.]
        # List of squared gradient moving averages,
        # where v_t = gamma_t * v_{t-1}  + (1 - gamma_t) g_t^2.
        self._v_list = []
        # List of normalization factors for squared gradient moving averages.
        self._v_normalization_factors_list = [0.]       

    def reinitialize(self, options):
        if options is not None:
            self._init_algorithm_params(options)
        self._init_optimization_params()
        
    def update_options(self, options):
        self._init_algorithm_params(options)
        
    def _optimize_one_step(self, objective_fun, theta):
        pass
    
    def optimize(self, objective_fun, theta, iterations):
        for _ in range(iterations):
            theta = self._optimize_one_step(objective_fun, theta)
        return theta
    
    def get_optimization_params(self, detailed=None):
        result = {}
        result['thetas'] = np.array(self._thetas)
        result['thetas_plus'] = np.array(self._thetas_plus)
        result['thetas_minus'] = np.array(self._thetas_minus)
        result['objective_values'] = np.array(self._objective_values)
        result['objective_values_plus'] = np.array(self._objective_values_plus)
        result['objective_values_minus'] = np.array(self._objective_values_minus)        
        return result

   