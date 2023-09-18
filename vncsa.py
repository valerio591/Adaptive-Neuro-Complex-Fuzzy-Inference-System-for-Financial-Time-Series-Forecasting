from torch import Tensor, tensor, rand, sin, cos, pow, sqrt, max, pi
from numpy import exp

def loss(params: Tensor, cmf_actual):
    # Unpack new parameters
    a,b,c,d = params.numpy()
    # Compute membership function in polar coordinates
    n = cmf_actual.size(0)
    mf_phase = tensor([(2*pi)*k/n if k != 0 else 0 for k in range(0,n)])
    mf_phase = mf_phase.view(*mf_phase.shape, 1)
    mf_rho = d*sin(a*mf_phase + b) + c
    # Transform into Cartesian form
    mf_real = mf_rho.mul(cos(mf_phase)).reshape(n)
    mf_imag = mf_rho.mul(sin(mf_phase)).reshape(n)
    distance = sqrt(pow(mf_real - cmf_actual.real, 2) + pow(mf_imag-cmf_actual.imag,2))
    return max(distance).item()

# Define logistic and Ulam von Neumann chaotic maps
def logistic_map(x):
    for j in x:
        assert 0 <= j <= 1, 'Input should belong to the interval [0,1]'
    return 4*x*(1-x)

def un_map(x, r=2):
    for j in x:
        assert -1 <= j <= 1, 'Input should belong to the interval [-1,1]'
    return 1-r*(x**2)

def vncsa(actual, loss_fn = loss, L:int = 2, M:int = 1000, T_init= 10, T_min = 0.1, T_decrease_rate = 0.06,
               d = 0.1, alpha = 0.99, omega = 0.95):
    # Define upper and lower bounds for control varibles
    lower = Tensor([-2*pi,-2*pi,0,0])
    upper = Tensor([2*pi,2*pi,1,1])
    # The algorithm begins by genrating an initial solution using the Logistic Map
    initial_solution = rand(lower.shape[0]) # Sample N observation from uniform distribution on (0,1], where N = number of control variables
    a,b,c,e = initial_solution
    condition = (0 <= e + c <= 1 and 0 <= e <= c and 0 <= c <= 1 and 0 <= e)
    while not condition:
        initial_solution = rand(lower.shape[0])
        a,b,c,e = initial_solution
        condition = (0 <= e + c <= 1 and 0 <= e <= c and 0 <= c <= 1 and 0 <= e)
    # Input the sample to logistic map to generate first chaotic solution 
    first = logistic_map(initial_solution) 
    current_solution = lower + (upper-lower)*first
    a,b,c,e = current_solution
    while not condition: 
        first = logistic_map(initial_solution) 
        current_solution = lower + (upper-lower)*first
        a,b,c,e = current_solution
        condition = (0 <= e + c <= 1 and 0 <= e <= c and 0 <= c <= 1 and 0 <= e)
    T = T_init
    D = d*(upper-lower)
    R = D
    current_best = (initial_solution, loss_fn(current_solution, actual))
    new_current_best = None
    initial_best = current_best

    # Build loop that evaluates the new solutions against the objective function
    while T > T_min:
        for step in range(L):
            # Generate new solution based on Ulam von Neumann Map
            neumann_iter = Tensor(M,lower.shape[0])
            neumann_iter[0] = rand(lower.shape[0])
            for i in range(1,M):
                neumann_iter[i] = un_map(neumann_iter[i-1])
                new_trial_solution = current_solution + neumann_iter[i]*D
                a,b,c,e = new_trial_solution
                condition = (0 <= e + c <= 1 and 0 <= e <= c and 0 <= c <= 1 and 0 <= e)
                if condition:
                    new_possible_best = (new_trial_solution, loss_fn(new_trial_solution, actual))
                    if new_possible_best[1] < current_best[1]:
                        current_best = new_possible_best
                        new_current_best = new_possible_best
                    else:
                        prob = exp(-(new_possible_best[1]-current_best[1])/T)
                        if rand(1).item() < prob:
                            new_current_best = new_possible_best
                    if  not new_current_best is None:
                        current_solution = new_current_best[0]
                    else:
                        pass
                else:
                    pass
        R = initial_best[0] - current_best[0]
        D = (1-alpha)*D + alpha*omega*R
        T = T*(1-T_decrease_rate)
        initial_best = current_best

    return current_best