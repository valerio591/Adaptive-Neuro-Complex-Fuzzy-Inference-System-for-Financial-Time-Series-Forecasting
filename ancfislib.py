import torch
from torch.utils.data import IterableDataset
from torch import nn
from collections import OrderedDict
from typing import Union
from vncsa import *

# Define function to use for overriding FUzzifyLayer backward method
def optimize_params(grad_output):
        new_params, error = vncsa(grad_output, T_init = 40)
        return new_params
    
# Fuzzify Variable 
class FuzzifyVariable():
    def __init__(self, input_size, mf_params, nmfs):
        self.nmfs = nmfs
        self.cmf = None
        n = input_size
        self.sample_phase = torch.tensor([(2*pi)*k/n if k != 0 else 0 for k in range(0,n)], requires_grad=False)
        if mf_params is not None:
            self.a, self.b, self.c, self.d = mf_params.values()
        else:
            self.a = self.b = torch.randn((1, nmfs), requires_grad = False)
            self.c = torch.rand(1, nmfs)
            self.d = torch.rand(1, nmfs)*(1-torch.max(self.c))
        # Check that params constraints are satisfied
        a,b,c,d = self.a, self.b, self.c, self.d
        if a is torch.Tensor:
            assert torch.all(0 <= d + c), 'Invalid parameters, constraint check failed: d + c >= 0'
            assert torch.all(d + c <= 1), 'Invalid parameters, constraint check failed: d + c <= 1'
            assert torch.all(0 <= d), 'Invalid parameters, constraint check failed: d >= 0'
            assert torch.all(d <= c), 'Invalid parameters, constraint check failed: d <= c'
            assert torch.all(0 <= c), 'Invalid parameters, constraint check failed: c >= 0'
            assert torch.all(c <= 1), 'Invalid parameters, constraint check failed: c <= 1'
    
    def forward(self):
        # Compute amplitude from phase
        a,b,c,d = self.a, self.b, self.c, self.d
        mf_phase = self.sample_phase
        mf_phase = mf_phase.view(*mf_phase.shape, 1)
        mf_rho = d*sin(a*mf_phase + b) + c
        # Transform into Cartesian form
        mf_real = mf_rho.mul(cos(mf_phase))
        mf_imag = mf_rho.mul(sin(mf_phase))
        # Save complex mf and enable gradient computation
        cmf = torch.complex(mf_real, mf_imag)
        cmf.requires_grad = True
        self.cmf = cmf
        return cmf
    
    @property
    def params(self):
        return torch.stack((self.a,self.b,self.c,self.d))
    
    def backward(self, grad_output):
        params = []
        for i in range(grad_output.shape[1]):
            param = optimize_params(grad_output[:,i])
            params.append(param)
        p = torch.stack(params)
        params = [p[:,i] for i in range(p.shape[1])]
        self.a, self.b, self.c, self.d = params
    
# FuzzifyLayer
class FuzzifyLayer(nn.Module):
    def __init__(self, cmf):
        super(FuzzifyLayer, self).__init__()
        self.cmf = nn.Parameter(cmf)

    # Implement forward function
    def forward(self, x: torch.Tensor):
        # Compute convolution
        einsum = torch.einsum('bj, bi -> ibj', (self.cmf, x))
        einsum = einsum.transpose(0,2)
        convolution_sum  = torch.sum(torch.sum(einsum, dim = 1), dim = 1)
        absolute_sum = torch.abs(convolution_sum)
        output = convolution_sum/(1+absolute_sum)
        self.output = output
        return output
    
# FiringStrenght Layer
class FiringStrenghtLayer(nn.Module):
    def __init__(self, mode:str = "univariate"):
        super(FiringStrenghtLayer, self).__init__()
        self.mode = mode
    
    def forward(self, x):
        if self.mode == "univariate":
            return x
        
# Normalization Layer
class NormalizationLayer(nn.Module):
    def __init__(self, mode:str = "univariate"):
        super(NormalizationLayer, self).__init__()
        self.mode = mode
        
        
    def forward(self, x):
        if self.mode == "univariate":
            normalizer = torch.sum(torch.tensor([*x.__abs__()]))
            output = x/normalizer
            return output
        
# DotProduct Layer
class DotProductLayer(nn.Module):
    def __init__(self):
        super(DotProductLayer, self).__init__()
        self.weights = []
    
    def forward(self,x):
        x_sum = torch.sum(x)
        output = x.real*x_sum.real + (x.imag * x_sum.imag)
        self.weights.append(output)
        return output
    
# LinearConsequent Layer
class LinearConsequentLayer(nn.Module):
    def __init__(self, n_rules, n_inputs, n_outputs = 1):
        super(LinearConsequentLayer, self).__init__()
        c_shape = torch.Size([n_rules, n_inputs, n_outputs])
        self._coeff = torch.ones(c_shape, dtype=torch.double)

    @property
    def coeff(self):
        '''
            Record the (current) coefficients for all the rules
            coeff.shape: n_rules * n_out * (n_in+1)
        '''
        return self._coeff

    @coeff.setter
    def coeff(self, new_coeff):
        '''
            Record new coefficients for all the rules
            coeff: for each rule, for each output variable:
                   a coefficient for each input variable, plus a constant
        '''
        assert new_coeff.shape == self.coeff.shape, \
            'Coeff shape should be {}, but is actually {}'\
            .format(self.coeff.shape, new_coeff.shape)
        self._coeff = new_coeff

    # Column of ones still to be added
    def fit_coeff(self, x, weights, y_actual):
        ''' x is a matrix of obserivation (each row of the matrix is the input vector of previos t-w observations at time t
        with w = 1,2,...,window. THe number of rows equals the number of observations on which the optimal autoegression parameters need to be
        computed). y_actual is the vector of observations at time t with the same lenght as the number of rows of x'''
        X = x # input vector 
        ones = torch.ones(X.size(0),1)
        X_ones =  torch.cat((ones, X), dim = 1)
        
        # Try with weighting x
        weighted_x = torch.einsum('sw, sn -> nsw', X_ones, torch.stack(weights))

        # # Try not weighting x
        # weighted_x = torch.einsum('sw, sn -> nsw', X_ones, torch.ones(torch.stack(weights).shape))

        # #! Try without weighted ys
        # weighted_y = torch.einsum('s, sn -> ns', y_actual, torch.ones_like(torch.stack(weights)))
        # regression = torch.linalg.lstsq(weighted_x, weighted_y)
        # coefficients = regression.solution
        # coeff = coefficients.view(*coefficients.shape, 1)
        # self.coeff = coeff

        #! Try with weighted ys
        weighted_y = torch.einsum('s, sn -> ns', y_actual, torch.stack(weights))
        regression = torch.linalg.lstsq(weighted_x, weighted_y)
        coefficients = regression.solution
        coeff = coefficients.view(*coefficients.shape, 1)
        self.coeff = coeff
        return coeff

    def forward(self, x, weights):
        '''
            Calculate: y = coeff * x + const   [NB: no weights yet]
                  x.shape: n_cases * n_in
              coeff.shape: n_rules * n_out * (n_in+1)
                  y.shape: n_cases * n_out * n_rules
        '''
        X = torch.cat((torch.ones(1),x[0]))
        weighted_x = torch.einsum('w, n -> nw', X, weights)
        y_pred = torch.einsum('nw, nwb -> n', weighted_x, self.coeff)
        # y_pred_f = torch.einsum('n, n -> n', y_pred, weights)
        # print(y_pred_f)
        # y_pred = torch.einsum('bi, bij -> bi', weighted_x, self._coeff)
        return y_pred
    
# WeightedSum Layer
class WeightedSumLayer(nn.Module):
    def __init__(self):
        super(WeightedSumLayer, self).__init__()
        
    def forward(self, x):
        y_pred = torch.sum(x, dim = 0)
        return y_pred
    
# Implement Final ANCFIS Module
class AncfisNet(torch.nn.Module):
    '''
        This is a container for the 5 layers of the ANFIS net.
        The forward pass maps inputs to outputs based on current settings,
        and then fit_coeff will adjust the TSK coeff using LSE.
    '''
    def __init__(self, window, cmf, nmfs, description = None):
        super(AncfisNet, self).__init__()
        self.description = description
        self.nmfs = nmfs
        cl = LinearConsequentLayer(self.nmfs, window+1)
        self.cmf = cmf
        self.layer = torch.nn.ModuleDict(OrderedDict([
            ('FuzzifyLayer', FuzzifyLayer(self.cmf)),
            ('FiringStrenghtLayer', FiringStrenghtLayer()),
            ('NormalizationLayer', NormalizationLayer()),
            ('DotProductLayer', DotProductLayer()),
            ('LinearConsequentLayer', cl),
            ('WeightedSumLayer', WeightedSumLayer())
            ]))

    @property
    def coeff(self):
        return self.layer['LinearConsequentLayer'].coeff

    @coeff.setter
    def coeff(self, new_coeff):
        self.layer['LinearConsequentLayer'].coeff = new_coeff

    def fit_coeff(self, x, y_actual):
        '''
            Do a forward pass (to get weights), then fit to y_actual.
            Does nothing for a non-hybrid ANFIS, so we have same interface.
        '''
        # self.layer['LinearConsequentLayer'].coeff =
        self.layer['LinearConsequentLayer'].fit_coeff(x, self.layer['DotProductLayer'].weights, y_actual)
        self.layer['DotProductLayer'].weights = []

    def membership_functions(self):
        '''
            Return an iterator over this system's input variables.
            Yields tuples of the form (var-name, FuzzifyVariable-object)
        '''
        return self.layer['FuzzifyLayer'].cmf

    def forward(self, x):
        '''
            Forward pass: run x thru the five layers and return the y values.
            I save the outputs from each layer to an instance variable,
            as this might be useful for comprehension/debugging.
        '''
        self.fuzzifyl = self.layer['FuzzifyLayer'](x)
        self.firing = self.layer['FiringStrenghtLayer'](self.fuzzifyl)
        self.normalize = self.layer['NormalizationLayer'](self.firing)
        self.dp = self.layer['DotProductLayer'](self.normalize)
        # Save weights for linear consequent update
        self.consequent = self.layer['LinearConsequentLayer'](x, self.dp)
        self.sum = self.y_pred = self.layer['WeightedSumLayer'](self.consequent)
        return self.y_pred

# Implement IterableDataset for training loop
class IterableData(IterableDataset):
    def __init__(self, window:int, x:torch.Tensor, y:torch.Tensor, num_batches = 10):
        super(IterableData).__init__()
        self.data = (x,y)
        self.window = window
        self.num_batches = num_batches
        self.shifts = x.shape[0] - self.window
        self.X = torch.stack([x.roll(-s)[:window].reshape(window) for s in range(self.shifts)]).reshape(self.num_batches, self.shifts//self. num_batches, self.window)  
        self.y = y[self.window:]
        self.y = self.y.reshape(1,*self.y.shape).reshape(self.num_batches, self.shifts//self.num_batches)
        self.iter = iter(range(self.num_batches))
    
    def __len__(self):
         return self.x.shape[0]

    def __iter__(self):
            idx = next(self.iter)
            yield self.X[idx], self.y[idx]