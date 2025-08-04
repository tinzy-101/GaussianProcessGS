import torch
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import MaternKernel, LCMKernel, ScaleKernel
from MOGP.kernels import WendlandKernel
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal


#Default class 
class IndependentMultiTaskGPModel(gpytorch.models.ExactGP):
    
    def __init__(self, train_x, train_y, likelihood, num_tasks, nu=0.5):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = ScaleKernel(
            MaternKernel(nu=nu, batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks])
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal.from_batch_mvn(MultivariateNormal(mean_x, covar_x))
    
#LCM class with Matern kernels nu=0.5
class LCMMultiTaskGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, num_tasks, rank, nu=0.5):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(ConstantMean(), num_tasks=num_tasks)
        
        
        base_kernels = [
            MaternKernel(nu=nu, ard_num_dims=train_x.shape[-1]) for _ in range(6)
        ]
        
        self.covar_module = LCMKernel(base_kernels, num_tasks=num_tasks, rank=rank)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)

#LCM class with mixed materns kernels
class LCMMixedMaternModel(gpytorch.models.ExactGP):
    
    def __init__(self, train_x, train_y, likelihood, num_tasks, rank):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(ConstantMean(), num_tasks=num_tasks)
        
        # Mixed Matern kernels: 3 with nu=0.5, 3 with nu=2.5
        base_kernels = [
            MaternKernel(nu=0.5, ard_num_dims=train_x.shape[-1]),   # first latent kernel
            MaternKernel(nu=0.5, ard_num_dims=train_x.shape[-1]),   # second latent kernel
            MaternKernel(nu=0.5, ard_num_dims=train_x.shape[-1]),   # third latent kernel
            MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1]),   # fourth latent kernel
            MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1]),   # fifth latent kernel
            MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1])    # sixth latent kernel
        ]
        
        self.covar_module = LCMKernel(base_kernels, num_tasks=num_tasks, rank=rank)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)

#LCM class with mixed matern and wendland kernels
class LCMMaternWendlandModel(gpytorch.models.ExactGP):
   
    def __init__(self, train_x, train_y, likelihood, num_tasks, rank):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(ConstantMean(), num_tasks=num_tasks)
        
        # Mixed Matern 2.5 and Wendland kernels
        base_kernels = [
            MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1]),   # first latent kernel
            MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1]),   # second latent kernel
            MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1]),   # third latent kernel
            WendlandKernel(ard_num_dims=train_x.shape[-1]),          # fourth latent kernel
            WendlandKernel(ard_num_dims=train_x.shape[-1]),          # fifth latent kernel
            WendlandKernel(ard_num_dims=train_x.shape[-1])           # sixth latent kernel
        ]
        
        self.covar_module = LCMKernel(base_kernels, num_tasks=num_tasks, rank=rank)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)

