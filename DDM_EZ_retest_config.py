import numpy as np
import bayesflow as bf
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from simulator import DDM_EZ_retest_config as sim_v2

import random 
random.seed(78901212)

def draw_prior():
    """Generates random draws from the prior."""
    v1 =np.random.uniform(low=0.0,high=5.0)
    v2 =np.random.uniform(low=-5.0,high=0.0)

    a  =np.random.uniform(low=0.6,high=3.0)
    ndt =np.random.uniform(low=0.1,high=0.7)
   # alpha =np.random.uniform(low=1.0,high=2.0)

    return np.array([v1,v2,a, ndt])

def prior_N(n_min=50, n_max=1000):
    """A prior fo]r the number of observation (will be called internally at each backprop step)."""
    return  np.random.randint(n_min, n_max+1, 2)


prior = bf.simulation.Prior(prior_fun=draw_prior,param_names=[r'$v1$',r'$v2$', r'$a$', r'$ndt$'])
var_num_obs = bf.simulation.ContextGenerator(non_batchable_context_fun=prior_N)
simulator = bf.simulation.Simulator(simulator_fun=sim_v2.simulate_diffusion_2_conds, context_generator=var_num_obs)
generative_model = bf.simulation.GenerativeModel(prior,simulator,name='EZ_retest')

prior_means, prior_stds = prior.estimate_means_and_stds()


def configurator(sim_dict):
    """Configures the outputs of a generative model for interaction with 
    BayesFlow modules."""
    
    out = dict()
    # These will be passed through the summary network. In this case,
    # it's just the data, but it can be other stuff as well.
    data = sim_dict['sim_data'].astype(np.float32)
    
    # Extract prior draws and z-standardize with previously computed means
    params = ((sim_dict['prior_draws'].astype(np.float32)) - prior_means) / prior_stds
    #params = (params - prior_means) / prior_stds
    
    
    # Remove a batch if it contains nan, inf or -inf
    idx_keep = np.all(np.isfinite(data), axis=(1, 2))
    if not np.all(idx_keep):
        print('Invalid value encountered...removing from batch')
        
    out['summary_conditions'] = data[idx_keep]
    out['parameters'] = params[idx_keep]

    
    
    
    
    # These will be concatenated to the outputs of the summary network
    # Convert N to log N since neural nets cant deal well with large numbers
    N = np.log(sim_dict['sim_non_batchable_context'])
    # Repeat N for each sim (since shared across batch), notice the
    # extra dimension needed
    N_vec = N * np.ones((data.shape[0], 1), dtype=np.float32)
    out['direct_conditions'] = N_vec
    return out

summary_net = bf.networks.InvariantNetwork(summary_dim=64,num_equiv=3)
inference_net = bf.networks.InvertibleNetwork(num_params=len(prior.param_names),num_coupling_layers=6)
amortizer = bf.amortizers.AmortizedPosterior(inference_net, summary_net, name='v2_equive')


trainer = bf.trainers.Trainer(
    amortizer=amortizer, 
    generative_model=generative_model, 
    configurator=configurator,
    checkpoint_path='./checkpoints/50_1000_EZ_DDM_retest_v2_EZ_config')

num_val = 300
val_sims = generative_model(num_val)

h = trainer.train_online(epochs=200, iterations_per_epoch=1000, batch_size=128, validation_sims=val_sims)
f = bf.diagnostics.plot_losses(h['train_losses'], h['val_losses'])
plt.savefig('./Figures_DDM/EZ_DDM_loss_retest.png')


num_test = 1000
num_posterior_draws_recovery = 5000
new_sims = configurator(generative_model(num_test))

param_names = [r'$v1$',r'$v2$',r'$a$',r'$ndt$',]

posterior_draws = amortizer.sample(new_sims, n_samples=num_posterior_draws_recovery)
fig = bf.diagnostics.plot_recovery(posterior_draws, new_sims['parameters'],param_names = param_names )
plt.savefig('./Figures_DDM/EZ_DDM_recovery_retest.png')

f = bf.diagnostics.plot_sbc_ecdf(posterior_draws, new_sims['parameters'], stacked=True, 
                       difference=True, legend_fontsize=12, fig_size=(10, 8))

plt.savefig('./Figures_DDM/EZ_DDM_ecdf_retest.png')

f = bf.diagnostics.plot_sbc_histograms(posterior_draws, new_sims['parameters'], num_bins=10,param_names = param_names)
plt.savefig('./Figures_DDM/EZ_DDM_SBC_retest.png')

f = bf.diagnostics.plot_sbc_ecdf(posterior_draws, new_sims['parameters'])
plt.savefig('./Figures_DDM/EZ_DDM_ecdf_sbc_retest.png')