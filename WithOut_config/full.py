import numpy as np
from numba import njit
import bayesflow as bf
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from simulator import full as sim_v2
import random 

random.seed(4321)

def draw_prior():
    """Generates random draws from the prior."""
    v1=np.random.uniform(low=0.0,high=6.0)
    v2 =np.random.uniform(low=-6.0,high=0.0)
    sv =np.random.uniform(low=0.0,high=2.0)
    zr =np.random.uniform(low=0.3,high=0.7)
    szr =np.random.uniform(low=0.0,high=0.6)
    a  =np.random.uniform(low=0.8,high=3.0)
    ndt =np.random.uniform(low=0.3,high=1.0)
    sndt =np.random.uniform(low=0.0,high=0.4)
    alpha =np.random.uniform(low=1.0,high=2.0)

    return np.array([v1,v2,sv,zr,szr,a, ndt,sndt,alpha])

def prior_N(n_min=100, n_max=1000):
    """A prior fo]r the number of observation (will be called internally at each backprop step)."""
    return np.random.randint(n_min, n_max+1)

prior = bf.simulation.Prior(prior_fun=draw_prior,param_names=[r'$v1$',r'$v2$',r'$sv$',r'$zr$',r'$szr$', r'$a$', r'$ndt$',r'$sndt$',r'$alpha$'])
var_num_obs = bf.simulation.ContextGenerator(non_batchable_context_fun=prior_N)
simulator = bf.simulation.Simulator(simulator_fun=sim_v2.simulate_diffusion_2_conds, context_generator=var_num_obs)
generative_model = bf.simulation.GenerativeModel(prior,simulator,name='v2_equiv')

param_names=[r'$v1$',r'$v2$',r'$sv$',r'$zr$',r'$szr$', r'$a$', r'$ndt$',r'$sndt$',r'$alpha$']

def configurator(sim_dict):
    """Configures the outputs of a generative model for interaction with 
    BayesFlow modules."""
    
    out = dict()
    # These will be passed through the summary network. In this case,
    # it's just the data, but it can be other stuff as well.
    out['summary_conditions'] = sim_dict['sim_data'].astype(np.float32)
    
    # Extract prior draws and z-standardize with previously computed means
    out['parameters'] = sim_dict['prior_draws'].astype(np.float32)
    #params = (params - prior_means) / prior_stds
    
    
    return out

summary_net = bf.networks.InvariantNetwork(summary_dim=128,num_equiv=3)
inference_net = bf.networks.InvertibleNetwork(num_params=len(prior.param_names),num_coupling_layers=4)
amortizer = bf.amortizers.AmortizedPosterior(inference_net, summary_net, name='full')

trainer = bf.trainers.Trainer(
    amortizer=amortizer, 
    generative_model=generative_model, 
    configurator=configurator,
    checkpoint_path='./checkpoints/full_py')

num_val = 300
val_sims = generative_model(num_val)

h = trainer.train_online(epochs=200, iterations_per_epoch=1000, batch_size=64, validation_sims=val_sims)

f = bf.diagnostics.plot_losses(h['train_losses'], h['val_losses'])

plt.savefig('./figures_full/loss.png')

num_test = 1000
num_posterior_draws_recovery = 5000
new_sims = configurator(generative_model(num_test))


posterior_draws = amortizer.sample(new_sims, n_samples=num_posterior_draws_recovery)
fig = bf.diagnostics.plot_recovery(posterior_draws, new_sims['parameters'],param_names = param_names )
plt.savefig('./figures_full/recovery.png')

f = bf.diagnostics.plot_sbc_ecdf(posterior_draws, new_sims['parameters'], stacked=True, 
                       difference=True, legend_fontsize=12, fig_size=(10, 8))

plt.savefig('./figures_full/ecdf.png')

f = bf.diagnostics.plot_sbc_histograms(posterior_draws, new_sims['parameters'], num_bins=10,param_names = param_names)
plt.savefig('./figures_full/SBC.png')

f = bf.diagnostics.plot_sbc_ecdf(posterior_draws, new_sims['parameters'])
plt.savefig('./figures_full/ecdf_sbc.png')


