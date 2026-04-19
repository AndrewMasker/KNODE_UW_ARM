from jax import config
config.update("jax_enable_x64", True)
import os
#os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import argparse
import collections
from Core.GP_Modules import GP
import equinox as eqx
import flax.nnx as nnx
from flax.jax_utils import prefetch_to_device
from functools import partial
import gpjax as gpx
from gpjax.kernels.computations import DenseKernelComputation
from gpjax.parameters import Parameter
import grain

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import (Array,Float,install_import_hook)
import json
import matplotlib.pyplot as plt
import mujoco.mjx as mjx
import numpy as np
import optax
import os
import time
import Utils.Data_Utils as data_utils
import Utils.Mujoco_Utils as mj_utils
import zarr




"""
ARGUMENT AND SPECIFICATION PARSING
"""

def return_Arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g" , "--gp_specs" , type = str , default = "gp_fluid")
    
    args = parser.parse_args()
    
    gp_path = "Library/GP_Specifications/" + args.gp_specs + ".json"
        
    return gp_path
    
    
def parse_GP_Specs(path_to_gp_specs):
    with open(path_to_gp_specs) as f:
        d = json.load(f)
    gp_options = d
    return gp_options

    
"""
SAVING/LOADING MODEL
"""

def get_New_Name(name , *args):
    if len(args) == 0:
        if os.path.exists(os.getcwd() + "/Models/" + name):
            return get_New_Name(name , 1)
        else:
            return name
    else:
        if os.path.exists(os.getcwd() + "/Models/" + name + "_" + str(args[0])):
            return get_New_Name(name , args[0] + 1)
        else:
            return name + "_" + str(args[0])
            

def save_Model(model , loss_list , model_options , training_options):
    
    name = get_New_Name(training_options["name"])
    save_path = os.getcwd() + "/Models/" + name
    
    os.makedirs(save_path)
    
    eqx.tree_serialise_leaves(save_path + "/model.eqx" , model)
    
    with open(save_path + "/model_options.json", 'w') as f:
        json.dump(model_options, f)
    with open(save_path + "/training_options.json", 'w') as f:
        json.dump(training_options, f)
    with open(save_path + "/loss_list.json", 'w') as f:
        json.dump(loss_list, f)
        
    return save_path
        
        
def load_Model_From_Options(gp_options , posterior , inputs_mean , inputs_std , outputs_mean , outputs_std):
    dataset_path = "Datasets/" + gp_options["residual_dataset_name"] + ".zarr"
    root = zarr.open(dataset_path)
    qvel_max = jnp.array(root.attrs["dataset_options"]["qvel_max"])
    
    training_options = root.attrs["training_options"]
    
    model_xml = mj_utils.create_XML_From_Options(training_options)
    mjx_model , mjx_data = mj_utils.init_mjx_model(model_xml)
    
    key = jr.key(int(time.time()))
    
    
    if gp_options["should_load"]:
        load_path = os.getcwd() + "/Models/" + training_options["load_name"] + "/"
        model_options = parse_Model_Specs(load_path + "model_options.json")
        
        model = KNODE(key , mjx_model , model_options["width"] , model_options["depth"] , qvel_max)
        
        model = eqx.tree_deserialise_leaves(load_path + "model.eqx" , model)
        
    else:
        model = GP(key , mjx_model , posterior , inputs_mean , inputs_std , outputs_mean , outputs_std)
        
    
    return model , mjx_data
    
"""
VISUAL AIDS
"""

def visualize_Trained_Model(model , gp_options , loss_list):
    root = zarr.open(os.getcwd() + "/Datasets/" + gp_options["residual_dataset_name"] + ".zarr" , mode = 'r')
    training_options = root.attrs["training_options"]
    dt = root.attrs.get("dataset_options")["step_size"]
    xml = root.attrs.get("xml")
    mj_model , data = mj_utils.init_mujoco_model(xml)
    
    _ , mjx_data = mj_utils.init_mjx_model(xml)
    
    trajectory_number = gp_options["which_to_show_at_end"]
    keys = sorted(root.keys(), key=lambda k: int(k.split('_')[-1]))
    t = keys[trajectory_number]
    print(t)
    d = root[t]#"trajectory_" + str(trajectory_number)
    gt_qpos_traj = d["qpos"][:]
#    print(d["qpos"])
    
    step_size = root.attrs["dataset_options"]["step_size"]
    num_steps = root.attrs["dataset_options"]["num_steps"]
    ts = jnp.arange(0 , step_size*(num_steps) , step = step_size)
    
    init_states = {"qpos": jnp.array(d["qpos"][0 , :]) , "qvel": jnp.array(d["qvel"][0 , :])}
    ctrl = jnp.array(d["ctrl"][0 , :])
    
    KNODE_traj = model(ts , mjx_data , init_states , ctrl , with_learned = jnp.array(1.0))
    KNODE_qpos_traj = np.array(KNODE_traj["qpos"])
    
    nominal_traj = model(ts , mjx_data , init_states , ctrl , with_learned = jnp.array(0.0))
    nominal_qpos_traj = np.array(nominal_traj["qpos"])
    
    
    

    l2 = gp_options["l2"]
    l4 = gp_options["l4"]
    
    
    
             
    print(l2*jnp.mean((KNODE_traj["qpos"] - gt_qpos_traj)**2) + l4*jnp.mean((KNODE_traj["qvel"] - d["qvel"][:])**2))
    

    
    fig , (ax1 , ax2 , ax3 , ax4 , ax5 , ax6) = plt.subplots(1 , 6)
    
    loss_inds = np.arange(1 , len(loss_list) + 1)
    ax1.plot(loss_inds , loss_list)
    ax2.loglog(loss_inds , loss_list)
    
    
    ax3.plot(KNODE_traj["qpos"] , color = 'green')
    ax3.plot(nominal_traj["qpos"] , color = 'red')
    ax3.plot(d["qpos"] , color = 'black')
    
    ax4.plot(KNODE_traj["qvel"] , color = 'green')
    ax4.plot(nominal_traj["qvel"] , color = 'red')
    ax4.plot(d["qvel"] , color = 'black')
    
    KNODE_acc = (KNODE_traj["qvel"][1:] - KNODE_traj["qvel"][:-1])/dt
    nominal_acc = (nominal_traj["qvel"][1:] - nominal_traj["qvel"][:-1])/dt
    gt_acc = (d["qvel"][1:] - d["qvel"][:-1])/dt
    
    ax5.plot(KNODE_acc , color = 'green')
    ax5.plot(nominal_acc , color = 'red')
    ax5.plot(gt_acc , color = 'black')
    
    ax6.plot(KNODE_acc - nominal_acc , color = 'green')
#    ax5.plot(nominal_acc , color = 'red')
    ax6.plot((gt_acc - nominal_acc) , color = 'black')
    
   
    plt.show(block = False)
    
    
    plt.show()
    
    mj_utils.visualize_GT_vs_Nominal_vs_KNODE_Trajectory(training_options["model_name"] , gt_qpos_traj , nominal_qpos_traj , KNODE_qpos_traj)
    
#def plot_Loss(loss_list ):#, save_path):
#    save_path = save_path + "/Plots"
#    os.make_dirs(save_path)
#    

    
    

"""
MODEL TRAINING
"""

class VelocityKernel(gpx.kernels.AbstractKernel):
    kernels: nnx.List# = gpx.kernels.RBF(active_dims=[0, 1])
#    kernel1: gpx.kernels.AbstractKernel# = gpx.kernels.RBF(active_dims=[0, 1])
    def __init__(self, nq , nu):
                
        kernels = []#gpx.kernels.RBF(active_dims=list(range(2*nq+nu)))   
        for i in range(nq):
            kernels.append(gpx.kernels.RBF(active_dims=list(range(2*nq+nu))))
        self.kernels = nnx.List(kernels)
#        self.kernel1 = gpx.kernels.RBF(active_dims=list(range(2*nq+nu)))
        super().__init__(compute_engine=DenseKernelComputation())

        
    def __call__(self, X: Float[Array, "1 D"], Xp: Float[Array, "1 D"]) -> Float[Array, "1"]:
                
        # standard RBF-SE kernel is x and x' are on the same output, otherwise returns 0

        z = jnp.array(X[-1], dtype=int)
                
        zp = jnp.array(Xp[-1], dtype=int)

                
        # achieve the correct value via 'switches' that are either 1 or 0
                
        is_same_output = jnp.equal(z , zp)

        kernel_funs = [lambda k=k: k(X, Xp) for k in self.kernels]
        kernel_val = jax.lax.switch(z, kernel_funs)
                
        return jnp.where(is_same_output , kernel_val , jnp.array(0.0).astype(jnp.float64))#k0_switch * self.kernel0(X, Xp) + k1_switch * self.kernel1(X, Xp)
        
#def initialise_gp(kernel, mean, dataset):
#    
#    prior = gpx.gps.Prior(mean_function=mean, kernel=kernel)
#        
#    likelihood = gpx.likelihoods.Gaussian(num_datapoints=dataset.n, obs_stddev=jnp.array([1.0e-3], dtype=jnp.float32))
#        
#    posterior = prior * likelihood
#        
#    return posterior


def rearrange_Sample(gp_state , nq):
    gp_state_repeated = jnp.repeat(jnp.expand_dims(gp_state , 0) , repeats = nq , axis = 0)
    gp_input = jnp.concatenate((gp_state_repeated , jnp.expand_dims(jnp.arange(nq,dtype=jnp.float64) , 1)) , axis = 1)
    return gp_input

@partial(jax.jit , static_argnums = 4)
def rearrange_Trajectory(qpos_traj , qvel_traj , ctrl_traj , resi_traj , nq):
    
    input_traj = jnp.concatenate((qpos_traj , qvel_traj , ctrl_traj) , axis = 1)
    
    rearranged_traj = jax.vmap(rearrange_Sample , in_axes = (0 , None))(input_traj , nq)
    
    s = rearranged_traj.shape
    samples = s[0]*s[1]
    inputs = s[2]
    
    rearranged_traj = jnp.reshape(rearranged_traj , (samples , inputs))
    rearranged_resi_traj = jnp.ravel(resi_traj)
    
    return rearranged_traj , rearranged_resi_traj
    
    

def build_Multioutput_Dataset(gp_options) -> gpx.Dataset:
    
    root = zarr.open(os.getcwd() + "/Datasets/" + gp_options["residual_dataset_name"] + ".zarr" , mode = 'r')
    
    keys = sorted(root.keys(), key=lambda k: int(k.split('_')[-1]))
    
    subset_size = gp_options["subset_size"]
    if subset_size == None:
        subset_size = len(keys)
    elif subset_size > len(keys):
        raise ValueError("Subset Size Cannot Exceed Size of Dataset")
    
    traj = root[keys[0]]
    nt , nq = traj["qpos"].shape
    _ , nu = traj["ctrl"].shape
    
    final_input_dims = (nq*nt*subset_size , 2*nq + nu + 1)
    final_output_dims = (nq*nt*subset_size)
    

    inputs_list = []
    outputs_list = []
    

    for i in range(subset_size):
        traj = root[keys[i]]
        qpos_traj = jnp.array(traj["qpos"] , dtype=jnp.float64)
        qvel_traj = jnp.array(traj["qvel"] , dtype=jnp.float64)
        ctrl_traj = jnp.array(traj["ctrl"] , dtype=jnp.float64)
        resi_traj = jnp.array(traj["res_qacc"] , dtype=jnp.float64)
        
        rearranged_traj , rearranged_resi_traj = rearrange_Trajectory(qpos_traj , qvel_traj , ctrl_traj , resi_traj , nq)
        
        inputs_list.append(rearranged_traj)
        outputs_list.append(rearranged_resi_traj)
        
        if i%100 == 0:
            print(i)
        
    inputs = jnp.concatenate(inputs_list , 0)
    outputs = jnp.expand_dims(jnp.concatenate(outputs_list , 0),1)
 
    inputs_mean  = jnp.concatenate((jnp.mean(inputs[:,:-1],  axis=0) , jnp.array([0.,]).astype(jnp.float64)) , axis=0)
    inputs_std   = jnp.concatenate((jnp.std(inputs[:,:-1],   axis=0) + 1e-8 , jnp.array([1.,]).astype(jnp.float64)) , axis=0)
    outputs_mean = jnp.mean(outputs, axis=0)
    outputs_std  = jnp.std(outputs,  axis=0) + 1e-8

    inputs  = (inputs  - inputs_mean)  / inputs_std
    outputs = (outputs - outputs_mean) / outputs_std
 
    return gpx.Dataset(inputs, outputs) , inputs_mean , inputs_std , outputs_mean , outputs_std
    
#def optimise_mll(posterior, dataset, key):
#    opt_posterior, history = gpx.fit(
#        model=posterior,
#        objective=lambda p, d: -gpx.objectives.collapsed_elbo(p, d),
#        train_data=dataset,
#        optim=optax.adamw(learning_rate=1e-2),
#        num_iters=500,
#        batch_size=256,
#        key=key,
#    )
#    return opt_posterior, history


def train_GP(gp_options):
    
    
#    model , mjx_data = load_Model_From_Options(model_options , training_options)
#    loss_list = []
    

    
    dataset_path = "Datasets/" + gp_options["residual_dataset_name"] + ".zarr"
    root = zarr.open(os.getcwd() + "/Datasets/" + gp_options["residual_dataset_name"] + ".zarr" , mode = 'r')
    keys = sorted(root.keys(), key=lambda k: int(k.split('_')[-1]))
    traj = root[keys[0]]
    nt , nq = traj["qpos"].shape
    _ , nu = traj["ctrl"].shape

    train_dataset, inputs_mean , inputs_std , outputs_mean , outputs_std = build_Multioutput_Dataset(gp_options)
    
    # Build the GP model
    mean   = gpx.mean_functions.Zero()
    kernel = VelocityKernel(nq, nu)
    prior  = gpx.gps.Prior(mean_function=mean, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(
        num_datapoints=train_dataset.n,
        obs_stddev=jnp.array(1., dtype=jnp.float64),
    )
    
    # Inducing points: these are learned GP parameters, NOT data you're throwing away.
    # Rule of thumb: 200-1000 is usually enough. More = more accurate but slower.
    key = jr.key(int(time.time()))
    num_inducing = gp_options["num_inducing"]
    perm = jr.permutation(key, train_dataset.n)[:num_inducing]
    inducing_inputs = train_dataset.X[perm]  # shape: (num_inducing, D)

    posterior = gpx.variational_families.ExpectationVariationalGaussian(
        posterior=prior*likelihood,
        inducing_inputs=inducing_inputs,
        jitter = 1e-3
    )
    
    graphdef, state = nnx.split(posterior)
    print(state)  # pretty-prints the full parameter tree with types and values
    
#    poop()
    
    # Mini-batch training — all data is used, just in chunks of batch_size
    opt_posterior, history = gpx.fit(
        model=posterior,
        objective=lambda p, d: -gpx.objectives.elbo(p, d),
        train_data=train_dataset,
#        optim=optax.chain(optax.clip_by_global_norm(1.0),optax.adam(learning_rate=1e-4)),
        optim=optax.adam(learning_rate=1e-3),
        num_iters=2000,
        batch_size=1000,   # tune this based on your GPU memory
        key=key,
    )
    
    graphdef, state = nnx.split(opt_posterior)
    print(state)  # pretty-prints the full parameter tree with types and values
    poop()

#    opt_posterior , history = gpx.fit_scipy(model = posterior,    
#        objective=lambda p, d: -gpx.objectives.elbo(p, d)/train_dataset.n,  
#        train_data= train_dataset)
                
    return opt_posterior , train_dataset , history , inputs_mean , inputs_std , outputs_mean , outputs_std
    


"""
CALL MAIN FUNCTION
"""

if __name__ == "__main__":
    gp_path = return_Arguments()
    gp_options = parse_GP_Specs(gp_path)
    opt_posterior , train_data , history , inputs_mean , inputs_std , outputs_mean , outputs_std = train_GP(gp_options)
    
    model , _ = load_Model_From_Options(gp_options , opt_posterior , inputs_mean , inputs_std , outputs_mean , outputs_std)
    
#    print(model.knode_dynamics.res_mixing_matrix)
#    print(model.knode_dynamics.mjx_mixing_matrix)
#    if training_options["should_save"]:
#        save_path = save_Model(model , loss_list , model_options , training_options)
#    plot_Loss(loss_list)
    visualize_Trained_Model(model , gp_options , history)
#    plt.show()
    
    
    
    
    
    
    

