import os
import argparse
import equinox as eqx
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import json
import matplotlib.pyplot as plt
import mujoco.mjx as mjx
import numpy as np
import time
import Utils.Data_Utils as data_utils
import Utils.Mujoco_Utils as mj_utils
import zarr




"""
ARGUMENT AND SPECIFICATION PARSING
"""

def return_Arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t" , "--test_specs" , type = str , default = "test_new")
    
    args = parser.parse_args()
    
    test_path = "Library/Test_Specifications/" + args.test_specs + ".json"
        
    return test_path
    
    
def parse_Specs(path_to_test_specs):
    with open(path_to_test_specs) as f:
        d = json.load(f)
    test_options = d
    return test_options
    
    
def get_New_Name(name , *args):
    if len(args) == 0:
        if os.path.exists(os.getcwd() + "/Test_Outputs/" + name):
            return get_New_Name(name , 1)
        else:
            return name
    else:
        if os.path.exists(os.getcwd() + "/Test_Outputs/" + name + "_" + str(args[0])):
            return get_New_Name(name , args[0] + 1)
        else:
            return name + "_" + str(args[0])
    

"""
LOADING MODEL
"""

def get_Training_Options(test_options):
    load_name = test_options["load_name"]
    model_folder_path = "Models/" + load_name
    
    training_options_path = model_folder_path + "/training_options.json"
    with open(training_options_path) as f:
        training_options = json.load(f)
        
    return training_options
        
def get_Model_Options(test_options):
    load_name = test_options["load_name"]
    model_folder_path = "Models/" + load_name
    
    model_options_path = model_folder_path + "/model_options.json"
    with open(model_options_path) as f:
        model_options = json.load(f)
    
    return model_options

def load_Old_Model(test_options):

    from Core.KNODE_Modules_Old import KNODE
    
    training_options = get_Training_Options(test_options)
    model_options = get_Model_Options(test_options)

    dataset_path = "Datasets/" + training_options["dataset_name"] + ".zarr"
    root = zarr.open(dataset_path)
    qvel_max = jnp.array(root.attrs["dataset_options"]["qvel_max"])
    
    model_xml = mj_utils.create_XML_From_Options(training_options)
    mjx_model , mjx_data = mj_utils.init_mjx_model(model_xml)
    
    key = jr.key(int(time.time()))
    model = KNODE(key , mjx_model , model_options["width"] , model_options["depth"] , qvel_max)
    
    load_path = os.getcwd() + "/Models/" + test_options["load_name"] + "/"
    model = eqx.tree_deserialise_leaves(load_path + "model.eqx" , model)
        
    return model , mjx_data
    
def load_New_Model(test_options):

    from Core.KNODE_Modules import KNODE
    
    training_options = get_Training_Options(test_options)
    model_options = get_Model_Options(test_options)

    dataset_path = "Datasets/" + training_options["dataset_name"] + ".zarr"
    root = zarr.open(dataset_path)
    qvel_max = jnp.array(root.attrs["dataset_options"]["qvel_max"])
    
    model_xml = mj_utils.create_XML_From_Options(training_options)
    mjx_model , mjx_data = mj_utils.init_mjx_model(model_xml)
    
    key = jr.key(int(time.time()))
    model = KNODE(key , mjx_model , model_options["width"] , model_options["depth"] , qvel_max)
    
    load_path = os.getcwd() + "/Models/" + test_options["load_name"] + "/"
    model = eqx.tree_deserialise_leaves(load_path + "model.eqx" , model)
        
    return model , mjx_data
    
def load_SINDy_Model(test_options):

    from Core.SINDy_Modules import SINDy_Residual , KNODE_SINDy
    from Core.Visualize_SINDy import build_SINDy_Library
    
    training_options = get_Training_Options(test_options)

    model_xml            = mj_utils.create_XML_From_Options(training_options)
    mjx_model, mjx_data  = mj_utils.init_mjx_model(model_xml)

    library = build_SINDy_Library(training_options, mjx_model)
    
    sindy_residual = SINDy_Residual(library, mjx_model.nq, jnp.zeros((library.n_library_features , mjx_model.nq)))
    model = KNODE_SINDy(mjx_model , sindy_residual)
    
    load_path = os.getcwd() + "/Models/" + test_options["load_name"] + "/"
    
    model = eqx.tree_deserialise_leaves(load_path + "model.eqx" , model)
    
    return model , mjx_data
    
"""
JOB FUNCTIONS
"""


def run_Single_Trajectory(model , mjx_data , gt_qpos_traj , gt_qvel_traj , ctrl , ts):
    
    init_states = {"qpos": gt_qpos_traj[0 , :] , "qvel": gt_qvel_traj[0 , :]}
    
    learned_traj = model(ts , mjx_data , init_states , ctrl , with_learned = jnp.array(1.0))
    learned_qpos_traj = learned_traj["qpos"]
    learned_qvel_traj = learned_traj["qvel"]
    
    nominal_traj = model(ts , mjx_data , init_states , ctrl , with_learned = jnp.array(0.0))
    nominal_qpos_traj = nominal_traj["qpos"]
    nominal_qvel_traj = nominal_traj["qvel"]
    
    return learned_qpos_traj , learned_qvel_traj , nominal_qpos_traj , nominal_qvel_traj
    
@eqx.filter_jit
def run_Trajectorys(model , mjx_data , ts , gt_qpos_trajs , gt_qvel_trajs , ctrls):
    
    output_tuple = jax.vmap(run_Single_Trajectory , in_axes = (None , None , 0 , 0 , 0 , None))(model , mjx_data , gt_qpos_trajs , gt_qvel_trajs , ctrls , ts)
    
    learned_qpos_traj , learned_qvel_traj , nominal_qpos_traj , nominal_qvel_traj = output_tuple
    
    return learned_qpos_traj , learned_qvel_traj , nominal_qpos_traj , nominal_qvel_traj , gt_qpos_trajs , gt_qvel_trajs
    
    
"""
JOB RUNNING
"""

def run_Statistics_Job(model , mjx_data , job_options , job_name , save_path):
    
    root = zarr.open(os.getcwd() + "/Datasets/" + job_options["dataset_name"] + ".zarr" , mode = 'r')
    keys = sorted(root.keys(), key=lambda k: int(k.split('_')[-1]))
    step_size = root.attrs["dataset_options"]["step_size"]
    num_steps = root.attrs["dataset_options"]["num_steps"]
    ts = jnp.arange(0 , step_size*(num_steps) , step = step_size)
    
    if job_options["subset_size"] == "None":
        subset_size = len(keys)
    elif job_options["subset_size"] > len(keys):
        raise ValueError("Subset Size Cannot Exceed Dataset Size")
    else:
        subset_size = job_options["subset_size"]
        
    batch_size = job_options["batch_size"]
    if subset_size%batch_size != 0:
        raise ValueError("Subset Size Must Be Divisible By Batch Size")
        
    num_batches = subset_size // batch_size
    
    learned_qpos_traj_list = []
    learned_qvel_traj_list = []
    nominal_qpos_traj_list = []
    nominal_qvel_traj_list = []
    gt_qpos_traj_list = []
    gt_qvel_traj_list = []
    
    for batch in range(num_batches):
        gt_qpos_traj_batch_list = []
        gt_qvel_traj_batch_list = []
        ctrl_batch_list = []
        for trajectory_number in range(batch*batch_size , (batch+1)*batch_size):
            d = root[keys[trajectory_number]]
    
            gt_qpos_traj = np.array(d["qpos"][:])
            gt_qvel_traj = np.array(d["qvel"][:])
            ctrl = np.array(d["ctrl"][0 , :])
            
            gt_qpos_traj_batch_list.append(gt_qpos_traj)
            gt_qvel_traj_batch_list.append(gt_qvel_traj)
            ctrl_batch_list.append(ctrl)
            
        gt_qpos_traj_batch = jnp.array(np.stack(gt_qpos_traj_batch_list , axis = 0))
        gt_qvel_traj_batch = jnp.array(np.stack(gt_qvel_traj_batch_list , axis = 0))
        ctrl_batch = jnp.array(np.stack(ctrl_batch_list , axis = 0))
        
        output_tuple = run_Trajectorys(model , mjx_data , ts , gt_qpos_traj_batch , gt_qvel_traj_batch , ctrl_batch)
        learned_qpos_traj , learned_qvel_traj , nominal_qpos_traj , nominal_qvel_traj , gt_qpos_traj , gt_qvel_traj = output_tuple
        
        learned_qpos_traj_list.append(learned_qpos_traj)
        learned_qvel_traj_list.append(learned_qvel_traj)
        nominal_qpos_traj_list.append(nominal_qpos_traj)
        nominal_qvel_traj_list.append(nominal_qvel_traj)
        gt_qpos_traj_list.append(gt_qpos_traj)
        gt_qvel_traj_list.append(gt_qvel_traj)
        
        
        print(f"Batch {batch} of {num_batches} completed.")
        
    learned_qpos_trajs = jnp.concatenate(learned_qpos_traj_list , axis=0)
    learned_qvel_trajs = jnp.concatenate(learned_qvel_traj_list , axis=0)
    nominal_qpos_trajs = jnp.concatenate(nominal_qpos_traj_list , axis=0)
    nominal_qvel_trajs = jnp.concatenate(nominal_qvel_traj_list , axis=0)
    gt_qpos_trajs = jnp.concatenate(gt_qpos_traj_list , axis=0)
    gt_qvel_trajs = jnp.concatenate(gt_qvel_traj_list , axis=0)
        
        
    learned_rmse_qpos = jnp.sqrt(jnp.mean((learned_qpos_trajs - gt_qpos_trajs)**2))
    nominal_rmse_qpos = jnp.sqrt(jnp.mean((nominal_qpos_trajs - gt_qpos_trajs)**2))
    
    learned_rmse_qvel = jnp.sqrt(jnp.mean((learned_qvel_trajs - gt_qvel_trajs)**2))
    nominal_rmse_qvel = jnp.sqrt(jnp.mean((nominal_qvel_trajs - gt_qvel_trajs)**2))
        
    print("Learned RMSE qpos:")
    print(learned_rmse_qpos)
    print("Nominal RMSE qpos:")
    print(nominal_rmse_qpos)
    
    print("Learned RMSE qvel:")
    print(learned_rmse_qvel)
    print("Nominal RMSE qvel:")
    print(nominal_rmse_qvel)
    
    l2 = 589.0
    l4 = 3.5
    print("Learned Loss")         
    print(l2*jnp.mean((learned_qpos_trajs - gt_qpos_trajs)**2) + l4*jnp.mean((learned_qvel_trajs - gt_qvel_trajs)**2))
    print("Nominal Loss")
    print(l2*jnp.mean((nominal_qpos_trajs - gt_qpos_trajs)**2) + l4*jnp.mean((nominal_qvel_trajs - gt_qvel_trajs)**2))
    
        

    
"""
MAIN FUNCTION
"""    
    
def main():
    test_path = return_Arguments()
    test_options = parse_Specs(test_path)
    
    if test_options["model_type"] == "old":
        model , mjx_data = load_Old_Model(test_options)
    elif test_options["model_type"]=="new":
        model , mjx_data = load_New_Model(test_options)
    elif test_options["model_type"]=="sindy":
        model , mjx_data = load_SINDy_Model(test_options)
    else:
        raise ValueError(f"{test_options['model_type']} is not a known model type.")
    
    save_name = get_New_Name(test_options["save_name"])
    save_path = "Test_Outputs/" + save_name
    
    jobs = test_options["jobs"]
    
    for job_name , job_options in jobs.items():
        
        job_type = job_options["type"]
        if job_type == "statistics":
            run_Statistics_Job(model , mjx_data , job_options , job_name , save_path)
            
            
    print(test_options["load_name"])
        
    
if __name__ == "__main__":
    main()
    
    
    
    
    
