#import os
#os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import argparse
#import collections
#from Core.KNODE_Modules import KNODE
#import equinox as eqx
#from flax.jax_utils import prefetch_to_device
#import grain

from Core.Dataset_Generation import get_New_Name
from Core.Train_Model_UVMS import load_Model_From_Options #This will need to be changed depending on which type
import jax
import jax.numpy as jnp
#import jax.random as jr
#import jax.tree_util as jtu
import json
#import matplotlib.pyplot as plt
import mujoco.mjx as mjx
import numpy as np
#import optax
#import os
import time
import Utils.Data_Utils as data_utils
import Utils.Mujoco_Utils as mj_utils
import zarr





def return_Arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m" , "--model_specs" , type = str , default = "model_scalped")
    parser.add_argument("-t" , "--training_specs" , type = str , default = "training_fluid_uvms")
    parser.add_argument("-d" , "--dataset_name" , type = str , default = "uvms_no_coll")
    
    args = parser.parse_args()
    
    model_path = "Library/Model_Specifications/" + args.model_specs + ".json"
    training_path = "Library/Training_Specifications/" + args.training_specs + ".json"
    dataset_path = "Datasets/" + args.dataset_name + ".zarr"
        
    return model_path , training_path , dataset_path
    
def parse_Model_Specs(path_to_model_specs):
    with open(path_to_model_specs) as f:
        d = json.load(f)
    model_options = d
    return model_options
    
def parse_Training_Specs(path_to_training_specs):
    with open(path_to_training_specs) as f:
        d = json.load(f)
    training_options = d
    return training_options
    

    
    
def main(model_options , training_options , dataset_path):
    
    model , mjx_data = load_Model_From_Options(model_options , training_options)
    
    @jax.jit
    def internal(mjx_data , ctrl , states , mjx_model):
        args = (mjx_data , ctrl , jnp.array(0.0))
        
        batched_knode_dynam = jax.vmap(model.knode_dynamics , in_axes = (None , {"qpos":0 , "qvel":0} , None))
        
        states_vel = batched_knode_dynam(0 , states , args)
        
        nominal_qacc = states_vel["qvel"]
        
        gt_forward = lambda states , ctrl: mjx.forward(mjx_model , mjx_data.replace(qpos = states["qpos"] , qvel = states["qvel"] , ctrl = ctrl))
        
        batched_gt_forward = jax.vmap(gt_forward , in_axes = ({"qpos":0 , "qvel":0} , None))
        
        gt_mjx_data_batched = batched_gt_forward(states , ctrl)
        gt_qacc = gt_mjx_data_batched.qacc
        
        res_qacc = gt_qacc - nominal_qacc
        
        return res_qacc
    
    root = zarr.open(dataset_path)
    step_size = root.attrs["dataset_options"]["step_size"]
    num_steps = root.attrs["dataset_options"]["num_steps"]
    ts = jnp.arange(0 , step_size*(num_steps) , step = step_size)
    
    
    keys = sorted(root.keys(), key=lambda k: int(k.split('_')[-1]))
    
    dataset_options = root.attrs["dataset_options"]
    model_xml = mj_utils.create_XML_From_Options(dataset_options)
    mjx_model , mjx_data = mj_utils.init_mjx_model(model_xml)
    
    
    name = get_New_Name(dataset_options["name"] + "_res")
    save_root = zarr.open("Datasets/" + name + ".zarr" , mode="a")
    
    save_root.attrs["xml"] = model_xml
    save_root.attrs["dataset_options"] = data_utils.convert_Dict_to_Zarr_Saveable(dataset_options)
    save_root.attrs["model_options"] = data_utils.convert_Dict_to_Zarr_Saveable(model_options)
    save_root.attrs["training_options"] = data_utils.convert_Dict_to_Zarr_Saveable(training_options)
    
    
    for trajectory_number in range(len(keys)):
    
        t = keys[trajectory_number]
        d = root[t]
        gt_qpos_traj = jnp.array(d["qpos"][:])
        gt_qvel_traj = jnp.array(d["qvel"][:])
        states = {"qpos":gt_qpos_traj , "qvel":gt_qvel_traj}
#        init_states = {"qpos": jnp.array(d["qpos"][0 , :]) , "qvel": jnp.array(d["qvel"][0 , :])}
        ctrl = jnp.array(d["ctrl"][0 , :])
        
        res_qacc = internal(mjx_data , ctrl , states , mjx_model)
        
        
        group = save_root.create_group(f"{t}")
        
        group.create_array("qpos" , data = gt_qpos_traj)
        group.create_array("qvel" , data = gt_qvel_traj)
        group.create_array("ctrl" , data = d["ctrl"][:])
        group.create_array("res_qacc" , data = res_qacc)
        
        print(trajectory_number)
        
        
    
    
    
if __name__ == "__main__":
    model_path , training_path , dataset_path = return_Arguments()
    model_options = parse_Model_Specs(model_path)
    training_options = parse_Training_Specs(training_path)
    main(model_options , training_options , dataset_path)
    
