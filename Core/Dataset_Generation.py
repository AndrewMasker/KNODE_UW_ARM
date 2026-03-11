from functools import partial
import jax
import jax.random as jr
import jax.numpy as jnp
import json
from mujoco import mjx
import mujoco
from mujoco import viewer
import os
import sys
import time
import Utils.Data_Utils as data_utils
import Utils.Mujoco_Utils as mj_utils
from robot_descriptions import panda_mj_description
from pathlib import Path
import zarr


"""
ARGUMENT AND SPECIFICATION PARSING
"""

def return_Arguments():
    if len(sys.argv)>1:
        # Put input variables here
        path_to_dataset_specs = "/Library/Dataset_Specifications/" + sys.argv[1] + ".json"
    else:
        # Put default variables here
        path_to_dataset_specs = "/Library/Dataset_Specifications/dataset_1_specs.json"
        
    return os.getcwd() + path_to_dataset_specs
    

def parse_Dataset_Specs(path_to_dataset_specs):
    with open(path_to_dataset_specs) as f:
        d = json.load(f)
    dataset_options = d
    return dataset_options
    

"""
DATASET GENERATION
"""

def create_Sample(key , mjx_model , mjx_data , ranges , dataset_options):
    qpos_key , qvel_key , ctrl_key = jr.split(key , 3)
    
    qpos = jr.uniform(qpos_key , shape = (mjx_data.qpos.shape[0],) , minval = ranges[0][: , 0] , maxval = ranges[0][: , 1])
    qvel = jr.uniform(qvel_key , shape = (mjx_data.qvel.shape[0],) , minval = -ranges[1] , maxval = ranges[1])
    ctrl = jr.uniform(ctrl_key , shape = (mjx_data.ctrl.shape[0],) , minval = ranges[2][: , 0] , maxval = ranges[2][: , 1])
    
    mjx_data = mjx_data.replace(qpos = qpos)
    mjx_data = mjx_data.replace(qvel = qvel)
    mjx_data = mjx_data.replace(ctrl = ctrl)
    
    N_STEPS = dataset_options["num_steps"]
    def step_fn(data, _):
        data = mjx.step(mjx_model, data)
        return data, (data.qpos , data.qvel , data.ctrl)
    
    _, traj_tuple = jax.lax.scan(step_fn, mjx_data, None, length=N_STEPS)

    return traj_tuple

@partial(jax.jit , static_argnums = (4,))
def create_Batch(key , mjx_model , mjx_data , ranges , dataset_options):
    dataset_options = dict(dataset_options)
    keys = jr.split(key , dataset_options["batch_size"])
    trajs_tuple = jax.vmap(create_Sample , in_axes = (0 , None , None , None , None))(keys , mjx_model , mjx_data , ranges , dataset_options)
    return trajs_tuple
    

def get_New_Name(name , *args):
    if len(args) == 0:
        if os.path.exists(os.getcwd() + "/Datasets/" + name + ".zarr"):
            return get_New_Name(name , 1)
        else:
            return name
    else:
        if os.path.exists(os.getcwd() + "/Datasets/" + name + "_" + str(args[0]) + ".zarr"):
            return get_New_Name(name , args[0] + 1)
        else:
            return name + "_" + str(args[0])
            

def main(dataset_options):
    
    model_xml = mj_utils.create_XML_From_Options(dataset_options)
    mjx_model , mjx_data = mj_utils.init_mjx_model(model_xml)
    
    qpos_ranges = mjx_model.jnt_range
    qvel_max = jnp.array(dataset_options["qvel_max"])
    ctrl_ranges = mjx_model.actuator_ctrlrange
    ranges = (qpos_ranges , qvel_max , ctrl_ranges)
    
    name = get_New_Name(dataset_options["name"])
    root = zarr.open("Datasets/" + name + ".zarr" , mode="a")
    
    root.attrs["xml"] = model_xml
    root.attrs["dataset_options"] = data_utils.convert_Dict_to_Zarr_Saveable(dataset_options)
    
    dataset_options_tuple = tuple(sorted(data_utils.convert_Dict_Entries_to_Jax_Hashable(dataset_options).items()))
    
    for batch_id in range(dataset_options['num_batches']):
            t1 = time.time()
            print(f"Starting Batch: {batch_id} of {dataset_options['num_batches']} Total Batches.")
            key = jax.random.key(int(time.time()))
            trajs_tuple = create_Batch(key , mjx_model , mjx_data , ranges , dataset_options_tuple)
            trajs_qpos = trajs_tuple[0]
            trajs_qvel = trajs_tuple[1]
            trajs_ctrl = trajs_tuple[2]
        
            for sample_id in range(dataset_options['batch_size']):
                group = root.create_group(f"trajectory_{batch_id*dataset_options['batch_size'] + sample_id}")
                
                group.create_array("qpos" , data = trajs_qpos[sample_id , : , :])
                group.create_array("qvel" , data = trajs_qvel[sample_id , : , :])
                group.create_array("ctrl" , data = trajs_ctrl[sample_id , : , :])
        
            print(f"Finishing Batch: {batch_id} of {dataset_options['num_batches']} Total Batches. Time Elapsed: {time.time() - t1}"),
            

"""
CALL MAIN FUNCTION
"""

if __name__ == "__main__":
    path_to_dataset_specs = return_Arguments()
    dataset_options = parse_Dataset_Specs(path_to_dataset_specs)
    main(dataset_options)
    
    
    
    
    
    
    


