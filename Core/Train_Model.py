import os
#os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import argparse
import collections
from Core.KNODE_Modules import KNODE
import equinox as eqx
from flax.jax_utils import prefetch_to_device
import grain

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
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
    parser.add_argument("-m" , "--model_specs" , type = str , default = "model_2")
    parser.add_argument("-t" , "--training_specs" , type = str , default = "training_1")
    
    args = parser.parse_args()
    
    model_path = "Library/Model_Specifications/" + args.model_specs + ".json"
    training_path = "Library/Training_Specifications/" + args.training_specs + ".json"
        
    return model_path , training_path
    
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
        
        
def load_Model_From_Options(model_options , training_options):
    dataset_path = "Datasets/" + training_options["dataset_name"] + ".zarr"
    root = zarr.open(dataset_path)
    qvel_max = jnp.array(root.attrs["dataset_options"]["qvel_max"])
    
    model_xml = mj_utils.create_XML_From_Options(training_options)
    mjx_model , mjx_data = mj_utils.init_mjx_model(model_xml)
    
    key = jr.key(int(time.time()))
    
    
    if training_options["should_load"]:
        load_path = os.getcwd() + "/Models/" + training_options["load_name"] + "/"
        model_options = parse_Model_Specs(load_path + "model_options.json")
        
        model = KNODE(key , mjx_model , model_options["width"] , model_options["depth"] , qvel_max)
        
        model = eqx.tree_deserialise_leaves(load_path + "model.eqx" , model)
        
    else:
        model = KNODE(key , mjx_model , model_options["width"] , model_options["depth"] , qvel_max)
        
    
    return model , mjx_data
    
"""
VISUAL AIDS
"""

def visualize_Trained_Model(model , model_options , training_options , loss_list):
    root = zarr.open(os.getcwd() + "/Datasets/" + training_options["dataset_name"] + ".zarr" , mode = 'r')
    dt = root.attrs.get("dataset_options")["step_size"]
    xml = root.attrs.get("xml")
    mj_model , data = mj_utils.init_mujoco_model(xml)
    
    _ , mjx_data = mj_utils.init_mjx_model(xml)
    
    trajectory_number = training_options["which_to_show_at_end"]
    d = root["trajectory_" + str(trajectory_number)]
    gt_qpos_traj = d["qpos"][:]
    
    step_size = root.attrs["dataset_options"]["step_size"]
    num_steps = root.attrs["dataset_options"]["num_steps"]
    ts = jnp.arange(0 , step_size*(num_steps) , step = step_size)
    
    init_states = {"qpos": jnp.array(d["qpos"][0 , :]) , "qvel": jnp.array(d["qvel"][0 , :])}
    ctrl = jnp.array(d["ctrl"][0 , :])
    
    KNODE_traj = model(ts , mjx_data , init_states , ctrl , with_learned = True)
    KNODE_qpos_traj = np.array(KNODE_traj["qpos"])
    
    nominal_traj = model(ts , mjx_data , init_states , ctrl , with_learned = False)
    nominal_qpos_traj = np.array(nominal_traj["qpos"])

    
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

def trajectory_Train(model , mjx_data , num_batches , steps_strategy , lr , training_options , filter_spec , loss_list):

    dataset_path = "Datasets/" + training_options["dataset_name"] + ".zarr"
    root = zarr.open(dataset_path)
    step_size = root.attrs["dataset_options"]["step_size"]
    num_steps = root.attrs["dataset_options"]["num_steps"]
    ts = jnp.arange(0 , step_size*(num_steps) , step = step_size)
    
    
    dataset = data_utils.zarr_Trajectory_Dataset(dataset_path , training_options["subset_size"])
    dataset = (dataset
               .shuffle(seed = int(time.time()))
               .batch(batch_size = training_options["batch_size"] , drop_remainder = True)
               .repeat())
    
    iter_loader = dataset.to_iter_dataset()    
#        iter_loader = prefetch_to_device(dataset , size = 10 , devices = None)
    iter_loader = iter(iter_loader)
    
    def prefetch(loader , size = 2):
        queue = collections.deque()
        def enqueue(n):
            for _ in range(n):
                queue.append(jtu.tree_map(jax.device_put , next(loader)))
        enqueue(size)
        while queue:
            yield queue.popleft()
            enqueue(1)
    iter_loader =  prefetch(iter_loader , size=5)
    
    optim = optax.adabelief(lr)
    diff_model , static_model = eqx.partition(model , filter_spec)
    opt_state = optim.init(eqx.filter(diff_model , eqx.is_inexact_array))
    
    l0 = training_options["l0"]
    l1 = training_options["l1"]
    l2 = training_options["l2"]
    l3 = training_options["l3"]
    l4 = training_options["l4"]
    
    @eqx.filter_value_and_grad
    def grad_loss(diff_model , static_model , ts , gt_trajs , ctrls , mjx_data , init_states):
        model = eqx.combine(diff_model , static_model)
        pred_trajs = jax.vmap(model , (None , None , 0 , 0))(ts , mjx_data , init_states , ctrls)
        get_data = lambda given_qpos , given_qvel : mjx.com_vel(model.knode_dynamics.mjx_model , mjx.com_pos(model.knode_dynamics.mjx_model , mjx.kinematics(model.knode_dynamics.mjx_model , mjx_data.replace(qpos=given_qpos , qvel=given_qvel))))
        get_data_traj = jax.vmap(jax.vmap(get_data , in_axes = (0 , 0) , out_axes = 0) , in_axes = (0 , 0) , out_axes = 0)

        pred_data_trajs = get_data_traj(pred_trajs["qpos"]  , pred_trajs["qvel"])
        gt_data_trajs = get_data_traj(gt_trajs["qpos"] , gt_trajs["qvel"])
        
        pred_xpos_trajs = pred_data_trajs.xpos
        gt_xpos_trajs = gt_data_trajs.xpos
        
        pred_xquat_trajs = pred_data_trajs.xquat
        gt_xquat_trajs = gt_data_trajs.xquat
        
        pred_qpos_trajs = pred_trajs["qpos"]
        gt_qpos_trajs = gt_trajs["qpos"]
        
        pred_cvel_trajs = pred_data_trajs.cvel
        gt_cvel_trajs = gt_data_trajs.cvel
        
        pred_qvel_trajs = pred_trajs["qvel"]
        gt_qvel_trajs = gt_trajs["qvel"]
        
#            
        return(l0*jnp.mean((pred_xpos_trajs - gt_xpos_trajs)**2) + 
               l1*jnp.mean((pred_xquat_trajs - gt_xquat_trajs)**2) + 
               l2*jnp.mean((pred_qpos_trajs - gt_qpos_trajs)**2) + 
               l3*jnp.mean((pred_cvel_trajs - gt_cvel_trajs)**2) + 
               l4*jnp.mean((pred_qvel_trajs - gt_qvel_trajs)**2))
               
#            return (l2*jnp.mean((pred_qpos_trajs - gt_qpos_trajs)**2) + 
#                    l4*jnp.mean((pred_qvel_trajs - gt_qvel_trajs)**2))
#                   
    
    @eqx.filter_jit
    def make_step(model , ts , gt_trajs , ctrls , mjx_data):
        diff_model , static_model = eqx.partition(model , filter_spec)
        init_states = {"qpos": gt_trajs["qpos"][: , 0 , :] , "qvel": gt_trajs["qvel"][: , 0 , :]}
        loss , grads = grad_loss(diff_model , static_model , ts , gt_trajs , ctrls , mjx_data , init_states)
#            print(grads.knode_dynamics.residual_network.out_scale)
#            print(grads.knode_dynamics.residual_network.mlp.layers[0].weight)
#            print(grads.knode_dynamics.mjx_mixing_matrix)
#            print(grads.knode_dynamics.res_mixing_matrix)
#            poop()
        updates , opt_states = optim.update(grads , opt_state)
        diff_model = eqx.apply_updates(diff_model , updates)
        model = eqx.combine(diff_model , static_model)
        return loss , model , opt_states

    _num_steps = int(num_steps*steps_strategy)
    
    for batch_id in range(num_batches):
        t1 = time.time()
        gt_trajs , ctrls = next(iter_loader)
        gt_trajs = jtu.tree_map(jax.device_put , gt_trajs)
        ctrls = jax.device_put(ctrls)
        _ts = ts[0:_num_steps]
        key = jr.key(int(time.time()))
        start_step = int(jr.randint(key , (1,) , minval = 0 , maxval = num_steps - _num_steps)[0])
        _gt_trajs = {"qpos": gt_trajs["qpos"][: , start_step:start_step+_num_steps , :] , "qvel": gt_trajs["qvel"][: , start_step:start_step+_num_steps , :]}
        loss , model , opt_state = make_step(model , _ts , _gt_trajs , ctrls , mjx_data)
        
        loss_list.append(loss.item())
        
        if batch_id%training_options["print_every"] == 0 or (batch_id+1)==num_batches:
            print(f"Batch: {batch_id+1} of {num_batches}. Loss: {loss}. Elapsed time: {time.time() - t1}")
            print(model.knode_dynamics.residual_network.vector)
            
    return model , loss_list


def accel_Train_From_Trajectory(model , mjx_data , num_batches , steps_strategy , lr , training_options , filter_spec , loss_list):

    dataset_path = "Datasets/" + training_options["dataset_name"] + ".zarr"
    root = zarr.open(dataset_path)
    step_size = root.attrs["dataset_options"]["step_size"]
    num_steps = root.attrs["dataset_options"]["num_steps"]
    ts = jnp.arange(0 , step_size*(num_steps) , step = step_size)
    
    
    dataset = data_utils.zarr_Trajectory_Dataset(dataset_path , training_options["subset_size"])
    dataset = (dataset
               .shuffle(seed = int(time.time()))
               .batch(batch_size = training_options["batch_size"] , drop_remainder = True)
               .repeat())
    
    iter_loader = dataset.to_iter_dataset()    
#        iter_loader = prefetch_to_device(dataset , size = 10 , devices = None)
    iter_loader = iter(iter_loader)
    
    def prefetch(loader , size = 2):
        queue = collections.deque()
        def enqueue(n):
            for _ in range(n):
                queue.append(jtu.tree_map(jax.device_put , next(loader)))
        enqueue(size)
        while queue:
            yield queue.popleft()
            enqueue(1)
    iter_loader =  prefetch(iter_loader , size=5)
    
    optim = optax.adabelief(lr)
    diff_model , static_model = eqx.partition(model , filter_spec)
    opt_state = optim.init(eqx.filter(diff_model , eqx.is_inexact_array))
    
    l0 = training_options["l0"]
    l1 = training_options["l1"]
    l2 = training_options["l2"]
    l3 = training_options["l3"]
    l4 = training_options["l4"]
    l5 = training_options["l5"]
    
    @eqx.filter_value_and_grad
    def grad_loss(diff_model , static_model , ts , qacc_act , ctrls , mjx_data , states):
        model = eqx.combine(diff_model , static_model)
#        pred_trajs = jax.vmap(model.knode_dynamics , (None , None , 0 , 0))(ts , mjx_data , init_states , ctrls)
        
        args = (mjx_data , ctrls , True)
        
#        qacc_pred = jax.vmap(model.knode_dynamics , (None , None , 0 , 0))(ts , state , args)
        
        qacc_pred = jax.vmap(jax.vmap(model.knode_dynamics , (None , 0 , (None , 0 , None))) , in_axes = (0 , 1 , (None , None , None)) , out_axes = 1)(ts , states , args)["qvel"]
        
#        get_data = lambda given_qpos , given_qvel : mjx.com_vel(model.knode_dynamics.mjx_model , mjx.com_pos(model.knode_dynamics.mjx_model , mjx.kinematics(model.knode_dynamics.mjx_model , mjx_data.replace(qpos=given_qpos , qvel=given_qvel))))
#        get_data_traj = jax.vmap(jax.vmap(get_data , in_axes = (0 , 0) , out_axes = 0) , in_axes = (0 , 0) , out_axes = 0)

#        print(qacc_pred)
#        print(qacc_act)
#        poop()
#        
#        print(qacc_pred.shape)
        return l5*jnp.mean((qacc_pred - qacc_act)**2)
             
#                   
    
    @eqx.filter_jit
    def make_step(model , ts , gt_trajs , ctrls , mjx_data):
        diff_model , static_model = eqx.partition(model , filter_spec)
        
        qacc_act = jnp.gradient(gt_trajs["qvel"] , ts , axis = 1)
#        print(qacc_act.shape)
        
        states = gt_trajs
#        print(states["qpos"].shape)
#        print(ctrls.shape)
        loss , grads = grad_loss(diff_model , static_model , ts , qacc_act , ctrls , mjx_data , states)
#            print(grads.knode_dynamics.residual_network.out_scale)
#            print(grads.knode_dynamics.residual_network.mlp.layers[0].weight)
#            print(grads.knode_dynamics.mjx_mixing_matrix)
#            print(grads.knode_dynamics.res_mixing_matrix)
#            poop()
        updates , opt_states = optim.update(grads , opt_state)
        diff_model = eqx.apply_updates(diff_model , updates)
        model = eqx.combine(diff_model , static_model)
        return loss , model , opt_states

    _num_steps = int(num_steps*steps_strategy)
    
    for batch_id in range(num_batches):
        t1 = time.time()
        gt_trajs , ctrls = next(iter_loader)
        gt_trajs = jtu.tree_map(jax.device_put , gt_trajs)
        ctrls = jax.device_put(ctrls)
        _ts = ts[0:_num_steps]
        key = jr.key(int(time.time()))
        start_step = int(jr.randint(key , (1,) , minval = 0 , maxval = num_steps - _num_steps)[0])
        _gt_trajs = {"qpos": gt_trajs["qpos"][: , start_step:start_step+_num_steps , :] , "qvel": gt_trajs["qvel"][: , start_step:start_step+_num_steps , :]}
        loss , model , opt_state = make_step(model , _ts , _gt_trajs , ctrls , mjx_data)
        
        loss_list.append(loss.item())
        
        if batch_id%training_options["print_every"] == 0 or (batch_id+1)==num_batches:
            print(f"Batch: {batch_id+1} of {num_batches}. Loss: {loss}. Elapsed time: {time.time() - t1}")
            print(model.knode_dynamics.residual_network.vector)
            
    return model , loss_list



def train_Model(model_options , training_options):
    
    
    model , mjx_data = load_Model_From_Options(model_options , training_options)
    loss_list = []
    

    
    filter_spec = jtu.tree_map(lambda _: False , model)
    
                              
#    filter_spec = eqx.tree_at(lambda tree: tree.knode_dynamics.residual_network.mlp,
#                      filter_spec , 
#                      replace = True)
#    filter_spec = eqx.tree_at(lambda tree: tree.knode_dynamics.residual_network.out_scale,
#                      filter_spec , 
#                      replace = True)
    filter_spec = eqx.tree_at(lambda tree: tree.knode_dynamics.mjx_mixing_matrix,
                      filter_spec , 
                      replace = True)
    filter_spec = eqx.tree_at(lambda tree: tree.knode_dynamics.res_mixing_matrix,
                      filter_spec , 
                      replace = True)
    filter_spec = eqx.tree_at(lambda tree: tree.knode_dynamics.residual_network.vector,
                      filter_spec , 
                      replace = True)
#    
    
    
    for num_batches , steps_strategy , lr in zip(training_options["num_batches_strategy"] , training_options["steps_strategy"] , training_options["lr_strategy"]):
        model , loss_list = trajectory_Train(model , mjx_data , num_batches , steps_strategy , lr , training_options , filter_spec , loss_list)
        
#        model , loss_list = accel_Train_From_Trajectory(model , mjx_data , num_batches , steps_strategy , lr , training_options , filter_spec , loss_list)
        
                
    return model , loss_list
    


"""
CALL MAIN FUNCTION
"""

if __name__ == "__main__":
    model_path , training_path = return_Arguments()
    model_options = parse_Model_Specs(model_path)
    training_options = parse_Training_Specs(training_path)
    model , loss_list = train_Model(model_options , training_options)
#    print(model.knode_dynamics.residual_network.vector)
#    print(model.knode_dynamics.res_mixing_matrix)
#    print(model.knode_dynamics.mjx_mixing_matrix)
    if training_options["should_save"]:
        save_path = save_Model(model , loss_list , model_options , training_options)
#    plot_Loss(loss_list)
    visualize_Trained_Model(model , model_options , training_options , loss_list)
    plt.show()
    
    
    
    
    
    
    

