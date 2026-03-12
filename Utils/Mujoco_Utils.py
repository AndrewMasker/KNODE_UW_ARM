import argparse
import mujoco
import mujoco.viewer as mj_viewer
import numpy as np
from mujoco import mjx
import os
from pathlib import Path
import time
import zarr

def get_XML_From_Name(name):
    return Path(os.getcwd() + "/Library/XML_Models/" + name + ".txt").read_text()

def create_XML_From_Options(options):
    model_xml = get_XML_From_Name(options["model_name"])
    
    if options["is_gravity"]:
        model_xml = model_xml.replace("<!-- GRAVITY_FLAG -->" , get_XML_From_Name("gravity_string")).replace("GRAVITY_FLAG" , "0 0 -9.81")
    else:
        model_xml = model_xml.replace("<!-- GRAVITY_FLAG -->" , get_XML_From_Name("gravity_string")).replace("GRAVITY_FLAG" , "0 0 0")     
    if options["is_fluid"]:
        fluid_options_xml = get_XML_From_Name("fluid_options_string")
        if options["is_const_wind"]:
            fluid_options_xml = fluid_options_xml.replace("WIND_FLAG" , options["wind"])
        fluid_options_xml = fluid_options_xml.replace("DENSITY_FLAG" , options["density"])
        fluid_options_xml = fluid_options_xml.replace("VISCOSITY_FLAG" , options["viscosity"])
        model_xml = model_xml.replace("<!-- FLUID_FLAG -->" , fluid_options_xml)
    model_xml = model_xml.replace("<!-- TIMESTEP_FLAG -->" , get_XML_From_Name("timestep_string").replace("TIMESTEP_FLAG" , str(options["step_size"])))
    return model_xml

# This function takes a path to a txt file with the xml definition of the
# and converts it into the mjx_model and data
def init_mjx_model(xml):
    model = mujoco.MjModel.from_xml_string(xml)
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.make_data(model)
    
    return mjx_model , mjx_data
    
def init_mujoco_model(xml):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    
    return model , data

def visualize_mujoco_model(model , data):
    with mj_viewer.launch_passive(model , data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model , data)
            time.sleep(.002)
            viewer.sync()
            
# python3 -c "import Utils.Mujoco_Utils as mj_utils; mj_utils.visualize_Saved_Trajectory('dataset_1',1)"
def visualize_Saved_Trajectory(dataset_name , trajectory_number):
    root = zarr.open(os.getcwd() + "/Datasets/" + dataset_name + ".zarr" , mode = 'r')
    
    xml = root.attrs.get("xml")
    model , data = init_mujoco_model(xml)
    
    d = root["trajectory_" + str(trajectory_number)]
    qpos_traj = d["qpos"][:]
    ctrl_traj = d["ctrl"][:]
    
    with mj_viewer.launch_passive(model , data) as viewer:
        while viewer.is_running():
            for i in range(qpos_traj.shape[0]):
                data.qpos = qpos_traj[i , :]
                data.ctrl = ctrl_traj[i , :]
                mujoco.mj_forward(model, data)
                time.sleep(.002*10)
                viewer.sync()
                
                
def visualize_Mujoco_XML(name):
    xml = get_XML_From_Name(name)
    
    model , data = init_mujoco_model(xml)
    
    with mj_viewer.launch(model , data) as viewer:
        while viewer.is_running():            
            mujoco.mj_forward(model, data)
            time.sleep(.002)
            viewer.sync()
                

def visualize_GT_vs_Nominal_vs_KNODE_Trajectory(model_name , gt_qpos_traj , nominal_qpos_traj , KNODE_qpos_traj):

    xml = get_XML_From_Name(model_name + "_comparison")

    spec1 = mujoco.MjSpec.from_string(xml)
    spec2 = mujoco.MjSpec.from_string(xml)
    

#    
    model = spec1.compile()
    data = mujoco.MjData(model)
    nq = model.nq // 3
    
    
    with mj_viewer.launch_passive(model , data) as viewer:
        while viewer.is_running():
            for i in range(KNODE_qpos_traj.shape[0]):
                
                # Assign states to the dual model
                data.qpos[:nq] = KNODE_qpos_traj[i]
                data.qpos[nq:2*nq] = gt_qpos_traj[i]
                data.qpos[2*nq:3*nq] = nominal_qpos_traj[i]
        
                mujoco.mj_forward(model, data)
                time.sleep(.002*10)
                viewer.sync()
    
    
def return_Arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t" , "--type" , type = str , default = "traj_vis")
    parser.add_argument("-n" , "--name" , type = str , default = "mjx_panda_orig")
    parser.add_argument("-j" , "--traj" , type = int , default = 0)
    
    
    args = parser.parse_args()
    
    type_ = args.type
    name = args.name
    traj = args.traj
        
    return type_ , name , traj
    
def main():
    type_ , name , traj = return_Arguments()
    
    if type_ == "xml_vis":
        visualize_Mujoco_XML(name)
    elif type_ == "traj_vis":
        visualize_Saved_Trajectory(name , traj)
    
if __name__ == "__main__":
    main()
    
    
    
    
