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
    model_xml = model_xml.replace("<!-- TIMESTEP_FLAG -->" , get_XML_From_Name("timestep_string").replace("TIMESTEP_FLAG" , options["step_size"]))
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
    
    with mj_viewer.launch_passive(model , data) as viewer:
        while viewer.is_running():
            for i in range(qpos_traj.shape[0]):
                data.qpos = qpos_traj[i , :]
                mujoco.mj_forward(model, data)
                time.sleep(.002)
                viewer.sync()
                

def visualize_GT_vs_ML_Trajectory(model_name , gt_qpos_traj , ml_qpos_traj):

    xml = get_XML_From_Name(model_name + "_duplicated")

    spec1 = mujoco.MjSpec.from_string(xml)
    spec2 = mujoco.MjSpec.from_string(xml)
    
#    # Prefix everything in the second model to avoid collisions
#    prefix = "gt_"
#    for obj_list in [spec2.bodies, spec2.joints, spec2.geoms, spec2.meshes, spec2.materials]:
#        for item in obj_list:
#            if item.name:
#                item.name = prefix + item.name
#    print(dir(spec2))
#    spec2_root = spec2.worldbody.first_body()
#    attachment_body = spec1.worldbody
#    
#    spec2_root.attach(attachment_body)
#    spec1.attach(spec2 , frame = spec1.worldbody)
#    
    model = spec1.compile()
    data = mujoco.MjData(model)
    nq = model.nq // 2
    
    
    with mj_viewer.launch_passive(model , data) as viewer:
        while viewer.is_running():
            for i in range(ml_qpos_traj.shape[0]):
                
                # Assign states to the dual model
                data.qpos[:nq] = ml_qpos_traj[i]
                data.qpos[nq:2*nq] = gt_qpos_traj[i]
        
                mujoco.mj_forward(model, data)
                time.sleep(.002)
                viewer.sync()
    
    
    
    
    
    
    
    
