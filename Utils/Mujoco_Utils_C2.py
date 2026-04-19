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
    
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_LIMIT
    
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.make_data(model)
    
    return mjx_model , mjx_data
    
def init_mujoco_model(xml):
    model = mujoco.MjModel.from_xml_string(xml)
    
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_LIMIT
    
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
    
    model = spec1.compile()
    data = mujoco.MjData(model)
    nq = model.nq // 3
    
    with mj_viewer.launch_passive(model , data) as viewer:
        while viewer.is_running():
            for i in range(KNODE_qpos_traj.shape[0]):
                data.qpos[:nq] = KNODE_qpos_traj[i]
                data.qpos[nq:2*nq] = gt_qpos_traj[i]
                data.qpos[2*nq:3*nq] = nominal_qpos_traj[i]
        
                mujoco.mj_forward(model, data)
                time.sleep(.002*10)
                viewer.sync()


def render_GT_vs_Nominal_vs_KNODE_Video(
    model_name,
    gt_qpos_traj,
    nominal_qpos_traj,
    KNODE_qpos_traj,
    save_path,
    fps=50,
    width=2560,
    height=1440,
    playback_speed=1.0,
):
    """
    Render a side-by-side comparison of GT / Learned / Nominal trajectories to a
    video file using MuJoCo's offscreen renderer.  Entirely non-blocking — no
    viewer window is opened.

    The comparison XML is expected to contain three copies of the robot arranged
    side-by-side, each consuming `nq = model.nq // 3` generalised coordinates in
    the order:  [Learned | GT | Nominal].

    Parameters
    ----------
    model_name      : str   base name used to look up  <model_name>_comparison.txt
    gt_qpos_traj    : array (T, nq_single)
    nominal_qpos_traj: array (T, nq_single)
    KNODE_qpos_traj : array (T, nq_single)
    save_path       : str   destination path WITHOUT extension
    fps             : int   frames per second in the saved file
    width, height   : int   render resolution
    playback_speed  : float fraction of real-time to encode  (1.0 = real-time)

    Saves
    -----
    <save_path>.mp4   (falls back to <save_path>.gif if ffmpeg is unavailable)
    """
    try:
        import imageio
    except ImportError:
        print("    Warning: imageio not installed — skipping video render.")
        print("             Install with:  pip install imageio imageio-ffmpeg")
        return

    # ── Build comparison model ─────────────────────────────────────────────
    xml   = get_XML_From_Name(model_name + "_comparison")
    model = mujoco.MjModel.from_xml_string(xml)
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_LIMIT
    data  = mujoco.MjData(model)
    nq    = model.nq // 3

    # ── Set up offscreen renderer ──────────────────────────────────────────
    renderer = mujoco.Renderer(model, height=height, width=width)

    # ── Collect frames ─────────────────────────────────────────────────────
    n_frames   = KNODE_qpos_traj.shape[0]
    # Subsample if playback_speed > 1 so the video runs faster
    step       = max(1, round(playback_speed))
    frame_indices = range(0, n_frames, step)
    frames     = []

    for i in frame_indices:
        data.qpos[:nq]        = KNODE_qpos_traj[i]
        data.qpos[nq:2*nq]   = gt_qpos_traj[i]
        data.qpos[2*nq:3*nq] = nominal_qpos_traj[i]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        pixels = renderer.render().copy()   # (H, W, 3) uint8

        # ── Overlay text labels via PIL ────────────────────────────────────
        pixels = _annotate_comparison_frame(pixels, width)
        frames.append(pixels)

    renderer.close()

    # ── Save video ─────────────────────────────────────────────────────────
    actual_fps = int(fps / max(1, step) * playback_speed)
    actual_fps = max(1, actual_fps)

    try:
        out_path = save_path + ".mp4"
        imageio.mimsave(out_path, frames, fps=actual_fps, quality=8, macro_block_size=None)
        print(f"    Video saved: {out_path}")
    except Exception as e:
        # Fall back to GIF (no ffmpeg required)
        try:
            out_path = save_path + ".gif"
            duration = 1000 // actual_fps   # ms per frame for GIF
            imageio.mimsave(out_path, frames, duration=duration, loop=0)
            print(f"    Video saved (GIF fallback): {out_path}  (mp4 error: {e})")
        except Exception as e2:
            print(f"    Warning: could not save video — mp4: {e}  gif: {e2}")


def _annotate_comparison_frame(pixels, total_width):
    """
    Draw 'Learned / GT / Nominal' labels into the top of a comparison frame.
    Falls back silently if Pillow is not installed.
    """
    try:
        from PIL import Image, ImageDraw
        img  = Image.fromarray(pixels)
        draw = ImageDraw.Draw(img)
        w3   = total_width // 3
        # Colours match the matplotlib palette used in Test_Suite
        labels = [
            ("Learned", w3 // 2,     (60, 200, 60)),
            ("GT",      w3 + w3 // 2,(255, 255, 255)),
            ("Nominal", 2*w3 + w3//2,(200, 60, 60)),
        ]
        for text, cx, colour in labels:
            # rough text width: ~6px per char
            tx = cx - len(text) * 3
            draw.text((tx, 8), text, fill=colour)
        return np.array(img)
    except ImportError:
        return pixels


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
