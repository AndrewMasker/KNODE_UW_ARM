import os
import sys
from Utils import Mujoco_Utils as mj_utils


def return_Arguments():
    if len(sys.argv)>1:
        # Put input variables here
        model_name = sys.argv[1]
    else:
        # Put default variables here
        model_name = "robotic_arm_1"
        
    return model_name


def main():
    model_name = return_Arguments()

    model , data = mj_utils.init_mujoco_model(mj_utils.get_XML_From_Name(model_name))
    data.qvel[1] = .001
    mj_utils.visualize_mujoco_model(model , data)



if __name__ == "__main__":
    main()
