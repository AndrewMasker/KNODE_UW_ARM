import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

def return_Arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n" , "--name" , type = str , default = "test_just_2_7")
    parser.add_argument("-t" , "--tasks" , nargs = "+" , type = str , default = ["mean" , "inverse_mean" , "std" , "vis"])
    
    args = parser.parse_args()
    
    save_path = "Models/" + args.name 
    
    return save_path , args.tasks

def get_LL_from_SP(save_path):
    with open(save_path + "/loss_list.json") as f:
        d = json.load(f)
    return list(d)

def get_Mean(save_path):
    loss_list = get_LL_from_SP(save_path)
    mean = sum(loss_list) / len(loss_list)
    print(f"Mean              : {mean}")
    
def get_Inverse_Mean(save_path):
    loss_list = get_LL_from_SP(save_path)
    mean = sum(loss_list) / len(loss_list)
    print(f"Inverse Mean      : {1/mean}")
    
def get_Std(save_path):
    loss_list = get_LL_from_SP(save_path)
    mean = sum(loss_list) / len(loss_list)
    diff_square = [(loss - mean)**2 for loss in loss_list]
    std = sum(diff_square) / (len(diff_square)-1)
    
    print(f"Standard Deviation: {std}")
    

def vis_Loss(save_path):
    loss_list = get_LL_from_SP(save_path)
    fig , (ax1 , ax2) = plt.subplots(1 , 2)
    loss_inds = np.arange(1 , len(loss_list) + 1)
    ax1.plot(loss_inds , loss_list)
    ax2.loglog(loss_inds , loss_list)
    plt.show()
    

task_to_function_mapping = {
    'mean': get_Mean,
    'std': get_Std , 
    'inverse_mean': get_Inverse_Mean , 
    'vis': vis_Loss
}
        
def main():
    save_path , tasks = return_Arguments()
    
    for task in tasks:
        task_to_function_mapping[task](save_path)
        
        
if __name__ == "__main__":
    main()
