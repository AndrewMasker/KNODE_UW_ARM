import argparse
import json

def return_Arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n" , "--name" , type = str , default = "l4_test_03052026")
    parser.add_argument("-t" , "--tasks" , nargs = "+" , type = str , default = ["mean" , "inverse_mean" , "std"])
    
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

task_to_function_mapping = {
    'mean': get_Mean,
    'std': get_Std , 
    'inverse_mean': get_Inverse_Mean
}
        
def main():
    save_path , tasks = return_Arguments()
    
    for task in tasks:
        task_to_function_mapping[task](save_path)
        
        
if __name__ == "__main__":
    main()
