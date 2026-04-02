import grain
import jax.numpy as jnp
import numpy as np
import zarr

def convert_Dict_to_Zarr_Saveable(dicty):
    new_dicty = {}
    for k , v in dicty.items():
        if isinstance(v , jnp.ndarray):
            new_dicty[k] = np.asarray(v).tolist()
        elif type(v) == dict:
            new_dicty[k] = convert_Dict_Entries_to_Zarr_Saveable(v)
        else:
            new_dicty[k] = v

    return new_dicty
    
def convert_Dict_Entries_to_Jax_Hashable(dicty):
    new_dicty = {}
    for k , v in dicty.items():
        if isinstance(v , list):
            new_dicty[k] = tuple(v)
        else:
            new_dicty[k] = v

    return new_dicty

def convert_Zarr_Saveable_to_Dict(dicty):
    new_dicty = {}
    for k , v in dicty.items():
        if isinstance(v , list):
            new_dicty[k] = jnp.asarray(v)
        elif type(v) == dict:
            new_dicty[k] = convert_Dict_Entries_to_Zarr_Saveable(v)
        else:
            new_dicty[k] = v

    return new_dicty

class zarr_Trajectory_Dataset(grain.MapDataset):
    def __init__(self , root_path , subset_size):
        super().__init__()
        self.root = zarr.open(root_path , mode = 'r')
        self.keys = list(self.root.keys())
        self.len_keys = len(self.keys)
        if subset_size == "None":
            self.length = self.len_keys
        else:
            if subset_size > self.len_keys:
                raise ValueError("Subset Size Cannot Be Greater Than Dataset Size")
            else:
                self.length = subset_size

    def __len__(self):
        return self.length

    def __getitem__(self , idx):
        grp = self.root[self.keys[idx%self.length]]
        traj = {"qpos": grp['qpos'][:] , "qvel": grp['qvel'][:]}
        ctrl = grp['ctrl'][0 , :]

        return traj , ctrl

def collate_fn(batch):

    batched_data = np.stack([sample[0] for sample in batch])
    batched_label = np.stack([sample[1] for sample in batch])

    return (batched_data , batched_label)
    







        
