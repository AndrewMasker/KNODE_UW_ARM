import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import mujoco
from mujoco import mjx
from pathlib import Path


# This function takes a path to a txt file with the xml definition of the
# and converts it into the mjx_model and data
def init_mjx_model(path_to_model_xml):
    xml = Path(path_to_model_xml).read_text()
    
    model = mujoco.MjModel.from_xml_string(xml)
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.make_data(model)
    
    return mjx_model , mjx_data
    

# This class holds the neural network velocity field residuals of KNODE
class Residual_Network(eqx.Module):
    out_scale: jax.Array
    mlp: eqx.nn.MLP
    
    def __init__(self , input_num , output_num , width , depth , key):
        self.out_scale = jnp.array(1.0)
        self.mlp = eqx.nn.MLP(
            in_size = input_num , 
            out_size = output_num , 
            width_size = width , 
            depth = depth , 
            activation = jax.nn.relu , 
            final_activation = jax.nn.tanh , 
            key = key)
        
    def __call__(self , y):
        return self.out_scale * self.mlp(y)
    

# This class holds the combined state dynamics
class KNODE_Dynamics(eqx.Module):
    residual_network: Residual_Network
    mjx_mixing_matrix: jax.Array
    res_mixing_matrix: jax.Array
    mjx_model: mjx.Model = eqx.field(static = True)
    
    def __init__(self , mjx_model , network_args , key):
        self.mjx_model = mjx_model
        self.residual_network = Residual_Network(network_args)
        self.mjx_mixing_matrix = jnp.eye(network_args[1]) + jr.uniform(key , (network_args[1] , network_args[1]) , minval=-.1 , maxval=.1)
        self.res_mixing_matrix = jr.uniform(key , (network_args[1] , network_args[1]) , minval=-.1 , maxval=.1)
        
    def __call__(self , t , state , args):
        qpos = state["qpos"]
        qvel = state["qvel"]
        
        mjx_data , ctrl = args
        mjx_data = mjx_data.replace(qpos = qpos , qvel = qvel , ctrl = ctrl)
        mjx_data = mjx.forward(self.mjx_model , mjx_data)
        mjx_state_vel = jnp.concatenate((qvel , mjx_data.qacc))
        
        res_state_vel = self.residual_network(jnp.concatenate((qpos , qvel , ctrl)))
        
        final_state_vel = mjx_state_vel @ self.mjx_mixing_matrix + res_state_vel @ self.res_mixing_matrix
        final_qpos_vel , final_qvel_vel = jnp.split(final_state_vel , 2)
        
        return {"qpos": final_qpos_vel , "qvel": final_qvel_vel}
        
        
class KNODE(eqx.Module):
    knode_dynamics: KNODE_Dynamics
    
    def __init__(self , mjx_model , width , depth , key):
        key1 , key2 = jr.split(key , 2)
        
        output_num = 2*jnp.size(mjx_model.nq)
        input_num = output_num + jnp.size(mjx_model.n_ctrl)
        network_args = (input_num , output_num , width , depth , key1)
        
        self.knode_dynamics = KNODE_Dynamics(mjx_model , network_args , key2)
        
    def __call__(self , ts , mjx_data , init_state , ctrl):
        mjx_data = mjx_data.replace(qpos = init_state["qpos"] , qvel = init_state["qvel"] , ctrl = ctrl)
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.knode_dynamics) ,
            diffrax.Tsit5() ,
            t0 = ts[0] ,
            t1 = ts[-1] ,
            y0 = init_state ,
            args = (mjx_data , ctrl) ,
            stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6) ,
            saveat = diffrax.SaveAt(ts=ts)
            )
        return solution.ys
        
    
    










































    