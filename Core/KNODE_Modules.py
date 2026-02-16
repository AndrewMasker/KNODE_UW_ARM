import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from mujoco import mjx

# This class holds the neural network velocity field residuals of KNODE
class Residual_Network(eqx.Module):
    out_scale: jax.Array
    mlp: eqx.nn.MLP
#    fe_sin_matrix: jax.Array# = eqx.field(static = True)
#    fe_cos_matrix: jax.Array# = eqx.field(static = True)
    input_norm_add:     jax.Array# = eqx.field(static = True)
    input_norm_mult:    jax.Array# = eqx.field(static = True)
    output_denorm_add:  jax.Array# = eqx.field(static = True)
    output_denorm_mult: jax.Array# = eqx.field(static = True)

    
    def __init__(self , input_num , output_num , width , depth , fe_std , num_fe_features , qpos_ranges , qvel_max , ctrl_ranges , key):
        key1 , key2 , key3 = jr.split(key , 3)
        self.out_scale = jnp.array(1)
        self.mlp = eqx.nn.MLP(
            in_size = input_num, 
            out_size = output_num, 
            width_size = width , 
            depth = depth , 
            activation = jax.nn.relu ,
            key = key1)
        qpos_add = -qpos_ranges[: , 0]
        qvel_add =  qvel_max
        ctrl_add = -ctrl_ranges[: , 0]
        qpos_mult = 2/(qpos_ranges[: , 1] - qpos_ranges[: , 0])
        qvel_mult = 1/(qvel_max)
        ctrl_mult = 2/(ctrl_ranges[: , 1] - ctrl_ranges[: , 0])
        
        self.input_norm_add  = jnp.concatenate((qpos_add , qvel_add , ctrl_add) , axis=0)
        self.input_norm_mult = jnp.concatenate((qpos_mult , qvel_mult , ctrl_mult) , axis=0)
        self.output_denorm_add = -jnp.concatenate((qpos_add , qvel_add) , axis=0)
        self.output_denorm_mult = 1/jnp.concatenate((qpos_mult , qvel_mult) , axis=0)
        
#        self.fe_sin_matrix = jr.uniform(key2 , shape = (input_num , num_fe_features))*fe_std
#        self.fe_cos_matrix = jr.uniform(key3 , shape = (input_num , num_fe_features))*fe_std
        
    def __call__(self , y):
        normalized_input = self.input_norm_mult*(y + self.input_norm_add) - 1
#        fe_sin_input = jnp.sin(normalized_input @ self.fe_sin_matrix)
#        fe_cos_input = jnp.cos(normalized_input @ self.fe_cos_matrix)
#        fe_input = jnp.concatenate((fe_sin_input , fe_cos_input) , 0)
        mlp_output = self.mlp(normalized_input)
        limited_mlp_output = self.out_scale * jax.nn.tanh(mlp_output/self.out_scale)
        denormalized_output = (limited_mlp_output + 1)*self.output_denorm_mult + self.output_denorm_add
        return denormalized_output
    

# This class holds the combined state dynamics
class KNODE_Dynamics(eqx.Module):
    residual_network: Residual_Network
    mjx_mixing_matrix: jax.Array
    res_mixing_matrix: jax.Array
    mjx_model: mjx.Model# = eqx.field(static = True)
    
    def __init__(self , mjx_model , network_args , key):
        self.mjx_model = mjx_model
        self.residual_network = Residual_Network(*network_args)
        self.mjx_mixing_matrix = jnp.eye(network_args[1]) + jr.uniform(key , (network_args[1] , network_args[1]) , minval=-.1 , maxval=.1)
        self.res_mixing_matrix = jr.uniform(key , (network_args[1] , network_args[1]) , minval=-.1 , maxval=.1)
        
    def __call__(self , t , state , args):
        qpos = state["qpos"]
        qvel = state["qvel"]
        
        mjx_data , ctrl , with_learned = args
        mjx_data = mjx_data.replace(qpos = qpos , qvel = qvel , ctrl = ctrl)
#        mjx_data = mjx.forward(self.mjx_model , mjx_data)
#        qacc = mjx_data.qacc

        mjx_data = mjx.kinematics(self.mjx_model , mjx_data)
        mjx_data = mjx.com_pos(self.mjx_model , mjx_data)
        mjx_data = mjx.crb(self.mjx_model , mjx_data)
        mjx_data = mjx.factor_m(self.mjx_model , mjx_data)
        mjx_data = mjx.rne(self.mjx_model , mjx_data)
        mjx_data = mjx.passive(self.mjx_model , mjx_data)
        mjx_data = mjx.fwd_actuation(self.mjx_model , mjx_data)
        #mjx_data = mjx.collision(self.mjx_model , mjx_data)
        #mjx_data = mjx.make_constraint(self.mjx_model , mjx_data)
        mjx_data = mjx.fwd_acceleration(self.mjx_model , mjx_data)
        qacc = mjx_data.qacc
        
        mjx_state_vel = jnp.concatenate((qvel , qacc))
      
        if with_learned:
            res_state_vel = self.residual_network(jnp.concatenate((qpos , qvel , ctrl)))
            final_state_vel = mjx_state_vel @ self.mjx_mixing_matrix + res_state_vel @ self.res_mixing_matrix
        else:
            final_state_vel = mjx_state_vel
        
        final_qpos_vel , final_qvel_vel = jnp.split(final_state_vel , 2)
        
        return {"qpos": final_qpos_vel , "qvel": final_qvel_vel}
        
        
class KNODE(eqx.Module):
    knode_dynamics: KNODE_Dynamics
    
    def __init__(self , key , mjx_model , width , depth , fe_std , num_fe_features , qvel_max ):#, mjx_data):
        key1 , key2 = jr.split(key , 2)
        
        output_num = 2*mjx_model.nq
        input_num = output_num + mjx_model.nu
        
        qpos_ranges = mjx_model.jnt_range
        ctrl_ranges = mjx_model.actuator_ctrlrange
        
        network_args = (input_num , output_num , width , depth , fe_std , num_fe_features , qpos_ranges , qvel_max , ctrl_ranges , key1)
        
        self.knode_dynamics = KNODE_Dynamics(mjx_model , network_args , key2)
        
        
    def __call__(self , ts , mjx_data , init_state , ctrl , with_learned=True):
        mjx_data = mjx_data.replace(qpos = init_state["qpos"] , qvel = init_state["qvel"] , ctrl = ctrl)
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.knode_dynamics) ,
            diffrax.Tsit5() ,
            t0 = ts[0] ,
            t1 = ts[-1] ,
            dt0 = ts[1] - ts[0] , 
            y0 = init_state ,
            args = (mjx_data , ctrl , with_learned) ,
            stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-3) ,
            saveat = diffrax.SaveAt(ts=ts)
            )
        return solution.ys
        
    
    










































    
