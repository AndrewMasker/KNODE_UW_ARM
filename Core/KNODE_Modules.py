import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from mujoco import mjx
import Utils.MJX_Diff as mjx_diff

# This class holds the neural network velocity field residuals of KNODE
class Residual_Network(eqx.Module):
#    out_scale: jax.Array
#    mlp: eqx.nn.MLP
#    fe_sin_matrix: jax.Array# = eqx.field(static = True)
#    fe_cos_matrix: jax.Array# = eqx.field(static = True)
    input_norm_add:     jax.Array# = eqx.field(static = True)
    input_norm_mult:    jax.Array# = eqx.field(static = True)
    output_denorm_add:  jax.Array# = eqx.field(static = True)
    output_denorm_mult: jax.Array# = eqx.field(static = True)
    body_num: int = eqx.field(static = True)
#    vectors: jax.Array
    body_networks: list

    
    def __init__(self , body_num , input_num , output_num , width , depth , qpos_ranges , qvel_max , ctrl_ranges , * ,  key):
        keys = jr.split(key , body_num)
#        self.out_scale = jnp.array(10.0) #500
        self.body_networks = []
        for i in range(body_num):
            self.body_networks.append(eqx.nn.MLP(
                in_size = 10, 
                out_size = 6 , #output_num, 
                width_size = width , 
                depth = depth , 
                activation = jax.nn.gelu ,
                key = keys[i]))
                
        qpos_add = -qpos_ranges[: , 0]
        qvel_add =  qvel_max
        ctrl_add = -ctrl_ranges[: , 0]
        qpos_mult = 2/(qpos_ranges[: , 1] - qpos_ranges[: , 0])
        qvel_mult = 1/(qvel_max)
        ctrl_mult = 2/(ctrl_ranges[: , 1] - ctrl_ranges[: , 0])
#        

        self.input_norm_add  = jnp.concatenate((qpos_add , qvel_add , ctrl_add) , axis=0)
        self.input_norm_mult = jnp.concatenate((qpos_mult , qvel_mult , ctrl_mult) , axis=0)
        self.output_denorm_add = -qvel_add
        self.output_denorm_mult = 1/qvel_mult
        
        self.body_num = body_num

        
#        self.vectors = (jr.uniform(key1 , shape = (body_num , 6))-.5)*2*10
        
    def __call__(self , y):
        normalized_input = y #self.input_norm_mult*(y + self.input_norm_add) - 1
        
        vectors = self.body_networks[0](y[0])
        vectors = jnp.expand_dims(vectors , 0)
        for i in range(1 , self.body_num):
            vectors = jnp.concatenate((vectors , jnp.expand_dims(self.body_networks[i](y[i]) , 0)) , 0)
#       
        return vectors#self.vectors#denormalized_output
    

# This class holds the combined state dynamics
class KNODE_Dynamics(eqx.Module):
    residual_network: Residual_Network
    mjx_mixing_matrix: jax.Array
    res_mixing_matrix: jax.Array
    mjx_model: mjx.Model# = eqx.field(static = True)
    
    def __init__(self , mjx_model , network_args , key):
        keys = jr.split(key , 3)
        self.mjx_model = mjx_model
        self.residual_network = Residual_Network(*network_args , key=keys[0])
        self.res_mixing_matrix = jr.uniform(keys[1] , (network_args[1] , network_args[1]) , minval=-1.0 , maxval=1.0)
        self.mjx_mixing_matrix = jnp.eye(network_args[1]) + jr.uniform(keys[2] , (network_args[1] , network_args[1]) , minval=-.1 , maxval=.1)
        
        
    def __call__(self , t , state , args):
        qpos = state["qpos"]
        qvel = state["qvel"]
        
        mjx_data , ctrl , with_learned = args
        mjx_data = mjx_data.replace(qpos = qpos , qvel = qvel , ctrl = ctrl)
        
        mjx_data = mjx_diff.fwd_position(self.mjx_model , mjx_data)
        mjx_data = mjx_diff.fwd_velocity(self.mjx_model , mjx_data)
        
        xquat = mjx_data.xquat
        cvel = mjx_data.cvel
        bf_w = cvel[:,:3]
        bf_v = cvel[:,3:]
        
        rot_mat = mjx_data.xmat
#        print(rot_mat.shape)
#        print(bf_w.shape)
#        quit()
        
        bf_to_world = lambda mat , x: mat @ x
        
        bf_to_world_mapped = jax.vmap(bf_to_world , (0 , 0))
        
        world_w = bf_to_world_mapped(rot_mat , bf_w)
        world_v = bf_to_world_mapped(rot_mat , bf_v)
        
        network_inputs = jnp.concatenate((xquat , world_w , world_v) , 1)
        
        vectors = self.residual_network(network_inputs)*with_learned
        mjx_data = mjx_data.replace(xfrc_applied=vectors)
            

            
        mjx_data = mjx_diff.forward(self.mjx_model , mjx_data)# , with_learned*self.residual_network(jnp.concatenate((qpos , qvel , ctrl))))
        mjx_qacc = mjx_data.qacc

        final_qacc = mjx_qacc# + res_state_vel @ self.res_mixing_matrix
    
    
        return {"qpos": qvel , "qvel": final_qacc}
        
        
class KNODE(eqx.Module):
    knode_dynamics: KNODE_Dynamics
    
    def __init__(self , key , mjx_model , width , depth , qvel_max ):
        
        output_num = mjx_model.nq
        input_num = mjx_model.nu#2*output_num + mjx_model.nu
        
        qpos_ranges = mjx_model.jnt_range
        ctrl_ranges = mjx_model.actuator_ctrlrange
        
        network_args = (mjx_model.nbody , input_num , output_num , width , depth , qpos_ranges , qvel_max , ctrl_ranges)
        
        self.knode_dynamics = KNODE_Dynamics(mjx_model , network_args , key)
        
        
    def __call__(self , ts , mjx_data , init_state , ctrl , with_learned=True):
        mjx_data = mjx_data.replace(qpos = init_state["qpos"] , qvel = init_state["qvel"] , ctrl = ctrl)
        
#        
#        qpos = init_state["qpos"]
#        qvel = init_state["qvel"]
#        
#        #mjx_data , ctrl , with_learned = args
#        mjx_data = mjx_data.replace(qpos = qpos , qvel = qvel , ctrl = ctrl)
#        #mjx_data = mjx.forward(self.mjx_model , mjx_data)
#        #qacc = mjx_data.qacc
#
#        mjx_data = mjx.kinematics(self.knode_dynamics.mjx_model , mjx_data)
#        mjx_data = mjx.com_pos(self.knode_dynamics.mjx_model , mjx_data)
#        mjx_data = mjx.com_vel(self.knode_dynamics.mjx_model, mjx_data)
#        mjx_data = mjx.crb(self.knode_dynamics.mjx_model , mjx_data)
#        mjx_data = mjx.factor_m(self.knode_dynamics.mjx_model , mjx_data)
#        mjx_data = mjx.rne(self.knode_dynamics.mjx_model , mjx_data)
#        mjx_data = mjx.passive(self.knode_dynamics.mjx_model , mjx_data)
#        mjx_data = mjx.fwd_actuation(self.knode_dynamics.mjx_model , mjx_data)
#        #mjx_data = mjx.collision(self.mjx_model , mjx_data)
#        #mjx_data = mjx.make_constraint(self.mjx_model , mjx_data)
#        mjx_data = mjx.fwd_acceleration(self.knode_dynamics.mjx_model , mjx_data)
#        mjx_qacc = mjx_data.qacc_smooth
#        
#        print(mjx_qacc)
#        poop()
        
        
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.knode_dynamics) ,
            diffrax.Euler() ,
            t0 = ts[0] ,
            t1 = ts[-1] ,
            dt0 = ts[1] - ts[0] , 
            y0 = init_state ,
            args = (mjx_data , ctrl , with_learned) ,
#            stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-3) ,
            saveat = diffrax.SaveAt(ts=ts)
            )
        return solution.ys
        
    
    










































    
