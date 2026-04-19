import diffrax
import equinox as eqx
import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
from mujoco import mjx
import Utils.MJX_Diff as mjx_diff
    
    
def rearrange_Sample(gp_state , nq):
    gp_state_repeated = jnp.repeat(jnp.expand_dims(gp_state , 0) , repeats = nq , axis = 0)
    gp_input = jnp.concatenate((gp_state_repeated , jnp.expand_dims(jnp.arange(nq,dtype=jnp.float64) , 1)) , axis = 1)
    return gp_input

# This class holds the combined state dynamics
class GP_Dynamics(eqx.Module):
    posterior: gpx.variational_families.ExpectationVariationalGaussian
#    train_data: gpx.Dataset = eqx.field(static=True)
    mjx_model: mjx.Model
    inputs_mean: jax.Array
    inputs_std: jax.Array
    outputs_mean: jax.Array
    outputs_std: jax.Array
    output_num: int = eqx.field(static=True)
    
    def __init__(self , mjx_model , posterior , gp_args , inputs_mean , inputs_std , outputs_mean , outputs_std , key):
        keys = jr.split(key , 3)
        self.mjx_model = mjx_model
        self.posterior = posterior
#        self.train_data = train_data
        self.output_num = gp_args[2]
        
        self.inputs_mean = inputs_mean
        self.inputs_std = inputs_std
        self.outputs_mean = outputs_mean 
        self.outputs_std = outputs_std

        
    def __call__(self , t , state , args):
        qpos = state["qpos"]
        qvel = state["qvel"]
        
        mjx_data , ctrl , with_learned = args
        mjx_data = mjx_data.replace(qpos = qpos , qvel = qvel , ctrl = ctrl)
            
        mjx_data = mjx_diff.forward(self.mjx_model , mjx_data)
        mjx_qacc = mjx_data.qacc
        
        gp_state = jnp.concatenate((qpos , qvel , ctrl) , axis = 0)
        gp_input = rearrange_Sample(gp_state , self.output_num)
        gp_input = (gp_input - self.inputs_mean)/self.inputs_std
        
        gp_latent = self.posterior.predict(gp_input.astype(jnp.float64))
        
        gp_residual = gp_latent.mean * self.outputs_std.squeeze() + self.outputs_mean

        final_qacc = mjx_qacc + gp_residual.astype(jnp.float32)*with_learned
    
    
        return {"qpos": qvel , "qvel": final_qacc}
        
        
class GP(eqx.Module):
    gp_dynamics: GP_Dynamics
    
    def __init__(self , key , mjx_model , posterior, inputs_mean , inputs_std , outputs_mean , outputs_std):
        
        output_num = mjx_model.nq
        input_num = 2*output_num + mjx_model.nu
       
        
        gp_args = (mjx_model.nbody , input_num , output_num )# , qpos_ranges , qvel_max , ctrl_ranges)
        
        self.gp_dynamics = GP_Dynamics(mjx_model , posterior , gp_args , inputs_mean , inputs_std , outputs_mean , outputs_std , key)
        
        
    def __call__(self , ts , mjx_data , init_state , ctrl , with_learned=jnp.array(1.0)):
        mjx_data = mjx_data.replace(qpos = init_state["qpos"] , qvel = init_state["qvel"] , ctrl = ctrl)
        

        
        
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.gp_dynamics) ,
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
        
    
    










































    
