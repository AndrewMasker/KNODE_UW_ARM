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
    mlp: eqx.nn.MLP
#    fe_sin_matrix: jax.Array# = eqx.field(static = True)
#    fe_cos_matrix: jax.Array# = eqx.field(static = True)
    input_norm_add:     jax.Array# = eqx.field(static = True)
    input_norm_mult:    jax.Array# = eqx.field(static = True)
    output_denorm_add:  jax.Array# = eqx.field(static = True)
    output_denorm_mult: jax.Array# = eqx.field(static = True)
#    body_num: int = eqx.field(static = True)
#    vectors: jax.Array
#    body_networks: list

    
    def __init__(self , input_num , output_num , width , depth , qpos_ranges , qvel_max , ctrl_ranges , * ,  key):
#        keys = jr.split(key , body_num)
#        self.out_scale = jnp.array(10.0) #500

        self.mlp = eqx.nn.MLP(
            in_size = input_num, 
            out_size = output_num, 
            width_size = width , 
            depth = depth , 
            activation = jax.nn.gelu ,
            key = key)
            
                
        qpos_add = -qpos_ranges[: , 0]
        qvel_add =  qvel_max
        ctrl_add = -ctrl_ranges[: , 0]
        qpos_mult = 2/(qpos_ranges[: , 1] - qpos_ranges[: , 0])
        qvel_mult = 1/(qvel_max)
        ctrl_mult = 2/(ctrl_ranges[: , 1] - ctrl_ranges[: , 0])
        

        self.input_norm_add  = jnp.concatenate((jnp.zeros(4,) , qpos_add , qvel_add , ctrl_add) , axis=0)
        self.input_norm_mult = jnp.concatenate((jnp.ones(4,) , qpos_mult , qvel_mult , ctrl_mult) , axis=0)
        self.output_denorm_add = -qvel_add
        self.output_denorm_mult = 1/qvel_mult
        
#        self.body_num = body_num

        
#        self.vectors = (jr.uniform(key1 , shape = (body_num , 6))-.5)*2*10
        
    def __call__(self , y):
        y = y[3:]
#        print(y.shape)
        normalized_input = self.input_norm_mult*(y + self.input_norm_add) - 1
        mlp_output = self.mlp(normalized_input)
#        limited_mlp_output = self.out_scale * jax.nn.tanh(mlp_output/self.out_scale)
        denormalized_output = (mlp_output + 1)*self.output_denorm_mult + self.output_denorm_add
        return denormalized_output
    
def quat_dot(q, omega):
    # q = [w, x, y, z], omega = [wx, wy, wz]
    w, x, y, z = q
    ox, oy, oz = omega
    
    return 0.5 * jnp.array([
        -x*ox - y*oy - z*oz,
         w*ox + y*oz - z*oy,
         w*oy - x*oz + z*ox,
         w*oz + x*oy - y*ox
    ])

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
#        print(qpos.shape)
#        print(qvel.shape)
        
        mjx_data , ctrl , with_learned = args
        mjx_data = mjx_data.replace(qpos = qpos , qvel = qvel , ctrl = ctrl)
        
#        print(ctrl.shape)
#        
#        print(jnp.concatenate((qpos , qvel , ctrl)).shape)
#        poop()
        res_state_vel = with_learned*self.residual_network(jnp.concatenate((qpos , qvel , ctrl)))
        mjx_data = mjx_diff.forward(self.mjx_model , mjx_data)
        mjx_qacc = mjx_data.qacc

        final_qacc = mjx_qacc + res_state_vel @ self.res_mixing_matrix

        q     = qpos[3:7]
        omega = qvel[3:6]

        qpos_dot = jnp.concatenate([
            qvel[0:3],           # position dot = linear velocity
            quat_dot(q, omega),  # quaternion dot
            qvel[6:]             # remaining arm joints
        ])
    
    
        return {"qpos": qpos_dot , "qvel": final_qacc}
        
        
class ManifoldEuler(diffrax.AbstractWrappedSolver):
    solver: diffrax.AbstractSolver = diffrax.Euler()
    term_structure = diffrax.ODETerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def init(self, terms, t0, t1, y0, args):
        return self.solver.init(terms, t0, t1, y0, args)

    def func(self, terms, t0, y0, args):
        return self.solver.func(terms, t0, y0, args)

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        y1, err, dense, solver_state, result = self.solver.step(
            terms, t0, t1, y0, args, solver_state, made_jump
        )
        q = y1["qpos"][3:7]
        norm = jnp.linalg.norm(q)
        q = q / jnp.where(norm > 1e-6, norm, 1.0)
        y1 = {"qpos": y1["qpos"].at[3:7].set(q), "qvel": y1["qvel"]}
        return y1, err, dense, solver_state, result
        
        
class KNODE(eqx.Module):
    knode_dynamics: KNODE_Dynamics
    
    def __init__(self , key , mjx_model , width , depth , qvel_max ):
        
        output_num = mjx_model.nv
        input_num = mjx_model.nq - 3 + mjx_model.nv + mjx_model.nu
#        
#        print(mjx_model.nq)
#        print(mjx_model.nv)
#        print(mjx_model.nu)
#        print(input_num)
#        print(output_num)
#        poop()
        
        qpos_ranges = mjx_model.jnt_range[1:]
        ctrl_ranges = mjx_model.actuator_ctrlrange
        
        network_args = (input_num , output_num , width , depth , qpos_ranges , qvel_max , ctrl_ranges)
        
        self.knode_dynamics = KNODE_Dynamics(mjx_model , network_args , key)
        
        
    def __call__(self , ts , mjx_data , init_state , ctrl , with_learned=jnp.array(1.0)):
        mjx_data = mjx_data.replace(qpos = init_state["qpos"] , qvel = init_state["qvel"] , ctrl = ctrl)
        

        
        
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.knode_dynamics) ,
            ManifoldEuler() ,
            t0 = ts[0] ,
            t1 = ts[-1] ,
            dt0 = ts[1] - ts[0] , 
            y0 = init_state ,
            args = (mjx_data , ctrl , with_learned) ,
#            stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-3) ,
            saveat = diffrax.SaveAt(ts=ts)
            )
        return solution.ys
        
    
    










































    
