import functools
from typing import Optional, Sequence

import jax
from jax import numpy as jp
import mujoco
from mujoco.mjx._src import collision_driver
from mujoco.mjx._src import constraint
from mujoco.mjx._src import derivative
from mujoco.mjx._src import math
from mujoco.mjx._src import passive
from mujoco.mjx._src import scan
from mujoco.mjx._src import sensor
from mujoco.mjx._src import smooth
from mujoco.mjx._src import solver
from mujoco.mjx._src import support
# pylint: disable=g-importing-member
from mujoco.mjx._src.types import BiasType
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import DataJAX
from mujoco.mjx._src.types import DisableBit
from mujoco.mjx._src.types import DynType
from mujoco.mjx._src.types import GainType
from mujoco.mjx._src.types import Impl
from mujoco.mjx._src.types import IntegratorType
from mujoco.mjx._src.types import JointType
from mujoco.mjx._src.types import Model
from mujoco.mjx._src.types import ModelJAX
from mujoco.mjx._src.types import TrnType
# pylint: enable=g-importing-member
import mujoco.mjx.warp as mjxw
import numpy as np


def named_scope(fn, name: str = ''):
  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    with jax.named_scope(name or getattr(fn, '__name__')):
      res = fn(*args, **kwargs)
    return res

  return wrapper


@named_scope
def fwd_position(m: Model, d: Data) -> Data:
  """Position-dependent computations."""
  # TODO(robotics-simulation): tendon
  d = smooth.kinematics(m, d)
  d = smooth.com_pos(m, d)
#  d = smooth.camlight(m, d)
#  d = smooth.tendon(m, d)
  d = smooth.crb(m, d)
#  d = smooth.tendon_armature(m, d)
  d = smooth.factor_m(m, d)
#  d = collision_driver.collision(m, d)
#  d = constraint.make_constraint(m, d)
  d = smooth.transmission(m, d)
  return d


@named_scope
def fwd_velocity(m: Model, d: Data) -> Data:
  """Velocity-dependent computations."""
#  if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
#    raise ValueError('fwd_velocity requires JAX backend implementation.')

  d = d.tree_replace({
      '_impl.actuator_velocity': d._impl.actuator_moment @ d.qvel,
      '_impl.ten_velocity': d._impl.ten_J @ d.qvel,
  })
  d = smooth.com_vel(m, d)
  d = passive.passive(m, d)
  d = smooth.rne(m, d)
#  d = smooth.tendon_bias(m, d)
  return d



@named_scope
def fwd_actuation(m: Model, d: Data) -> Data:
  """Actuation-dependent computations."""
  if not isinstance(d._impl, DataJAX):
    raise ValueError('fwd_actuation requires JAX backend implementation.')
  if not m.nu or m.opt.disableflags & DisableBit.ACTUATION:
    return d.replace(
        act_dot=jp.zeros((m.na,)),
        qfrc_actuator=jp.zeros((m.nv,)),
    )

  ctrl = d.ctrl
  if not m.opt.disableflags & DisableBit.CLAMPCTRL:
    ctrlrange = jp.where(
        m.actuator_ctrllimited[:, None],
        m.actuator_ctrlrange,
        jp.array([-jp.inf, jp.inf]),
    )
    ctrl = jp.clip(ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

  # act_dot for stateful actuators
  def get_act_dot(dyn_typ, dyn_prm, ctrl, act):
    if dyn_typ == DynType.NONE:
      act_dot = jp.array(0.0)
    elif dyn_typ == DynType.INTEGRATOR:
      act_dot = ctrl
    elif dyn_typ in (DynType.FILTER, DynType.FILTEREXACT):
      act_dot = (ctrl - act) / jp.clip(dyn_prm[0], mujoco.mjMINVAL)
    elif dyn_typ == DynType.MUSCLE:
      act_dot = support.muscle_dynamics(ctrl, act, dyn_prm)
    else:
      raise NotImplementedError(f'dyntype {dyn_typ.name} not implemented.')
    return act_dot

  act_dot = jp.zeros((m.na,))
  if m.na:
    act_dot = scan.flat(
        m,
        get_act_dot,
        'uuua',
        'a',
        m.actuator_dyntype,
        m.actuator_dynprm,
        ctrl,
        d.act,
        group_by='u',
    )

  ctrl_act = ctrl
  if m.na:
    act_last_dim = d.act[m.actuator_actadr + m.actuator_actnum - 1]
    ctrl_act = jp.where(m.actuator_actadr == -1, ctrl, act_last_dim)

  def get_force(*args):
    gain_t, gain_p, bias_t, bias_p, len_, vel, ctrl_act, len_range, acc0 = args

    typ, prm = GainType(gain_t), gain_p
    if typ == GainType.FIXED:
      gain = prm[0]
    elif typ == GainType.AFFINE:
      gain = prm[0] + prm[1] * len_ + prm[2] * vel
    elif typ == GainType.MUSCLE:
      gain = support.muscle_gain(len_, vel, len_range, acc0, prm)
    else:
      raise RuntimeError(f'unrecognized gaintype {typ.name}.')

    typ, prm = BiasType(bias_t), bias_p
    bias = jp.array(0.0)
    if typ == BiasType.AFFINE:
      bias = prm[0] + prm[1] * len_ + prm[2] * vel
    elif typ == BiasType.MUSCLE:
      bias = support.muscle_bias(len_, len_range, acc0, prm)

    return gain * ctrl_act + bias

  force = scan.flat(
      m,
      get_force,
      'uuuuuuuuu',
      'u',
      m.actuator_gaintype,
      m.actuator_gainprm,
      m.actuator_biastype,
      m.actuator_biasprm,
      d.actuator_length,
      d._impl.actuator_velocity,
      ctrl_act,
      jp.array(m.actuator_lengthrange),
      jp.array(m.actuator_acc0),
      group_by='u',
  )

#  # tendon total force clamping
#  if np.any(m.tendon_actfrclimited):
#    (tendon_actfrclimited_id,) = np.nonzero(m.tendon_actfrclimited)
#    actuator_tendon = m.actuator_trntype == TrnType.TENDON
#
#    force_mask = [
#        actuator_tendon & (m.actuator_trnid[:, 0] == tendon_id)
#        for tendon_id in tendon_actfrclimited_id
#    ]
#    force_ids = np.concatenate([np.nonzero(mask)[0] for mask in force_mask])
#    force_mat = np.array(force_mask)[:, force_ids]
#    tendon_total_force = force_mat @ force[force_ids]
#
#    force_scaling = jp.where(
#        tendon_total_force < m.tendon_actfrcrange[tendon_actfrclimited_id, 0],
#        m.tendon_actfrcrange[tendon_actfrclimited_id, 0] / tendon_total_force,
#        1,
#    )
#    force_scaling = jp.where(
#        tendon_total_force > m.tendon_actfrcrange[tendon_actfrclimited_id, 1],
#        m.tendon_actfrcrange[tendon_actfrclimited_id, 1] / tendon_total_force,
#        force_scaling,
#    )
#
#    tendon_forces = force[force_ids] * (force_mat.T @ force_scaling)
#    force = force.at[force_ids].set(tendon_forces)
#
  forcerange = jp.where(
      m.actuator_forcelimited[:, None],
      m.actuator_forcerange,
      jp.array([-jp.inf, jp.inf]),
  )
  force = jp.clip(force, forcerange[:, 0], forcerange[:, 1])

  qfrc_actuator = d._impl.actuator_moment.T @ force

  if m.ngravcomp:
    # actuator-level gravity compensation, skip if added as passive force
    qfrc_actuator += d.qfrc_gravcomp * m.jnt_actgravcomp[m.dof_jntid]

  # clamp qfrc_actuator
  actfrcrange = jp.where(
      m.jnt_actfrclimited[:, None],
      m.jnt_actfrcrange,
      jp.array([-jp.inf, jp.inf]),
  )
  actfrcrange = actfrcrange[m.dof_jntid]
  qfrc_actuator = jp.clip(qfrc_actuator, actfrcrange[:, 0], actfrcrange[:, 1])

  d = d.replace(
      act_dot=act_dot, qfrc_actuator=qfrc_actuator, actuator_force=force
  )
  return d



@named_scope
def fwd_acceleration(m: Model, d: Data , residual) -> Data:
  """Add up all non-constraint forces, compute qacc_smooth."""
  qfrc_applied = d.qfrc_applied + support.xfrc_accumulate(m, d)
  qfrc_smooth = d.qfrc_passive - d.qfrc_bias + d.qfrc_actuator + qfrc_applied + residual
  qacc_smooth = smooth.solve_m(m, d, qfrc_smooth)
  d = d.replace(qfrc_smooth=qfrc_smooth, qacc_smooth=qacc_smooth)
  return d




@named_scope
def forward(m: Model, d: Data , residual) -> Data:
  """Forward dynamics."""
#  if m.impl == Impl.WARP and d.impl == Impl.WARP and mjxw.WARP_INSTALLED:
#    from mujoco.mjx.warp import forward as mjxw_forward  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
#    return mjxw_forward.forward(m, d)

#  if not isinstance(m._impl, ModelJAX) or not isinstance(d._impl, DataJAX):
#    raise ValueError('forward requires JAX backend implementation.')

  d = fwd_position(m, d)
#  d = sensor.sensor_pos(m, d)
  d = fwd_velocity(m, d)
#  d = sensor.sensor_vel(m, d)
  d = fwd_actuation(m, d)
  d = fwd_acceleration(m, d , residual)

#  if d._impl.efc_J.size == 0:
#    d = d.replace(qacc=d.qacc_smooth)
#    return d

#  d = named_scope(solver.solve)(m, d)
  d = d.replace(qacc=d.qacc_smooth)
#  d = sensor.sensor_acc(m, d)

  return d
  
  
  
  
  
