
"""Offline RL with Fisher Divergence Critic Regularization (BC-KL)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin
import tensorflow as tf
import agent
import numpy as np
from scripts import networks
from scripts import policies
from scripts import utils


CLIP_EPS = 1e-3  # Epsilon for clipping actions.
ALPHA_MAX = 500.0


@gin.configurable
class Agent(agent.Agent):
  """fish_brc agent class."""

  def __init__(
      self,
      alpha=0.1,
      alpha_entropy=0.0,
      train_alpha_entropy=False,
      target_entropy=None,
      behavior_ckpt_file=None,
      ensemble_q_lambda=1.0,
      n_action_samples=1,
      reward_bonus=5.0,
      **kwargs):
    self._alpha = alpha
    self._train_alpha_entropy = train_alpha_entropy
    self._alpha_entropy = alpha_entropy
    self._target_entropy = target_entropy
    self._behavior_ckpt_file = behavior_ckpt_file
    self._ensemble_q_lambda = ensemble_q_lambda
    self._n_action_samples = n_action_samples
    self._reward_bonus = reward_bonus
    super(Agent, self).__init__(**kwargs)

  def _build_fns(self):
    self._agent_module = AgentModule(modules=self._modules)
    self._q_fns = self._agent_module.q_nets
    self._p_fn = self._agent_module.p_net
    self._b_fn = self._agent_module.b_net
    # entropy regularization
    if self._target_entropy is None:
      self._target_entropy = - self._a_dim
    self._get_alpha_entropy = self._agent_module.get_alpha_entropy
    self._agent_module.assign_alpha_entropy(self._alpha_entropy)

  def _get_q_vars(self):
    return self._agent_module.q_source_variables

  def _get_p_vars(self):
    return self._agent_module.p_variables

  def _get_b_vars(self):
    return self._agent_module.b_variables

  def _get_q_weight_norm(self):
    weights = self._agent_module.q_source_weights
    norms = []
    for w in weights:
      norm = tf.reduce_sum(tf.square(w))
      norms.append(norm)
    return tf.add_n(norms)

  def _get_p_weight_norm(self):
    weights = self._agent_module.p_weights
    norms = []
    for w in weights:
      norm = tf.reduce_sum(tf.square(w))
      norms.append(norm)
    return tf.add_n(norms)

  def _get_b_weight_norm(self):
    weights = self._agent_module.b_weights
    norms = []
    for w in weights:
      norm = tf.reduce_sum(tf.square(w))
      norms.append(norm)
    return tf.add_n(norms)

  def ensemble_q(self, qs):
    lambda_ = self._ensemble_q_lambda
    return (lambda_ * tf.reduce_min(qs, axis=-1)
            + (1 - lambda_) * tf.reduce_max(qs, axis=-1))

  def _ensemble_q2_target(self, q2_targets):
    return self.ensemble_q(q2_targets)

  def _ensemble_q1(self, q1s):
    return self.ensemble_q(q1s)

  def _residual_q(self, q_func, states, actions):
    q = q_func(states, actions)
    log_prob = self._b_fn.get_log_density(
      states, utils.clip_by_eps(actions, -self._a_max, self._a_max, CLIP_EPS))
    return q + log_prob

  def _build_q_loss(self, batch):
    s1 = batch.s1
    s2 = batch.s2
    a1 = batch.a1
    r = batch.reward + self._reward_bonus
    dsc = batch.discount

    _, a1_p, log_pi_a1_p = self._p_fn(s1)
    # duplicate state n times.
    s2 = tf.tile(s2, [self._n_action_samples, 1])
    _, a2_p, log_pi_a2_p = self._p_fn(s2)

    # compute q target
    q2_targets = []
    q1_preds = []
    for q_fn, q_fn_target in self._q_fns:
      q2_target_ = self._residual_q(q_fn_target, s2, a2_p)
      q1_pred = self._residual_q(q_fn, s1, a1)
      q1_preds.append(q1_pred)
      q2_targets.append(q2_target_)
    q2_targets = tf.stack(q2_targets, axis=-1)
    q2_target = self._ensemble_q2_target(q2_targets)
    q2_target = tf.reshape(q2_target, [self._n_action_samples, -1])
    q2_target = tf.reduce_max(q2_target, 0)  # maximum in Q
    v2_target = q2_target - self._get_alpha_entropy() * log_pi_a2_p
    q1_target = tf.stop_gradient(r + dsc * self._discount * v2_target)

    # q's gradient to action
    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape2:
      tape2.watch([a1_p])

      q1_reg = self._q_fns[0][0](s1, a1_p)
      q2_reg = self._q_fns[0][1](s1, a1_p)

    q1_grads = tape2.gradient(q1_reg, a1_p)
    q2_grads = tape2.gradient(q2_reg, a1_p)

    q1_grad_norm = tf.reduce_sum(tf.square(q1_grads), axis=-1)
    q2_grad_norm = tf.reduce_sum(tf.square(q2_grads), axis=-1)

    del tape2

    q_reg = tf.reduce_mean(q1_grad_norm + q2_grad_norm)

    # q loss
    q_losses = []
    for q1_pred in q1_preds:
      q_loss_ = tf.reduce_mean(tf.square(q1_pred - q1_target))
      q_losses.append(q_loss_)
    q_loss = tf.add_n(q_losses)
    q_w_norm = self._get_q_weight_norm()
    norm_loss = self._weight_decays * q_w_norm
    loss = q_loss + self._alpha * q_reg + norm_loss
    # info
    info = collections.OrderedDict()
    info['q_loss'] = q_loss
    info['q_reg'] = q_reg
    info['q_norm'] = q_w_norm
    info['r_mean'] = tf.reduce_mean(r)
    info['dsc_mean'] = tf.reduce_mean(dsc)
    info['q2_target_mean'] = tf.reduce_mean(q2_target)
    info['q1_target_mean'] = tf.reduce_mean(q1_target)
    return loss, info

  def _build_p_loss(self, batch):
    # read from batch
    s = batch.s1
    _, a_p, apn_logp = self._p_fn(s)
    q1 = self._residual_q(self._q_fns[0][0], s, a_p)
    q2 = self._residual_q(self._q_fns[1][0], s, a_p)
    q = tf.minimum(q1, q2)
    p_loss = tf.reduce_mean(
      self._get_alpha_entropy() * apn_logp - q)
    p_w_norm = self._get_p_weight_norm()
    norm_loss = self._weight_decays * p_w_norm
    loss = p_loss + norm_loss
    # info
    info = collections.OrderedDict()
    # info['p_loss'] = p_loss
    # info['p_norm'] = p_w_norm
    info['H(p(s))'] = tf.reduce_mean(apn_logp)

    return loss, info

  def _build_ae_loss(self, batch):
    s = batch.s1
    _, _, log_pi_a = self._p_fn(s)
    alpha = self._get_alpha_entropy()
    ae_loss = tf.reduce_mean(alpha * (- log_pi_a - self._target_entropy))
    # info
    info = collections.OrderedDict()
    info['ae_loss'] = ae_loss
    info['alpha_entropy'] = alpha
    return ae_loss, info

  def _get_source_target_vars(self):
    return (self._agent_module.q_source_variables,
            self._agent_module.q_target_variables)

  def _build_optimizers(self):
    opts = self._optimizers
    self._q_optimizer = utils.get_optimizer(opts[0][0])(lr=opts[0][1])
    self._p_optimizer = utils.get_optimizer(opts[1][0])(lr=opts[1][1])
    self._ae_optimizer = utils.get_optimizer(opts[3][0])(lr=opts[3][1])
    # if len(self._weight_decays) == 1:
    #   self._weight_decays = tuple([self._weight_decays[0]] * 3)

  @tf.function
  def _optimize_step(self, batch):
    info = collections.OrderedDict()
    if tf.equal(self._global_step % self._update_freq, 0):
      source_vars, target_vars = self._get_source_target_vars()
      self._update_target_fns(source_vars, target_vars)
    q_info = self._optimize_q(batch)
    p_info = self._optimize_p(batch)
    if self._train_alpha_entropy:
      ae_info = self._optimize_ae(batch)
    info.update(p_info)
    info.update(q_info)
    if self._train_alpha_entropy:
      info.update(ae_info)
    return info

  def _optimize_q(self, batch):
    vars_ = self._q_vars
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(vars_)
      loss, info = self._build_q_loss(batch)
    grads = tape.gradient(loss, vars_)
    grads_and_vars = tuple(zip(grads, vars_))
    self._q_optimizer.apply_gradients(grads_and_vars)
    return info

  def _optimize_p(self, batch):
    vars_ = self._p_vars
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(vars_)
      loss, info = self._build_p_loss(batch)
    grads = tape.gradient(loss, vars_)
    grads_and_vars = tuple(zip(grads, vars_))
    self._p_optimizer.apply_gradients(grads_and_vars)
    return info

  def _optimize_ae(self, batch):
    vars_ = self._ae_vars
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(vars_)
      loss, info = self._build_ae_loss(batch)
    grads = tape.gradient(loss, vars_)
    grads_and_vars = tuple(zip(grads, vars_))
    self._ae_optimizer.apply_gradients(grads_and_vars)
    return info

  def _build_test_policies(self):
    policy = policies.DeterministicSoftPolicy(
        a_network=self._agent_module.p_net)
    self._test_policies['main'] = policy
    policy = policies.MaxQSoftPolicy(
        a_network=self._agent_module.p_net,
        q_network=self._agent_module.q_nets[0][0],
        )
    self._test_policies['max_q'] = policy

  def _build_online_policy(self):
    return policies.RandomSoftPolicy(
        a_network=self._agent_module.p_net,
        )

  def _init_vars(self, batch):
    self._build_q_loss(batch)
    self._build_p_loss(batch)
    self._q_vars = self._get_q_vars()
    self._p_vars = self._get_p_vars()
    self._b_vars = self._get_b_vars()
    self._ae_vars = self._agent_module.ae_variables

  def _build_checkpointer(self):
    state_ckpt = tf.train.Checkpoint(
        policy=self._agent_module.p_net,
        q=self._agent_module.q_nets[0][0],
        )
    behavior_ckpt = tf.train.Checkpoint(
        policy=self._agent_module.b_net)
    return dict(state=state_ckpt, behavior=behavior_ckpt)

  def save(self, ckpt_name):
    step = self._global_step.numpy()
    self._checkpointer['state'].write(ckpt_name+ '_' + str(step))

  def restore_behavior_model(self, ckpt_dir):
    self._checkpointer['behavior'].restore(ckpt_dir)


class AgentModule(agent.AgentModule):
  """Models in a brac_primal agent."""

  def _build_modules(self):
    self._q_nets = []
    n_q_fns = self._modules.n_q_fns
    for _ in range(n_q_fns):
      self._q_nets.append(
          [self._modules.q_net_factory(),
           self._modules.q_net_factory(),]  # source and target
          )
    self._p_net = self._modules.p_net_factory()
    self._b_net = self._modules.b_net_factory()
    self._alpha_entropy_var = tf.Variable(1.0)

  def get_alpha_entropy(self):
    return utils.relu_v2(self._alpha_entropy_var)

  def assign_alpha_entropy(self, alpha):
    self._alpha_entropy_var.assign(alpha)

  @property
  def ae_variables(self):
    return [self._alpha_entropy_var]

  @property
  def q_nets(self):
    return self._q_nets

  @property
  def q_source_weights(self):
    q_weights = []
    for q_net, _ in self._q_nets:
      q_weights += q_net.weights
    return q_weights

  @property
  def q_target_weights(self):
    q_weights = []
    for _, q_net in self._q_nets:
      q_weights += q_net.weights
    return q_weights

  @property
  def q_source_variables(self):
    vars_ = []
    for q_net, _ in self._q_nets:
      vars_ += q_net.trainable_variables
    return tuple(vars_)

  @property
  def q_target_variables(self):
    vars_ = []
    for _, q_net in self._q_nets:
      vars_ += q_net.trainable_variables
    return tuple(vars_)

  @property
  def p_net(self):
    return self._p_net

  @property
  def p_weights(self):
    return self._p_net.weights

  @property
  def p_variables(self):
    return self._p_net.trainable_variables

  @property
  def b_net(self):
    return self._b_net

  @property
  def b_weights(self):
    return self._b_net.weights

  @property
  def b_variables(self):
    return self._b_net.trainable_variables


def get_modules(model_params, a_max, a_dim):
  """Get agent modules."""
  model_params, n_q_fns, _ = model_params
  def q_net_factory():
    return networks.CriticNetwork(
        fc_layer_params=model_params[0])
  def p_net_factory():
    return networks.ActorNetwork(
        a_dim, a_max,
        fc_layer_params=model_params[1])
  def b_net_factory():
    return networks.ActorNetwork(
        a_dim, a_max,
        fc_layer_params=model_params[1])
  modules = utils.Flags(
      q_net_factory=q_net_factory,
      p_net_factory=p_net_factory,
      b_net_factory=b_net_factory,
      n_q_fns=n_q_fns,
      )
  return modules


class Config(agent.Config):

  def _get_modules(self):
    return get_modules(
        self._agent_flags.model_params,
        self._agent_flags.a_max,
        self._agent_flags.a_dim,
    )