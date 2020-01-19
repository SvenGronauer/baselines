import tensorflow as tf
from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype

import gym


class PolicyWithValue(tf.Module):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, ac_space, policy_network, value_network=None, estimate_q=False):
        """
        Parameters:
        ----------
        ac_space        action space

        policy_network  keras network for policy

        value_network   keras network for value

        estimate_q      q value or v value

        """

        self.policy_network = policy_network
        self.value_network = value_network or policy_network
        self.estimate_q = estimate_q
        self.initial_state = None

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(policy_network.out_shape, ac_space, init_scale=0.01)
        # self.pdtype = make_pdtype(policy_network.output_shape, ac_space, init_scale=0.01)

        if estimate_q:
            assert isinstance(ac_space, gym.spaces.Discrete)
            self.value_fc = fc(self.value_network.out_shape, 'q', ac_space.n)
            # self.value_fc = fc(self.value_network.output_shape, 'q', ac_space.n)
        else:
            self.value_fc = fc(self.value_network.out_shape, 'vf', 1)
            # self.value_fc = fc(self.value_network.output_shape, 'vf', 1)

    @tf.function
    def step(self, observation, training):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation: np.ndarray
            batched observation data
        training:   bool
            Sample from action distribution if True, else take the mode of prob. dist.

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """
        latent = self.policy_network(observation, training=training)  # applies e.g. dropout in train mode
        pd, pi = self.pdtype.pdfromlatent(latent)
        action = pd.sample() if training else pd.mode()  # do not sample during evaluation
        neglogp = pd.neglogp(action)
        value_latent = self.value_network(observation, training=training)
        vf = tf.squeeze(self.value_fc(value_latent), axis=1)
        return action, vf, None, neglogp

    @tf.function
    def value(self, observation, training):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        Returns:
        -------
        value estimate
        """
        value_latent = self.value_network(observation, training=training)
        result = tf.squeeze(self.value_fc(value_latent), axis=1)
        return result

