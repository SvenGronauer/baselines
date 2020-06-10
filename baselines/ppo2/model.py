import tensorflow as tf
from baselines.common.policies import PolicyWithValue

MPI = None


class Model(tf.Module):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    def __init__(self,
                 *,
                 env,
                 policy_network,
                 value_network,
                 ent_coef,
                 vf_coef,
                 max_grad_norm,
                 network_kwargs):
        super(Model, self).__init__(name='PPO2Model')
        ac_space = env.action_space
        self.train_model = PolicyWithValue(ac_space,
                                           policy_network,
                                           value_network,
                                           estimate_q=False,
                                           norm_apply=network_kwargs['norm_apply'],
                                           norm_type=network_kwargs['norm_type'],
                                           norm_coefficient=network_kwargs['norm_coefficient'])
        # if MPI is not None:
        #   self.optimizer = MpiAdamOptimizer(MPI.COMM_WORLD, self.train_model.trainable_variables)
        # else:
        self.optimizer = tf.keras.optimizers.Adam()
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.step = self.train_model.step
        self.value = self.train_model.value
        self.initial_state = self.train_model.initial_state
        self.loss_names = ['Loss/Pi', 'Loss/Value', 'Entropy', 'ApproxKL', 'ClipFrac']
        # if MPI is not None:
        #   sync_from_root(self.variables)

        self.set_env_params(*env.ob_rms.get_values())

    def train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpac_old, states=None):
        grads, pg_loss, vf_loss, entropy, approxkl, clipfrac = self.get_grad(
            cliprange, obs, returns, masks, actions, values, neglogpac_old)
        # if MPI is not None:
        #     self.optimizer.apply_gradients(grads, lr)
        # else:
        self.optimizer.learning_rate = lr
        grads_and_vars = zip(grads, self.train_model.trainable_variables)
        self.optimizer.apply_gradients(grads_and_vars)

        return pg_loss, vf_loss, entropy, approxkl, clipfrac

    @tf.function
    def get_grad(self, cliprange, obs, returns, masks, actions, values, neglogpac_old):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - tf.reduce_mean(advs)) / (tf.keras.backend.std(advs) + 1e-8)

        with tf.GradientTape() as tape:
            policy_latent = self.train_model.policy_network(obs, training=True)  # called by train() method
            pd, _ = self.train_model.pdtype.pdfromlatent(policy_latent)
            neglogpac = pd.neglogp(actions)
            entropy = tf.reduce_mean(pd.entropy())
            vpred = self.train_model.value(obs, training=True)  # called by train() method
            vpredclipped = values + tf.clip_by_value(vpred - values, -cliprange, cliprange)
            vf_losses1 = tf.square(vpred - returns)
            vf_losses2 = tf.square(vpredclipped - returns)
            vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

            ratio = tf.exp(neglogpac_old - neglogpac)
            pg_losses1 = -advs * ratio
            pg_losses2 = -advs * tf.clip_by_value(ratio, 1-cliprange, 1+cliprange)
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses1, pg_losses2))

            approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - neglogpac_old))
            clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0), cliprange), tf.float32))
            loss = pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef

        var_list = self.train_model.trainable_variables
        grads = tape.gradient(loss, var_list)
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        # if MPI is not None:
        #     grads = tf.concat([tf.reshape(g, (-1,)) for g in grads], axis=0)
        return grads, pg_loss, vf_loss, entropy, approxkl, clipfrac

    def get_env_params(self):
        return self.env_obs_mean.numpy(), self.env_obs_var.numpy(), self.env_obs_count.numpy()

    def set_env_params(self, mean, var, count):
        self.env_obs_mean = tf.Variable(mean, name='env_obs_mean')
        self.env_obs_var = tf.Variable(var, name='env_obs_var')
        self.env_obs_count = tf.Variable(count, name='env_obs_count')