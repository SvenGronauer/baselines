import numpy as np
import tensorflow as tf


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init


def conv(scope, *, nf, rf, stride, activation, pad='valid', init_scale=1.0, data_format='channels_last'):
    with tf.name_scope(scope):
        layer = tf.keras.layers.Conv2D(filters=nf, kernel_size=rf, strides=stride, padding=pad,
                                       data_format=data_format, kernel_initializer=ortho_init(init_scale))
    return layer


def fc(input_shape,
       scope,
       nh,
       *,
       init_scale=1.0,
       init_bias=0.0,
       norm_apply=False,
       norm_type='L2',
       norm_coefficient=1.0e-4):
    """Note: No dropout is applied since this is the last affine layer of network.

    """

    if norm_apply:
        assert norm_type == 'L1' or norm_type == 'L2', 'Choose L1 or L2 norm.'
        assert norm_coefficient > 0.0, 'Norm co-efficient must be greater than zero.'
        kernel_regularizer = tf.keras.regularizers.l1(norm_coefficient) \
            if norm_type == 'L1' else tf.keras.regularizers.l2(norm_coefficient)
        bias_regularizer = tf.keras.regularizers.l1(norm_coefficient) \
            if norm_type == 'L1' else tf.keras.regularizers.l2(norm_coefficient)
    else:
        kernel_regularizer = None
        bias_regularizer = None

    with tf.name_scope(scope):
        layer = tf.keras.layers.Dense(units=nh, kernel_initializer=ortho_init(init_scale),
                                      bias_initializer=tf.keras.initializers.Constant(init_bias),
                                      kernel_regularizer=kernel_regularizer,
                                      bias_regularizer=bias_regularizer)
        layer.build(input_shape)
    return layer


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done) # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]


class InverseLinearTimeDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, nupdates, name="InverseLinearTimeDecay"):
        super(InverseLinearTimeDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.nupdates = nupdates
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name):
            initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            step_t = tf.cast(step, dtype)
            nupdates_t = tf.convert_to_tensor(self.nupdates, dtype=dtype)
            tf.assert_less(step_t, nupdates_t)
            return initial_learning_rate * (1. - step_t / nupdates_t)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "nupdates": self.nupdates,
            "name": self.name
        }
