import numpy as np
import tensorflow as tf
from baselines.a2c.utils import ortho_init, conv

mapping = {}

def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk


def nature_cnn(input_shape, **conv_kwargs):
    """
    CNN from Nature paper.
    """
    print('input shape is {}'.format(input_shape))
    x_input = tf.keras.Input(shape=input_shape, dtype=tf.uint8)
    h = x_input
    h = tf.cast(h, tf.float32) / 255.
    h = conv('c1', nf=32, rf=8, stride=4, activation='relu', init_scale=np.sqrt(2))(h)
    h2 = conv('c2', nf=64, rf=4, stride=2, activation='relu', init_scale=np.sqrt(2))(h)
    h3 = conv('c3', nf=64, rf=3, stride=1, activation='relu', init_scale=np.sqrt(2))(h2)
    h3 = tf.keras.layers.Flatten()(h3)
    h3 = tf.keras.layers.Dense(units=512, kernel_initializer=ortho_init(np.sqrt(2)),
                               name='fc1', activation='relu')(h3)
    network = tf.keras.Model(inputs=[x_input], outputs=[h3])
    return network

@register("mlp")
def mlp(num_layers=2,
        num_hidden=64,
        activation=tf.tanh,
        dropout=False,
        dropout_rate=0.5,
        norm_apply=True,
        norm_type='L2',
        norm_coefficient=1.0e-4,
        **kwargs):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)

    num_hidden: int                 size of fully-connected layers (default: 64)

    activation:                     activation function (default: tf.tanh)

    Returns:
    -------

    function that builds fully connected network with a given input tensor / placeholder
    """
    # def old_network_fn(input_shape):
    #     print('input shape is {}'.format(input_shape))
    #     x_input = tf.keras.Input(shape=input_shape)
    #     # h = tf.keras.layers.Flatten(x_input)
    #     h = x_input
    #     for i in range(num_layers):
    #       h = tf.keras.layers.Dense(units=num_hidden, kernel_initializer=ortho_init(np.sqrt(2)),
    #                                 name='mlp_fc{}'.format(i), activation=activation)(h)
    #
    #     network = tf.keras.Model(inputs=[x_input], outputs=[h])
    #     return network
    #
    # return old_network_fn

    def network_fn(input_shape):

        class MyModel(tf.keras.Model):
            def __init__(self, in_shape):
                super(MyModel, self).__init__()
                self.num_layers = num_layers
                self.in_shape = in_shape
                self.norm_apply = norm_apply
                self._layers = self.build_layers()
                self.dropout = tf.keras.layers.Dropout(dropout_rate) if dropout else None

                self.init_weights_biases()
                self.out_shape = (None, num_hidden)

            def build_layers(self):
                _layers = []
                for i in range(self.num_layers):
                    if self.norm_apply:
                        assert norm_type == 'L1' or norm_type == 'L2', 'Choose L1 or L2 norm.'
                        assert norm_coefficient > 0.0, 'Norm co-efficient must be greater than zero.'
                        kernel_regularizer = tf.keras.regularizers.l1(norm_coefficient) \
                            if norm_type == 'L1' else tf.keras.regularizers.l2(norm_coefficient)
                        bias_regularizer = tf.keras.regularizers.l1(norm_coefficient) \
                            if norm_type == 'L1' else tf.keras.regularizers.l2(norm_coefficient)
                    else:
                        kernel_regularizer = None
                        bias_regularizer = None
                    d = tf.keras.layers.Dense(units=num_hidden,
                                              kernel_initializer=ortho_init(np.sqrt(2)),
                                              kernel_regularizer=kernel_regularizer,
                                              bias_regularizer=bias_regularizer,
                                              name=f'mlp_fc{i}', activation=activation)
                    _layers.append(d)
                return _layers

            def call(self, inputs, training):
                x = inputs
                for lay in self._layers:
                    x = lay(x)
                    if self.dropout is not None:
                        x = self.dropout(x, training=training)
                return x

            def init_weights_biases(self):
                """ perform forward-pass to init weights and biases"""
                fake_pass_shape = (1,) + self.in_shape
                self.call(tf.ones(fake_pass_shape), training=False)

        net = MyModel(input_shape)
        return net
    
    return network_fn


@register("cnn")
def cnn(**conv_kwargs):
    def network_fn(input_shape):
        return nature_cnn(input_shape, **conv_kwargs)
    return network_fn


@register("conv_only")
def conv_only(convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], **conv_kwargs):
    '''
    convolutions-only net
    Parameters:
    ----------
    conv:       list of triples (filter_number, filter_size, stride) specifying parameters for each layer.
    Returns:
    function that takes tensorflow tensor as input and returns the output of the last convolutional layer
    '''

    def network_fn(input_shape):
        print('input shape is {}'.format(input_shape))
        x_input = tf.keras.Input(shape=input_shape, dtype=tf.uint8)
        h = x_input
        h = tf.cast(h, tf.float32) / 255.
        with tf.name_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                h = tf.keras.layers.Conv2D(
                    filters=num_outputs, kernel_size=kernel_size, strides=stride,
                    activation='relu', **conv_kwargs)(h)

        network = tf.keras.Model(inputs=[x_input], outputs=[h])
        return network
    return network_fn


def get_network_builder(name):
    """
    If you want to register your own network outside models.py, you just need:

    Usage Example:
    -------------
    from baselines.common.models import register
    @register("your_network_name")
    def your_network_define(**net_kwargs):
        ...
        return network_fn

    """
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError('Unknown network type: {}'.format(name))
