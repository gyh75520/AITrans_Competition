from abc import ABC

import numpy as np
import tensorflow as tf
from gym.spaces import Discrete

from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm
from stable_baselines.common.distributions import make_proba_dist_type
from stable_baselines.common.input import observation_input


def conv1d(input_tensor, scope, n_filters=128, filter_size=4, stride=1,
           pad='VALID', init_scale=1.0):
    """
    Creates a 1d convolutional layer for TensorFlow

    :param input_tensor: (TensorFlow Tensor) The input tensor for the convolution
    :param scope: (str) The TensorFlow variable scope
    :param n_filters: (int) The number of filters
    :param filter_size: (int) The filter size
    :param stride: (int) The stride of the convolution
    :param pad: (str) The padding type ('VALID' or 'SAME')
    :param init_scale: (int) The initialization scale
    :return: (TensorFlow Tensor) 1d convolutional layer
    """
    bshape = [1, 1, n_filters]
    bias_var_shape = [1, n_filters, 1]
    n_input = input_tensor.get_shape()[2].value
    wshape = [filter_size, n_input, n_filters]
    with tf.variable_scope(scope):
        w_init = tf.random_normal_initializer(mean=0, stddev=0.02)
        weight = tf.get_variable("w", wshape, initializer=w_init)
        bias = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0))
        bias = tf.reshape(bias, bshape)
        return bias + tf.nn.conv1d(input_tensor, weight, stride=stride, padding=pad)


def nature_cnn(scaled_images, **kwargs):
    """
    CNN from Nature paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


class BasePolicy(ABC):
    """
    The base policy object

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectivly
    :param add_action_ph: (bool) whether or not to create an action placeholder
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, scale=False,
                 obs_phs=None, add_action_ph=False):
        self.n_env = n_env
        self.n_steps = n_steps
        with tf.variable_scope("input", reuse=False):
            if obs_phs is None:
                self.obs_ph, self.processed_x = observation_input(ob_space, n_batch, scale=scale)
            else:
                self.obs_ph, self.processed_x = obs_phs
            self.masks_ph = tf.placeholder(tf.float32, [n_batch], name="masks_ph")  # mask (done t-1)
            self.states_ph = tf.placeholder(tf.float32, [self.n_env, n_lstm * 2], name="states_ph")  # states
            self.action_ph = None
            if add_action_ph:
                self.action_ph = tf.placeholder(dtype=ac_space.dtype, shape=(None,) + ac_space.shape, name="action_ph")
        self.sess = sess
        self.reuse = reuse
        self.ob_space = ob_space
        self.ac_space = ac_space

    def step(self, obs, state=None, mask=None):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp
        """
        raise NotImplementedError

    def proba_step(self, obs, state=None, mask=None):
        """
        Returns the action probability for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) the action probability
        """
        raise NotImplementedError


class ActorCriticPolicy(BasePolicy):
    """
    Policy object that implements actor critic

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, scale=False):
        super(ActorCriticPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=n_lstm,
                                                reuse=reuse, scale=scale)
        self.pdtype = make_proba_dist_type(ac_space)
        self.is_discrete = isinstance(ac_space, Discrete)
        self.policy = None
        self.proba_distribution = None
        self.value_fn = None
        self.deterministic_action = None
        self.initial_state = None

    def _setup_init(self):
        """
        sets up the distibutions, actions, and value
        """
        with tf.variable_scope("output", reuse=True):
            assert self.policy is not None and self.proba_distribution is not None and self.value_fn is not None
            self.action = self.proba_distribution.sample()
            self.deterministic_action = self.proba_distribution.mode()
            self.neglogp = self.proba_distribution.neglogp(self.action)
            self.policy_proba = self.policy
            if self.is_discrete:
                self.policy_proba = tf.nn.softmax(self.policy_proba)
            self._value = self.value_fn[:, 0]

    def step(self, obs, state=None, mask=None, deterministic=False):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp
        """
        raise NotImplementedError

    def proba_step(self, obs, state=None, mask=None):
        """
        Returns the action probability for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) the action probability
        """
        raise NotImplementedError

    def value(self, obs, state=None, mask=None):
        """
        Returns the value for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) The associated value of the action
        """
        raise NotImplementedError


class LstmPolicy(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using LSTMs.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network before the LSTM layer  (if None, default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param layer_norm: (bool) Whether or not to use layer normalizing LSTMs
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, layers=None,
                 cnn_extractor=nature_cnn, layer_norm=False, feature_extraction="cnn", **kwargs):
        super(LstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                                         scale=(feature_extraction == "cnn"))

        if layers is None:
            layers = [64, 64]

        with tf.variable_scope("model", reuse=reuse):
            if feature_extraction == "cnn":
                extracted_features = cnn_extractor(self.processed_x, **kwargs)
            else:
                activ = tf.tanh
                extracted_features = tf.layers.flatten(self.processed_x)
                for i, layer_size in enumerate(layers):
                    extracted_features = activ(linear(extracted_features, 'pi_fc' + str(i), n_hidden=layer_size,
                                                      init_scale=np.sqrt(2)))
            input_sequence = batch_to_seq(extracted_features, self.n_env, n_steps)
            masks = batch_to_seq(self.masks_ph, self.n_env, n_steps)
            rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                         layer_norm=layer_norm)
            rnn_output = seq_to_batch(rnn_output)
            value_fn = linear(rnn_output, 'vf', 1)

            self.proba_distribution, self.policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(rnn_output, rnn_output)

        self.value_fn = value_fn
        self.initial_state = np.zeros((self.n_env, n_lstm * 2), dtype=np.float32)
        self._setup_init()

    def AC_init(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse, feature_extraction):
        super(LstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                                         scale=(feature_extraction == "cnn"))

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run([self.deterministic_action, self._value, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})
        else:
            return self.sess.run([self.action, self._value, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self._value, {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})


class FeedForwardPolicy(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="cnn", **kwargs):
        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256,
                                                reuse=reuse, scale=(feature_extraction == "cnn"))
        if layers is None:
            layers = [64, 64]

        with tf.variable_scope("model", reuse=reuse):
            if feature_extraction == "cnn":
                extracted_features = cnn_extractor(self.processed_x, **kwargs)
                value_fn = linear(extracted_features, 'vf', 1)
                pi_latent = extracted_features
                vf_latent = extracted_features
            else:
                activ = tf.tanh
                processed_x = tf.layers.flatten(self.processed_x)
                pi_h = processed_x
                vf_h = processed_x
                for i, layer_size in enumerate(layers):
                    pi_h = activ(linear(pi_h, 'pi_fc' + str(i), n_hidden=layer_size, init_scale=np.sqrt(2)))
                    vf_h = activ(linear(vf_h, 'vf_fc' + str(i), n_hidden=layer_size, init_scale=np.sqrt(2)))
                value_fn = linear(vf_h, 'vf', 1)
                pi_latent = pi_h
                vf_latent = vf_h

            self.proba_distribution, self.policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self.value_fn = value_fn
        self.initial_state = None
        self._setup_init()

    def AC_init(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse, feature_extraction):
        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256,
                                                reuse=reuse, scale=(feature_extraction == "cnn"))

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self._value, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self._value, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self._value, {self.obs_ph: obs})


class CnnPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(CnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="cnn", **_kwargs)


class CnnLstmPolicy(LstmPolicy):
    """
    Policy object that implements actor critic, using LSTMs with a CNN feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs):
        super(CnnLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                                            layer_norm=False, feature_extraction="cnn", **_kwargs)


class CnnLnLstmPolicy(LstmPolicy):
    """
    Policy object that implements actor critic, using a layer normalized LSTMs with a CNN feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs):
        super(CnnLnLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                                              layer_norm=True, feature_extraction="cnn", **_kwargs)


class MlpPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(MlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="mlp", **_kwargs)


class MlpLstmPolicy(LstmPolicy):
    """
    Policy object that implements actor critic, using LSTMs with a MLP feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs):
        super(MlpLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                                            layer_norm=False, feature_extraction="mlp", **_kwargs)


class MlpLnLstmPolicy(LstmPolicy):
    """
    Policy object that implements actor critic, using a layer normalized LSTMs with a MLP feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs):
        super(MlpLnLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                                              layer_norm=True, feature_extraction="mlp", **_kwargs)


class LstmCustPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=512, reuse=False, layers=None,
                 layer_norm=False, feature_extraction="not cnn", **kwargs):
        # super(LstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
        #                                  scale=(feature_extraction == "cnn"))
        # add this function to LstmPolicy to init ActorCriticPolicy
        self.AC_init(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse, feature_extraction)

        with tf.variable_scope("model", reuse=reuse):
            print('----------------------------------------------')
            print('using LstmCustPolicy')
            print('self.processed_x', self.processed_x)
            batch_size = self.n_env * self.n_steps
            past_frame_num = 8
            activ = tf.nn.relu
            n_hiddens = 128

            # throughput, download_time, chunk_size, buffer_size, last_bit_rate, rebuf, play_time_len, end_delay, last_target_buffer

            throughput = self.processed_x[:, :past_frame_num]
            throughput = tf.reshape(throughput, [-1, past_frame_num, 1])
            throughput_conv = activ(conv1d(throughput, scope='throughput_conv1d'))

            download_time = self.processed_x[:, past_frame_num:2 * past_frame_num]
            download_time = tf.reshape(download_time, [-1, past_frame_num, 1])
            download_time_conv = activ(conv1d(download_time, scope='download_time_conv1d'))

            chunk_size = self.processed_x[:, 2 * past_frame_num:2 * past_frame_num + 4]
            chunk_size = tf.reshape(chunk_size, [-1, 4, 1])
            chunk_size_conv = activ(conv1d(chunk_size, scope='chunk_size_conv1d'))

            buffer_size = self.processed_x[:, 2 * past_frame_num + 4:2 * past_frame_num + 5]
            buffer_size_dense = activ(linear(buffer_size, scope='buffer_size_dense', n_hidden=n_hiddens))

            last_bit_rate = self.processed_x[:, 2 * past_frame_num + 5:2 * past_frame_num + 6]
            last_bit_rate_dense = activ(linear(last_bit_rate, scope='last_bit_rate_dense', n_hidden=n_hiddens))

            rebuf = self.processed_x[:, 2 * past_frame_num + 6:2 * past_frame_num + 7]
            rebuf_dense = activ(linear(rebuf, scope='rebuf_dense', n_hidden=n_hiddens))

            play_time_len = self.processed_x[:, 2 * past_frame_num + 7:2 * past_frame_num + 8]
            play_time_len_dense = activ(linear(play_time_len, scope='play_time_len_dense', n_hidden=n_hiddens))

            end_delay = self.processed_x[:, 2 * past_frame_num + 8:2 * past_frame_num + 9]
            end_delay_dense = activ(linear(end_delay, scope='end_delay_dense', n_hidden=n_hiddens))

            last_target_buffer = self.processed_x[:, 2 * past_frame_num + 9:2 * past_frame_num + 10]
            last_target_buffer_dense = activ(linear(last_target_buffer, scope='last_target_buffer_dense', n_hidden=n_hiddens))

            input = [tf.reshape(throughput_conv, [batch_size, -1]), tf.reshape(download_time_conv, [batch_size, -1]), tf.reshape(chunk_size_conv, [batch_size, -1]),
                     buffer_size_dense, last_bit_rate_dense, rebuf_dense, play_time_len_dense, end_delay_dense, last_target_buffer_dense]

            input = tf.concat(input, axis=1)
            print('input', input)
            input_sequence = batch_to_seq(input, self.n_env, n_steps)
            masks = batch_to_seq(self.masks_ph, self.n_env, n_steps)
            rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                         layer_norm=layer_norm)
            rnn_output = seq_to_batch(rnn_output)
            value_fn = linear(rnn_output, 'vf', 1)

            self.proba_distribution, self.policy, self.q_value = self.pdtype.proba_distribution_from_latent(rnn_output, rnn_output)

        self.value_fn = value_fn
        self.initial_state = np.zeros((self.n_env, n_lstm * 2), dtype=np.float32)
        self._setup_init()

class LstmCust3Policy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=512, reuse=False, layers=None,
                 layer_norm=False, feature_extraction="not cnn", **kwargs):
        # super(LstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
        #                                  scale=(feature_extraction == "cnn"))
        # add this function to LstmPolicy to init ActorCriticPolicy
        self.AC_init(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse, feature_extraction)

        with tf.variable_scope("model", reuse=reuse):
            print('----------------------------------------------')
            print('using LstmCust3Policy')
            print('self.processed_x', self.processed_x)
            batch_size = self.n_env * self.n_steps
            past_frame_num = 32
            activ = tf.nn.relu
            n_hiddens = 128

            # throughput, download_time, chunk_size, buffer_size, last_bit_rate, rebuf, play_time_len, end_delay, last_target_buffer

            throughput = self.processed_x[:, :past_frame_num]
            throughput = tf.reshape(throughput, [-1, past_frame_num, 1])
            throughput_conv = activ(conv1d(throughput, scope='throughput_conv1d'))
            print('throughput_conv', throughput_conv)

            download_time = self.processed_x[:, past_frame_num:2 * past_frame_num]
            download_time = tf.reshape(download_time, [-1, past_frame_num, 1])
            download_time_conv = activ(conv1d(download_time, scope='download_time_conv1d'))
            print('download_time_conv', download_time_conv)

            buffer_size = self.processed_x[:, 2 * past_frame_num:2 * past_frame_num + 1]
            buffer_size_dense = activ(linear(buffer_size, scope='buffer_size_dense', n_hidden=n_hiddens))

            last_bit_rate = self.processed_x[:, 2 * past_frame_num + 1:2 * past_frame_num + 2]
            last_bit_rate_dense = activ(linear(last_bit_rate, scope='last_bit_rate_dense', n_hidden=n_hiddens))

            rebuf = self.processed_x[:, 2 * past_frame_num + 2:2 * past_frame_num + 3]
            rebuf_dense = activ(linear(rebuf, scope='rebuf_dense', n_hidden=n_hiddens))

            switch_num = self.processed_x[:, 2 * past_frame_num + 3:2 * past_frame_num + 4]
            switch_num_dense = activ(linear(switch_num, scope='switch_num_dense', n_hidden=n_hiddens))

            end_delay = self.processed_x[:, 2 * past_frame_num + 4:2 * past_frame_num + 5]
            end_delay_dense = activ(linear(end_delay, scope='end_delay_dense', n_hidden=n_hiddens))

            last_target_buffer = self.processed_x[:, 2 * past_frame_num + 5:2 * past_frame_num + 6]
            last_target_buffer_dense = activ(
                linear(last_target_buffer, scope='last_target_buffer_dense', n_hidden=n_hiddens))

            input = [tf.reshape(throughput_conv, [batch_size, -1]), tf.reshape(download_time_conv, [batch_size, -1]),
                     buffer_size_dense, last_bit_rate_dense, rebuf_dense, switch_num_dense, end_delay_dense,
                     last_target_buffer_dense]

            input = tf.concat(input, axis=1)
            print('input', input)
            input_sequence = batch_to_seq(input, self.n_env, n_steps)
            masks = batch_to_seq(self.masks_ph, self.n_env, n_steps)
            rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                         layer_norm=layer_norm)
            rnn_output = seq_to_batch(rnn_output)
            value_fn = linear(rnn_output, 'vf', 1)

            self.proba_distribution, self.policy, self.q_value = self.pdtype.proba_distribution_from_latent(rnn_output, rnn_output)

        self.value_fn = value_fn
        self.initial_state = np.zeros((self.n_env, n_lstm * 2), dtype=np.float32)
        self._setup_init()

class LstmCust4Policy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=512, reuse=False, layers=None,
                 layer_norm=False, feature_extraction="not cnn", **kwargs):
        # super(LstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
        #                                  scale=(feature_extraction == "cnn"))
        # add this function to LstmPolicy to init ActorCriticPolicy
        self.AC_init(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse, feature_extraction)

        with tf.variable_scope("model", reuse=reuse):
            print('----------------------------------------------')
            print('using LstmCust4Policy')
            print('self.processed_x', self.processed_x)
            batch_size = self.n_env * self.n_steps
            past_frame_num = 16
            activ = tf.nn.relu
            n_hiddens = 128

            # throughput, download_time, chunk_size, buffer_size, last_bit_rate, rebuf, play_time_len, end_delay, last_target_buffer

            throughput = self.processed_x[:, :past_frame_num]
            throughput = tf.reshape(throughput, [-1, past_frame_num, 1])
            throughput_conv = activ(conv1d(throughput, scope='throughput_conv1d'))

            download_time = self.processed_x[:, past_frame_num:2 * past_frame_num]
            download_time = tf.reshape(download_time, [-1, past_frame_num, 1])
            download_time_conv = activ(conv1d(download_time, scope='download_time_conv1d'))

            chunk_size = self.processed_x[:, 2 * past_frame_num:2 * past_frame_num + 4]
            chunk_size = tf.reshape(chunk_size, [-1, 4, 1])
            chunk_size_conv = activ(conv1d(chunk_size, scope='chunk_size_conv1d'))

            buffer_size = self.processed_x[:, 2 * past_frame_num + 4:2 * past_frame_num + 5]
            buffer_size_dense = activ(linear(buffer_size, scope='buffer_size_dense', n_hidden=n_hiddens))

            last_bit_rate = self.processed_x[:, 2 * past_frame_num + 5:2 * past_frame_num + 9]
            last_bit_rate = tf.reshape(last_bit_rate,[-1,4,1])
            last_bit_rate_conv = activ(conv1d(last_bit_rate, scope='last_bit_rate_conv1d'))

            rebuf = self.processed_x[:, 2 * past_frame_num + 9:2 * past_frame_num + 10]
            rebuf_dense = activ(linear(rebuf, scope='rebuf_dense', n_hidden=n_hiddens))

            play_time_len = self.processed_x[:, 2 * past_frame_num + 10:2 * past_frame_num + 11]
            play_time_len_dense = activ(linear(play_time_len, scope='play_time_len_dense', n_hidden=n_hiddens))

            end_delay = self.processed_x[:, 2 * past_frame_num + 11:2 * past_frame_num + 12]
            end_delay_dense = activ(linear(end_delay, scope='end_delay_dense', n_hidden=n_hiddens))

            last_target_buffer = self.processed_x[:, 2 * past_frame_num + 12:2 * past_frame_num + 13]
            last_target_buffer_dense = activ(linear(last_target_buffer, scope='last_target_buffer_dense', n_hidden=n_hiddens))

            input = [tf.reshape(throughput_conv, [batch_size, -1]), tf.reshape(download_time_conv, [batch_size, -1]), tf.reshape(chunk_size_conv, [batch_size, -1]),
                     buffer_size_dense, tf.reshape(last_bit_rate_conv,[batch_size, -1]), rebuf_dense, end_delay_dense,play_time_len_dense,last_target_buffer_dense]

            input = tf.concat(input, axis=1)
            print('input', input)
            input_sequence = batch_to_seq(input, self.n_env, n_steps)
            masks = batch_to_seq(self.masks_ph, self.n_env, n_steps)
            rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                         layer_norm=layer_norm)
            rnn_output = seq_to_batch(rnn_output)
            value_fn = linear(rnn_output, 'vf', 1)

            self.proba_distribution, self.policy, self.q_value = self.pdtype.proba_distribution_from_latent(rnn_output, rnn_output)

        self.value_fn = value_fn
        self.initial_state = np.zeros((self.n_env, n_lstm * 2), dtype=np.float32)
        self._setup_init()


class LstmCust5Policy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=512, reuse=False, layers=None,
                 layer_norm=False, feature_extraction="not cnn", **kwargs):
        # super(LstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
        #                                  scale=(feature_extraction == "cnn"))
        # add this function to LstmPolicy to init ActorCriticPolicy
        self.AC_init(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse, feature_extraction)

        with tf.variable_scope("model", reuse=reuse):
            print('----------------------------------------------')
            print('using LstmCust5Policy add cdn_newest_id, download_id')
            print('self.processed_x', self.processed_x)
            batch_size = self.n_env * self.n_steps
            past_frame_num = 16
            activ = tf.nn.relu
            n_hiddens = 128

            # throughput, download_time, chunk_size, buffer_size, last_bit_rate, rebuf, play_time_len, end_delay, last_target_buffer

            throughput = self.processed_x[:, :past_frame_num]
            throughput = tf.reshape(throughput, [-1, past_frame_num, 1])
            throughput_conv = activ(conv1d(throughput, scope='throughput_conv1d'))

            download_time = self.processed_x[:, past_frame_num:2 * past_frame_num]
            download_time = tf.reshape(download_time, [-1, past_frame_num, 1])
            download_time_conv = activ(conv1d(download_time, scope='download_time_conv1d'))

            chunk_size = self.processed_x[:, 2 * past_frame_num:2 * past_frame_num + 4]
            chunk_size = tf.reshape(chunk_size, [-1, 4, 1])
            chunk_size_conv = activ(conv1d(chunk_size, scope='chunk_size_conv1d'))

            buffer_size = self.processed_x[:, 2 * past_frame_num + 4:2 * past_frame_num + 5]
            buffer_size_dense = activ(linear(buffer_size, scope='buffer_size_dense', n_hidden=n_hiddens))

            last_bit_rate = self.processed_x[:, 2 * past_frame_num + 5:2 * past_frame_num + 9]
            last_bit_rate = tf.reshape(last_bit_rate,[-1,4,1])
            last_bit_rate_conv = activ(conv1d(last_bit_rate, scope='last_bit_rate_conv1d'))

            rebuf = self.processed_x[:, 2 * past_frame_num + 9:2 * past_frame_num + 10]
            rebuf_dense = activ(linear(rebuf, scope='rebuf_dense', n_hidden=n_hiddens))

            play_time_len = self.processed_x[:, 2 * past_frame_num + 10:2 * past_frame_num + 11]
            play_time_len_dense = activ(linear(play_time_len, scope='play_time_len_dense', n_hidden=n_hiddens))

            end_delay = self.processed_x[:, 2 * past_frame_num + 11:2 * past_frame_num + 12]
            end_delay_dense = activ(linear(end_delay, scope='end_delay_dense', n_hidden=n_hiddens))

            last_target_buffer = self.processed_x[:, 2 * past_frame_num + 12:2 * past_frame_num + 13]
            last_target_buffer_dense = activ(linear(last_target_buffer, scope='last_target_buffer_dense', n_hidden=n_hiddens))

            cdn_newest_id = self.processed_x[:, 2 * past_frame_num + 13:2 * past_frame_num + 14]
            cdn_newest_id_dense = activ(linear(cdn_newest_id, scope='cdn_newest_id_dense', n_hidden=n_hiddens))

            download_id = self.processed_x[:, 2 * past_frame_num + 14:2 * past_frame_num + 15]
            download_id_dense = activ(linear(download_id, scope='download_id_dense', n_hidden=n_hiddens))

            input = [tf.reshape(throughput_conv, [batch_size, -1]), tf.reshape(download_time_conv, [batch_size, -1]), tf.reshape(chunk_size_conv, [batch_size, -1]),
                     buffer_size_dense, tf.reshape(last_bit_rate_conv,[batch_size, -1]), rebuf_dense, end_delay_dense, play_time_len_dense, last_target_buffer_dense]

            input = tf.concat(input, axis=1)
            print('input', input)
            input_sequence = batch_to_seq(input, self.n_env, n_steps)
            masks = batch_to_seq(self.masks_ph, self.n_env, n_steps)
            rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                         layer_norm=layer_norm)
            rnn_output = seq_to_batch(rnn_output)
            value_fn = linear(rnn_output, 'vf', 1)

            self.proba_distribution, self.policy, self.q_value = self.pdtype.proba_distribution_from_latent(rnn_output, rnn_output)

        self.value_fn = value_fn
        self.initial_state = np.zeros((self.n_env, n_lstm * 2), dtype=np.float32)
        self._setup_init()

class LstmTestPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=512, reuse=False, layers=None,
                 layer_norm=False, feature_extraction="not cnn", **kwargs):
        # super(LstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
        #                                  scale=(feature_extraction == "cnn"))
        # add this function to LstmPolicy to init ActorCriticPolicy
        self.AC_init(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse, feature_extraction)

        with tf.variable_scope("model", reuse=reuse):
            print('----------------------------------------------')
            print('using LstmTestPolicy')
            print('self.processed_x', self.processed_x)
            batch_size = self.n_env * self.n_steps
            past_frame_num = 16
            activ = tf.nn.relu
            n_hiddens = 128

            # throughput, download_time, chunk_size, buffer_size, last_bit_rate, rebuf, play_time_len, end_delay, last_target_buffer

            throughput = self.processed_x[:, :past_frame_num]
            throughput = tf.reshape(throughput, [-1, past_frame_num, 1])
            throughput_conv = activ(conv1d(throughput, scope='throughput_conv1d'))

            download_time = self.processed_x[:, past_frame_num:2 * past_frame_num]
            download_time = tf.reshape(download_time, [-1, past_frame_num, 1])
            download_time_conv = activ(conv1d(download_time, scope='download_time_conv1d'))

            chunk_size = self.processed_x[:, 2 * past_frame_num:2 * past_frame_num + 4]
            chunk_size = tf.reshape(chunk_size, [-1, 4, 1])
            chunk_size_conv = activ(conv1d(chunk_size, scope='chunk_size_conv1d'))

            buffer_size = self.processed_x[:, 2 * past_frame_num + 4:2 * past_frame_num + 5]
            buffer_size_dense = activ(linear(buffer_size, scope='buffer_size_dense', n_hidden=n_hiddens))

            last_bit_rate = self.processed_x[:, 2 * past_frame_num + 5:2 * past_frame_num + 6]
            last_bit_rate_dense = activ(linear(last_bit_rate, scope='last_bit_rate_dense', n_hidden=n_hiddens))

            rebuf = self.processed_x[:, 2 * past_frame_num + 6:2 * past_frame_num + 7]
            rebuf_dense = activ(linear(rebuf, scope='rebuf_dense', n_hidden=n_hiddens))

            play_time_len = self.processed_x[:, 2 * past_frame_num + 7:2 * past_frame_num + 8]
            play_time_len_dense = activ(linear(play_time_len, scope='play_time_len_dense', n_hidden=n_hiddens))

            end_delay = self.processed_x[:, 2 * past_frame_num + 8:2 * past_frame_num + 9]
            end_delay_dense = activ(linear(end_delay, scope='end_delay_dense', n_hidden=n_hiddens))

            last_target_buffer = self.processed_x[:, 2 * past_frame_num + 9:2 * past_frame_num + 10]
            last_target_buffer_dense = activ(linear(last_target_buffer, scope='last_target_buffer_dense', n_hidden=n_hiddens))

            # input = [tf.reshape(throughput_conv, [batch_size, -1]), tf.reshape(download_time_conv, [batch_size, -1]), tf.reshape(chunk_size_conv, [batch_size, -1]),
            #          buffer_size_dense, last_bit_rate_dense, rebuf_dense, play_time_len_dense, end_delay_dense, last_target_buffer_dense]

            input = [tf.reshape(throughput_conv, [batch_size, -1]),buffer_size_dense, end_delay_dense]

            input = tf.concat(input, axis=1)
            print('input', input)
            input_sequence = batch_to_seq(input, self.n_env, n_steps)
            masks = batch_to_seq(self.masks_ph, self.n_env, n_steps)
            rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                         layer_norm=layer_norm)
            rnn_output = seq_to_batch(rnn_output)
            value_fn = linear(rnn_output, 'vf', 1)

            self.proba_distribution, self.policy, self.q_value = self.pdtype.proba_distribution_from_latent(rnn_output, rnn_output)

        self.value_fn = value_fn
        self.initial_state = np.zeros((self.n_env, n_lstm * 2), dtype=np.float32)
        self._setup_init()


class FeedForwardCustPolicy(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None,
                 feature_extraction="mlp", **kwargs):
        # super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256,
                            # reuse = reuse, scale = (feature_extraction == "cnn"))
        # self.AC_init(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse, feature_extraction)
        self.AC_init(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse, feature_extraction)
        if layers is None:
            layers = [128]

        with tf.variable_scope("model", reuse=reuse):
            print('----------------------------------------------')
            print('using FeedForwardCustPolicy')
            batch_size = self.n_env * self.n_steps
            past_frame_num = 8
            activ = tf.nn.relu
            n_hiddens = 128

            # throughput, download_time, chunk_size, buffer_size, last_bit_rate, rebuf, play_time_len, end_delay, last_target_buffer

            throughput = self.processed_x[:, :past_frame_num]
            throughput = tf.reshape(throughput, [-1, past_frame_num, 1])
            throughput_conv = activ(conv1d(throughput, scope='throughput_conv1d'))

            download_time = self.processed_x[:, past_frame_num:2 * past_frame_num]
            download_time = tf.reshape(download_time, [-1, past_frame_num, 1])
            download_time_conv = activ(conv1d(download_time, scope='download_time_conv1d'))

            chunk_size = self.processed_x[:, 2 * past_frame_num:2 * past_frame_num + 4]
            chunk_size = tf.reshape(chunk_size, [-1, 4, 1])
            chunk_size_conv = activ(conv1d(chunk_size, scope='chunk_size_conv1d'))

            buffer_size = self.processed_x[:, 2 * past_frame_num + 4:2 * past_frame_num + 5]
            buffer_size_dense = activ(linear(buffer_size, scope='buffer_size_dense', n_hidden=n_hiddens))

            last_bit_rate = self.processed_x[:, 2 * past_frame_num + 5:2 * past_frame_num + 6]
            last_bit_rate_dense = activ(linear(last_bit_rate, scope='last_bit_rate_dense', n_hidden=n_hiddens))

            rebuf = self.processed_x[:, 2 * past_frame_num + 6:2 * past_frame_num + 7]
            rebuf_dense = activ(linear(rebuf, scope='rebuf_dense', n_hidden=n_hiddens))

            play_time_len = self.processed_x[:, 2 * past_frame_num + 7:2 * past_frame_num + 8]
            play_time_len_dense = activ(linear(play_time_len, scope='play_time_len_dense', n_hidden=n_hiddens))

            end_delay = self.processed_x[:, 2 * past_frame_num + 8:2 * past_frame_num + 9]
            end_delay_dense = activ(linear(end_delay, scope='end_delay_dense', n_hidden=n_hiddens))

            last_target_buffer = self.processed_x[:, 2 * past_frame_num + 9:2 * past_frame_num + 10]
            last_target_buffer_dense = activ(linear(last_target_buffer, scope='last_target_buffer_dense', n_hidden=n_hiddens))

            input = [tf.reshape(throughput_conv, [batch_size, -1]), tf.reshape(download_time_conv, [batch_size, -1]), tf.reshape(chunk_size_conv, [batch_size, -1]),
                     buffer_size_dense, last_bit_rate_dense, rebuf_dense, play_time_len_dense, end_delay_dense, last_target_buffer_dense]

            input = tf.concat(input, axis=1)
            processed_x = input
            # processed_x = end_delay_dense
            print('input', processed_x)

            activ = tf.tanh
            # processed_x = tf.layers.flatten(self.processed_x)

            pi_h = processed_x
            vf_h = processed_x
            for i, layer_size in enumerate(layers):
                pi_h = activ(linear(pi_h, 'pi_fc' + str(i), n_hidden=layer_size, init_scale=np.sqrt(2)))
                vf_h = activ(linear(vf_h, 'vf_fc' + str(i), n_hidden=layer_size, init_scale=np.sqrt(2)))
            value_fn = linear(vf_h, 'vf', 1)
            pi_latent = pi_h
            vf_latent = vf_h

            self.proba_distribution, self.policy, self.q_value = self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self.value_fn = value_fn
        self.initial_state = None
        self._setup_init()


class LstmCust2Policy(LstmPolicy):
    # __module__ = None

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=128, reuse=False, layers=None,
                 layer_norm=False, feature_extraction="not cnn", **kwargs):
        # super(LstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
        #                                  scale=(feature_extraction == "cnn"))
        # add this function to LstmPolicy to init ActorCriticPolicy
        self.AC_init(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse, feature_extraction)

        with tf.variable_scope("model", reuse=reuse):
            print('----------------------------------------------')
            print('using LstmCust2Policy')
            print('self.processed_x', self.processed_x)
            batch_size = self.n_env * self.n_steps
            past_frame_num = 8
            activ = tf.nn.relu
            n_hiddens = 128

            # throughput, download_time, chunk_size, buffer_size, last_bit_rate, rebuf, play_time_len, end_delay

            throughput = self.processed_x[:, past_frame_num - 1:past_frame_num]
            throughput_dense = activ(linear(throughput, scope='throughput_dense', n_hidden=n_hiddens))

            download_time = self.processed_x[:, 2 * past_frame_num - 1:2 * past_frame_num]
            download_time_dense = activ(linear(download_time, scope='download_time_dense', n_hidden=n_hiddens))

            chunk_size = self.processed_x[:, 2 * past_frame_num:2 * past_frame_num + 4]
            chunk_size = tf.reshape(chunk_size, [-1, 4, 1])
            chunk_size_conv = activ(conv1d(chunk_size, scope='chunk_size_conv1d'))

            buffer_size = self.processed_x[:, 2 * past_frame_num + 4:2 * past_frame_num + 5]
            buffer_size_dense = activ(linear(buffer_size, scope='buffer_size_dense', n_hidden=n_hiddens))

            last_bit_rate = self.processed_x[:, 2 * past_frame_num + 5:2 * past_frame_num + 6]
            last_bit_rate_dense = activ(linear(last_bit_rate, scope='last_bit_rate_dense', n_hidden=n_hiddens))

            end_delay = self.processed_x[:, 2 * past_frame_num + 6:2 * past_frame_num + 7]
            end_delay_dense = activ(linear(end_delay, scope='end_delay_dense', n_hidden=n_hiddens))

            input = [throughput_dense, download_time_dense, tf.reshape(chunk_size_conv, [batch_size, -1]), buffer_size_dense, last_bit_rate_dense, end_delay_dense]

            input = tf.concat(input, axis=1)
            print('input', input)
            input_sequence = batch_to_seq(input, self.n_env, n_steps)
            masks = batch_to_seq(self.masks_ph, self.n_env, n_steps)
            rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                         layer_norm=layer_norm)
            rnn_output = seq_to_batch(rnn_output)
            value_fn = linear(rnn_output, 'vf', 1)

            self.proba_distribution, self.policy, self.q_value = self.pdtype.proba_distribution_from_latent(rnn_output, rnn_output)

        self.value_fn = value_fn
        self.initial_state = np.zeros((self.n_env, n_lstm * 2), dtype=np.float32)
        self._setup_init()


_policy_registry = {
    ActorCriticPolicy: {
        "CnnPolicy": CnnPolicy,
        "CnnLstmPolicy": CnnLstmPolicy,
        "CnnLnLstmPolicy": CnnLnLstmPolicy,
        "MlpPolicy": MlpPolicy,
        "MlpLstmPolicy": MlpLstmPolicy,
        "MlpLnLstmPolicy": MlpLnLstmPolicy,
    }
}


def get_policy_from_name(base_policy_type, name):
    """
    returns the registed policy from the base type and name

    :param base_policy_type: (BasePolicy) the base policy object
    :param name: (str) the policy name
    :return: (base_policy_type) the policy
    """
    if base_policy_type not in _policy_registry:
        raise ValueError("Error: the policy type {} is not registered!".format(base_policy_type))
    if name not in _policy_registry[base_policy_type]:
        raise ValueError("Error: unknown policy type {}, the only registed policy type are: {}!"
                         .format(name, list(_policy_registry[base_policy_type].keys())))
    return _policy_registry[base_policy_type][name]


def register_policy(name, policy):
    """
    returns the registed policy from the base type and name

    :param name: (str) the policy name
    :param policy: (subclass of BasePolicy) the policy
    """
    sub_class = None
    for cls in BasePolicy.__subclasses__():
        if issubclass(policy, cls):
            sub_class = cls
            break
    if sub_class is None:
        raise ValueError("Error: the policy {} is not of any known subclasses of BasePolicy!".format(policy))

    if sub_class not in _policy_registry:
        _policy_registry[sub_class] = {}
    if name in _policy_registry[sub_class]:
        raise ValueError("Error: the name {} is alreay registered for a different policy, will not override."
                         .format(name))
    _policy_registry[sub_class][name] = policy
