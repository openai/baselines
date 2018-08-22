import tensorflow as tf

from stable_baselines.common.input import observation_input

# ================================================================
# Placeholders
# ================================================================


class TfInput(object):
    def __init__(self, name="(unnamed)"):
        """
        Generalized Tensorflow placeholder. The main differences are:
            - possibly uses multiple placeholders internally and returns multiple values
            - can apply light postprocessing to the value feed to placeholder.

        :param name: (str) the input name
        """
        self.name = name

    def get(self):
        """
        Return the tf variable(s) representing the possibly postprocessed value
        of placeholder(s).

        :return: (TensorFlow Tensor) the placeholder
        """
        raise NotImplementedError

    def make_feed_dict(self, data):
        """
        Given data input it to the placeholder(s).

        :return: (dict) the given data input
        """
        raise NotImplementedError


class PlaceholderTfInput(TfInput):
    def __init__(self, placeholder):
        """
        Wrapper for regular tensorflow placeholder.

        :param placeholder: (TensorFlow Tensor)
        """
        super().__init__(placeholder.name)
        self._placeholder = placeholder

    def get(self):
        return self._placeholder

    def make_feed_dict(self, data):
        return {self._placeholder: data}


class Uint8Input(PlaceholderTfInput):
    def __init__(self, shape, name=None):
        """
        Takes input in uint8 format which is cast to float32 and divided by 255
        before passing it to the model.

        On GPU this ensures lower data transfer times.

        :param shape: ([int]) shape of the tensor.
        :param name: (str) name of the underlying placeholder
        """

        super().__init__(tf.placeholder(tf.uint8, [None] + list(shape), name=name))
        self._shape = shape
        self._output = tf.cast(super().get(), tf.float32) / 255.0

    def get(self):
        return self._output


class ObservationInput(PlaceholderTfInput):
    def __init__(self, observation_space, name=None):
        """
        Creates an input placeholder tailored to a specific observation space

        :param observation_space: (Gym Space) observation space of the environment. Should be one of the gym.spaces
            types
        :param name: (str) tensorflow name of the underlying placeholder
        """
        is_image = len(observation_space.shape) == 3
        inpt, self.processed_inpt = observation_input(observation_space, name=name, scale=is_image)
        super().__init__(inpt)

    def get(self):
        return self.processed_inpt
