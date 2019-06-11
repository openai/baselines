Dealing with NaNs and infs
==========================

During the training of a model on a given environment, it is possible that the RL model becomes completely
corrupted when a NaN or an inf is given or returned from the RL model.

How and why?
------------

The issue arises then NaNs or infs do not crash, but simply get propagated through the training,
until all the floating point number converge to NaN or inf. This is in line with the
`IEEE Standard for Floating-Point Arithmetic (IEEE 754) <https://ieeexplore.ieee.org/document/4610935>`_ standard, as it says:

.. note::
    Five possible exceptions can occur:
        - Invalid operation (:math:`\sqrt{-1}`, :math:`\inf \times 1`, :math:`\text{NaN}\ \mathrm{mod}\ 1`, ...) return NaN
        - Division by zero:
            - if the operand is not zero (:math:`1/0`, :math:`-2/0`, ...) returns :math:`\pm\inf`
            - if the operand is zero (:math:`0/0`) returns signaling NaN
        - Overflow (exponent too high to represent) returns :math:`\pm\inf`
        - Underflow (exponent too low to represent) returns :math:`0`
        - Inexact (not representable exactly in base 2, eg: :math:`1/5`) returns the rounded value (ex: :code:`assert (1/5) * 3 == 0.6000000000000001`)

And of these, only ``Division by zero`` will signal an exception, the rest will propagate invalid values quietly.

In python, dividing by zero will indeed raise the exception: ``ZeroDivisionError: float division by zero``,
but ignores the rest.

The default in numpy, will warn: ``RuntimeWarning: invalid value encountered``
but will not halt the code.

And the worst of all, Tensorflow will not signal anything

.. code-block:: python

  import tensorflow as tf
  import numpy as np

  print("tensorflow test:")

  a = tf.constant(1.0)
  b = tf.constant(0.0)
  c = a / b

  sess = tf.Session()
  val = sess.run(c)  # this will be quiet
  print(val)
  sess.close()

  print("\r\nnumpy test:")

  a = np.float64(1.0)
  b = np.float64(0.0)
  val = a / b  # this will warn
  print(val)

  print("\r\npure python test:")

  a = 1.0
  b = 0.0
  val = a / b  # this will raise an exception and halt.
  print(val)

Unfortunately, most of the floating point operations are handled by Tensorflow and numpy,
meaning you might get little to no warning when a invalid value occurs.

Numpy parameters
----------------

Numpy has a convenient way of dealing with invalid value: `numpy.seterr <https://docs.scipy.org/doc/numpy/reference/generated/numpy.seterr.html>`_,
which defines for the python process, how it should handle floating point error.

.. code-block:: python

  import numpy as np

  np.seterr(all='raise')  # define before your code.

  print("numpy test:")

  a = np.float64(1.0)
  b = np.float64(0.0)
  val = a / b  # this will now raise an exception instead of a warning.
  print(val)

but this will also avoid overflow issues on floating point numbers:

.. code-block:: python

  import numpy as np

  np.seterr(all='raise')  # define before your code.

  print("numpy overflow test:")

  a = np.float64(10)
  b = np.float64(1000)
  val = a ** b  # this will now raise an exception
  print(val)

but will not avoid the propagation issues:

.. code-block:: python

  import numpy as np

  np.seterr(all='raise')  # define before your code.

  print("numpy propagation test:")

  a = np.float64('NaN')
  b = np.float64(1.0)
  val = a + b  # this will neither warn nor raise anything
  print(val)
  
Tensorflow parameters
---------------------

Tensorflow can add checks for detecting and dealing with invalid value: `tf.add_check_numerics_ops <https://www.tensorflow.org/api_docs/python/tf/add_check_numerics_ops>`_ and `tf.check_numerics <https://www.tensorflow.org/api_docs/python/tf/debugging/check_numerics>`_,
however they will add operations to the Tensorflow graph and raise the computation time.

.. code-block:: python

  import tensorflow as tf

  print("tensorflow test:")

  a = tf.constant(1.0)
  b = tf.constant(0.0)
  c = a / b
  
  check_nan = tf.add_check_numerics_ops()  # add after your graph definition.

  sess = tf.Session()
  val, _ = sess.run([c, check_nan])  # this will now raise an exception
  print(val)
  sess.close()

but this will also avoid overflow issues on floating point numbers:

.. code-block:: python

  import tensorflow as tf
  
  print("tensorflow overflow test:")
  
  check_nan = []  # the list of check_numerics operations

  a = tf.constant(10)
  b = tf.constant(1000)
  c = a ** b  
  
  check_nan.append(tf.check_numerics(c, ""))  # check the 'c' operations
  
  sess = tf.Session()
  val, _ = sess.run([c] + check_nan)  # this will now raise an exception
  print(val)
  sess.close()

and catch propagation issues:

.. code-block:: python

  import tensorflow as tf

  print("tensorflow propagation test:")
  
  check_nan = []  # the list of check_numerics operations

  a = tf.constant('NaN')
  b = tf.constant(1.0)
  c = a + b
  
  check_nan.append(tf.check_numerics(c, ""))  # check the 'c' operations
  
  sess = tf.Session()
  val, _ = sess.run([c] + check_nan)  # this will now raise an exception
  print(val)
  sess.close()


VecCheckNan Wrapper
-------------------

In order to find when and from where the invalid value originated from, stable-baselines comes with a ``VecCheckNan`` wrapper.

It will monitor the actions, observations, and rewards, indicating what action or observation caused it and from what.

.. code-block:: python

  import gym
  from gym import spaces
  import numpy as np

  from stable_baselines import PPO2
  from stable_baselines.common.vec_env import DummyVecEnv, VecCheckNan

  class NanAndInfEnv(gym.Env):
      """Custom Environment that raised NaNs and Infs"""
      metadata = {'render.modes': ['human']}

      def __init__(self):
          super(NanAndInfEnv, self).__init__()
          self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
          self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)

      def step(self, _action):
          randf = np.random.rand()
          if randf > 0.99:
              obs = float('NaN')
          elif randf > 0.98:
              obs = float('inf')
          else:
              obs = randf
          return [obs], 0.0, False, {}

      def reset(self):
          return [0.0]

      def render(self, mode='human', close=False):
          pass

  # Create environment
  env = DummyVecEnv([lambda: NanAndInfEnv()])
  env = VecCheckNan(env, raise_exception=True)

  # Instantiate the agent
  model = PPO2('MlpPolicy', env)

  # Train the agent
  model.learn(total_timesteps=int(2e5))  # this will crash explaining that the invalid value originated from the environment.

RL Model hyperparameters
------------------------

Depending on your hyperparameters, NaN can occurs much more often.
A great example of this: https://github.com/hill-a/stable-baselines/issues/340

Be aware, the hyperparameters given by default seem to work in most cases,
however your environment might not play nice with them.
If this is the case, try to read up on the effect each hyperparameters has on the model,
so that you can try and tune them to get a stable model. Alternatively, you can try automatic hyperparameter tuning (included in the rl zoo).

Missing values from datasets
----------------------------

If your environment is generated from an external dataset, do not forget to make sure your dataset does not contain NaNs.
As some datasets will sometimes fill missing values with NaNs as a surrogate value.

Here is some reading material about finding NaNs: https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html

And filling the missing values with something else (imputation): https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4

