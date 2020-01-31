import tensorflow as tf
import functools

from baselines.common.tf_util import initialize

from baselines.common.running_mean_std import RunningMeanStd

from baselines.common.tf_util import get_session, save_variables, load_variables


# Helpers functions
def conv2d(inputs, filters, kernel_size, strides, activation):
	return tf.layers.conv2d(inputs = inputs,
								filters = filters,
								kernel_size = kernel_size,
								strides = strides,
								activation = activation)

def fc(inputs, units, activation):
	return tf.layers.dense(inputs = inputs,
								units = units,
								activation = activation)

class RND():	
	def __init__(self, ob_space, proportion_of_exp_used_for_predictor_update):
		self.sess = sess = get_session()

		self.proportion_of_exp_used_for_predictor_update = proportion_of_exp_used_for_predictor_update

		# CREATE THE PLACEHOLDERS
		# Remember that we pass only one frame, not a stack of frame hence ([None, 84, 84, 1]) 
		self.NEXT_STATE = tf.placeholder(tf.float32, [None,] + list(ob_space.shape[:2])+[1], name="NEXT_STATE")

		# These two are for the observation normalization (mean and std)
		self.RND_OBS_MEAN = tf.placeholder(tf.float32, list(ob_space.shape[:2])+[1])
		self.RND_OBS_STD = tf.placeholder(tf.float32, list(ob_space.shape[:2])+[1])

		# Build our RunningMeanStd object for observation normalization
		self.rnd_ob_rms = RunningMeanStd(shape=list(ob_space.shape[:2])+[1])

		# Build our RunningMeanStd for intrinsic reward normalization (mandatory since IR are non-stationary rewards)
		self.rnd_ir_rms = RunningMeanStd()

		# These two are for the observation normalization (mean and std)
		self.pred_next_feature_ = tf.placeholder(tf.float32, [None, 512])
		self.target_next_feature_ = tf.placeholder(tf.float32, [None, 512])


		# Here we create the two networks of our RND:
		# - Predictor model (trained)
		# - Target model (untrained)

		with tf.variable_scope('rnd_ppo_model'):
			# Clip and normalize the obs
			self.normalized_ob =  tf.clip_by_value((self.NEXT_STATE - self.RND_OBS_MEAN) / self.RND_OBS_STD, -5.0, 5.0)
			self.predictor_conv1 = conv2d(self.normalized_ob, 32, 8, 4, tf.nn.leaky_relu)
			self.predictor_conv2 = conv2d(self.predictor_conv1, 64, 4, 2, tf.nn.leaky_relu)
			self.predictor_conv3 = conv2d(self.predictor_conv2, 64, 3, 1, tf.nn.leaky_relu)

			self.predictor_flattened = tf.layers.flatten(self.predictor_conv3)

			self.predictor_fc1 = fc(self.predictor_flattened, 512, tf.nn.leaky_relu)

			self.predictor_fc2 = fc(self.predictor_fc1, 512, tf.nn.leaky_relu)

			self.pred_next_feature = fc(self.predictor_fc2, 512, None)

		with tf.variable_scope('rnd_target'):
			#self.normalized_ob =  tf.clip_by_value((self.NEXT_STATE - self.RND_OBS_MEAN) / self.RND_OBS_STD, -5.0, 5.0)
			self.target_conv1 = conv2d(self.normalized_ob, 32, 8, 4, tf.nn.leaky_relu)
			self.target_conv2 = conv2d(self.target_conv1, 64, 4, 2, tf.nn.leaky_relu)
			self.target_conv3 = conv2d(self.target_conv2, 64, 3, 1, tf.nn.leaky_relu)

			self.target_flattened = tf.layers.flatten(self.target_conv3)

			self.target_next_feature = fc(self.target_flattened, 512, None)

			#self.intrinsic_rewards = tf.reduce_mean(tf.square(self.target_next_feature - self.pred_next_feature))
		
		# CALCULATE THE LOSS
		# The loss is just the negative of IR
		self.rnd_loss = -tf.reduce_mean(tf.square(self.target_next_feature - self.pred_next_feature))
		
		# Here we train the predictor network with only a proportion of experience defined in proportion_of_exp_used_for_predictor_update
		mask = tf.random_uniform(shape=tf.shape(self.rnd_loss), minval=0., maxval=1., dtype=tf.float32)
		mask = tf.cast(mask < self.proportion_of_exp_used_for_predictor_update, tf.float32)
		self.rnd_loss = tf.reduce_sum(mask * self.rnd_loss) / tf.maximum(tf.reduce_sum(mask), 1.)


	def calculate_intrinsic_rewards(self, next_states):
		# First we predict the pred_next_state and target_next_state for each observations of the batch
		pred_nstate, target_nstate = self.sess.run([self.pred_next_feature, self.target_next_feature], {
			self.NEXT_STATE: next_states,
			self.RND_OBS_MEAN: self.rnd_ob_rms.mean,
			# Transform var into standard deviation
			self.RND_OBS_STD: self.rnd_ob_rms.var ** 0.5
		})
	
		#Then, we calculate the intrinsic reward for each.
		intrinsic_rewards = []
		for e in range(len(pred_nstate)):
			IR = 0
			IR = ((target_nstate[e] - pred_nstate[e])**2).mean() 
			intrinsic_rewards.append(IR)
		return intrinsic_rewards
	