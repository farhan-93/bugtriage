import numpy as np
import tensorflow as tf

class TextCNN(object):
	def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
		# Placeholders for input, output and dropout
		self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
		self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
		self.learning_rate = tf.placeholder(tf.float32)


		# Keeping track of l2 regularization loss (optional)
		l2_loss = tf.constant(0.0)


		with tf.device('/cpu:0'), tf.name_scope('embedding'):
			self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name='W')

			self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
			self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)



		pooled_outputs = []
		for i, filter_size in enumerate(filter_sizes):
			with tf.name_scope('conv-maxpool-%s' % filter_size):
				# Convolution Layer
				filter_shape = [filter_size, embedding_size, 1, num_filters]
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
				b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
				conv = tf.nn.conv2d(
					self.embedded_chars_expanded,
					W,
					strides=[1, 1, 1, 1],
					padding='VALID',
					name='conv')
				# Apply nonlinearity
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
				# Maxpooling over the outputs
				pooled = tf.nn.max_pool(
					h,
					ksize=[1, sequence_length - filter_size + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name='pool')
				pooled_outputs.append(pooled)


		num_filters_total = num_filters * len(filter_sizes)
		self.h_pool = tf.concat(pooled_outputs,3)
		self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])


		with tf.name_scope('dropout'):
			self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)


		with tf.name_scope('output'):
			W = tf.get_variable(
				'W',
				shape=[num_filters_total, num_classes],
				initializer=tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
			l2_loss += tf.nn.l2_loss(W)
			l2_loss += tf.nn.l2_loss(b)
			self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
			self.predictions = tf.argmax(self.scores, 1, name='predictions')
						
		# CalculateMean cross-entropy loss
		with tf.name_scope('loss'):
			losses = tf.nn.softmax_cross_entropy_with_logits(labels = self.input_y, logits = self.scores)            
			self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

		# Accuracy
		with tf.name_scope('accuracy'):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

		with tf.name_scope('num_correct'):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.num_correct = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct')
			#top_2 Accuracy
			top_2_correct_predications=tf.nn.in_top_k(self.scores,tf.argmax(self.input_y, 1),2,name="top_2_correct_predications")
			self.k_2_accuracy = tf.reduce_mean(tf.cast(top_2_correct_predications, 'float'), name='accuracy_2')
			self.num_correct_2 = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct_2')
			self.k_2_num_correct = tf.reduce_sum(tf.cast(top_2_correct_predications, 'float'), name='k_2_num_correct')

			#top_3 Accuracy
			top_3_correct_predications=tf.nn.in_top_k(self.scores,tf.argmax(self.input_y, 1),3,name="top_3_correct_predications")
			self.k_3_accuracy = tf.reduce_mean(tf.cast(top_3_correct_predications, 'float'), name='accuracy_3')
			self.num_correct_3 = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct_3')
			self.k_3_num_correct = tf.reduce_sum(tf.cast(top_3_correct_predications, 'float'), name='k_3_num_correct')

			#top_4 Accuracy
			top_4_correct_predications=tf.nn.in_top_k(self.scores,tf.argmax(self.input_y, 1),4,name="top_4_correct_predications")
			self.k_4_accuracy = tf.reduce_mean(tf.cast(top_4_correct_predications, 'float'), name='accuracy_4')
			self.num_correct_4 = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct_4')
			self.k_4_num_correct = tf.reduce_sum(tf.cast(top_4_correct_predications, 'float'), name='k_4_num_correct')

			#top_5 Accuracy
			top_5_correct_predications=tf.nn.in_top_k(self.scores,tf.argmax(self.input_y, 1),5,name="top_5_correct_predications")
			self.k_5_accuracy = tf.reduce_mean(tf.cast(top_5_correct_predications, 'float'), name='accuracy_5')
			self.num_correct_5 = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct_5')
			self.k_5_num_correct = tf.reduce_sum(tf.cast(top_5_correct_predications, 'float'), name='k_5_num_correct')

			#top_6 Accuracy
			top_6_correct_predications=tf.nn.in_top_k(self.scores,tf.argmax(self.input_y, 1),6,name="top_6_correct_predications")
			self.k_6_accuracy = tf.reduce_mean(tf.cast(top_6_correct_predications, 'float'), name='accuracy_6')
			self.num_correct_6 = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct_6')
			self.k_6_num_correct = tf.reduce_sum(tf.cast(top_6_correct_predications, 'float'), name='k_6_num_correct')

			#top_3 Accuracy
			top_7_correct_predications=tf.nn.in_top_k(self.scores,tf.argmax(self.input_y, 1),7,name="top_7_correct_predications")
			self.k_7_accuracy = tf.reduce_mean(tf.cast(top_7_correct_predications, 'float'), name='accuracy_7')
			self.num_correct_7 = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct_7')
			self.k_7_num_correct = tf.reduce_sum(tf.cast(top_7_correct_predications, 'float'), name='k_7_num_correct')

			#top_8 Accuracy
			top_8_correct_predications=tf.nn.in_top_k(self.scores,tf.argmax(self.input_y, 1),8,name="top_8_correct_predications")
			self.k_8_accuracy = tf.reduce_mean(tf.cast(top_8_correct_predications, 'float'), name='accuracy_8')
			self.num_correct_8 = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct_8')
			self.k_8_num_correct = tf.reduce_sum(tf.cast(top_8_correct_predications, 'float'), name='k_8_num_correct')

			#top_9 Accuracy
			top_9_correct_predications=tf.nn.in_top_k(self.scores,tf.argmax(self.input_y, 1),9,name="top_9_correct_predications")
			self.k_9_accuracy = tf.reduce_mean(tf.cast(top_9_correct_predications, 'float'), name='accuracy_9')
			self.num_correct_9 = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct_9')
			self.k_9_num_correct = tf.reduce_sum(tf.cast(top_9_correct_predications, 'float'), name='k_9_num_correct')

			#top_10 Accuracy
			top_10_correct_predications=tf.nn.in_top_k(self.scores,tf.argmax(self.input_y, 1),10,name="top_10_correct_predications")
			self.k_10_accuracy = tf.reduce_mean(tf.cast(top_10_correct_predications, 'float'), name='accuracy_10')
			self.num_correct_10 = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct_10')
			self.k_10_num_correct = tf.reduce_sum(tf.cast(top_10_correct_predications, 'float'), name='k_10_num_correct')



		
