import os
import sys
import json
import time
import logging
import data_helper
import numpy as np
import tensorflow as tf
from text_cnn import TextCNN
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
import yaml
import math
import datetime
from time import gmtime

logging.getLogger().setLevel(logging.INFO)

tf.flags.DEFINE_boolean("enable_word_embeddings", True, "Enable/disable the word embedding (default: True)")
tf.flags.DEFINE_float("decay_coefficient", 2.5, "Decay coefficient (default: 2.5)")

start_time="";
end_time="";



def train_cnn():
	FLAGS = tf.flags.FLAGS
	"""Step 0: load sentences, labels, and training parameters"""
	with open("config.yml", 'r') as ymlfile:
		cfg = yaml.load(ymlfile)
	if FLAGS.enable_word_embeddings and cfg['word_embeddings']['default'] is not None:
		embedding_name = cfg['word_embeddings']['default']
		embedding_dimension = cfg['word_embeddings'][embedding_name]['dimension']
	else:
		embedding_dimension = 300

	path=""
	train_file = sys.argv[1]
	x_raw, y_raw, df, labels = data_helper.load_data_and_labels(train_file)

	parameter_file = sys.argv[2]
	params = json.loads(open(parameter_file).read())

	"""Step 1: pad each sentence to the same length and map each word to an id"""
	max_document_length = max([len(x.split(' ')) for x in x_raw])
	logging.info('The maximum length of all sentences: {}'.format(max_document_length))
	vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
	x = np.array(list(vocab_processor.fit_transform(x_raw)))
	y = np.array(y_raw)
	

	"""Step 2: split the original dataset into train and test sets"""
	#x_test=x[132795:147550]
	#y_test=y[132795:147550]
	#print(np.shape(x_test))
	#x_ =x[0:132795]
	#y_ =y[0:132795]

	#x_concat=x[1040:]
	#y_concat=y[1040:]

	#x_=np.concatenate((x_,x_concat),axis=0)
	#y_=np.concatenate((y_,y_concat),axis=0)
	#x_, x_test, y_, y_test = train_test_split(x, y, test_size=0.1)
	#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)


	"""Step 3: shuffle the train set and split the train set into train and dev sets"""
	#shuffle_indices = np.random.permutation(np.arange(len(y_)))
	shuffle_indices = np.random.permutation(np.arange(len(y)))

	#x_shuffled = x_[shuffle_indices]
	x_shuffled = x[shuffle_indices]
	#y_shuffled = y_[shuffle_indices]
	y_shuffled = y[shuffle_indices]
	x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, test_size=0.1,random_state=1)

	"""Step 4: save the labels into labels.json since predict.py needs it"""
	with open('./labels.json', 'w') as outfile:
		json.dump(labels, outfile, indent=4)

	#logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
	logging.info('x_train: {}, x_dev: {}'.format(len(x_train), len(x_dev)))
	#logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))
	logging.info('y_train: {}, y_dev: {}'.format(len(y_train), len(y_dev)))

	"""Step 5: build a graph and cnn object"""
	graph = tf.Graph()
	with graph.as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			cnn = TextCNN(
				#sequence_length=x_train.shape[1],
				sequence_length=x_train.shape[1],
				num_classes=y_train.shape[1],
				vocab_size=len(vocab_processor.vocabulary_),
				embedding_size=params['embedding_dim'],
				filter_sizes=list(map(int, params['filter_sizes'].split(","))),
				num_filters=params['num_filters'],
				l2_reg_lambda=params['l2_reg_lambda'])

			global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.AdamOptimizer(cnn.learning_rate)
			grads_and_vars = optimizer.compute_gradients(cnn.loss)
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

			timestamp = str(int(time.time()))
			out_dir = os.path.abspath(os.path.join(os.path.curdir, "netbeans_trained_model_" + timestamp))

			checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
			checkpoint_prefix = os.path.join(checkpoint_dir, "model")
			if not os.path.exists(checkpoint_dir):
				os.makedirs(checkpoint_dir)
			saver = tf.train.Saver(tf.all_variables())

	# #sunro

	# 	for epochs in range(args.num_epochs): 
	# 		train_sets.shuffle_train_data()
	# 		for batch_x, batch_y in train_sets.batches():
	# 			start = time.time()
	# 			current_step = tf.train.global_step(sess, model.global_step)
	# 			feed = {model.train_data: batch_x, model.targets: batch_y,model.dropout: args.dropout}
    #             #_, summaries, global_step, loss, accuracy = sess.run(
    #             #    [model.train_op, train_summary_op, model.global_step, model.loss, model.accuracy], feed_dict=feed)
    #             #end = time.time()
    #             #train_summary_writer.add_summary(summaries, global_step)
    #             #print("{}/{} ({} epochs) step, loss : {:.6f}, accuracy : {:.3f}, time/batch : {:.3f}sec"
    #             #      .format(current_step, train_sets.num_batches * args.num_epochs,
    #             #              epochs, loss, accuracy, end - start))
	# 			_, summaries, global_step, loss, accuracy,k_2_accuracy,k_3_accuracy,k_4_accuracy,k_5_accuracy,k_6_accuracy,k_7_accuracy,k_8_accuracy,k_9_accuracy,k_10_accuracy = sess.run([model.train_op, train_summary_op, model.global_step, model.loss, model.accuracy,model.k_2_accuracy,model.k_3_accuracy,model.k_4_accuracy,model.k_5_accuracy,model.k_6_accuracy,model.k_7_accuracy,model.k_8_accuracy,model.k_9_accuracy,model.k_10_accuracy], feed_dict=feed)
	# 			end = time.time()
	# 			train_summary_writer.add_summary(summaries, global_step)
	# 			print("{}/{} ({} epochs) step, loss : {:.6f}, accuracy : {:.3f},Top-2-Accuracy{:.3f},Top-3-Accuracy{:.3f},Top-4-Accuracy{:.3f}, Top-5-Accuracy{:.3f}, Top-6-Accuracy{:.3f}, Top-7-Accuracy{:.3f}, Top-8-Accuracy{:.3f}, Top-9-Accuracy{:.3f}, Top-10-Accuracy{:.3f} time/batch : {:.3f}sec"
    #                   .format(current_step, train_sets.num_batches * args.num_epochs, epochs, loss, accuracy,k_2_accuracy,k_3_accuracy,k_4_accuracy,k_5_accuracy,k_6_accuracy,k_7_accuracy,k_8_accuracy,k_9_accuracy,k_10_accuracy, end - start))


			# One training step: train the model with one batch
			def train_step(x_batch, y_batch,learning_rate):
				feed_dict = {
					cnn.input_x: x_batch,
					cnn.input_y: y_batch,
					cnn.dropout_keep_prob: params['dropout_keep_prob'],
					cnn.learning_rate: learning_rate}
				_, step, loss, acc,k_2_accuracy,k_3_accuracy,k_4_accuracy,k_5_accuracy,k_6_accuracy,k_7_accuracy,k_8_accuracy,k_9_accuracy,k_10_accuracy = sess.run([train_op, global_step, cnn.loss, cnn.accuracy,cnn.k_2_accuracy,cnn.k_3_accuracy,cnn.k_4_accuracy,cnn.k_5_accuracy,cnn.k_6_accuracy,cnn.k_7_accuracy,cnn.k_8_accuracy,cnn.k_9_accuracy,cnn.k_10_accuracy], feed_dict)
				print("Train Step: step {}, loss {:.6f}, acc {:.3f},Top-2-Accuracy{:.3f},Top-3-Accuracy{:.3f},Top-4-Accuracy{:.3f}, Top-5-Accuracy{:.3f}, Top-6-Accuracy{:.3f}, Top-7-Accuracy{:.3f}, Top-8-Accuracy{:.3f}, Top-9-Accuracy{:.3f}, Top-10-Accuracy{:.3f}".format( step, loss, acc,k_2_accuracy,k_3_accuracy,k_4_accuracy,k_5_accuracy,k_6_accuracy,k_7_accuracy,k_8_accuracy,k_9_accuracy,k_10_accuracy))

			# One evaluation step: evaluate the model with one batch
			def dev_step(x_batch, y_batch):
				feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: 1.0}

				step, loss,  acc,k_2_accuracy,k_3_accuracy,k_4_accuracy,k_5_accuracy,k_6_accuracy,k_7_accuracy,k_8_accuracy,k_9_accuracy,k_10_accuracy, num_correct,scores,k_2_num_correct,k_3_num_correct,k_4_num_correct,k_5_num_correct,k_6_num_correct,k_7_num_correct,k_8_num_correct,k_9_num_correct,k_10_num_correct= sess.run([global_step, cnn.loss, cnn.accuracy,cnn.k_2_accuracy,cnn.k_3_accuracy,cnn.k_4_accuracy,cnn.k_5_accuracy,cnn.k_6_accuracy,cnn.k_7_accuracy,cnn.k_8_accuracy,cnn.k_9_accuracy,cnn.k_10_accuracy, cnn.num_correct,cnn.scores,cnn.k_2_num_correct,cnn.k_3_num_correct,cnn.k_4_num_correct,cnn.k_5_num_correct,cnn.k_6_num_correct,cnn.k_7_num_correct,cnn.k_8_num_correct,cnn.k_9_num_correct,cnn.k_10_num_correct], feed_dict)
				top_k_predications=tf.nn.top_k(scores,5)
				#print(num_correct)
				#print(k_num_correct)
				print("Dev Step: step {}, loss {:g}, acc {:g},Top-2-Accuracy{:g},Top-3-Accuracy{:g},Top-4-Accuracy{:g}, Top-5-Accuracy{:g}, Top-6-Accuracy{:g}, Top-7-Accuracy{:g}, Top-8-Accuracy{:g}, Top-9-Accuracy{:g}, Top-10-Accuracy{:g}".format( step, loss, acc,k_2_accuracy,k_3_accuracy,k_4_accuracy,k_5_accuracy,k_6_accuracy,k_7_accuracy,k_8_accuracy,k_9_accuracy,k_10_accuracy))
				return num_correct,k_2_num_correct,k_3_num_correct,k_4_num_correct,k_5_num_correct,k_6_num_correct,k_7_num_correct,k_8_num_correct,k_9_num_correct,k_10_num_correct

			# Save the word_to_id map since predict.py needs it
			vocab_processor.save(os.path.join(out_dir, "vocab"))
			sess.run(tf.global_variables_initializer())
			# GLoVE Embedding

			if FLAGS.enable_word_embeddings and cfg['word_embeddings']['default'] is not None:
				vocabulary = vocab_processor.vocabulary_
				initW = None
				if embedding_name == 'word2vec':
					print("Load word2vec file {}".format(cfg['word_embeddings']['word2vec']['path']))
					initW = data_helper.load_embedding_vectors_word2vec(vocabulary,cfg['word_embeddings']['word2vec']['path'],cfg['word_embeddings']['word2vec']['binary'])
					print("word2vec file has been loaded")
				elif embedding_name == 'glove':
					print("Load glove file {}".format(cfg['word_embeddings']['glove']['path']))
					initW = data_helper.load_embedding_vectors_glove(vocabulary,cfg['word_embeddings']['glove']['path'],embedding_dimension)
					print("glove file has been loaded\n")
					
				sess.run(cnn.W.assign(initW))

			# It uses dynamic learning rate with a high value at the beginning to speed up the training
			max_learning_rate = 0.005
			min_learning_rate = 0.0001
			decay_speed = FLAGS.decay_coefficient*len(y_train)/params['batch_size']
			# Training starts here
			train_batches = data_helper.batch_iter(list(zip(x_train, y_train)), params['batch_size'], params['num_epochs'])
			best_accuracy, best_at_step = 0, 0
			counter = 0
			#start_time=gmtime();
			"""Step 6: train the cnn model with x_train and y_train (batch by batch)"""
			for train_batch in train_batches:
				#learning_rate = 0.001
				learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-counter/decay_speed)
				counter += 1
				x_train_batch, y_train_batch = zip(*train_batch)
				train_step(x_train_batch, y_train_batch,learning_rate)
				current_step = tf.train.global_step(sess, global_step)

				"""Step 6.1: evaluate the model with x_dev and y_dev (batch by batch)"""
				if current_step % params['evaluate_every'] == 0:
					dev_batches = data_helper.batch_iter(list(zip(x_dev, y_dev)), params['batch_size'], 1)
					total_dev_correct = 0
					k_2_total_dev_correct = 0
					k_3_total_dev_correct = 0
					k_4_total_dev_correct = 0
					k_5_total_dev_correct = 0
					k_6_total_dev_correct = 0
					k_7_total_dev_correct = 0
					k_8_total_dev_correct = 0
					k_9_total_dev_correct = 0
					k_10_total_dev_correct = 0
					for dev_batch in dev_batches:
						x_dev_batch, y_dev_batch = zip(*dev_batch)
						num_dev_correct,k_2_num_dev_correct,k_3_num_dev_correct,k_4_num_dev_correct,k_5_num_dev_correct,k_6_num_dev_correct,k_7_num_dev_correct,k_8_num_dev_correct,k_9_num_dev_correct,k_10_num_dev_correct = dev_step(x_dev_batch, y_dev_batch)
						total_dev_correct += num_dev_correct
						k_2_total_dev_correct += k_2_num_dev_correct
						k_3_total_dev_correct += k_3_num_dev_correct
						k_4_total_dev_correct += k_4_num_dev_correct
						k_5_total_dev_correct += k_5_num_dev_correct
						k_6_total_dev_correct += k_6_num_dev_correct
						k_7_total_dev_correct += k_7_num_dev_correct
						k_8_total_dev_correct += k_8_num_dev_correct
						k_9_total_dev_correct += k_9_num_dev_correct
						k_10_total_dev_correct += k_10_num_dev_correct

					dev_accuracy = float(total_dev_correct) / len(y_dev)
					k_2_dev_accuracy = float(k_2_total_dev_correct) / len(y_dev)
					k_3_dev_accuracy = float(k_3_total_dev_correct) / len(y_dev)
					k_4_dev_accuracy = float(k_4_total_dev_correct) / len(y_dev)
					k_5_dev_accuracy = float(k_5_total_dev_correct) / len(y_dev)
					k_6_dev_accuracy = float(k_6_total_dev_correct) / len(y_dev)
					k_7_dev_accuracy = float(k_7_total_dev_correct) / len(y_dev)
					k_8_dev_accuracy = float(k_8_total_dev_correct) / len(y_dev)
					k_9_dev_accuracy = float(k_9_total_dev_correct) / len(y_dev)
					k_10_dev_accuracy = float(k_10_total_dev_correct) / len(y_dev)
					print("\n\n")
					logging.critical('Accuracy on dev set: {}'.format(dev_accuracy))
					logging.critical('Top-2 Accuracy on dev set: {}'.format(k_2_dev_accuracy))
					logging.critical('Top-3 Accuracy on dev set: {}'.format(k_3_dev_accuracy))
					logging.critical('Top-4 Accuracy on dev set: {}'.format(k_4_dev_accuracy))
					logging.critical('Top-5 Accuracy on dev set: {}'.format(k_5_dev_accuracy))
					logging.critical('Top-6 Accuracy on dev set: {}'.format(k_6_dev_accuracy))
					logging.critical('Top-7 Accuracy on dev set: {}'.format(k_7_dev_accuracy))
					logging.critical('Top-8 Accuracy on dev set: {}'.format(k_8_dev_accuracy))
					logging.critical('Top-9 Accuracy on dev set: {}'.format(k_9_dev_accuracy))
					logging.critical('Top-10 Accuracy on dev set: {}'.format(k_10_dev_accuracy))
					print("\n\n")

					"""Step 6.2: save the model if it is the best based on accuracy of the dev set"""
					if dev_accuracy >= best_accuracy:
						best_accuracy, best_at_step = dev_accuracy, current_step
						path = saver.save(sess, checkpoint_prefix, global_step=current_step)
						logging.critical('Saved model {} at step {}'.format(path, best_at_step))
						logging.critical('Best accuracy {} at step {}'.format(best_accuracy, best_at_step))

			# """Step 7: predict x_test (batch by batch)"""
			# end_time=gmtime();
			# print("\n\n")
			# print("Start Time:",start_time)
			# print("End Time:",end_time)
			# test_batches = data_helper.batch_iter(list(zip(x_test, y_test)), params['batch_size'], 1)
			# total_test_correct = 0
			# k_2_total_test_correct = 0
			# k_3_total_test_correct = 0
			# k_4_total_test_correct = 0
			# k_5_total_test_correct = 0
			# k_6_total_test_correct = 0
			# k_7_total_test_correct = 0
			# k_8_total_test_correct = 0
			# k_9_total_test_correct = 0
			# k_10_total_test_correct = 0
			# for test_batch in test_batches:
			# 	x_test_batch, y_test_batch = zip(*test_batch)
			# 	num_test_correct,k_2_num_test_correct,k_3_num_test_correct,k_4_num_test_correct,k_5_num_test_correct,k_6_num_test_correct,k_7_num_test_correct,k_8_num_test_correct,k_9_num_test_correct,k_10_num_test_correct = dev_step(x_test_batch, y_test_batch)
			# 	total_test_correct += num_test_correct
			# 	k_2_total_test_correct += k_2_num_test_correct
			# 	k_3_total_test_correct += k_3_num_test_correct
			# 	k_4_total_test_correct += k_4_num_test_correct
			# 	k_5_total_test_correct += k_5_num_test_correct
			# 	k_6_total_test_correct += k_6_num_test_correct
			# 	k_7_total_test_correct += k_7_num_test_correct
			# 	k_8_total_test_correct += k_8_num_test_correct
			# 	k_9_total_test_correct += k_9_num_test_correct
			# 	k_10_total_test_correct += k_10_num_test_correct


			# test_accuracy = float(total_test_correct) / len(y_test)
			# k_2_test_accuracy = float(k_2_total_test_correct) / len(y_test)
			# k_3_test_accuracy = float(k_3_total_test_correct) / len(y_test)
			# k_4_test_accuracy = float(k_4_total_test_correct) / len(y_test)
			# k_5_test_accuracy = float(k_5_total_test_correct) / len(y_test)
			# k_6_test_accuracy = float(k_6_total_test_correct) / len(y_test)
			# k_7_test_accuracy = float(k_7_total_test_correct) / len(y_test)
			# k_8_test_accuracy = float(k_8_total_test_correct) / len(y_test)
			# k_9_test_accuracy = float(k_9_total_test_correct) / len(y_test)
			# k_10_test_accuracy = float(k_10_total_test_correct) / len(y_test)
			# print("\n\n")
			# logging.critical('Accuracy on test set is {} '.format(test_accuracy))
			# logging.critical('Top-2 Accuracy on test set is {}'.format(k_2_test_accuracy))
			# logging.critical('Top-3 Accuracy on test set is {}'.format(k_3_test_accuracy))
			# logging.critical('Top-4 Accuracy on test set is {}'.format(k_4_test_accuracy))
			# logging.critical('Top-5 Accuracy on test set is {}'.format(k_5_test_accuracy))
			# logging.critical('Top-6 Accuracy on test set is {}'.format(k_6_test_accuracy))
			# logging.critical('Top-7 Accuracy on test set is {}'.format(k_7_test_accuracy))
			# logging.critical('Top-8 Accuracy on test set is {}'.format(k_8_test_accuracy))
			# logging.critical('Top-9 Accuracy on test set is {}'.format(k_9_test_accuracy))
			# logging.critical('Top-10 Accuracy on test set is {}'.format(k_10_test_accuracy))
			
			print("\n\n")
			logging.critical('The training is complete')
			end_time=gmtime();
			print("\n\n")
			print("Start Time:",start_time)
			print("End Time:",end_time)

if __name__ == '__main__':
	# python3 train.py ./data/consumer_complaints.csv.zip ./parameters.json
	start_time=gmtime();
	print("Start Time:",start_time)
	train_cnn()
