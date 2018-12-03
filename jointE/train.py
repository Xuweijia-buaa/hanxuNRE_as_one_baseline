import tensorflow as tf
import numpy as np
#import time
import datetime
#import os
import network
import json
#from sklearn.metrics import average_precision_score
#import sys
import threading
import subprocess

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
pure_KB=True
use_embedding=False
transE=True
transD=not transE
# 3 keep, one random
if use_embedding:
    #embedding_file='/media/xuweijia/06CE5DF1CE5DD98F/word_embedding/word2vec_glove.txt'                  # 400000,100 n_in_V:71647/149532
    embedding_file='/media/xuweijia/06CE5DF1CE5DD98F/word_embedding/hanxu_vec.txt'                        # 114042 50  n_in_V:40194/149532     
    #name=embedding_file.split('/')[-1].split('.')[:-1][0]    
    #embedding_file='/media/xuweijia/06CE5DF1CE5DD98F/word_embedding/glove840B300d.txt'                    # glove 300  n_in_V: 77076/149532
    name=embedding_file.split('/')[-1].split('.')[:-1][0]+'_keepwd'

    export_path = "../transX_embedding_{}/".format(name)
else:
    export_path = "../transX/"

f = open(export_path + "config.json", 'r')
config = json.loads(f.read())
f.close()

word_vec = np.load(export_path + 'vec.npy') if use_embedding else None
embedding_size=config['word_size'] if use_embedding else 100                      # ~

# g++ init.cpp -o init.so -fPIC -shared -pthread -O3 -march=native
import ctypes
ll = ctypes.cdll.LoadLibrary   
lib = ll("./init.so")        # can use function in init.cpp
# lib.setInPath(export_path)  lib.setInPath("../data/")
lib.init()

if transE:
    transX='transE'
elif transD:
    transX='transD'

FLAGS = tf.app.flags.FLAGS
# flags.DEFINE_interger/float()来添加命令行参数
# test.py --max_epoch 5000
tf.app.flags.DEFINE_integer('max_epoch',5000,'maximum of training epochs')         #  50 text epoch, 1000 kg epoch (n_bkg=100)     # 500 epoch text:  4794 batch KB
tf.app.flags.DEFINE_float('nbatch_kg',500,'entity numbers used each training time') #  100  batch_size 400       nb++  B--  (500,100)

tf.app.flags.DEFINE_integer('win_size',3,'dropout rate')
tf.app.flags.DEFINE_string('model_dir',export_path[:-1]+'_model_dir/','path to store model')

tf.app.flags.DEFINE_float('margin',1.0,'entity numbers used each training time')
tf.app.flags.DEFINE_float('learning_rate_kg',0.001,'learning rate for kg')
tf.app.flags.DEFINE_integer('ent_total',lib.getEntityTotal(),'total of entities')
tf.app.flags.DEFINE_integer('rel_total',lib.getRelationTotal(),'total of relations')
tf.app.flags.DEFINE_integer('tri_total',lib.getTripleTotal(),'total of triples')
tf.app.flags.DEFINE_integer('katt_flag', 1, '1 for katt, 0 for att')

tf.app.flags.DEFINE_string('model', 'cnn', 'neural models to encode sentences')
tf.app.flags.DEFINE_integer('max_length',config['fixlen'],'maximum of number of words in one sentence')
tf.app.flags.DEFINE_integer('pos_num', config['maxlen'] * 2 + 1,'number of position embedding vectors')
tf.app.flags.DEFINE_integer('num_classes', config['textual_rel_total'],'maximum of relations')

tf.app.flags.DEFINE_integer('hidden_size',230,'hidden feature size')
tf.app.flags.DEFINE_integer('pos_size',5,'position embedding size')

tf.app.flags.DEFINE_integer('batch_size',400,'entity numbers used each training time')
tf.app.flags.DEFINE_float('learning_rate',0.5,'learning rate for nn')
tf.app.flags.DEFINE_float('weight_decay',0.00001,'weight_decay')
tf.app.flags.DEFINE_float('keep_prob',0.5,'dropout rate')
tf.app.flags.DEFINE_string('summary_dir','./summary','path to store summary_dir')

subprocess.call(['mkdir', '-p', FLAGS.model_dir])
# tf.app.flags.FLAGS可以从对应的命令行参数取出参数, original all string
FLAGS.batch_size=int(FLAGS.batch_size)
FLAGS.max_length=int(FLAGS.max_length)
FLAGS.num_classes=int(FLAGS.num_classes)
FLAGS.pos_size=int(FLAGS.pos_size) 
FLAGS.hidden_size=int(FLAGS.hidden_size) 
FLAGS.ent_total=int(FLAGS.ent_total)
FLAGS.rel_total=int(FLAGS.rel_total)
FLAGS.pos_num=int(FLAGS.pos_num)
FLAGS.nbatch_kg=int(FLAGS.nbatch_kg)
FLAGS.tri_total=int(FLAGS.tri_total)
FLAGS.win_size=int(FLAGS.win_size)

def MakeSummary(name, value):
	"""Creates a tf.Summary proto with the given name and value."""
	summary = tf.Summary()
	val = summary.value.add()
	val.tag = str(name)
	val.simple_value = float(value)
	return summary

def make_shape(array,last_dim):
	output = []
	for i in array:
		for j in i:
			output.append(j)
	output = np.array(output)
	if np.shape(output)[-1]==last_dim:
		return output
	else:
		print ('Make Shape Error!')

def main(_):

	print ('reading word embedding')
	word_vec = np.load(export_path + 'vec.npy') if use_embedding else None
	print ('reading training data')
	
	instance_triple = np.load(export_path + 'train_instance_triple.npy')
	instance_scope = np.load(export_path + 'train_instance_scope.npy')
	train_len = np.load(export_path + 'train_len.npy')
	train_label = np.load(export_path + 'train_label.npy') # relation idx for each sample
	train_word = np.load(export_path + 'train_word.npy')
	train_pos1 = np.load(export_path + 'train_pos1.npy')
	train_pos2 = np.load(export_path + 'train_pos2.npy')
	train_mask = np.load(export_path + 'train_mask.npy')
	train_head = np.load(export_path + 'train_head.npy')
	train_tail = np.load(export_path + 'train_tail.npy')

	print ('reading finished')
	print ('mentions 		: %d' % (len(instance_triple)))
	print ('sentences		: %d' % (len(train_len)))
	print ('relations		: %d' % (FLAGS.num_classes))
	print ('position size 	: %d' % (FLAGS.pos_size))
	print ('hidden size		: %d' % (FLAGS.hidden_size))
    # train_label: all sample's relation idx
    # count different relations numbers in all samples, give different weights
	reltot = {}
	for index, i in enumerate(train_label):
		if not i in reltot:
			reltot[i] = 1.0
		else:
			reltot[i] += 1.0
	for i in reltot:
		reltot[i] = 1/(reltot[i] ** (0.05)) 
	print ('building network...')
	sess = tf.Session()
	if FLAGS.model.lower() == "cnn":
		model = network.CNN(use_embedding,embedding_size,is_training = True,word_embeddings = word_vec)
	elif FLAGS.model.lower() == "pcnn":
		model = network.PCNN(use_embedding,embedding_size,is_training = True,word_embeddings = word_vec)
	elif FLAGS.model.lower() == "lstm":
		model = network.RNN(use_embedding,embedding_size,is_training = True,word_embeddings = word_vec, cell_name = "LSTM", simple_position = True)
	elif FLAGS.model.lower() == "gru":
		model = network.RNN(use_embedding,embedding_size,is_training = True,word_embeddings = word_vec, cell_name = "GRU", simple_position = True)
	elif FLAGS.model.lower() == "bi-lstm" or FLAGS.model.lower() == "bilstm":
		model = network.BiRNN(use_embedding,embedding_size,is_training = True, word_embeddings = word_vec, cell_name = "LSTM", simple_position = True)
	elif FLAGS.model.lower() == "bi-gru" or FLAGS.model.lower() == "bigru":
		model = network.BiRNN(use_embedding,embedding_size,is_training = True, word_embeddings = word_vec, cell_name = "GRU", simple_position = True)
	# model once sure, just one model, different train_epoch, optimizer.
    
    
	global_step = tf.Variable(0,name='global_step',trainable=False)
	global_step_kg = tf.Variable(0,name='global_step_kg',trainable=False)
	tf.summary.scalar('learning_rate', FLAGS.learning_rate)
	tf.summary.scalar('learning_rate_kg', FLAGS.learning_rate_kg)

   # text op
	optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)   # sgd
	grads_and_vars = optimizer.compute_gradients(model.loss)
	train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)
   # kg op
	optimizer_kg = tf.train.GradientDescentOptimizer(FLAGS.learning_rate_kg)
	grads_and_vars_kg = optimizer_kg.compute_gradients(model.loss_kg)
	train_op_kg = optimizer_kg.apply_gradients(grads_and_vars_kg, global_step = global_step_kg)

	merged_summary = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
	sess.run(tf.global_variables_initializer())
	#saver = tf.train.Saver(max_to_keep=None)
	saver = tf.train.Saver()
	print ('building finished')

	def train_kg(coord):
       #train_step_kg(ph, pt, pr, nh, nt, nr),from c++
		def train_step_kg(pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch):
			feed_dict = {
				model.pos_h: pos_h_batch,
				model.pos_t: pos_t_batch,
				model.pos_r: pos_r_batch,
				model.neg_h: neg_h_batch,
				model.neg_t: neg_t_batch,
				model.neg_r: neg_r_batch
			}
			_, step, loss = sess.run(
				[train_op_kg, global_step_kg, model.loss_kg], feed_dict)
			return loss

		batch_size = (int(FLAGS.tri_total) // int(FLAGS.nbatch_kg))   # 100,600/  300,200          /200,300  /  1000,60
		#batch_size = (int(FLAGS.ent_total) // int(FLAGS.nbatch_kg))  # should not be FLAGS.tri_total
       # B. defi each element is np32, 32 wei
		ph = np.zeros(batch_size, dtype = np.int32)                   # use to store batch's ex e1 
		pt = np.zeros(batch_size, dtype = np.int32)                   # e2
		pr = np.zeros(batch_size, dtype = np.int32)                   # r
		nh = np.zeros(batch_size, dtype = np.int32)                   # n_e1
		nt = np.zeros(batch_size, dtype = np.int32)                   # n_e2
		nr = np.zeros(batch_size, dtype = np.int32)                   # n_r
        #ph.__array_interface__['data'] :2-tuple whose first argument is an integer (a long integer if necessary) that points to the data-area storing the array contents
        # array's first element's address
		ph_addr = ph.__array_interface__['data'][0]
		pt_addr = pt.__array_interface__['data'][0]
		pr_addr = pr.__array_interface__['data'][0]
		nh_addr = nh.__array_interface__['data'][0]
		nt_addr = nt.__array_interface__['data'][0]
		nr_addr = nr.__array_interface__['data'][0]
        # define type in c   
        # ctypes.c_void_p==void *    ctypes.c_int=int
		lib.getBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
		times_kg = 0
        # coord.request_stop() let it stop at some time
        # pure batch. B. continuelly train batch. no concept of epoch. just have a size
		while not coord.should_stop():
			times_kg += 1
			res = 0.0
			#print(type(FLAGS.nbatch_kg))
			#print(FLAGS.nbatch_kg)
			for batch in range(int(FLAGS.nbatch_kg)):
				lib.getBatch(ph_addr, pt_addr, pr_addr, nh_addr, nt_addr, nr_addr, batch_size)
                
				res += train_step_kg(ph, pt, pr, nh, nt, nr)
			time_str = datetime.datetime.now().isoformat()
			print ("KB batch %d time %s | loss : %f" % (times_kg, time_str, res))
			if pure_KB and times_kg % 20000 == 0:
				print ('saving model...')
				# path = saver.save(sess,FLAGS.model_dir+FLAGS.model+str(FLAGS.katt_flag)+transX, global_step=current_step)
				path = saver.save(sess,FLAGS.model_dir+FLAGS.model+str(FLAGS.katt_flag)+transX+'_pureKB_'+str(pure_KB)+'_epoch'+str(times_kg)+'_nkb'+str(FLAGS.nbatch_kg)+'_win'+str(FLAGS.win_size)+'_'+str(embedding_size))
              # tf.app.flags.DEFINE_string('model_dir','./model/','path to store model')
              # tf.app.flags.DEFINE_string('checkpoint_path','./model/','path to store model')
              # saver.restore(sess, FLAGS.checkpoint_path + FLAGS.model+str(FLAGS.katt_flag)+"-"+str(3664*iters))
				print ('have savde model to '+path)
			if pure_KB and times_kg==160000:
				coord.request_stop()


	def train_nn(coord):
    # train_step(train_head[index], train_tail[index], train_word[index,:], train_pos1[index,:], train_pos2[index,:], train_mask[index,:], train_len[index],train_label[index], label_, np.array(scope), weights)
    # all from numpy
		def train_step(head, tail, word, pos1, pos2, mask, leng, label_index, label, scope, weights):
			feed_dict = {
				model.head_index: head,
				model.tail_index: tail,
				model.word: word,
				model.pos1: pos1,
				model.pos2: pos2,
				model.mask: mask,
				model.len : leng,
				model.label_index: label_index, # B, real relation idx
				model.label: label,             # B,|R|.   real pos value 1. other pos 0
				model.scope: scope,
				model.keep_prob: FLAGS.keep_prob,
				model.weights: weights
			}
			_, step, loss, summary, output, correct_predictions = sess.run([train_op, global_step, model.loss, merged_summary, model.output, model.correct_predictions], feed_dict)
			summary_writer.add_summary(summary, step)
			return output, loss, correct_predictions

		train_order = list(range(len(instance_triple)))

		save_epoch = 150

		for one_epoch in range(FLAGS.max_epoch):

			print('epoch '+str(one_epoch+1)+' starts!')
			np.random.shuffle(train_order)
			s1 = 0.0
			s2 = 0.0
			tot1 = 0.0
			tot2 = 0.0
			losstot = 0.0
			for i in range(int(len(train_order)/float(FLAGS.batch_size))):
				input_scope = np.take(instance_scope, train_order[i * FLAGS.batch_size:(i+1)*FLAGS.batch_size], axis=0)
				index = []
				scope = [0]
				label = []      # sample's true relation idx
				weights = []
				for num in input_scope:
					index = index + list(range(num[0], num[1] + 1))
					label.append(train_label[num[0]])
					if train_label[num[0]] > 53:
						pass
					scope.append(scope[len(scope)-1] + num[1] - num[0] + 1)
					weights.append(reltot[train_label[num[0]]])
				label_ = np.zeros((FLAGS.batch_size, FLAGS.num_classes))
				label_[np.arange(FLAGS.batch_size), label] = 1
				# correct_predictions:B, if each sample predict true(1), else 0   cnn's output
				output, loss, correct_predictions = train_step(train_head[index], train_tail[index], train_word[index,:], train_pos1[index,:], train_pos2[index,:], train_mask[index,:], train_len[index],train_label[index], label_, np.array(scope), weights)
				num = 0
				s = 0
				losstot += loss
				for num in correct_predictions:
#					if label[s] == 0:
#						tot1 += 1.0
#						if num:
#							s1+= 1.0
					tot2 += 1.0
					if num:
						s2 += 1.0
					s = s + 1

				time_str = datetime.datetime.now().isoformat()
				print ("epoch %d batch %d time %s | loss : %f, accuracy: %f" % (one_epoch, i, time_str, loss, s2 / tot2))
				current_step = tf.train.global_step(sess, global_step)

			if (one_epoch + 1) % save_epoch == 0:
				print ('epoch '+str(one_epoch+1)+' has finished')
				print ('saving model...')
				# path = saver.save(sess,FLAGS.model_dir+FLAGS.model+str(FLAGS.katt_flag)+transX, global_step=current_step)
				path = saver.save(sess,FLAGS.model_dir+FLAGS.model+str(FLAGS.katt_flag)+transX+'_pureKB_'+str(pure_KB)+'_epoch'+str(one_epoch)+'_nkb'+str(FLAGS.nbatch_kg)+'_win'+str(FLAGS.win_size)+'_'+str(embedding_size))
              # tf.app.flags.DEFINE_string('model_dir','./model/','path to store model')
              # tf.app.flags.DEFINE_string('checkpoint_path','./model/','path to store model')
              # saver.restore(sess, FLAGS.checkpoint_path + FLAGS.model+str(FLAGS.katt_flag)+"-"+str(3664*iters))
				print ('have savde model to '+path)

		coord.request_stop()


	coord = tf.train.Coordinator()
	threads = []
	threads.append(threading.Thread(target=train_kg, args=(coord,)))
	if not pure_KB:
		threads.append(threading.Thread(target=train_nn, args=(coord,)))
	for t in threads: t.start()
	coord.join(threads)

if __name__ == "__main__":
	tf.app.run() # 执行程序中main函数，并解析命令行参数！
