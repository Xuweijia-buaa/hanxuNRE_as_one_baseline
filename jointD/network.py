import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

FLAGS = tf.app.flags.FLAGS
class NN(object):

	def calc(self, e, t, r):
		return e + tf.reduce_sum(e * t, 1, keep_dims = True) * r

	def __init__(self, use_embedding,embedding_size,is_training,word_embeddings, simple_position = False):

		FLAGS.batch_size=int(FLAGS.batch_size)
		FLAGS.max_length=int(FLAGS.max_length)
		FLAGS.num_classes=int(FLAGS.num_classes)
		FLAGS.pos_size=int(FLAGS.pos_size) 
		FLAGS.hidden_size=int(FLAGS.hidden_size) 
		FLAGS.ent_total=int(FLAGS.ent_total)
		FLAGS.rel_total=int(FLAGS.rel_total)
		FLAGS.pos_num=int(FLAGS.pos_num)
        
		self.max_length = FLAGS.max_length
		self.num_classes = FLAGS.num_classes
		self.word_size = embedding_size
		self.hidden_size = FLAGS.hidden_size
		if FLAGS.model.lower() == "cnn":
			self.output_size = FLAGS.hidden_size
		elif FLAGS.model.lower() == "pcnn":
			self.output_size = FLAGS.hidden_size * 3
		elif FLAGS.model.lower() == "lstm":
			self.output_size = FLAGS.hidden_size
		elif FLAGS.model.lower() == "gru":
			self.output_size = FLAGS.hidden_size
		elif FLAGS.model.lower() == "bi-lstm" or FLAGS.model.lower() == "bilstm":
			self.output_size = FLAGS.hidden_size * 2
		elif FLAGS.model.lower() == "bi-gru" or FLAGS.model.lower() == "bigru":
			self.output_size = FLAGS.hidden_size * 2
		self.margin = FLAGS.margin
		# placeholders for text models
		self.word = tf.placeholder(dtype=tf.int32,shape=[None, self.max_length], name='input_word')
		self.pos1 = tf.placeholder(dtype=tf.int32,shape=[None, self.max_length], name='input_pos1')
		self.pos2 = tf.placeholder(dtype=tf.int32,shape=[None, self.max_length], name='input_pos2')
		self.mask = tf.placeholder(dtype=tf.int32,shape=[None, self.max_length],name='input_mask')
		self.len = tf.placeholder(dtype=tf.int32,shape=[None],name='input_len')
		self.label_index = tf.placeholder(dtype=tf.int32,shape=[None], name='label_index')
		self.head_index = tf.placeholder(dtype=tf.int32,shape=[None], name='head_index')
		self.tail_index = tf.placeholder(dtype=tf.int32,shape=[None], name='tail_index')
		self.label = tf.placeholder(dtype=tf.float32,shape=[FLAGS.batch_size, self.num_classes], name='input_label')
		self.scope = tf.placeholder(dtype=tf.int32,shape=[FLAGS.batch_size+1], name='scope')	
		self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
		self.weights = tf.placeholder(dtype=tf.float32,shape=[FLAGS.batch_size])
		# placeholders for kg models
		self.pos_h = tf.placeholder(tf.int32, [None])
		self.pos_t = tf.placeholder(tf.int32, [None])
		self.pos_r = tf.placeholder(tf.int32, [None])
		self.neg_h = tf.placeholder(tf.int32, [None])
		self.neg_t = tf.placeholder(tf.int32, [None])
		self.neg_r = tf.placeholder(tf.int32, [None])
        
		if word_embeddings.any()!=None:
			word_embeddings=word_embeddings.astype('float32')
		with tf.name_scope("embedding-layers"):
			# word embeddings
			temp_word_embedding = tf.get_variable(initializer=word_embeddings[FLAGS.ent_total:,:],name = 'temp_word_embedding',dtype=tf.float32) if use_embedding else \
                                 tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform = False),shape =[FLAGS.ent_total,self.word_size],name = 'temp_word_embedding',dtype=tf.float32)
			ent_embedding = tf.get_variable(name = "ent_embedding",shape = [FLAGS.ent_total, self.word_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
			unk_word_embedding = tf.get_variable('unk_embedding',[self.word_size], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
			self.word_embedding = tf.concat([
				ent_embedding,
				temp_word_embedding,
				tf.reshape(unk_word_embedding,[1, self.word_size]),
				tf.reshape(tf.constant(np.zeros(self.word_size, dtype=np.float32)),[1, self.word_size]) ],0)
			self.relation_matrix = tf.get_variable('relation_matrix',[self.num_classes, self.output_size],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
			self.bias = tf.get_variable('bias',[self.num_classes],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
			# position embeddings
			if simple_position:
				temp_pos_array = np.zeros((FLAGS.pos_num + 1, FLAGS.pos_size), dtype=np.float32)
				temp_pos_array[(FLAGS.pos_num - 1) // 2] = np.ones(FLAGS.pos_size, dtype=np.float32)
				self.pos1_embedding = tf.constant(temp_pos_array)
				self.pos2_embedding = tf.constant(temp_pos_array)
			else:
				FLAGS.pos_size=int(FLAGS.pos_size)
				temp_pos1_embedding = tf.get_variable('temp_pos1_embedding',[FLAGS.pos_num,FLAGS.pos_size],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
				temp_pos2_embedding = tf.get_variable('temp_pos2_embedding',[FLAGS.pos_num,FLAGS.pos_size],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
				self.pos1_embedding = tf.concat([temp_pos1_embedding,tf.reshape(tf.constant(np.zeros(FLAGS.pos_size,dtype=np.float32)),[1, FLAGS.pos_size])],0)
				self.pos2_embedding = tf.concat([temp_pos2_embedding,tf.reshape(tf.constant(np.zeros(FLAGS.pos_size,dtype=np.float32)),[1, FLAGS.pos_size])],0)
			# relation embeddings and the transfer matrix between relations and textual relations
			self.rel_embeddings = tf.get_variable(name = "rel_embedding", shape = [FLAGS.rel_total, self.word_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
			self.transfer_matrix = tf.get_variable("transfer_matrix", [self.output_size, self.word_size])
			self.transfer_bias = tf.get_variable('transfer_bias', [self.word_size], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
			self.ent_transfer = tf.get_variable(name = "ent_transfer", shape = [FLAGS.ent_total, self.word_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
			self.rel_transfer = tf.get_variable(name = "rel_transfer", shape = [FLAGS.rel_total, self.word_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))

		with tf.name_scope("embedding-lookup"):
			# textual embedding-lookup 
			input_word = tf.nn.embedding_lookup(self.word_embedding, self.word)
			input_pos1 = tf.nn.embedding_lookup(self.pos1_embedding, self.pos1)
			input_pos2 = tf.nn.embedding_lookup(self.pos2_embedding, self.pos2)
			self.input_embedding = tf.concat(values = [input_word, input_pos1, input_pos2], axis = 2)
			# knowledge embedding-lookup 
			pos_h_e = tf.nn.embedding_lookup(self.word_embedding, self.pos_h)
			pos_t_e = tf.nn.embedding_lookup(self.word_embedding, self.pos_t)
			pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
			pos_h_t = tf.nn.embedding_lookup(self.ent_transfer, self.pos_h)
			pos_t_t = tf.nn.embedding_lookup(self.ent_transfer, self.pos_t)
			pos_r_t = tf.nn.embedding_lookup(self.rel_transfer, self.pos_r)
			neg_h_e = tf.nn.embedding_lookup(self.word_embedding, self.neg_h)
			neg_t_e = tf.nn.embedding_lookup(self.word_embedding, self.neg_t)
			neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)
			neg_h_t = tf.nn.embedding_lookup(self.ent_transfer, self.neg_h)
			neg_t_t = tf.nn.embedding_lookup(self.ent_transfer, self.neg_t)
			neg_r_t = tf.nn.embedding_lookup(self.rel_transfer, self.neg_r)
			pos_h_e = self.calc(pos_h_e, pos_h_t, pos_r_t)
			pos_t_e = self.calc(pos_t_e, pos_t_t, pos_r_t)
			neg_h_e = self.calc(neg_h_e, neg_h_t, neg_r_t)
			neg_t_e = self.calc(neg_t_e, neg_t_t, neg_r_t)
		with tf.name_scope("knowledge_graph"):
			pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keep_dims = True)
			neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keep_dims = True)
			self.loss_kg = tf.reduce_sum(tf.maximum(pos - neg + self.margin, 0))

	def transfer(self, x):
		res = tf.nn.bias_add(tf.matmul(x, self.transfer_matrix), self.transfer_bias)
		return res

	def att(self, x, is_training = True, dropout = True):
		with tf.name_scope("sentence-level-attention"):
			current_attention = tf.nn.embedding_lookup(self.relation_matrix, self.label_index)
			attention_logit = tf.reduce_sum(current_attention * x, 1)
			tower_repre = []
			for i in range(FLAGS.batch_size):
				sen_matrix = x[self.scope[i]:self.scope[i+1]]
				attention_score = tf.nn.softmax(tf.reshape(attention_logit[self.scope[i]:self.scope[i+1]], [1, -1]))
				final_repre = tf.reshape(tf.matmul(attention_score, sen_matrix),[self.output_size])
				tower_repre.append(final_repre)
			if dropout:
				stack_repre = tf.layers.dropout(tf.stack(tower_repre), rate = self.keep_prob, training = is_training)
			else:
				stack_repre = tf.stack(tower_repre)
		return stack_repre

	def katt(self, x, is_training = True, dropout = True):
		with tf.name_scope("knowledge-based-attention"):
			head = tf.nn.embedding_lookup(self.word_embedding, self.head_index)
			tail = tf.nn.embedding_lookup(self.word_embedding, self.tail_index)
			head_transfer = tf.nn.embedding_lookup(self.ent_transfer, self.head_index)
			tail_transfer = tf.nn.embedding_lookup(self.ent_transfer, self.tail_index)
			rel_transfer = tf.nn.embedding_lookup(self.rel_transfer, self.label_index)
			kg_att = self.calc(head, head_transfer, rel_transfer) - self.calc(tail, tail_transfer, rel_transfer)
			attention_logit = tf.reduce_sum(self.transfer(x) * kg_att, 1)
			tower_repre = []
			for i in range(FLAGS.batch_size):
				sen_matrix = x[self.scope[i]:self.scope[i+1]]
				attention_score = tf.nn.softmax(tf.reshape(attention_logit[self.scope[i]:self.scope[i+1]], [1, -1]))
				final_repre = tf.reshape(tf.matmul(attention_score, sen_matrix),[self.output_size])
				tower_repre.append(final_repre)
			if dropout:
				stack_repre = tf.layers.dropout(tf.stack(tower_repre), rate = self.keep_prob, training = is_training)
			else:
				stack_repre = tf.stack(tower_repre)
		return stack_repre

	def att_test(self, x, is_training = False):
		test_attention_logit = tf.matmul(x, tf.transpose(self.relation_matrix))
		return test_attention_logit

	def katt_test(self, x, is_training = False):
		
		head = tf.nn.embedding_lookup(self.word_embedding, self.head_index)
		tail = tf.nn.embedding_lookup(self.word_embedding, self.tail_index)
		head_transfer = tf.nn.embedding_lookup(self.ent_transfer, self.head_index)
		tail_transfer = tf.nn.embedding_lookup(self.ent_transfer, self.tail_index)

		kg_att = []
		for i in range(self.num_classes):
			each_att = tf.expand_dims(self.calc(head, head_transfer, tf.reshape(self.rel_transfer[i], [-1, self.word_size])) - self.calc(tail, tail_transfer,  tf.reshape(self.rel_transfer[i], [-1, self.word_size])), -1)
			kg_att.append(each_att)
		kg_att = tf.concat(kg_att, 2)
		x = tf.reshape(self.transfer(x), [-1, 1, self.word_size])
		test_attention_logit = tf.matmul(x, kg_att)
		return tf.reshape(test_attention_logit, [-1, self.num_classes])

class CNN(NN):

	def __init__(self, use_embedding,embedding_size,is_training,word_embeddings, simple_position = False):
		NN.__init__(self, use_embedding,embedding_size,is_training,word_embeddings, simple_position)
		if not is_training:
			FLAGS.test_batch_size=int(FLAGS.test_batch_size)
		with tf.name_scope("conv-maxpool"):
			input_sentence = tf.expand_dims(self.input_embedding, axis=1)
			x = tf.layers.conv2d(inputs = input_sentence, filters=FLAGS.hidden_size, kernel_size=[1,3], strides=[1, 1], padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()) 
			x = tf.reduce_max(x, axis=2)
			x = tf.nn.relu(tf.squeeze(x))

		if FLAGS.katt_flag != 0:
			stack_repre = self.katt(x, is_training)
		else:
			stack_repre = self.att(x, is_training)

		with tf.name_scope("loss"):
			logits = tf.matmul(stack_repre, tf.transpose(self.relation_matrix)) + self.bias
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=logits))
			self.loss = tf.losses.softmax_cross_entropy(onehot_labels = self.label, logits = logits, weights = self.weights)
			self.output = tf.nn.softmax(logits)
			tf.summary.scalar('loss',self.loss)
			self.predictions = tf.argmax(logits, 1, name="predictions")
			self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, 1))
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

		if not is_training:
			with tf.name_scope("test"):
				if FLAGS.katt_flag != 0:
					test_attention_logit = self.katt_test(x)
				else:
					test_attention_logit = self.att_test(x)
				test_tower_output = []
				for i in range(FLAGS.test_batch_size):
					test_attention_score = tf.nn.softmax(tf.transpose(test_attention_logit[self.scope[i]:self.scope[i+1],:]))
					final_repre = tf.matmul(test_attention_score, x[self.scope[i]:self.scope[i+1]])
					logits = tf.matmul(final_repre, tf.transpose(self.relation_matrix)) + self.bias
					output = tf.diag_part(tf.nn.softmax(logits))
					test_tower_output.append(output)
				test_stack_output = tf.reshape(tf.stack(test_tower_output),[FLAGS.test_batch_size, self.num_classes])
				self.test_output = test_stack_output


class PCNN(NN):

	def __init__(self, use_embedding,embedding_size,is_training, word_embeddings,simple_position = False):
		NN.__init__(self, use_embedding,embedding_size,is_training,word_embeddings, simple_position)
		if not is_training:
			FLAGS.test_batch_size=int(FLAGS.test_batch_size)
		with tf.name_scope("conv-maxpool"):
			mask_embedding = tf.constant([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
			pcnn_mask = tf.nn.embedding_lookup(mask_embedding, self.mask)
			input_sentence = tf.expand_dims(self.input_embedding, axis=1)
			x = tf.layers.conv2d(inputs = input_sentence, filters=FLAGS.hidden_size, kernel_size=[1,3], strides=[1, 1], padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
			x = tf.reshape(x, [-1, self.max_length, FLAGS.hidden_size, 1])
			x = tf.reduce_max(tf.reshape(pcnn_mask, [-1, 1, self.max_length, 3]) * tf.transpose(x,[0, 2, 1, 3]), axis = 2)
			x = tf.nn.relu(tf.reshape(x,[-1, self.output_size]))

		if FLAGS.katt_flag != 0:
			stack_repre = self.katt(x, is_training)
		else:
			stack_repre = self.att(x, is_training)

		with tf.name_scope("loss"):
			logits = tf.matmul(stack_repre, tf.transpose(self.relation_matrix)) + self.bias
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=logits))
			self.loss = tf.losses.softmax_cross_entropy(onehot_labels = self.label, logits = logits, weights = self.weights)
			self.output = tf.nn.softmax(logits)
			tf.summary.scalar('loss',self.loss)
			self.predictions = tf.argmax(logits, 1, name="predictions")
			self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, 1))
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

		if not is_training:
			with tf.name_scope("test"):
				if FLAGS.katt_flag != 0:
					test_attention_logit = self.katt_test(x)
				else:
					test_attention_logit = self.att_test(x)
				test_tower_output = []
				for i in range(FLAGS.test_batch_size):
					test_attention_score = tf.nn.softmax(tf.transpose(test_attention_logit[self.scope[i]:self.scope[i+1],:]))
					final_repre = tf.matmul(test_attention_score, x[self.scope[i]:self.scope[i+1]])
					logits = tf.matmul(final_repre, tf.transpose(relation_matrix)) + bias
					output = tf.diag_part(tf.nn.softmax(logits))
					test_tower_output.append(output)
				test_stack_output = tf.reshape(tf.stack(test_tower_output),[FLAGS.test_batch_size, self.num_classes])
				self.test_output = test_stack_output

class RNN(NN):

	def get_rnn_cell(self, dim, cell_name = 'lstm'):
		if isinstance(cell_name,list) or isinstance(cell_name, tuple):
			if len(cell_name) == 1:
				return get_rnn_cell(dim, cell_name[0])
			cells = [get_rnn_cell(dim, c) for c in cell_name]
			return tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
		if cell_name.lower() == 'lstm':
			return tf.contrib.rnn.BasicLSTMCell(dim, state_is_tuple=True)
		elif cell_name.lower() == 'gru':
			return tf.contrib.rnn.GRUCell(dim)
		raise NotImplementedError

	def __init__(self, use_embedding,embedding_size,is_training, word_embeddings, cell_name, simple_position = False):
		NN.__init__(self, use_embedding,embedding_size,is_training, word_embeddings, simple_position)
		input_sentence = tf.layers.dropout(self.input_embedding, rate = self.keep_prob, training = is_training)
		with tf.name_scope('rnn'):
			cell = self.get_rnn_cell(FLAGS.hidden_size, cell_name)
			outputs, states = tf.nn.dynamic_rnn(cell, input_sentence,
											sequence_length = self.len,
											dtype = tf.float32,
											scope = 'dynamic-rnn')
			if isinstance(states, tuple):
				states = states[0]
			x = states

		if FLAGS.katt_flag != 0:
			stack_repre = self.katt(x, is_training, False)
		else:
			stack_repre = self.att(x, is_training, False)

		with tf.name_scope("loss"):
			logits = tf.matmul(stack_repre, tf.transpose(self.relation_matrix)) + self.bias
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=logits))
			self.loss = tf.losses.softmax_cross_entropy(onehot_labels = self.label, logits = logits, weights = self.weights)
			self.output = tf.nn.softmax(logits)
			tf.summary.scalar('loss',self.loss)
			self.predictions = tf.argmax(logits, 1, name="predictions")
			self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, 1))
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")
	
		if not is_training:
			with tf.name_scope("test"):
				if FLAGS.katt_flag != 0:
					test_attention_logit = self.katt_test(x)
				else:
					test_attention_logit = self.att_test(x)
				test_tower_output = []
				for i in range(FLAGS.test_batch_size):
					test_attention_score = tf.nn.softmax(tf.transpose(test_attention_logit[self.scope[i]:self.scope[i+1],:]))
					final_repre = tf.matmul(test_attention_score, x[self.scope[i]:self.scope[i+1]])
					logits = tf.matmul(final_repre, tf.transpose(relation_matrix)) + bias
					output = tf.diag_part(tf.nn.softmax(logits))
					test_tower_output.append(output)
				test_stack_output = tf.reshape(tf.stack(test_tower_output),[FLAGS.test_batch_size, self.num_classes])
				self.test_output = test_stack_output

class BiRNN(NN):

	def get_rnn_cell(self, dim, cell_name = 'lstm'):
		if isinstance(cell_name,list) or isinstance(cell_name, tuple):
			if len(cell_name) == 1:
				return get_rnn_cell(dim, cell_name[0])
			cells = [get_rnn_cell(dim, c) for c in cell_name]
			return tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
		if cell_name.lower() == 'lstm':
			return tf.contrib.rnn.BasicLSTMCell(dim, state_is_tuple=True)
		elif cell_name.lower() == 'gru':
			return tf.contrib.rnn.GRUCell(dim)
		raise NotImplementedError

	def __init__(self, use_embedding,embedding_size,is_training, word_embeddings, cell_name, simple_position = False):
		NN.__init__(self, use_embedding,embedding_size,is_training, word_embeddings, simple_position)
		input_sentence = tf.layers.dropout(self.input_embedding, rate = self.keep_prob, training = is_training)
		with tf.name_scope('bi-rnn'):
			fw_cell = self.get_rnn_cell(FLAGS.hidden_size, cell_name)
			bw_cell = self.get_rnn_cell(FLAGS.hidden_size, cell_name)
			outputs, states = tf.nn.bidirectional_dynamic_rnn(
							fw_cell, bw_cell, input_sentence,
							sequence_length = self.len,
							dtype = tf.float32,
							scope = 'bi-dynamic-rnn')
			fw_states, bw_states = states
			if isinstance(fw_states, tuple):
				fw_states = fw_states[0]
				bw_states = bw_states[0]
			x = tf.concat(states, axis=1)

		if FLAGS.katt_flag != 0:
			stack_repre = self.katt(x, is_training, False)
		else:
			stack_repre = self.att(x, is_training, False)

		with tf.name_scope("loss"):
			logits = tf.matmul(stack_repre, tf.transpose(self.relation_matrix)) + self.bias
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=logits))
			self.loss = tf.losses.softmax_cross_entropy(onehot_labels = self.label, logits = logits, weights = self.weights)
			self.output = tf.nn.softmax(logits)
			tf.summary.scalar('loss',self.loss)
			self.predictions = tf.argmax(logits, 1, name="predictions")
			self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, 1))
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")
	
		if not is_training:
			with tf.name_scope("test"):
				if FLAGS.katt_flag != 0:
					test_attention_logit = self.katt_test(x)
				else:
					test_attention_logit = self.att_test(x)
				test_tower_output = []
				for i in range(FLAGS.test_batch_size):
					test_attention_score = tf.nn.softmax(tf.transpose(test_attention_logit[self.scope[i]:self.scope[i+1],:]))
					final_repre = tf.matmul(test_attention_score, x[self.scope[i]:self.scope[i+1]])
					logits = tf.matmul(final_repre, tf.transpose(relation_matrix)) + bias
					output = tf.diag_part(tf.nn.softmax(logits))
					test_tower_output.append(output)
				test_stack_output = tf.reshape(tf.stack(test_tower_output),[FLAGS.test_batch_size, self.num_classes])
				self.test_output = test_stack_output

