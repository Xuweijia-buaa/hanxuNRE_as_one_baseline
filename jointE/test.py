#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 22:50:34 2018

@author: xuweijia
"""
import json
import tensorflow as tf
import network
import numpy as np
import ctypes
pure_KB=True
transE=True
use_embedding=False
times_kg=160000 if pure_KB else 4999


transD=not transE
ee='30'
dev=True
test=not dev

# get config
if use_embedding:
    # embedding_file='/media/xuweijia/06CE5DF1CE5DD98F/word_embedding/word2vec_glove.txt'                   # 40000,100
    # embedding_file='/media/xuweijia/06CE5DF1CE5DD98F/word_embedding/hanxu_vec.txt'                        # 114042 50
    # name=embedding_file.split('/')[-1].split('.')[:-1][0]
    
    embedding_file='/media/xuweijia/06CE5DF1CE5DD98F/word_embedding/glove840B300d.txt'
    name=embedding_file.split('/')[-1].split('.')[:-1][0]+'_keepwd'
    
    export_path = "../transX_embedding_{}/".format(name)
else:
    export_path = "../transX/"
f = open(export_path + "config.json", 'r')
config = json.loads(f.read())
f.close()

word_vec = np.load(export_path + 'vec.npy') if use_embedding else None
embedding_size=config['word_size'] if use_embedding else 100


print(1,config["entity_total"])
print(2,config["rel_total"])
print(3,len(config["word2id"]))


ll = ctypes.cdll.LoadLibrary   
lib = ll("./init.so")
# lib.setInPath(export_path)
lib.init()
# get model_dir
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir',export_path[:-1]+'_model_dir/','path to store model')

tf.app.flags.DEFINE_float('nbatch_kg',500,'entity numbers used each training time')
tf.app.flags.DEFINE_float('margin',1.0,'entity numbers used each training time')
tf.app.flags.DEFINE_float('learning_rate_kg',0.001,'learning rate for kg')
tf.app.flags.DEFINE_float('ent_total',lib.getEntityTotal(),'total of entities')
tf.app.flags.DEFINE_float('rel_total',lib.getRelationTotal(),'total of relations')
tf.app.flags.DEFINE_float('tri_total',lib.getTripleTotal(),'total of triples')
tf.app.flags.DEFINE_float('katt_flag', 1, '1 for katt, 0 for att')

tf.app.flags.DEFINE_string('model', 'cnn', 'neural models to encode sentences')
tf.app.flags.DEFINE_float('max_length',config['fixlen'],'maximum of number of words in one sentence')
tf.app.flags.DEFINE_float('pos_num', config['maxlen'] * 2 + 1,'number of position embedding vectors')
tf.app.flags.DEFINE_float('num_classes', config['textual_rel_total'],'maximum of relations')

tf.app.flags.DEFINE_float('hidden_size',230,'hidden feature size')
tf.app.flags.DEFINE_float('pos_size',5,'position embedding size')
tf.app.flags.DEFINE_float('batch_size',131*2,'entity numbers used each training time')
tf.app.flags.DEFINE_float('learning_rate',0.1,'entity numbers used each training time')
tf.app.flags.DEFINE_float('weight_decay',0.00001,'weight_decay')
tf.app.flags.DEFINE_float('keep_prob',1.0,'dropout rate')
tf.app.flags.DEFINE_float('test_batch_size',131*2,'entity numbers used each test time')

FLAGS.batch_size=int(FLAGS.batch_size)
FLAGS.test_batch_size=int(FLAGS.test_batch_size)
FLAGS.max_length=int(FLAGS.max_length)
FLAGS.num_classes=int(FLAGS.num_classes)
FLAGS.pos_size=int(FLAGS.pos_size) 
FLAGS.hidden_size=int(FLAGS.hidden_size) 
FLAGS.ent_total=int(FLAGS.ent_total)
FLAGS.rel_total=int(FLAGS.rel_total)
FLAGS.tri_total=int(FLAGS.tri_total)
FLAGS.pos_num=int(FLAGS.pos_num)
FLAGS.nbatch_kg=int(FLAGS.nbatch_kg)
FLAGS.katt_flag=int(FLAGS.katt_flag)


if transE:
    transX='transE'
elif transD:
    transX='transD'
    
print( FLAGS.model_dir+FLAGS.model+str(FLAGS.katt_flag)+transX)

if dev:
    modeX='dev'
else:
    modeX='test'
    
exact_match=0
exact_match3=0
exact_match10=0
exclude_e1=0

pre='/home/xuweijia/my_drqa_up/data/final_test/'
# dev_contain_e_valid_cands_tokenized_all.json
contain_dev_file=pre+'dev_contain_e_valid_cands_tokenized_all.json'
contain_test_file=pre+'test_contain_e_valid_cands_tokenized_all.json'
#contain_dev_file=pre+'dev_contain_e.json'
#contain_test_file=pre+'test_contain_e.json'
if dev:
    with open(contain_dev_file,'r') as f:    # contain e  dev  3011 sp
        dev_samples=json.load(f)
        samples=dev_samples
else:
    with open(contain_test_file,'r') as f:   # contain e  test 2862 sp
        test_samples=json.load(f)
        samples=test_samples

N_entity=config["entity_total"]
N_relation=config["rel_total"]
     
   
eid2idx=config["e_dict"]# # from pure file
pid2idx=config["relation2id"]

sess = tf.Session()
model = network.CNN(use_embedding,embedding_size,is_training = False,word_embeddings = word_vec)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
# saver.save(sess,FLAGS.model_dir+FLAGS.model+str(FLAGS.katt_flag)+transX, global_step=current_step)
# tf.app.flags.DEFINE_string('model_dir','./model/','path to store model')
# tf.app.flags.DEFINE_string('checkpoint_path','./model/','path to store model')
# saver.restore(sess, FLAGS.checkpoint_path + FLAGS.model+str(FLAGS.katt_flag)+"-"+str(3664*iters))
FLAGS.katt_flag=int(FLAGS.katt_flag)
print( FLAGS.model_dir+FLAGS.model+str(FLAGS.katt_flag)+transX+'_pureKB_'+str(pure_KB)+'_epoch'+str(times_kg)+'_'+str(embedding_size))
saver.restore(sess, FLAGS.model_dir+FLAGS.model+str(FLAGS.katt_flag)+transX+'_pureKB_'+str(pure_KB)+'_epoch'+str(times_kg)+'_nkb'+str(FLAGS.nbatch_kg)+'_win'+str(FLAGS.win_size)+'_'+str(embedding_size))

if transE:
    ## sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keep_dims = True)
    entity_embedding=model.word_embedding.eval(session=sess)[:N_entity,:]          # E+E+1+1
    relation_embedding =model.rel_embeddings.eval(session=sess)          # R
    #entity_embedding = np.memmap(filename ,dtype='float32', shape=(N_entity,h),mode='r')
    #relation_embedding = np.memmap(p_filename ,dtype='float32', shape=(N_relation,h),mode='r')
elif transD:
    # transD
    def transferD(e,e_p,r_p):# h,hp,rp
        return e+np.sum(e*e_p)*r_p
    entity_embedding=model.word_embedding.eval(session=sess)[:N_entity,:]        # E+E+1+1,h
    relation_embedding=model.rel_embeddings.eval(session=sess)                   # R,h
    entity_transfer=model.ent_transfer.eval(session=sess)                            # 5,20   E,h
    relation_transfer=model.rel_transfer.eval(session=sess)            # 11,20  R,h
    #entity_embedding = np.memmap(filename ,dtype='float32', shape=(N_entity,h),mode='r')
    # relation_embedding = np.memmap(p_filename ,dtype='float32', shape=(N_relation,h),mode='r')
    #A_embedding=np.memmap(A_filename ,dtype='float32', shape=(N_entity+N_relation,h),mode='r')
    #relation_transfer=A_embedding[:N_relation]
    # entity_transfer=A_embedding[N_relation:]
samples_new=[]
for i,sample in enumerate(samples):
    print(i)
#    e1_id=sample['e1_id']
#    p_id=sample['p_id']
#    ans_id=sample['ans_id']
    e1_id,p_id,ans_id=sample['triple'][0]    

    if eid2idx.get(e1_id)!=None and pid2idx.get(p_id)!=None:
        e1_idx=eid2idx[e1_id]
        p_idx=pid2idx[p_id]
        
        eids=list(eid2idx.keys()) 
        pre_indexs=list(eid2idx.values())    #
        # compute final score
        # transD
        # pos_h_e = self.calc(pos_h_e, pos_h_t, pos_r_t)   e_numpy,e_trans,r_trans
        # sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keep_dims = True)
        # def calc(e, t, r):
		 #    return e + tf.reduce_sum(e * t, 1, keep_dims = True) * r
        if transE:
            final_scores=  np.sum(abs(entity_embedding[e1_idx]+relation_embedding[p_idx]-entity_embedding[pre_indexs]),1)
        elif transD:
            e1_vec=transferD(entity_embedding[e1_idx],entity_transfer[e1_idx],relation_transfer[p_idx])
            p_vec=relation_embedding[p_idx]
            final_scores=[]
            for index in pre_indexs:
                e2_vec=transferD(entity_embedding[index],entity_transfer[index],relation_transfer[p_idx])
                score=np.sum(abs(e1_vec+p_vec-e2_vec))
                final_scores.append(score)
                
        new_index=np.argsort(final_scores)
        predictions=list(np.array(eids)[new_index])
        prediction=predictions[0]
        correct=prediction in ans_id
        exact_match+=correct
        
        prediction=predictions[1] if prediction[0]==e1_id and len(predictions)>1 else predictions[0]
        correct=prediction in ans_id
        exclude_e1+=correct
        
        correct3=len(([p for p in predictions[:3] if p in ans_id]))!=0
        #correct3=any([p for p in predictions[:3] if p in ans_id])
        exact_match3+=correct3
        
        correct10=len(([p for p in predictions[:10] if p in ans_id]))!=0
        # correct10=any([p for p in predictions[:10] if p in ans_id])
        exact_match10+=correct10
        
        if len([e for e in ans_id if e in eids])!= len(ans_id):
            print('what!!!, all ans should in train e_ids')
            print(i)
            break
                    
    total=i+1
#exact_match_exist_rate = 100.0 * exact_match_exist/ total_have
if use_embedding==False:
    name=None
exact_match_rate = 100.0 * exact_match / total
exact_match_rate3 = 100.0 * exact_match3 / total
exact_match_rate10 = 100.0 * exact_match10 / total

exclude_e1_rate= 100.0 * exclude_e1 / total
print('transX:{} mode:{} size:{} name{} pure_KB{} :'.format(transX,modeX,embedding_size,name,pure_KB))
print({'exact_match': exact_match},{'total:':total}) 
print({'exact_match_rate': exact_match_rate})  
print({'exclude_e1_rate': exclude_e1_rate})  
print({'exact_match_rate3': exact_match_rate3})  
print({'exact_match_rate10': exact_match_rate10})  


#4421 4937
#4454 4881
#{'E': 45128} {'T': 67637} {'R': 78}
