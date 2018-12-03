#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 22:55:40 2018

@author: xuweijia
"""
import subprocess
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 16:17:42 2018

@author: xuweijia
"""
import json
import numpy as np
# train_file='/home/xuweijia/my_drqa/train_before_statis.json'
# '/media/xuweijia/06CE5DF1CE5DD98F/useful_old_data/new_NER_data/e_30/train_20w_list.json'
pre='/home/xuweijia/my_drqa_up/data/final_train/'
train_file=pre+'train_tokenized.json'
with open(train_file,'r') as f:
    samples=json.load(f)

max_L=0
for sample in samples:
    if len(sample['document'])>max_L:
        max_L=len(sample['document'])
#length of sentence
fixlen = max_L


transX_dir='transX'
subprocess.call(['mkdir', '-p', transX_dir])

#max length of position embedding is 100 (-100~+100)
maxlen = 100
def pos_embed(x):
	return max(0, min(x + maxlen, maxlen + maxlen + 1))
# word_dict: first half Q, second half w (include mention)
import copy
def build_dict(samples):
    e_set=set() # 58617
    r_set=set()
    triple_set=set()
    word_set=set()
    word_dict=dict()
    for sample in samples:
        e1_id,p_id,e2_id=sample['triple'][0]
        e_set.add(e1_id)
        e_set.add(e2_id)
        r_set.add(p_id)
        triple_set.add((e1_id,p_id,e2_id))  # only id that matters
        phrase_tokens=[w.lower() for w in sample['all_Q_tokens']['phrase_tokens']]
        # phrase_tokens=[w.lower() for w in sample['phrase_tokens']]
        word_set|=set(phrase_tokens)
        
    r_list=list(r_set)
    r_dict=dict(zip(r_list,range(len(r_list))))
    
    e_list=list(e_set)
    e_dict=dict(zip(e_list,range(len(e_list))))
    
    E=len(e_set)
    V=len(word_set)
    R=len(r_set)
    word_dict=copy.deepcopy(e_dict)
    for w in word_set:
        word_dict[w]=len(word_dict)
        
    word_dict['UNK'] = len(word_dict)
    word_dict['BLANK'] = len(word_dict)
    
    I=len(word_dict)
    T=len(triple_set) 
    
    f_train=open(transX_dir+'/triple2id.txt','w')
    f_e=open(transX_dir+'/entity2id.txt','w')
    f_r=open(transX_dir+'/relation2id.txt','w')
    f = open(transX_dir+'/config.json', "w")
    f.write(json.dumps({"word2id":word_dict,"relation2id":r_dict,"e_dict":e_dict, "fixlen":fixlen, "maxlen":maxlen, "entity_total":E, "word_total":I,"T":T, "rel_total":R, "textual_rel_total":R}))
    f.close()
    
    # triple e1,e2,r
    f_train.write(str(len(triple_set))+'\n') # pure id 
    triple_list=list(triple_set)
    for e1,p,e2 in triple_list:
        e1_id=word_dict[e1]
        e2_id=word_dict[e2]
        r_id=r_dict[p]
        f_train.write(str(e1_id)+'\t'+str(e2_id)+'\t'+str(r_id)+'\n')
    f_train.close()
    
    # Qid idx
    f_e.write(str(E)+'\n')
    for e in e_list:# original label 
        f_e.write(e+'\t'+str(e_dict[e])+'\n')                      # wiki  eid  -->  idx
    f_e.close()
    
    # Pid idx
    f_r.write(str(R)+'\n')
    for r in r_list:
        f_r.write(r+'\t'+str(r_dict[r])+'\n')
    f_r.close()   
    return word_dict,e_dict,r_dict,E,V,I,T,triple_set

word_dict,e_dict,r_dict,E,V,I,T,triple_set=build_dict(samples)
id2w=dict(zip(word_dict.values(),word_dict.keys()))

def sort_file(file):
    with open(file,'r') as f:
        samples=json.load(f)
    hash = {}
    for sample in samples:
        e1_id,p_id,e2_id=sample['triple'][0]  
        id1 = (e1_id,e2_id)
        id2 = p_id
        if not id1 in hash:
            hash[id1] = {}
        if not id2 in hash[id1]:
            hash[id1][id2] = []
        hash[id1][id2].append(sample) 
    new_list=[]
    for i in hash:          # each unique e1,e2
        for j in hash[i]:   # this e1,e2's r_idx
            for k in hash[i][j]:  # all line belong to e1,e2,r_idx
                new_list.append(k)
    with open(transX_dir+'/train_sort.json','w') as f:
        json.dump(new_list,f)

sort_file(train_file)

# "train_sort",  [textual_rel_total, rel_total]
def init_train_files(word_dict,r_dict,e_dict):
    print ('reading ' +' data...')
    with open(transX_dir+'/train_sort.json','r') as f:
        new_list=json.load(f)
    total = len(new_list)                    # sample number
    # n, 120 tokens(pos)
    sen_word = np.zeros((total, fixlen), dtype = np.int32) # each doc token's idx in word_dict (first 120 tokens)
    sen_pos1 = np.zeros((total, fixlen), dtype = np.int32) # each pos's relative pos to first  e  +100  (number restict to 0-201)
    sen_pos2 = np.zeros((total, fixlen), dtype = np.int32) # each pos's relative pos to second e  +100  (just accordding to emerge pos, no matter e1/e2)
    sen_mask = np.zeros((total, fixlen), dtype = np.int32) # sen's mask   before e1,e1-e2,after e2: mask 1,2,3    less than 120 tokens-->0
    sen_len = np.zeros((total), dtype = np.int32)          # real length for each sample's doc (120 or lower)
    sen_label = np.zeros((total), dtype = np.int32)        # relation idx for each sample
    sen_head = np.zeros((total), dtype = np.int32)         # Q1's idx for each sample
    sen_tail = np.zeros((total), dtype = np.int32)         # Q2's idx for each sample
                                                           # t1,t1,t2,t2,t2,t3,t4,t4
                                                           # len(triples)
    instance_scope = []                                    # each triple's samples sapn [0,1],[2,4],[5,5],[6,7]
    instance_triple = []                                   # all unique triples:  [(Q1,Q2,relation_idx1),(Q3,Q4,relation_idx2),(Q5,Q6,relation_idx3)]
    # each ex
    for s,sample in enumerate(new_list):
       en1_id,p_id,en2_id=sample['triple'][0]
       # ['Q_tokens_rep_mention'],['good'],['e1_mention'],['ans_mention'],['e1_pos'],['e2_pos']
       # sentence = sample['phrase_tokens']
       Q_sentence=sample['all_Q_tokens']['Q_tokens']
       #Q_sentence=sample['Q_tokens_rep_mention']
       relation = r_dict[p_id]
        
       en1pos = sample['all_Q_tokens']['e1_pos']
       en2pos = sample['all_Q_tokens']['e2_pos']
       sen_head[s] = word_dict[en1_id]
       sen_tail[s] = word_dict[en2_id]  # Q2's idx
        # first entity pos in sentence tokens
       en_first = min(en1pos,en2pos)
        # second entity pos in sentence tokens
       en_second = en1pos + en2pos - en_first
       #  pos
       for i in range(fixlen):
           sen_word[s][i] = word_dict['BLANK']       # len word_dict
           # restrict pos in -100-101 --> 0-201     others+100
           sen_pos1[s][i] = pos_embed(i - en1pos)
           sen_pos2[s][i] = pos_embed(i - en2pos)
           if i >= len(Q_sentence):    # doc < 100
               sen_mask[s][i] = 0
            # doc >100
           elif i - en_first<=0:    # before e1
               sen_mask[s][i] = 1
           elif i - en_second<=0:   # between e1,e2
               sen_mask[s][i] = 2
           else:                    # after e2
               sen_mask[s][i] = 3
       for i, word in enumerate(Q_sentence):
            if i >= fixlen:
                break
            if word in e_dict:
                sen_word[s][i] = word_dict[word]
            else:
                word=word.lower()
            if not word in word_dict:
                sen_word[s][i] = word_dict['UNK']
            else:
                sen_word[s][i] = word_dict[word]     # meet Q1,Q2,  Q1,Q2's idx
       sen_len[s] = min(fixlen, len(Q_sentence))
       sen_label[s] = relation
		#put the same entity pair sentences into a dict
       # Q1,Q2,relation_idx
       tup = (en1_id,en2_id,relation)
        # if have new_triple, append into instance_triple   [(Q1,Q2,relation_idx1),(Q3,Q4,relation_idx2),(Q5,Q6,relation_idx3)]
        # new triple come, get [s,s] --> when finish this triple, become [s,s+number_t]
       if instance_triple == [] or instance_triple[len(instance_triple) - 1] != tup:
           instance_triple.append(tup)
           instance_scope.append([s,s])
       instance_scope[len(instance_triple) - 1][1] = s
       if (s+1) % 100 == 0:
           print (s)
    return np.array(instance_triple), np.array(instance_scope), sen_len, sen_label, sen_word, sen_pos1, sen_pos2, sen_mask, sen_head, sen_tail

export_path=transX_dir+'/'
instance_triple, instance_scope, train_len, train_label, train_word, train_pos1, train_pos2, train_mask, train_head, train_tail = init_train_files(word_dict,r_dict,e_dict)
np.save(export_path+'train_instance_triple', instance_triple)
np.save(export_path+'train_instance_scope', instance_scope)
np.save(export_path+'train_len', train_len)
np.save(export_path+'train_label', train_label)
np.save(export_path+'train_word', train_word) # Q_doc , if meet entity, is Q's id, not mention's id.    share  embedding with mention (but mention it self is rare)
np.save(export_path+'train_pos1', train_pos1)
np.save(export_path+'train_pos2', train_pos2)
np.save(export_path+'train_mask', train_mask)
np.save(export_path+'train_head', train_head)
np.save(export_path+'train_tail', train_tail)
