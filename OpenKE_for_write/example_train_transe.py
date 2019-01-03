import config
import models
import json

con = config.Config()
#Input training files from benchmarks/FB15K/ folder.
# 7个文件放在一起　
con.set_in_path("./benchmarks/FB15K/")
#True: Input test files from the same folder.
con.set_log_on(1)
con.set_work_threads(8)
con.set_train_times(10) #(Epoch)
con.set_nbatches(100)   #每个ｅｐｏｃｈ　训练多少次	
con.set_alpha(0.001)    # learning rate
con.set_bern(0)         # 0:use the traditional sampling method  1: use the method in (Wang et al. 2014) denoted
con.set_dimension(100)  # dimensions of entity and relation embeddings. 
con.set_margin(1.0)
con.set_ent_neg_rate(1) # entity sample rate
con.set_rel_neg_rate(0)
con.set_opt_method("SGD")
#模型参数　Model parameters will be exported via torch.save() automatically.
con.set_export_files("./res/model.vec.pt")
#最后训练出的向量　Model parameters will be exported to json files automatically. a dict, entity+ embdeeing + relation embeddding
con.set_out_files("./res/embedding.vec.json")
con.init()
con.set_model(models.TransE)
con.run()
