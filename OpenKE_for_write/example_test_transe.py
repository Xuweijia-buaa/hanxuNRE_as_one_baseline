import config
import models
import numpy as np
import json
#(1) Set import files and OpenKE will automatically load models via torch.load().
con = config.Config()
con.set_in_path("./benchmarks/FB15K/")　　　　# 测试数据集所在位置
con.set_test_link_prediction(True)        # 测试link predict
con.set_test_triple_classification(True)  # 测试triple分类
con.set_work_threads(4)
con.set_dimension(100)
# 直接加载训练好的模型，比如./res/model.pt  　训练集对应"./res/model.vec.pt"  no need to load para, paras are in original model
con.set_import_files("./res/transe.pt")
con.init()
con.set_model(models.TransE)
con.test()
'''
#(2) Read model parameters from json files and manually load parameters. 
con = config.Config()
con.set_in_path("./benchmarks/FB15K/")
con.set_test_link_prediction(True)
con.set_test_triple_classification(True)
con.set_work_threads(4)
con.set_dimension(100)
con.init()
con.set_model(models.TransE)
f = open("./res/embedding.vec.json", "r")　# 先加载embedding vector
content = json.loads(f.read())   # a dict, include two embeddings
f.close()
con.set_parameters(content)               # 再把参数加载到模型上
con.test()

#(3) Manually load models via torch.load()
con = config.Config()
con.set_in_path("./benchmarks/FB15K/")
con.set_test_link_prediction(True)
con.set_test_triple_classification(True)
con.set_work_threads(4)
con.set_dimension(100)
con.init()
con.set_model(models.TransE)
con.import_variables("./res/model.vec.pt")  #torch.load().加载模型
con.test()
'''
