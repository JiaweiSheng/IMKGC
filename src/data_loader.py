import numpy as np
import pandas as pd
import torch

from src.knowledgegraph import KnowledgeGraph
from src.utils import get_language_list, get_all_edges, create_subgraph_list

import copy
import os
import pdb


# data_loader
class MKGDataset:
    def __init__(self, args, MAX_SAM=10000000000):
        self.data_dir = args.data_path + args.dataset
        self.entity_dir = self.data_dir + '/entity'
        self.kg_dir = self.data_dir + '/kg'
        self.align_dir = self.data_dir + '/seed_alignlinks'
        self.args = args

        self.MAX_SAM=MAX_SAM
        self.kg_names = args.langs # ['el', 'en', ...]
        self.num_kgs = len(self.kg_names)
    
    def load_data(self):
        kg_objects_dict, subgraph_list, all_entity2kgs, all_entity_global_index = self.create_KG_objects_and_subgraph()
        return kg_objects_dict, subgraph_list, all_entity2kgs, all_entity_global_index
    
    def create_KG_objects_and_subgraph(self):
        kg_objects_dict = {}
        seeds = self.load_align_links() # {(lang1, lang2): torch.LongTensor}
        pre_langs = []
        all_entity2kgs = {}  # 当前实体存在于哪些KG； {1:('en','ja'), ...}
        all_entity_global_index = {}  # 当前语言包括的哪些实体，及其全局id； {'en':[1,2,3], ...}

        for lang in self.kg_names: # 文件名列表
            kg_train_data, kg_val_data, kg_test_data, num_entities, num_relations = self.load_kg_data(lang)
            # 训练三元组，验证三元组，测试三元组，实体数目，关系数目
            
            kg_object = KnowledgeGraph(lang, kg_train_data, kg_val_data, kg_test_data, num_entities, num_relations, self.args.device)
            kg_object.get_global_h_t(seeds, pre_langs, all_entity2kgs, all_entity_global_index) # 处理pre_langs, all_entity2kgs, all_entity_global_index, 以及头尾实体的global id
            kg_objects_dict[lang] = kg_object
            
        self.num_entities = len(all_entity2kgs) # 全局实体数量
        self.num_relations = num_relations # 全局实体数量，因为关系是统一的

        edge_index, edge_type = get_all_edges(self.kg_dir, kg_objects_dict, all_entity_global_index) # edge_index, edge_relation
        graph_dir = '_'.join(self.kg_names)
        subgraph_list_path = os.path.join(self.data_dir, graph_dir, 'subgraph_list.graph')
        if not os.path.exists(subgraph_list_path):
            if not os.path.exists(os.path.join(self.data_dir, graph_dir)):
                os.mkdir(os.path.join(self.data_dir, graph_dir))

            subgraph_list = create_subgraph_list(edge_index, edge_type, self.num_entities, self.args.num_hops, self.args.k) # 以实体为中心的若干阶子图，所有的实体
            # 这应该是一张大图
            # pdb.set_trace()
            torch.save(subgraph_list, subgraph_list_path)
        else:
            print('passed!')
            # subgraph_list = torch.load(subgraph_list_path)
            subgraph_list = []
        
        return kg_objects_dict, subgraph_list, all_entity2kgs, all_entity_global_index
    
    def load_align_links(self):
        seeds = {}
        for f in os.listdir(self.align_dir):
            lang1 = f[:2]
            lang2 = f[3:5]
            links = pd.read_csv(os.path.join(self.align_dir, f), sep='\t',
                                header=None).values.astype(int) # [N_align, 2]
            
            links = torch.LongTensor(links) # 
            links = torch.unique(links, dim=0) 
            seeds[(lang1, lang2)] = torch.LongTensor(links) # [N_align, 2]
        # dict{'el-en':[]}
        return seeds
    

    
    def load_kg_data(self, language):

        train_df = pd.read_csv(os.path.join(self.kg_dir, language + '-train.tsv'), sep='\t', header=None, names=['head', 'relation', 'tail'])[:self.MAX_SAM]
        val_df = pd.read_csv(os.path.join(self.kg_dir, language + '-val.tsv'), sep='\t', header=None, names=['head', 'relation', 'tail'])[:self.MAX_SAM]
        test_df = pd.read_csv(os.path.join(self.kg_dir, language + '-test.tsv'), sep='\t', header=None, names=['head', 'relation', 'tail'])[:self.MAX_SAM]

        print('load_kg_data',len(train_df))
        
        entity_file = open(os.path.join(self.entity_dir, language + '.tsv'), encoding='utf-8')
        num_entities = len(entity_file.readlines())
        entity_file.close()
        
        relation_file = open(os.path.join(self.data_dir, 'relations.txt')) # 和语言无关，是个全局量
        num_relations = len(relation_file.readlines())
        relation_file.close()
        
        triples_train = train_df.values.astype(int)
        triples_val = val_df.values.astype(int)
        triples_test = test_df.values.astype(int)
        # 训练三元组，验证三元组，测试三元组
        return torch.LongTensor(triples_train), torch.LongTensor(triples_val), torch.LongTensor(triples_test), num_entities, num_relations
