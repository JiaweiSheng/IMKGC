#!/usr/bin/env python
# coding: utf-8
import time

import os
from os.path import join
print('Current working dir', os.getcwd())

import numpy as np
import torch
from src.data_loader import MKGDataset
from src.validate import Tester
from src.utils import nodes_to_k_graph, get_k_subgraph_list, get_language_list, get_negative_samples_graph, save_model
import logging
import argparse
import random
from random import SystemRandom
from tqdm import tqdm
from itertools import cycle
import pdb
from transformers import AdamW, get_linear_schedule_with_warmup


def set_logger(model_dir, args):
    '''
    Write logs to checkpoint and console
    '''
    experimentID = int(SystemRandom().random() * 100000)
    log_file = model_dir + "/" + '_'.join(args.langs) + "_train_" + args.model +'_'+args.v+ '_' +str(experimentID) + ".log"
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return experimentID


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models'
    )

    # Data loader related
    parser.add_argument('--remove_language', type=str, default='', help="remove kg")
    parser.add_argument('--k', default=10, type=int, help="how many nominations to consider")
    parser.add_argument('--num_hops', default=2, type=int, help="hop sampling")
    parser.add_argument('--data_path', default="dataset", type=str, help="data path")
    parser.add_argument('--dataset', default="dbp5l", type=str, help="dataset")
    parser.add_argument('--save', default="F", type=str)

    # Training Related
    parser.add_argument('--epoch_each', default=3, type=int, help="epochs for each KG")
    parser.add_argument('--round', default=50, type=int,help="rounds to train")
    parser.add_argument('--lr', '--learning_rate', default=5e-3, type=float, help="learning ratel")
    parser.add_argument('--batch_size', default=200, type=int, help="batch size for training")
    parser.add_argument('--optimizer', type=str, default="Adam", help='Adam, AdamW')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--scheduler', type=str, default="constant", help='constant, linear')
    parser.add_argument('--warmup', default=0, type=int, help="warmup rounds")
    
    # Model related
    parser.add_argument('--margin', default=0.3, type=float)
    parser.add_argument('--dim', default=256, type=int, help = 'kg embedding dimension')
    parser.add_argument('--n_layers_gnn', default=2, type=int,help="GNN layer")
    parser.add_argument('--encoder_hdim_gnn', default=256, type=int, help='dimension of GNN')
    parser.add_argument('--n_heads', default=1, type=int, help="heads of attention")
    parser.add_argument('--pretrain_dim', default=256, type=int, help="")

    # Others
    parser.add_argument('--device', default='cuda:0', type=str, help="which device to use")
    parser.add_argument('--test_batch_size', default=200, type=int, help="batch size for testing")
    parser.add_argument('--MAX_SAM', default=10000000000, type=int, help="subset of data for debugging")

    parser.add_argument('--v_gnn', default='same', type=str, help="")
    parser.add_argument('--v', default='', type=str, help="")
    parser.add_argument('--v_loss', default='', type=str, help="")
    parser.add_argument('--v_fusion', default='mean', type=str, help="")
    parser.add_argument('--v_ent', default='', type=str, help="")
    parser.add_argument('--v_rel', default='', type=str, help="")
    parser.add_argument('--flow', default='', type=str, help="")
    parser.add_argument('--K', default=10, type=int, help="")
    parser.add_argument('--v_note', default='', type=str, help="")

    parser.add_argument('--clamp', default=3., type=float, help = '')
    parser.add_argument('--lw', type=str, default='n',choices=['y', 'n'], help='')
    parser.add_argument('--model', default='imkgc', type=str, help="")
    parser.add_argument('--decoder', default='TransE', type=str, help="")
    parser.add_argument('--fusion', default='mean', type=str, help="")
    parser.add_argument('--temp', type=float, default=0.5, help='temp')

    parser.add_argument('--alpha', type=float, default=0.01, help='gamma')
    parser.add_argument('--beta', type=float, default=0.0001, help='beta')
    parser.add_argument('--gamma', type=float, default=0.001, help='alpha')
    parser.add_argument('--omega', type=float, default=0.01, help='')
    parser.add_argument('--loss_fusion', type=str, default='def', help='assist')

    parser.add_argument('--vq_loss_w', type=float, default=0.0001, help='')
    parser.add_argument('--reason_step', default=2, type=int, help="")
    parser.add_argument('--commit_loss', default=0.5, type=float, help="")
    parser.add_argument('--codebook_ratio', default=1., type=float, help="")
    return parser.parse_args(args)


def train_kgs(args, all_kgs, kg_objects_dict, kgname2idx, optimizer, num_epoch, model, scheduler=None):
    max_data = 0
    kg_dataloader_list = {}
    lang_nbatch = {}
    for lang in all_kgs:
        kg = kg_objects_dict[lang]
        kg_index = kgname2idx[lang]
        kg_dataloader = kg.generate_batch_data(kg.h_train, kg.r_train, kg.t_train, batch_size=args.batch_size, shuffle=True)
        kg_dataloader_list[lang] = cycle(kg_dataloader)
        max_data = max(max_data, len(kg_dataloader))
        lang_nbatch[lang] = len(kg_dataloader)
        logging.info('Lang {}: nbatch {}'.format(lang, len(kg_dataloader)))
    logging.info('Largest data has {} batches'.format(max_data))

    if args.lw == 'y':
        lang_lw = {}
        for lang in all_kgs:
            lang_lw[lang] = lang_nbatch[lang] / max_data
            logging.info('{}, {}, {}, {}'.format(lang, lang_nbatch[lang], max_data, lang_lw[lang]))

    for one_epoch in range(num_epoch): # 3
        logging.info('Epoch {:d}'.format(one_epoch))
        kg_loss = []
        kld_loss = []
        all_loss = []

        for i in range(max_data):
            random.shuffle(all_kgs)
            for lang in all_kgs:
                time0 = time.time()
                kg_dataloader = kg_dataloader_list[lang]
                kg_each = next(kg_dataloader)

                kg = kg_objects_dict[lang]
                kg_index = kgname2idx[lang]

                h_graph_batch_list = nodes_to_k_graph(kg.k_subgraph_list, kg_each[:, 0], args.device) # [2, ]
                t_graph_batch_list = nodes_to_k_graph(kg.k_subgraph_list, kg_each[:, 2], args.device) 
                
                batch_size = kg_each.shape[0]
                t_neg_index = get_negative_samples_graph(batch_size, kg.num_entities)
                t_neg_graph_batch_list = nodes_to_k_graph(kg.k_subgraph_list, t_neg_index, args.device)
                
                kg_each = kg_each.to(args.device)
                
                optimizer.zero_grad()
                # h_graph_batch_list, kg_each, t_graph_batch_list, t_neg_graph_batch_list, kg_index
                total_loss = model.forward_kg(h_graph_batch_list, kg_each, t_graph_batch_list, t_neg_graph_batch_list, kg_index)

                loss = total_loss['kg_loss'] + total_loss['kld_loss'] * args.beta +  total_loss['kg_cur_loss'] * args.alpha + total_loss['kg_assist_loss'] * args.omega +  total_loss['info_contrastive'] * args.gamma + total_loss['vq_loss'] * args.vq_loss_w

                if args.lw == 'y':
                    loss = loss * lang_lw[lang]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                all_loss.append(loss.item())
                kg_loss.append(total_loss['kg_loss'].item())
                kld_loss.append(total_loss['kld_loss'].item())
                
                
                if i % 10 == 0:
                    logging.info('Step {}: Lang: {}, Train KG Loss: all_loss {:.6f}, kg_loss {:.6f}, kg_cur_loss {:.6f}, kg_assist_loss {:.6f}, kld_loss {:.6f}, info_contrastive {:.6f}, vq_loss {:.6f}'.format(i, lang, loss.item(), total_loss['kg_loss'].item(), total_loss['kg_cur_loss'].item(), total_loss['kg_assist_loss'].item(), total_loss['kld_loss'].item(), total_loss['info_contrastive'].item(),total_loss['vq_loss'].item()))
                
                del loss
                torch.cuda.empty_cache()

        logging.info('Epoch {:d} [Train KG Loss: all_loss {:.6f}, kg_loss {:.6f}, kld_loss {:.6f}]'.format(one_epoch, np.mean(all_loss), np.mean(kg_loss), np.mean(kld_loss)))
        logging.info('\n')

def main(args):
    args.device = torch.device(args.device)
    args.entity_dim = args.dim
    args.relation_dim = args.entity_dim
    all_kgs = get_language_list(args.data_path + args.dataset + '/entity')

    if args.dataset == 'dbp5l':
        all_kgs = ['el', 'en', 'es', 'fr', 'ja'] # dbp5l
    elif args.dataset == 'depkg':
        all_kgs = ['de', 'es', 'fr', 'it', 'jp', 'uk'] # depkg
    elif args.dataset == 'dwy':
        all_kgs = ['db', 'wk', 'yg'] # dwy

    remove_lang = args.remove_language
    if remove_lang != '':
        remove_lang = remove_lang.split(',')
        for la in remove_lang:
            all_kgs.remove(la)

    print(f"Number of KGs is {len(all_kgs)}")
    args.langs = all_kgs
    kgname2idx = {}
    for i in range(len(all_kgs)):
        kgname2idx[all_kgs[i]] = i
    
    model_dir = join('./' + args.dataset + "/trained_model", '_'.join(all_kgs))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    experimentID = set_logger(model_dir, args)  # set logger
    logging.info('logger setting finished')
    
    # load data
    dataset = MKGDataset(args, MAX_SAM=args.MAX_SAM)
    
    kg_objects_dict, subgraph_list, all_entity2kgs, all_entity_global_index = dataset.load_data()  
    logging.info('subgraph_list loaded')

    all_entity2kgidx = {} 
    for global_id, langs in all_entity2kgs.items():
        langidx = set([kgname2idx[l] for l in langs])
        all_entity2kgidx[global_id] = langidx

    graph_dir = '_'.join(args.langs)
    for lang in kg_objects_dict.keys():
        kg_lang = kg_objects_dict[lang]
        kg_index = kgname2idx[lang] 
        node_index = kg_lang.entity_global_index  # entity global id
        k_subgraph_list = get_k_subgraph_list(subgraph_list, node_index, kg_index, dataset.num_kgs, all_entity2kgidx, dataset.num_entities, os.path.join(dataset.data_dir, graph_dir))
        logging.info('kg' + str(kg_index) + '_k_subgraph_list loaded')
        kg_lang.k_subgraph_list = k_subgraph_list
    args.num_entities = dataset.num_entities
    args.num_relations = dataset.num_relations
    args.num_kgs = dataset.num_kgs

    args.kgname2idx = kgname2idx

    del subgraph_list

    args.entity_dim = args.dim
    args.relation_dim = args.entity_dim

    # logging
    logging.info('remove language: %s' % (remove_lang))
    logging.info(f'languages: {args.langs}')
    logging.info(f'device: {args.device}')
    logging.info(f'batch_size: {args.batch_size}')
    logging.info(f'k: {args.k}')
    logging.info(f'num_hops: {args.num_hops}')
    logging.info(f'lr: {args.lr}')
    logging.info(f'margin: {args.margin}')
    logging.info(f'dim: {args.dim}')
    logging.info(f'experimentID: {experimentID}')
    logging.info(f'MAX_SAM: {args.MAX_SAM}')

    # Build Model
    if args.model == 'imkgc':
        from src.imkgc import IMKGC
        model = IMKGC(args).to(args.device)
    else:
        assert True, 'unimplemented'
    
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    num_steps = 0
    for lang in kg_objects_dict.keys():
        kg_lang = kg_objects_dict[lang]
        kg_dataloader = kg_lang.generate_batch_data(kg_lang.h_train, kg_lang.r_train, kg_lang.t_train, batch_size=args.batch_size, shuffle=True)
        num_steps += len(kg_dataloader)
    args.num_steps = num_steps

    if args.scheduler == 'constant':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
    elif args.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.epoch_each * args.num_steps*args.warmup, num_training_steps=args.round * args.epoch_each * args.num_steps)
    else:
        raise NotImplementedError
    logging.info('model initialization done')

    validator = Tester(args, kg_objects_dict, model, args.device, args.data_path + args.dataset)
    
    best_mrr = 0
    best_result = {}
    all_kgs_out = list.copy(all_kgs)

    logging.info(f'=== experimentID {experimentID} ===')
    for i in range(args.round):
        logging.info(f'Round: {i} begin!')
        model.train()

        train_kgs(args, all_kgs, kg_objects_dict, kgname2idx, optimizer, args.epoch_each, model, scheduler=scheduler)
        logging.info(f'round : {i} finished!')

        model.eval()
        with torch.no_grad():
            metrics_test2 = validator.test(is_val=False, is_filtered=True)  # Test set
            filename = "experiment_" + str(experimentID) + "_epoch_" + str(i) + '.ckpt'
            # save_model(model, model_dir, filename, args)

            mean_mrr = np.mean([metrics_test2[lang][2].item() for lang in all_kgs])
            logging.info(f'cur epoch: {i}, cur mean mrr: {mean_mrr}!')
            logging.info(f'Round {i} finished!')

            if best_mrr < mean_mrr:
                best_mrr = mean_mrr
                best_epoch = i
                best_result = metrics_test2
                best_filename = filename
                if args.save == 'T':
                    save_model(model, model_dir, best_filename, args)

            logging.info(f'best epoch: {best_epoch}, best mean mrr: {best_mrr}!')
            for lang in all_kgs_out:
                logging.info('{} filterd: {:.4f}, {:.4f}, {:.4f}'.format(lang, best_result[lang][0], best_result[lang][1], best_result[lang][2]))


if __name__ == "__main__":
    main(parse_args())
