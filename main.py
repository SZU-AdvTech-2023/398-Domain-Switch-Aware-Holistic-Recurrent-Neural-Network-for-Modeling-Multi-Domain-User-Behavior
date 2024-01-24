# -*- coding: utf-8 -*-



import os
import time

import numpy as np
import pandas as pd
import torch
import argparse
import json
from tqdm import tqdm

from GRU4Rec import GRU4Rec
from utils import *
import random

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--datasets_information', default='config_3.json', type=str)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--weight_decay', default=0.1, type=float)
parser.add_argument('--top_n', default=10, type = int)
parser.add_argument('--interval', default=50, type = int)
parser.add_argument('--early_stop', default=1, type = int)
parser.add_argument('--supplement_loss_weight', default=0.1, type = float)
parser.add_argument('--behavior_regularizer_weight', default=0.1, type = float)



args = parser.parse_args()

# # to ensure reproduction
def setup_seed(seed):
    torch.manual_seed(seed) # initialize CPU,GPU by seed
    torch.cuda.manual_seed(seed) # initialize current GPU by seed
    torch.cuda.manual_seed_all(seed) # initialize all GPUs by seed
    np.random.seed(seed) # initialize numpy by seed
    random.seed(seed) # initialize random library by seed
    torch.backends.cudnn.deterministic = True # avoid the algorithm isn't the deterministic algorithm

    os.environ['PYTHONHASHSEED'] = str(seed)

SEED = 2020
setup_seed(SEED)


if __name__ == '__main__':

    hyper_parameters = vars(args)
    t_save = time.gmtime()
    # notes = 'DS_HRNN_slw_behavior_regularizer'
    notes = 'GRU_Rec_sl_br'  # real l2_regularizer
    information = time.strftime('%Y_%m_%d_%H_%M_%S', t_save) + '-' + notes
    hyper_parameters['information'] = information  # record_information


    # get the information of the dataset
    datasets_information = json.load(open(args.datasets_information, 'r', encoding='utf-8'))
    # make files for saving results
    for domain in datasets_information['domains']:
        if not os.path.isdir(os.path.join(notes, domain + '_' + args.train_dir)):
            os.makedirs(os.path.join(notes, domain + '_' + args.train_dir))
    # global dataset
    dataset = data_partition(datasets_information,args)

    [item_sets, user_sets, max_item_id, max_user_id, item_set_in_all_domains, user_set_in_all_domains,domain_specific_user_train, domain_invariant_user_train, neg_cand, user_valid, user_test] = dataset
    num_batch = len(domain_invariant_user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    len_domains = {}
    # get the average training sequence length in all domains
    cc = 0.0
    for u in domain_invariant_user_train:
        cc += len(domain_invariant_user_train[u]['seq'])
    print('average training sequence length in all domains: %.2f' % (cc / len(domain_invariant_user_train)))
    len_domains['all'] = cc / len(domain_invariant_user_train)
    # get the average training sequence length in each domain
    cc = 0.0
    for domain in datasets_information['domains']:
        cc = 0.0
        count_seq = 0 # the count of the sequence
        for u in domain_specific_user_train:
            cc += len(domain_specific_user_train[u][domain]['seq'])
            count_seq += 1
        print('average training sequence length in %s : %.2f' % (domain, cc / count_seq))
        len_domains[domain] = cc / count_seq

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    sampler = WarpSampler(domain_invariant_user_train, user_set_in_all_domains, item_sets, batch_size=args.batch_size,
                          maxlen=args.maxlen, n_workers=3)

    model = GRU4Rec(max_user_id, max_item_id, args).to(args.device) # no ReLU activation in original SASRec implementation?

    # create dictionary for saving results
    ans_list = {}
    for domain in datasets_information['domains']:
        ans_list[domain] = []

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers

    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)

    model.train() # enable model training

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()


    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()
    count = 0 # early_stop

    # save the results in each domain
    best_result = {} # record best result during training
    val_hr_10, val_ndcg_10, test_hr_10, test_ndcg_10 = {},{},{},{}
    # initialize the best result to 0
    for domain in datasets_information['domains']:
        best_result[domain] = 0
        val_hr_10[domain],val_ndcg_10[domain] = 0,0
        test_hr_10[domain],test_ndcg_10[domain] = 0,0






    # training, validating, testing model
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1), ncols=80):
        if args.inference_only: break  # just to decrease identition
        sum_loss = 0
        for step in range(num_batch):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg, pos_domain_switch, domain_switch_behavior, next_behavior = sampler.next_batch()  # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_domain_switch = np.array(pos_domain_switch)
            domain_switch_behavior, next_behavior = np.array(domain_switch_behavior), np.array(next_behavior)
            pos_logits, neg_logits, pos_domain_switch_logits = model(u, seq, pos, neg, pos_domain_switch)
            behavior_regularizer_loss = model.behavior_regularizer(domain_switch_behavior, next_behavior)

            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape,
                                                                                                   device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            supplement_indices = np.where(pos_domain_switch != 0)
            supplement_loss = bce_criterion(pos_domain_switch_logits[supplement_indices], pos_labels[supplement_indices])
            loss += supplement_loss * args.supplement_loss_weight
            loss += behavior_regularizer_loss * args.behavior_regularizer_weight
            sum_loss += loss
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            optimizer.step()
            # print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs

        if epoch % args.interval == 0:
            args.training = False
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            flag = False #whether the performance of the proposed model improves
            print(t_valid,t_test)
            for domain in datasets_information['domains']:
                ans_list[domain].append([t_valid[domain][0], t_valid[domain][1],
                                         t_test[domain][0], t_test[domain][1]])
                print('domain:%s, epoch:%d, time: %f(s), valid (NDCG@%d: %.4f, HR@%d: %.4f), test (NDCG@%d: %.4f, HR@%d: %.4f)'
                      % (domain, epoch, T, args.top_n, t_valid[domain][0], args.top_n, t_valid[domain][1],
                         args.top_n, t_test[domain][0], args.top_n,
                         t_test[domain][1]))

                if t_valid[domain][0] + t_valid[domain][1] > best_result[domain]:
                    flag = True
                    # update results in all domain
                    for temp_domain in datasets_information['domains']:
                        best_result[temp_domain] = t_valid[temp_domain][0] + t_valid[temp_domain][1]
                        val_ndcg_10[temp_domain], val_hr_10[temp_domain], test_ndcg_10[temp_domain], test_hr_10[temp_domain]\
                            = t_valid[temp_domain][0], t_valid[temp_domain][1], t_test[temp_domain][0], t_test[temp_domain][1]

            if flag:
                count = 0
            else:
                count += 1 #the performance of the proposed model doesn't improve in this evaluation


            t0 = time.time()
            model.train()
            print('loss = ', sum_loss)
            args.training = True

        if count > args.early_stop:
            break

        # save the best results in all domain
    best_result_domains = {}
    parameters_domains = {}

    for domain in datasets_information['domains']:
        folder = domain + '_' + args.train_dir
        fname = '{}.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
        fname = fname.format(information, args.num_epochs, args.lr,
                             args.num_blocks, args.num_heads,
                             args.hidden_units,
                             args.maxlen)
        torch.save(model.state_dict(), os.path.join(notes, folder, fname))

        df = pd.DataFrame(data=ans_list[domain],
                          columns=['val_NDCG@{0:}'.format(args.top_n), 'val_HR@{0:}'.format(args.top_n),
                                   'test_NDCG@{0:}'.format(args.top_n), 'test_HR@{0:}'.format(args.top_n)])
        df.to_csv(path_or_buf=os.path.join(notes, domain + '_' + args.train_dir, 'result{0:}.csv'.format(information)),
                  index=False)
        # deleting something results in errors
        # f.close()
        # different join
        with open(os.path.join(notes, domain + '_' + args.train_dir, 'args{0:}.txt'.format(information)), 'w') as f:
            f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
        f.close()
        sampler.close()

        parameters_in_this_domain = {
            'sequence_length_in_current_domain': len_domains[domain],
            'sequence_length_in_all_domains': len_domains['all']
        }
        parameters_domains[domain] = {
            **hyper_parameters, **parameters_in_this_domain
        }
        with open(os.path.join(notes, domain + '_' + args.train_dir, '{0:}_parameters.json'.format(information)),
                  'w') as f:
            json.dump(
                parameters_domains[domain], f
            )

        with open(os.path.join(notes, domain + '_' + args.train_dir, '{0:}_results.json'.format(information)),
                  'w') as f:
            json.dump(
                {
                    'val_ndcg_10': val_ndcg_10[domain],
                    'val_hr_10': val_hr_10[domain],
                    'test_ndcg_10': test_ndcg_10[domain],
                    'test_hr_10': test_hr_10[domain]
                }, f
            )
        best_result_domains[domain] = {}
        best_result_domains[domain]['val_ndcg_10'] = val_ndcg_10[domain]
        best_result_domains[domain]['val_hr_10'] = val_hr_10[domain]
        best_result_domains[domain]['test_ndcg_10'] = test_ndcg_10[domain]
        best_result_domains[domain]['test_hr_10'] = test_hr_10[domain]

    if not os.path.isdir(os.path.join(notes, 'all')):
        os.makedirs(os.path.join(notes, 'all'))

    with open(os.path.join(notes, 'all', '{0:}_parameters.json'.format(information)), 'w') as f:
        json.dump(parameters_domains, f)

    with open(os.path.join(notes, 'all', '{0:}_results.json'.format(information)), 'w') as f:
        json.dump(best_result_domains, f)

    table_result = pd.DataFrame(best_result_domains)
    table_result.to_csv(path_or_buf=os.path.join(notes, 'all' , 'table_result{0:}.csv'.format(information)),
                  index=False)

    print("Done")

