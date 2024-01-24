import pandas as pd
import numpy as np
import json
from multiprocessing import Process, Queue
import copy
from tqdm import tqdm
import os

# set map domain_name into domain_id and vice versa
domain_map = {1:"Books", 2:"CDs_and_Vinyl", 3:"Movies_and_TV"}
reversed_domain_map = {"Books":1, "CDs_and_Vinyl":2, "Movies_and_TV":3}
# domain_map = {1:"Books"}
# reversed_domain_map = {"Books":1}




# sampler for batch generation
# get an item in item_set but not in ts
def random_neq(item_set, ts):
    t = np.random.choice(item_set)
    while t in ts:
        t = np.random.choice(item_set)
    return t


# get negative sequence, positive sequence and position sequence for each user
def sample_function(domain_invariant_user_train, user_set_in_all_domains, item_sets, batch_size, maxlen, result_queue,
                    SEED):
    def sample():
        user = np.random.choice(user_set_in_all_domains)

        # the user doesn't appear in the training dataset
        while user not in domain_invariant_user_train.keys() or len(domain_invariant_user_train[user]) <= 1:
            user = np.random.choice(user_set_in_all_domains)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        seq_domain_switch_flag = np.zeros([maxlen], dtype=np.bool_)
        # domain-switch in this position. -> 1
        # 它可能会被保留为布尔类型，而不是被隐式地转换为其他类型。
        pos_single_domain_compress = np.zeros([maxlen], dtype=np.int32) # The next interacted item in the same domain
        nxt = domain_invariant_user_train[user]['seq'][-1]
        nxt_dom = domain_invariant_user_train[user]['domain'][-1]
        idx = maxlen - 1

        nxt_single_domain_item = {} # the next predicted item in this domain
        nxt_single_domain_id = {} # record the position of next predicted item

        nxt_single_domain_item[nxt_dom] = domain_invariant_user_train[user]['seq'][-1]
        nxt_single_domain_id[nxt_dom] = maxlen - 1

        # get the seq in each domain
        seq_single_domain = np.zeros([len(reversed_domain_map.keys()), maxlen], dtype=np.int32)
        # pos_single_domain = np.zeros([len(reversed_domain_map.keys()), maxlen], dtype=np.int32)
        # neg_single_domain = np.zeros([len(reversed_domain_map.keys()), maxlen], dtype=np.int32)

        pos_single_domain_compress = np.zeros([maxlen], dtype=np.int32)
        neg_single_domain_compress = np.zeros([maxlen], dtype=np.int32)

        domain_switch_behavior = np.zeros([len(reversed_domain_map.keys()), maxlen], dtype=np.int32) # the nest item, the domain is same as the output
        next_behavior = np.zeros([len(reversed_domain_map.keys()), maxlen], dtype=np.int32) # the domain is different as the output
        domain_switch_behavior_id = {} # record the index of the last item id in domain_switch_behavior
        next_behavior_id = {} # record the index of the last item id in next behavior




        # [...,last but one]
        ts = set(domain_invariant_user_train[user]['seq'])
        for item_id, domain_id in zip(reversed(domain_invariant_user_train[user]['seq'][:-1]),
                                      reversed(domain_invariant_user_train[user]['domain'][:-1])):
            seq[idx] = item_id
            pos[idx] = nxt
            seq_domain_switch_flag[idx] = (domain_id != nxt_dom)
            if seq_domain_switch_flag[idx]:
                if nxt_dom not in next_behavior_id:
                    next_behavior_id[nxt_dom] = maxlen - 1
                    next_behavior[nxt_dom-1, next_behavior_id[nxt_dom]] = item_id
                else:
                    next_behavior_id[nxt_dom] -= 1
                    next_behavior[nxt_dom - 1, next_behavior_id[nxt_dom]] = item_id

                if domain_id not in domain_switch_behavior_id and domain_id in next_behavior_id:
                    domain_switch_behavior_id[domain_id] =  maxlen - 1
                    domain_switch_behavior[domain_id-1, domain_switch_behavior_id[domain_id]] = item_id
                elif domain_id in domain_switch_behavior_id and domain_id in next_behavior_id:
                    domain_switch_behavior_id[domain_id] -= 1
                    domain_switch_behavior[domain_id - 1, domain_switch_behavior_id[domain_id]] = item_id


                # if domain_id not in behavior_regularizer_id:
                #     behavior_regularizer_id[domain_id] = maxlen - 1
                #     next_behavior[domain_id-1,behavior_regularizer_id[domain_id]] = item_id
                # else:
                #     domain_switch_behavior[domain_id-1, behavior_regularizer_id[domain_id]] = item_id #[...,B]
                #     behavior_regularizer_id[domain_id] -= 1
                #     next_behavior[domain_id-1,behavior_regularizer_id[domain_id]] = item_id #[...,B,A]

                # the item is domain switches
            if nxt != 0: neg[idx] = random_neq(item_sets[domain_map[domain_id]], ts)
            if domain_id not in nxt_single_domain_id.keys():
                nxt_single_domain_id[domain_id] = idx
                nxt_single_domain_item[domain_id] = item_id
                seq_single_domain[domain_id-1, idx] = item_id
            else:
                seq_single_domain[domain_id - 1, idx] = item_id
                pos_single_domain_compress[idx] = nxt_single_domain_item[domain_id]
                neg_single_domain_compress[idx] = neg[idx]
                nxt_single_domain_id[domain_id] = idx  # useless, only alignment
                nxt_single_domain_item[domain_id] = item_id

            nxt = item_id
            nxt_dom = domain_id
            idx -= 1
            if idx == -1: break

        pos_domain_switch = pos_single_domain_compress * seq_domain_switch_flag # the judge of the domain switch

        # next_behavior has more elements than domain_switch_behavior
        # print(domain_invariant_user_train[user]['seq'])
        # print(domain_invariant_user_train[user]['domain'])
        # print(domain_switch_behavior)
        # print(next_behavior)
        for domain_id in domain_map:
            if domain_id in next_behavior_id.keys() and domain_id in domain_switch_behavior_id.keys():
                if next_behavior_id[domain_id] < domain_switch_behavior_id[domain_id]:
                    next_behavior[domain_id-1,next_behavior_id[domain_id]] = 0 # more than one
            if domain_id in next_behavior_id.keys() and domain_id not in domain_switch_behavior_id.keys():
                next_behavior[domain_id - 1, next_behavior_id[domain_id]] = 0  # more than one

        # print(domain_invariant_user_train[user]['seq'])
        # print(domain_invariant_user_train[user]['domain'])
        # print(domain_switch_behavior)
        # print(next_behavior)
        # next_behavior[domain_id-1, behavior_regularizer_id[domain_id]] = 0

        return (user, seq, pos, neg, pos_domain_switch, domain_switch_behavior, next_behavior)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, domain_invariant_user_train, user_set_in_all_domains, item_sets, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(domain_invariant_user_train,
                                                      user_set_in_all_domains,
                                                      item_sets,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()



# train/val/test data generation
def data_partition(datasets_information,args):
    #define the columns of the datasets
    names = ['user_id','item_id','timestamp','domain']
    types = {'user_id': np.int32, 'item_id': np.int32, 'timestamp': np.int32, 'domain': np.int32}
    #read training dataset, validation dataset and test dataset in all domains
    # keys: [domain] values: data
    all_train_datasets = {}
    all_validation_datasets = {}
    all_test_datasets = {}
    all_negative_datasets = {}
    data_path = datasets_information['data_path']
    for domain in datasets_information['domains']:
        all_train_datasets[domain] = pd.read_csv(r'{0:}/{1:}/train.csv'.format(data_path,domain),names=names,header=None,dtype=types)
        # sort by timestamp
        all_train_datasets[domain] = all_train_datasets[domain].sort_values(by=['timestamp'], ascending=True)
        all_validation_datasets[domain] = pd.read_csv(r'{0:}/{1:}/valid.csv'.format(data_path,domain),names=names,header=None,dtype=types)
        all_test_datasets[domain] = pd.read_csv(r'{0:}/{1:}/test.csv'.format(data_path,domain),names=names,header=None,dtype=types)
        all_negative_datasets[domain] = pd.read_csv(r'{0:}/{1:}/negative.csv'.format(data_path,domain),header=None)
    # get item sets and user sets in all domains respectively
    # one domain has one item set and one user set
    item_sets = {}
    user_sets = {}
    for domain in datasets_information['domains']:
        dfx = pd.DataFrame()
        dfx = pd.concat([dfx, all_train_datasets[domain]], axis=0)
        dfx = pd.concat([dfx, all_validation_datasets[domain]], axis=0)
        dfx = pd.concat([dfx, all_test_datasets[domain]], axis=0)
        item_sets[domain] = dfx.item_id.unique()
        user_sets[domain] = dfx.user_id.unique()
    # get max item_id and max user_id in all domains
    max_item_id = 0
    max_user_id = 0
    for domain in datasets_information['domains']:
        max_item_id = max(max_item_id, max(item_sets[domain]) )
        max_user_id = max(max_user_id, max(user_sets[domain]) )
    # get the unified training dataset in all domains
    unified_training_dataset = pd.DataFrame()
    for domain in datasets_information['domains']:
        unified_training_dataset = pd.concat([unified_training_dataset, all_train_datasets[domain]], axis = 0)
    unified_training_dataset = unified_training_dataset.sort_values(by=['timestamp'], ascending=True) # sort by timestamp
    # get item set and user set in all domains
    # all domains have only one item set and one user set
    # the int32 can't be saved
    # unified_training_dataset.user_id.unique()



    # validation/test [domain][user][seq/target/domain/target_timestamp/seq_timestamp]
    # the position is considered later
    # neg_cand [domain][user]

    # training [user][domain][seq/position/timestamp] domain-specific
    # position denotes the position
    # training [user][seq/domain/timestamp] domain-invariant
    # get training datasets for domain-invariant preference and domain-specific preference respectively.
    domain_specific_user_train = {} # In each domain, the user has one interaction sequence.
    domain_invariant_user_train = {}  # the user has only one interaction sequence in all domains
    # the users in the training dataset contains all users
    user_set_in_all_domains = unified_training_dataset.user_id.unique()
    # get training datasets for domain-invariant preference
    for user in user_set_in_all_domains:
        domain_invariant_user_train[int(user)] = {} # create empty dictionary
        domain_invariant_user_train[int(user)]['seq'] = unified_training_dataset[unified_training_dataset['user_id']
                                                                                 == user]['item_id'].values.tolist()
        domain_invariant_user_train[int(user)]['domain'] = unified_training_dataset[unified_training_dataset['user_id']
                                                                                 == user]['domain'].values.tolist()
        domain_invariant_user_train[int(user)]['timestamp'] = unified_training_dataset[unified_training_dataset['user_id']
                                                                                 == user]['timestamp'].values.tolist()
    # the position_id of the item that user interacted with in the last time in training dataset is maxlen - 1
    # record the position of items that the users interacted with
    # in the next steps
    # get training datasets for domain-specific preference
    # TypeError: keys must be str, int, float, bool or None, not int32
    # TypeError: Object of type int32 is not JSON serializable
    for user in user_set_in_all_domains:
        # set empty dictionary
        domain_specific_user_train[int(user)] = {}
        for domain in datasets_information['domains']:
            domain_specific_user_train[int(user)][domain] = {}
        for domain in datasets_information['domains']:
            domain_specific_user_train[int(user)][domain]['seq'] = all_train_datasets[domain][
                all_train_datasets[domain]['user_id'] == user]['item_id'].values.tolist()
            domain_specific_user_train[int(user)][domain]['timestamp'] = all_train_datasets[domain][
                all_train_datasets[domain]['user_id'] == user]['timestamp'].tolist()
            # get negative candidates for each user in all domains
    neg_cand = {}
    for domain in datasets_information['domains']:
        neg_cand[domain] = {}
        for user in user_set_in_all_domains:
            neg_cand[domain][int(user)] = all_negative_datasets[domain][all_negative_datasets[domain][0] == user].values[0][1:].tolist()
    # combine training dataset and validation dataset for testing
    unified_training_and_validation_dataset = pd.DataFrame()
    unified_training_and_validation_dataset = pd.concat([unified_training_and_validation_dataset,
                                                         unified_training_dataset], axis = 0)
    for domain in datasets_information['domains']:
        unified_training_and_validation_dataset = pd.concat([unified_training_and_validation_dataset,
                                                            all_validation_datasets[domain]], axis = 0)

    # combine training dataset, validation dataset and test dataset for getting the set
    unified_training_validation_and_test_dataset = pd.DataFrame()
    unified_training_validation_and_test_dataset = pd.concat([unified_training_validation_and_test_dataset,
                                                         unified_training_and_validation_dataset], axis=0)
    for domain in datasets_information['domains']:
        unified_training_validation_and_test_dataset = pd.concat([unified_training_validation_and_test_dataset,
                                                             all_test_datasets[domain]], axis=0)
    # item set should contain the items in the training dataset, the validation dataset and the test dataset
    item_set_in_all_domains = unified_training_validation_and_test_dataset.item_id.unique()
    user_set_in_all_domains = unified_training_validation_and_test_dataset.user_id.unique()


    # get validation data and test data for each user
    user_test = {}
    user_valid = {}
    for domain in datasets_information['domains']:
        # create empty dictionary
        user_test[domain] = {}
        user_valid[domain] = {}
        for user in user_set_in_all_domains:
            # the data framework for this user
            df_test = all_test_datasets[domain][all_test_datasets[domain]['user_id'] == user]
            df_valid = all_validation_datasets[domain][all_validation_datasets[domain]['user_id'] == user]


            # get sequence, target and domain that user interacted with for each user
            if df_test.shape[0] != 0:
                # create empty dictionary, not empty
                user_test[domain][int(user)] = {}
                # get historical items for current user
                df_for_user = unified_training_and_validation_dataset[
                    unified_training_and_validation_dataset['user_id'] == user]
                df_for_user = df_for_user[df_for_user['timestamp'] <= df_test.timestamp.values[0]] # The df_for_user doesn't contain the test dataset
                user_test[domain][int(user)]['seq'] = df_for_user.item_id.values.tolist()
                user_test[domain][int(user)]['target'] = int(df_test.item_id.values[0])
                user_test[domain][int(user)]['domain'] = df_for_user.domain.values.tolist()
                user_test[domain][int(user)]['seq_timestamp'] = df_for_user.timestamp.values.tolist()
                user_test[domain][int(user)]['target_timestamp'] = int(df_test.timestamp.values[0])

            if df_valid.shape[0] != 0:
                # create empty dictionary
                user_valid[domain][int(user)] = {}
                # get historical items for current user
                df_for_user = unified_training_dataset[
                    unified_training_dataset['user_id'] == user]
                df_for_user = df_for_user[df_for_user['timestamp'] <= df_valid.timestamp.values[0]]
                user_valid[domain][int(user)]['seq'] = df_for_user.item_id.values.tolist()
                user_valid[domain][int(user)]['target'] = int(df_valid.item_id.values[0])
                user_valid[domain][int(user)]['domain'] = df_for_user.domain.values.tolist()
                user_valid[domain][int(user)]['seq_timestamp'] = df_for_user.timestamp.values.tolist()
                user_valid[domain][int(user)]['target_timestamp'] = int(df_valid.timestamp.values[0])

    # save the data including domain_specific_user_train, domain_invariant_user_train, neg_cand, user_valid, user_test
    # the saved path
    saved_path = r'{0:}/all'.format(data_path)
    if not os.path.isdir(saved_path):
        os.makedirs(saved_path)

    #{dict} 多加了不该加的括号

    return [item_sets, user_sets, max_item_id, max_user_id, item_set_in_all_domains, user_set_in_all_domains,
            domain_specific_user_train, domain_invariant_user_train, neg_cand, user_valid, user_test]


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [item_sets, user_sets, max_item_id, max_user_id, item_set_in_all_domains, user_set_in_all_domains,
     domain_specific_user_train,
     domain_invariant_user_train, neg_cand, user_valid, user_test] = copy.deepcopy(dataset)

    # create dictionary for saving results
    ans_list = {}

    for domain in domain_map.values():
        # initialize parameters
        NDCG = 0.0
        valid_user = 0.0
        HT = 0.0


        # validation
        for u in tqdm(user_sets[domain], ncols=80):
            if not (u in user_valid[domain].keys() and len(user_valid[domain][u]['seq']) >= 1): continue

            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            for item_id in reversed(user_valid[domain][u]['seq']):
                seq[idx] = item_id
                idx -= 1
                if idx == -1: break

            item_idx = [user_valid[domain][u]['target']]
            item_idx = item_idx + neg_cand[domain][u]

            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
            predictions = predictions[0]

            rank = predictions.argsort().argsort()[0].item()

            valid_user += 1

            if rank < args.top_n:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1
            # if valid_user % 100 == 0:
            #     print('.', end="")
            #     sys.stdout.flush()

        # save results
        ans_list[domain] = [NDCG / valid_user, HT / valid_user]

    return ans_list


# evaluate on test set
def evaluate(model, dataset, args):
    [item_sets, user_sets, max_item_id, max_user_id, item_set_in_all_domains, user_set_in_all_domains,
     domain_specific_user_train,
     domain_invariant_user_train, neg_cand, user_valid, user_test] = copy.deepcopy(dataset)

    # create dictionary for saving results
    ans_list = {}

    for domain in domain_map.values():
        # initialize parameters
        NDCG = 0.0
        HT = 0.0
        valid_user = 0.0

        for u in tqdm(user_sets[domain], ncols=80):
            # user interaction length is short
            if not (u in user_test[domain].keys() and len(user_test[domain][u]['seq']) >= 1): continue

            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1

            for item_id in reversed(user_test[domain][u]['seq']):
                seq[idx] = item_id
                idx -= 1
                if idx == -1: break

            item_idx = [user_test[domain][u]['target']]
            item_idx = item_idx + neg_cand[domain][u]
            # for _ in range(100):
            #     t = np.random.choice(item_set)
            #     while t in rated: t = np.random.choice(item_set)
            #     item_idx.append(t)`

            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
            predictions = predictions[0]  # - for 1st argsort DESC

            rank = predictions.argsort().argsort()[0].item()

            valid_user += 1

            if rank < args.top_n:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1
        # save results
        # append[[]]
        ans_list[domain] = [NDCG / valid_user, HT / valid_user]

    return ans_list