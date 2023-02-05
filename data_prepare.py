# Import modules
import os
import csv
import random
import argparse
import itertools

# Settings
parser = argparse.ArgumentParser(description='Data preparation')
parser.add_argument('--dataset', type=str, default='CEDAR', help='dataset name, options: [CEDAR, BHSig-B, BHSig-H]')
args = parser.parse_args()

if args.dataset == 'CEDAR':
    num_people = 55 # number of writers
    true_times = 10  # number of real signatures
    false_times = 10 # number of forged signatures
    train_num = 50
elif args.dataset == 'BHSig-B':
    num_people = 100 # number of writers
    true_times = 24  # number of real signatures
    false_times = 30 # number of forged signatures
    train_num = 50
elif args.dataset == 'BHSig-H':
    num_people = 160 # number of writers
    true_times = 24  # number of real signatures
    false_times = 30 # number of forged signatures
    train_num = 100
random.seed(42)
save_path = 'datasets/%s'%args.dataset
true_data_path = 'datasets/%s/true'%args.dataset
false_data_path = 'datasets/%s/false'%args.dataset


names = os.listdir(true_data_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)
def make_samples(signers, lst1, lst2):
    samples = []
    for id_ in signers:
        lst3 = random.sample(lst2, len(lst1))
        true = list(itertools.zip_longest(lst1, [], fillvalue=1))
        true = list(map(lambda i: (id_, *i[0], i[1]), true))
        samples.extend(true)
        false = list(itertools.zip_longest(lst3, [], fillvalue=0))
        false = list(map(lambda i: (id_, *i[0], i[1]), false))
        samples.extend(false)
    return samples

def get_information(data):
    writer_id, x1, x2, y = data
    if y == 1:
        x1 = os.path.join(true_data_path, 'original_' + str(writer_id + 1) + f'_{x1 + 1}.png')
        x2 = os.path.join(true_data_path, 'original_' + str(writer_id + 1) + f'_{x2 + 1}.png')
    else:
        x1 = os.path.join(true_data_path, 'original_' + str(writer_id + 1) + f'_{x1 + 1}.png')
        x2 = os.path.join(false_data_path, 'forgeries_' + str(writer_id + 1) + f'_{x2 + 1}.png')
    return x1, x2, y

def make_csv(true_times, false_times, num):
    dataset = range(train_num)
    train_lst1 = list(itertools.combinations(range(0, true_times), 2))
    train_lst2 = list(itertools.product(range(0, true_times), range(0, false_times)))
    train_db = make_samples(dataset, train_lst1, train_lst2)
    train_db = list(map(get_information, train_db))
    
    test_dataset = range(train_num, num)
    test_lst1 = list(itertools.combinations(range(0, true_times), 2))
    test_lst2 = list(itertools.product(range(0, true_times), range(0, false_times)))
    test_db = make_samples(test_dataset, test_lst1, test_lst2)
    test_db = list(map(get_information, test_db))
    
    with open(os.path.join(save_path, 'train.csv'), 'wt', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(train_db)
    with open(os.path.join(save_path, 'test.csv'), 'wt', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(test_db)
        
make_csv(true_times, false_times, num_people)