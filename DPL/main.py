#-*- coding: utf-8 -*-

import os
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from parse import parse_args
from torch import sparse
from tqdm import tqdm
from data import *
from model import *
from evaluation import *
from negative_sampling import *
from cuda import *
# print(torch.__version__)
USE_CUDA = torch.cuda.is_available()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device('cuda' if USE_CUDA else 'cpu')
def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_data_path():
    directory = 'data/'
    if arg.dataset == '100k':
        total_file = directory + '/' + '100k.csv'
        train_file = directory + '/' + '100k_train.csv'
        test_file = directory + '/' + '100k_test.csv'
    elif arg.dataset == 'yahoo':
        total_file = directory + '/' + 'yahoo1.csv'
        train_file = directory + '/' + 'yahoo1_train.csv'
        test_file = directory + '/' + 'yahoo1_test.csv'
    elif arg.dataset == '1M':
        total_file = directory + '/' + '1m1.csv'
        train_file = directory + '/' + '1m1_train.csv'
        test_file = directory + '/' + '1m1_test.csv'
    elif arg.dataset == 'gowalla':
        total_file = directory + '/' + 'gowalla.csv'
        train_file = directory + '/' + 'gowalla_train.csv'
        test_file = directory + '/' + 'gowalla_test.csv'
    elif arg.dataset == 'amazon-book':
        total_file = directory + '/' + 'amazon-book.csv'
        train_file = directory + '/' + 'amazon-book_train.csv'
        test_file = directory + '/' + 'amazon-book_test.csv'
    elif arg.dataset == 'yelp2018':
        total_file = directory + '/' + 'yelp2018.csv'
        train_file = directory + '/' + 'yelp2018_train.csv'
        test_file = directory + '/' + 'yelp2018_test.csv'
    return total_file, train_file, test_file

def cdf_trans(cdf_u, alpha, tau_plus):
    tau_minus = 1 - tau_plus
    a = (1 - 2 * alpha) * (tau_minus - tau_plus) + 1e-3
    b = 2 * (alpha * tau_minus + (1 - alpha) * tau_plus)
    return (-b + torch.sqrt(b ** 2 + 4 * a * cdf_u)) / (2 * a)

def criterion(scores,arg):
    '''
    :param scores: bs * (1+M+N)
    :return: loss
    '''
    pos_scores = scores[:, 0:(arg.M+1) ]  # [bs * (1+M)]
    neg_scores = scores[:, (arg.M + 1):]  # [bs * N]
    assert neg_scores.shape[1] == arg.N and pos_scores.shape[1] == arg.M+1
    if arg.LOSS == 'BPR':
        pu_prob = torch.sigmoid (pos_scores[:, 0] - neg_scores[:, 0])
        loss = -torch.log(pu_prob).mean()

    elif arg.LOSS == 'DPL':
        assert arg.M >=1
        pu_prob = torch.sigmoid(pos_scores[:, 0:1] - neg_scores)       # bs * N
        pp_prob = torch.sigmoid(pos_scores[:, 0:1] - pos_scores[:,1:]) # bs * M
        pn_prob = pu_prob.mean(dim=-1) - (arg.tau_plus * pp_prob).mean(dim=-1)  # bs
        pn_prob = torch.clamp(pn_prob, min=1e-2) #This restriction is in place to avoid a negative probability value.
        loss = -torch.log(pn_prob).mean()
    else:
        print('Invalid loss function')
        raise Exception
    return loss

def log():
    if arg.log:
        path = arg.log_root
        if not os.path.exists(path):
            os.makedirs(path)
        file = path +'/' + arg.dataset +'_' + arg.LOSS + '_'+ arg.encoder +'_'+ str(arg.M) +'_'+ str(arg.N) +'_'+ str(arg.tau_plus) + '.txt'
        f = open(file, 'w')
        print('----------------loging----------------')
    else:
        f = sys.stdout
    return f


def get_numbers(file):
    '''
    :param file: data path
    :return:
    num_users: total number of users
    num_items: total number of items
    dividing_tensor: [|I|] element 1 represents hot items, while element 0 represents cold items.
    '''
    data = pd.read_csv(file, header=0, dtype='str', sep=',')
    userlist = list(data['user'].unique())
    itemlist = list(data['item'].unique())
    num_users, num_items = len(userlist), len(itemlist)
    return num_users, num_items


def load_train_data(path, num_item):
    data = pd.read_csv(path, header=0, sep=',')
    datapair = []
    popularity = np.zeros(num_item)
    train_tensor = torch.zeros(num_users, num_items)
    for i in data.itertuples():
        user, item, rating = getattr(i, 'user'), getattr(i, 'item'), getattr(i, 'rating')
        user, item = int(user), int(item)
        popularity[int(item)] += 1
        datapair.append([user, item])
        train_tensor[user, item] = 1
    prior = popularity / sum(popularity)
    return train_tensor.to_sparse(), datapair,prior


def load_test_data(path, num_users, num_items):
    data = pd.read_csv(path, header=0, sep=',')
    test_tensor = torch.zeros(num_users, num_items)
    for i in data.itertuples():
        user, item, rating = getattr(i, 'user'), getattr(i, 'item'), getattr(i, 'rating')
        user, item = int(user), int(item)
        test_tensor[user, item] = 1
    return test_tensor.bool()


def collect_G_Lap_Adj():
    G_Lap_tensor = convert_spmat_to_sptensor(dataset.Lap_mat)
    G_Adj_tensor = convert_spmat_to_sptensor(dataset.Adj_mat)
    G_Lap_tensor = G_Lap_tensor.to(device)
    G_Adj_tensor = G_Adj_tensor.to(device)
    return G_Lap_tensor, G_Adj_tensor



def model_init():
    # A new train
    model_path = r'.\model'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if arg.encoder == 'MF':
        model = MF(num_users, num_items, arg, device)
    if arg.encoder == 'LightGCN':
        g_laplace, g_adj = collect_G_Lap_Adj()
        model = LightGCN(num_users, num_items, arg, device, g_laplace, g_adj)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr, weight_decay=arg.l2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=arg.lr_dc_epoch, gamma=arg.lr_dc)
    return model, optimizer, scheduler


def train(train_loader,model, optimizer, epoch):
    print('-------------------------------------------', file=f)
    print('-------------------------------------------')
    print('epoch: ', epoch, file=f)
    print('epoch: ', epoch)
    print('start training: ', datetime.datetime.now(), file=f)
    print('start training: ', datetime.datetime.now())
    st = time.time()
    model.train()
    total_loss = []

    for index, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()
        batch = batch.to(device)           # [bs * (2+M+N)]
        # Calculate Loss
        scores = model.forward(batch)      # [bs * (1+M+N)]
        loss = criterion(scores, arg)

        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

    print('Loss:\t%.8f\tlr:\t%0.8f' % (np.mean(total_loss), optimizer.state_dict()['param_groups'][0]['lr']), file=f)
    print('Loss:\t%.8f\tlr:\t%0.8f' % (np.mean(total_loss), optimizer.state_dict()['param_groups'][0]['lr']))
    print('Training time:[%0.2f s]' % (time.time() - st))
    print('Training time:[%0.2f s]' % (time.time() - st), file=f)


def test():
    print('----------------', file=f)
    print('----------------')
    print('start evaluation: ', datetime.datetime.now(), file=f)
    print('start evaluation: ', datetime.datetime.now())
    model.eval()

    Pre_dic, Recall_dict, F1_dict, NDCG_dict = {}, {}, {}, {}
    sp = time.time()
    rating_mat = model.predict() # |U| * |V|
    rating_mat = erase(rating_mat)
    for k in arg.topk:
        metrices = topk_eval(rating_mat, k, test_tensor.to(device))
        precision, recall, F1, ndcg = metrices[0], metrices[1], metrices[2], metrices[3]
        Pre_dic[k] = precision
        Recall_dict[k] = recall
        F1_dict[k] = F1
        NDCG_dict[k] = ndcg
    print('Evaluation time:[%0.2f s]' % (time.time() - sp))
    print('Evaluation time:[%0.2f s]' % (time.time() - sp), file=f)
    return Pre_dic, Recall_dict, F1_dict, NDCG_dict


def erase(score):
    x = train_tensor.to(device) * (-1000)
    score = score + x
    return score


def print_epoch_result(real_epoch, Pre_dic, Recall_dict, F1_dict, NDCG_dict):

    for k in arg.topk:
        if Pre_dic[k] > best_result[k][0]:
            best_result[k][0], best_epoch[k][0] = Pre_dic[k], real_epoch
        if Recall_dict[k] > best_result[k][1]:
            best_result[k][1], best_epoch[k][1] = Recall_dict[k], real_epoch
        if F1_dict[k] > best_result[k][2]:
            best_result[k][2], best_epoch[k][2] = F1_dict[k], real_epoch
        if NDCG_dict[k] > best_result[k][3]:
            best_result[k][3], best_epoch[k][3] = NDCG_dict[k], real_epoch
        print(
            'Pre@%02d:\t%0.4f\tRecall@%02d:\t%0.4f\tF1@%02d:\t%0.4f\tNDCG@%02d:\t%0.4f' %
            (k, Pre_dic[k], k, Recall_dict[k], k, F1_dict[k], k, NDCG_dict[k]))
        print(
            'Pre@%02d:\t%0.4f\tRecall@%02d:\t%0.4f\tF1@%02d:\t%0.4f\tNDCG@%02d:\t%0.4f' %
            (k, Pre_dic[k], k, Recall_dict[k], k, F1_dict[k], k, NDCG_dict[k]),
            file=f)
    return best_result, best_epoch


def print_best_result(best_result, best_epoch):
    print('------------------best result-------------------', file=f)
    print('------------------best result-------------------')
    for k in arg.topk:
        print(
            'Best Result: Pre@%02d:\t%0.4f\tRecall@%02d:\t%0.4f\tF1@%02d:\t%0.4f\tNDCG@%02d:\t%0.4f\t[%0.2f s]' %
            (k, best_result[k][0], k, best_result[k][1], k, best_result[k][2], k, best_result[k][3],  (time.time() - t0)))
        print(
            'Best Result: Pre@%02d:\t%0.4f\tRecall@%02d:\t%0.4f\tF1@%02d:\t%0.4f\tNDCG@%02d:\t%0.4f\t[%0.2f s]' %
            (k, best_result[k][0], k, best_result[k][1], k, best_result[k][2], k, best_result[k][3],  (time.time() - t0)), file=f)

        print(
            'Best Epoch: Pre@%02d: %d\tRecall@%02d: %d\tF1@%02d: %d\tNDCG@%02d: %d\t[%0.2f s]' % (
                k, best_epoch[k][0], k, best_epoch[k][1], k, best_epoch[k][2], k, best_epoch[k][3],
                (time.time() - t0)))
        print(
            'Best Epoch: Pre@%02d: %d\tRecall@%02d: %d\tF1@%02d: %d\tNDCG@%02d: %d\t[%0.2f s]' % (
                k, best_epoch[k][0], k, best_epoch[k][1], k, best_epoch[k][2], k, best_epoch[k][3],
                (time.time() - t0)), file=f)
    print('------------------------------------------------', file=f)
    print('------------------------------------------------')
    print('Run time: %0.2f s' % (time.time() - t0), file=f)
    print('Run time: %0.2f s' % (time.time() - t0))


if __name__ == '__main__':
    t0 = time.time()
    arg = parse_args()
    f = log()
    print(arg)
    print(arg, file=f)


    init_seed(2022)
    total_file, train_file, test_file = get_data_path()
    num_users, num_items = get_numbers(total_file)

    # Load Data
    train_tensor, train_pair, prior = load_train_data(train_file, num_items)
    test_tensor = load_test_data(test_file, num_users, num_items) # This is a Boolean matrix with shape |U|*|I|, indicating which (u, i) pairs are present in the test set.

    dataset = Data(train_pair, arg, num_users, num_items)
    train_loader = DataLoader(dataset, batch_size=arg.batch_size, shuffle=True, collate_fn=dataset.collate_fn, drop_last=False, pin_memory=True, num_workers=arg.num_workers)

    # Init Model
    model, optimizer, scheduler = model_init()
    best_result = {}
    best_epoch = {}
    for k in arg.topk:
        best_result[k] = [0., 0., 0., 0.]
        best_epoch[k] = [0, 0, 0, 0]

    # Train and Test
    for epoch in range(arg.epochs):
        train(train_loader,model, optimizer, epoch)
        Pre_dic, Recall_dict, F1_dict, NDCG_dict = test()
        scheduler.step()
        best_result, best_epoch = print_epoch_result(epoch, Pre_dic, Recall_dict, F1_dict, NDCG_dict)
    print_best_result(best_result, best_epoch)
    f.close()


