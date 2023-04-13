import re

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

max_seq_len = 50

all_train_releases = {'activemq': 'activemq-5.0.0', 'camel': 'camel-1.4.0', 'derby': 'derby-10.2.1.6', 
                      'groovy': 'groovy-1_5_7', 'hbase': 'hbase-0.94.0',
                      'jruby': 'jruby-1.1', 'lucene': 'lucene-2.3.0', 'wicket': 'wicket-1.3.0-incubating-beta-1'}

all_eval_releases = {'activemq': ['activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'], 
                     'camel': ['camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'], 
                     'derby': ['derby-10.3.1.4', 'derby-10.5.1.1'], 
                     'groovy': ['groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'], 
                     'hbase': ['hbase-0.95.0', 'hbase-0.95.2'],
                     'jruby': ['jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'], 
                     'lucene': ['lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'], 
                     'wicket': ['wicket-1.3.0-beta2', 'wicket-1.5.3']}



all_releases = {'activemq': ['activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'], 
                     'camel': ['camel-1.4.0', 'camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'], 
                     'derby': ['derby-10.2.1.6', 'derby-10.3.1.4', 'derby-10.5.1.1'], 
                     'groovy': ['groovy-1_5_7', 'groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'], 
                     'hbase': ['hbase-0.94.0', 'hbase-0.95.0', 'hbase-0.95.2'], 'hive': ['hive-0.9.0', 'hive-0.10.0', 'hive-0.12.0'], 
                     'jruby': ['jruby-1.1', 'jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'], 
                     'lucene': ['lucene-2.3.0', 'lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'], 
                     'wicket': ['wicket-1.3.0-incubating-beta-1', 'wicket-1.3.0-beta2', 'wicket-1.5.3']}

all_projs = list(all_train_releases.keys())

file_lvl_gt = '../datasets/preprocessed_data/'


word2vec_dir = '../output/Word2Vec_model/' 

def get_df(rel, is_baseline=False):

    if is_baseline:
        df = pd.read_csv('../'+file_lvl_gt+rel+".csv")

    else:
        df = pd.read_csv(file_lvl_gt+rel+".csv")

    df = df.fillna('')

    df = df[df['is_blank']==False]
    df = df[df['is_test_file']==False]

    return df

def prepare_code2d(code_list, to_lowercase = False):
    '''
        input
            code_list (list): list that contains code each line (in str format)
        output
            code2d (nested list): a list that contains list of tokens with padding by '<pad>'
    '''
    code2d = []

    for c in code_list:
        c = re.sub('\\s+',' ',c)

        if to_lowercase:
            c = c.lower()


        token_list = c.strip().split()
        total_tokens = len(token_list)
        
        token_list = token_list[:max_seq_len]

        if total_tokens < max_seq_len:
            token_list = token_list + ['<pad>']*(max_seq_len-total_tokens)

        code2d.append(token_list)

    return code2d


def get_code2d_and_label(df, to_lowercase = False):
    code = list(df['code_line'])
    code2d = prepare_code2d(code, to_lowercase)
    all_line_label = list(df['line-label'])
    return code2d, all_line_label
    
def get_code3d_and_label(df, to_lowercase = False):
    '''
        input
            df (DataFrame): a dataframe from get_df()
        output
            code3d (nested list): a list of code2d from prepare_code2d()
            all_file_label (list): a list of file-level label
    '''

    code3d = []
    all_file_label = []

    # 取出所有.java文件
    for filename, group_df in df.groupby('filename'):

        file_label = bool(group_df['file-label'].unique())

        code = list(group_df['code_line'])

        code2d = prepare_code2d(code, to_lowercase)
        code3d.append(code2d)

        all_file_label.append(file_label)

    return code3d, all_file_label

def get_w2v_path():

    return word2vec_dir

def get_w2v_weight_for_deep_learning_models(word2vec_model, embed_dim):
    # word2vec_model.wv.syn0 为embedding matrix...
    word2vec_weights = torch.FloatTensor(word2vec_model.wv.syn0).cuda()
    
    # add zero vector for unknown(<UNK>) tokens
    word2vec_weights = torch.cat((word2vec_weights, torch.zeros(1,embed_dim).cuda()))

    return word2vec_weights


def pad_code(code_list_3d,max_sent_len,limit_sent_len=True, mode='train'):
    paded = []
    
    for file in code_list_3d:
        sent_list = []
        for line in file:
            new_line = line
            if len(line) > max_seq_len:
                new_line = line[:max_seq_len]
            sent_list.append(new_line)
            
        
        if mode == 'train':
            if max_sent_len-len(file) > 0:
                for i in range(0,max_sent_len-len(file)):
                    sent_list.append([0]*max_seq_len)

        if limit_sent_len:    
            paded.append(sent_list[:max_sent_len])
        else:
            paded.append(sent_list)
        
    return paded


def get_dataloader(code_vec, label_list,batch_size):
    y_tensor =  torch.cuda.FloatTensor([label for label in label_list])

    tensor_dataset = TensorDataset(torch.tensor(code_vec), y_tensor)

    dl = DataLoader(tensor_dataset,shuffle=True,batch_size=batch_size,drop_last=True)
    
    return dl


def get_x_vec(code_2d, word2vec):
    x_vec = [[word2vec.wv.vocab[token].index if token in word2vec.wv.vocab else len(word2vec.wv.vocab) for token in line]
         for line in code_2d]
    return x_vec


def get_adj_ones_frame(line_nums):
    row1 = torch.arange(0, line_nums - 1)
    col1 = row1 + 1
    row = torch.cat((row1, col1), dim=0)
    col = torch.cat((col1, row1), dim=0)
    row = row.resize(1, row.size(0))
    col = col.resize(1, col.size(0))
    adj = torch.cat((row, col), dim=0)
    return adj

def get_adj_frame(line_nums):
    file_name = "../datasets/pregraph/graph.txt"
    row_col = []
    row_list = []
    col_list = []
    epoch_nums = 0
    with open(file_name) as file_object:
        for line in file_object:
            row_col = re.findall(r"\d+", line)
            if int(row_col[0])>=epoch_nums*line_nums and int(row_col[0])<=(++epoch_nums*line_nums):
                row_list.append(int(row_col[0])-epoch_nums*line_nums)
                col_list.append(int(row_col[1])-epoch_nums*line_nums)


    row = torch.tensor(row_list, dtype=int)
    col = torch.tensor(col_list, dtype=int)
    row = row.resize(1, row.size(0))
    col = col.resize(1, col.size(0))
    adj = torch.cat((row, col), dim=0)
    return adj

import numpy as np
import scipy.sparse as sp
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def get_adj(line_nums):
    file_name = "../datasets/pregraph/graph.txt"
    edge_list = []
    with open(file_name) as file_object:
        for line in file_object:
            row_col = re.findall(r"\d+", line)
            if int(row_col[0] and row_col[1])<= line_nums-1 :
                edge_list.append(row_col)
    edges = np.array(edge_list)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(line_nums, line_nums),
                            dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).to_dense()

    normalized_adj = get_normalized_adj(adj.numpy())
    return normalized_adj

def get_normalized_adj(A):
    """
    Returns a tensor, the degree normalized adjacency matrix.
    """
    alpha = 0.8
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(A.shape[0]) + A_wave)
    return torch.from_numpy(A_reg.astype(np.float32))


def get_adj_ones(line_nums):
    # row_col = [0,0]
    edge_list = []
    for i in range(line_nums-1):
        row_col = [0,0]
        row_col[0] = str(i)
        row_col[1] = str(i+1)
        edge_list.append(row_col)
    edges = np.array(edge_list)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(line_nums, line_nums),
                            dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).to_dense()
    normalized_adj = get_normalized_adj(adj.numpy())
    return normalized_adj
