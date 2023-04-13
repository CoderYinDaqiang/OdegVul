import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from odegcn import ODEG

# Model structure
class HierarchicalAttentionNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim, word_gru_num_layers, sent_gru_num_layers, word_att_dim, sent_att_dim, use_layer_norm, dropout):
        """
        vocab_size: number of words in the vocabulary of the model
        embed_dim: dimension of word embeddings
        word_gru_hidden_dim: dimension of word-level GRU; biGRU output is double this size
        sent_gru_hidden_dim: dimension of sentence-level GRU; biGRU output is double this size
        word_gru_num_layers: number of layers in word-level GRU
        sent_gru_num_layers: number of layers in sentence-level GRU
        word_att_dim: dimension of word-level attention layer
        sent_att_dim: dimension of sentence-level attention layer
        use_layer_norm: whether to use layer normalization
        dropout: dropout rate; 0 to not use dropout
        """
        super(HierarchicalAttentionNetwork, self).__init__()

        self.sent_attention = SentenceAttention(
            vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim,
            word_gru_num_layers, sent_gru_num_layers, word_att_dim, sent_att_dim, use_layer_norm, dropout)

        self.fc = nn.Linear(2 * sent_gru_hidden_dim, 1)
        self.sig = nn.Sigmoid()

        self.use_layer_nome = use_layer_norm
        self.dropout = dropout

    def forward(self, line_tensor, adj):
        # line_lengths:batch_size个文件行(900行);sent_lengths:batch_size个长度为900的列表
        line_lengths = []
        sent_lengths = []

        for line in line_tensor:
            code_line = []
            line_lengths.append(len(line))

        
        line_tensor = line_tensor.type(torch.LongTensor)
        line_lengths = torch.tensor(line_lengths).type(torch.LongTensor).cuda()
        
        line_embeddings, word_att_weights, sent_att_weights, sents = self.sent_attention(line_tensor, line_lengths, adj)

        scores = self.fc(line_embeddings)
        final_scrs = self.sig(scores)

        return final_scrs, word_att_weights, sent_att_weights, sents

class SentenceAttention(nn.Module):
    """
    Sentence-level attention module. Contains a word-level attention module.
    """
    def __init__(self, vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim,
                word_gru_num_layers, sent_gru_num_layers, word_att_dim, sent_att_dim, use_layer_norm, dropout):
        super(SentenceAttention, self).__init__()

        # Word-level attention module
        self.word_attention = WordAttention(vocab_size, embed_dim, word_gru_hidden_dim, word_gru_num_layers, word_att_dim, use_layer_norm, dropout)

        # Bidirectional sentence-level GRU
        self.gru = nn.GRU(2 * word_gru_hidden_dim, sent_gru_hidden_dim, num_layers=sent_gru_num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)

        self.sage = SAGE(2 * word_gru_hidden_dim, 2 * sent_gru_hidden_dim, dropout)
        self.odeg = ODEG(2 * word_gru_hidden_dim, 1, time=1).cuda()

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(2 * sent_gru_hidden_dim, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

        # Sentence-level attention
        self.sent_attention = nn.Linear(2 * sent_gru_hidden_dim, sent_att_dim)

        # Sentence context vector u_s to take dot product with
        # This is equivalent to taking that dot product (Eq.10 in the paper),
        # as u_s is the linear layer's 1D parameter vector here
        self.sentence_context_vector = nn.Linear(sent_att_dim, 1, bias=False)

    def forward(self, line_tensor, line_lengths, adj):

        # # Sort code_tensor by decreasing order in length
        # line_lengths, line_perm_idx = line_lengths.sort(dim=0, descending=True)
        # line_tensor = line_tensor[line_perm_idx]
        #
        #
        # # Make a long batch of sentences by removing pad-sentences
        # # i.e. `code_tensor` was of size (num_code_tensor, padded_line_lengths, padded_sent_length)
        # # -> `packed_sents.data` is now of size (num_sents, padded_sent_length) 所有.java文件被压缩成了所有行的表示(14400,50)
        # packed_sents = pack_padded_sequence(line_tensor, lengths=line_lengths.tolist(), batch_first=True) #返回对象PackedSequence
        #
        # # effective batch size at each timestep
        # valid_bsz = packed_sents.batch_sizes
        #
        # # Make a long batch of sentence lengths by removing pad-sentences
        # # i.e. `sent_lengths` was of size (num_code_tensor, padded_line_lengths)
        # # -> `packed_sent_lengths.data` is now of size (num_sents) 所有.java文件被压缩成了所有行数的表示(14400,)
        # packed_sent_lengths = pack_padded_sequence(sent_lengths, lengths=line_lengths.tolist(), batch_first=True)

    
    
        # Word attention module  sents=Line_embedding(14400,64) which dim is 64;word_att_weights=(14400,50)
        # sents, word_att_weights = self.word_attention(packed_sents.data, packed_sent_lengths.data)
        sents, word_att_weights = self.word_attention(line_tensor, line_lengths)

        sents = self.dropout(sents)

        packed_sents = self.sage(sents,adj)
        packed_sents = self.sage(packed_sents, adj)
        packed_sents = self.sage(packed_sents, adj)
        line_tensor = packed_sents
        sent_att_weights = None

        # packed_sents = PackedSequence(packed_sents, valid_bsz)

        # if self.use_layer_norm:
        #     normed_sents = self.layer_norm(packed_sents)
        # else:
        #     normed_sents = packed_sents
        #
        # # Sentence attention
        # att = torch.tanh(self.sent_attention(normed_sents))# (u_l:(14400,64))
        # att = self.sentence_context_vector(att).squeeze(1) # (u_l) dot with (u_s context-vector) = (14400,)
        #
        # val = att.max()
        # att = torch.exp(att - val)
        #
        # sent_att_weights = att / torch.sum(att, dim=0, keepdim=True)
        #
        # line_tensor = packed_sents * sent_att_weights.unsqueeze(1)

        # # Restore as documents by repadding (16, 900)
        # att, _ = pad_packed_sequence(PackedSequence(att, valid_bsz), batch_first=True)
        # # 得到了所有.java文件中每一个line embedding的权重值...
        # sent_att_weights = att / torch.sum(att, dim=1, keepdim=True)
        #
        # # Restore as documents by repadding (16,900,64)
        # code_tensor, _ = pad_packed_sequence(packed_sents, batch_first=True)
        #
        # # Compute document vectors
        # code_tensor = code_tensor * sent_att_weights.unsqueeze(2) # (16,900,64)
        # code_tensor = code_tensor.sum(dim=1) # 聚合操作 (16,64)即 .java文件 embedding
        #
        # # Restore as documents by repadding (14400,50) -> (16,900,50)
        # word_att_weights, _ = pad_packed_sequence(PackedSequence(word_att_weights, valid_bsz), batch_first=True)
        #
        # # Restore the original order of documents (undo the first sorting)
        # _, code_tensor_unperm_idx = line_perm_idx.sort(dim=0, descending=False)
        # code_tensor = code_tensor[code_tensor_unperm_idx]
        #
        # word_att_weights = word_att_weights[code_tensor_unperm_idx]
        # sent_att_weights = sent_att_weights[code_tensor_unperm_idx]

        return line_tensor, word_att_weights, sent_att_weights, sents


class WordAttention(nn.Module):
    """
    Word-level attention module.
    """

    def __init__(self, vocab_size, embed_dim, gru_hidden_dim, gru_num_layers, att_dim, use_layer_norm, dropout):
        super(WordAttention, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        # output (batch, hidden_size) gru_hidden_dim:每一层中GRU的数量; batch_first:(seq_len,batch,feature)->(batch,seq_len,feature)
        self.gru = nn.GRU(embed_dim, gru_hidden_dim, num_layers=gru_num_layers, batch_first=True, bidirectional=True, dropout=dropout)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(2 * gru_hidden_dim, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

        # Maps gru output to `att_dim` sized tensor
        self.attention = nn.Linear(2 * gru_hidden_dim, att_dim)

        # Word context vector (u_w) to take dot-product with
        self.context_vector = nn.Linear(att_dim, 1, bias=False)

    def init_embeddings(self, embeddings):
        """
        Initialized embedding layer with pretrained embeddings.
        embeddings: embeddings to init with
        """
        self.embeddings.weight = nn.Parameter(embeddings)

    def freeze_embeddings(self, freeze=False):
        """
        Set whether to freeze pretrained embeddings.
        """
        self.embeddings.weight.requires_grad = freeze

    def forward(self, sents, sent_lengths):
        """
        sents: encoded sentence-level data; LongTensor (num_sents, pad_len, embed_dim);传入时，用所有行进行训练(14400,50)
        return: sentence embeddings, attention weights of words ;传入时，代表所有行数....(14400,)
        """
        # Sort sents by decreasing order in sentence lengths
        sent_lengths, sent_perm_idx = sent_lengths.sort(dim=0, descending=True)
        sents = sents[sent_perm_idx] # (14400,50)

        sents = self.embeddings(sents.cuda()) # (14400行,50词,50embedding dim)
        # 打包成一整行，加快GRU的运行速度，(720000词,50embedding dim)
        packed_words = pack_padded_sequence(sents, lengths=sent_lengths.tolist(), batch_first=True)

        # effective batch size at each timestep
        valid_bsz = packed_words.batch_sizes

        # Apply word-level GRU over word embeddings GRU隐藏层有32个但是是bidirectional (720000,50) -> (720000,64)
        packed_words, _ = self.gru(packed_words) #(h_it):得到所有GRU训练过的word embedding

        if self.use_layer_norm:
            normed_words = self.layer_norm(packed_words.data)
        else:
            normed_words = packed_words

        # Word Attenton
        att = torch.tanh(self.attention(normed_words.data)) # (u_it:(720000,64))
        att = self.context_vector(att).squeeze(1) # (u_it) to take dot-product with (u_w) = (720000,)

        val = att.max() # (u_w : context vector)
        att = torch.exp(att - val) # att.size: (n_words)

        # Restore as sentences by repadding (14400,50)
        att, _ = pad_packed_sequence(PackedSequence(att, valid_bsz), batch_first=True)
        # 得到了所有行中每一个 word embedding的权重值(softmax)...
        att_weights = att / torch.sum(att, dim=1, keepdim=True)

        # Restore as sentences by repadding sents为(h_it)
        sents, _ = pad_packed_sequence(packed_words, batch_first=True)

        # Compute sentence vectors 聚合操作
        sents = sents * att_weights.unsqueeze(2) # sent:(14400,50,64) att_weights.unsqueeze(2):(14400,50,64)
        sents = sents.sum(dim=1) # 聚合操作(14400,64) 即 line embedding

        # Restore the original order of sentences (undo the first sorting)
        _, sent_unperm_idx = sent_perm_idx.sort(dim=0, descending=False)
        sents = sents[sent_unperm_idx]

        att_weights = att_weights[sent_unperm_idx]

        return sents, att_weights

import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
class SAGE(nn.Module):
    def __init__(self, line_embed_dim, hid_dim, dropout):
        super().__init__()
        self.conv1 = SAGEConv(line_embed_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, hid_dim)
        # self.conv3 = GATConv(hid_dim * 4, hid_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, adj)
        return F.relu(x)