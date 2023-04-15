import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from odegcn import ODEG

# Model structure
class HierarchicalAttentionNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim, word_gru_num_layers, sent_gru_num_layers, word_att_dim, sent_att_dim, use_layer_norm, dropout):

        super(HierarchicalAttentionNetwork, self).__init__()

        self.sent_attention = SentenceAttention(
            vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim,
            word_gru_num_layers, sent_gru_num_layers, word_att_dim, sent_att_dim, use_layer_norm, dropout)

        self.fc = nn.Linear(2 * sent_gru_hidden_dim, 1)
        self.sig = nn.Sigmoid()

        self.use_layer_nome = use_layer_norm
        self.dropout = dropout

    def forward(self, line_tensor, adj):
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

        self.sage = SAGE(2 * word_gru_hidden_dim, 2 * sent_gru_hidden_dim, dropout)

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

        sents, word_att_weights = self.word_attention(line_tensor, line_lengths)

        sents = self.dropout(sents)
        packed_sents = self.sage(sents,adj)
        line_tensor = packed_sents
        sent_att_weights = None


        return line_tensor, word_att_weights, sent_att_weights, sents


class WordAttention(nn.Module):
    """
    Word-level attention module.
    """

    def __init__(self, vocab_size, embed_dim, gru_hidden_dim, gru_num_layers, att_dim, use_layer_norm, dropout):
        super(WordAttention, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embed_dim)

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

        # Sort sents by decreasing order in sentence lengths
        sent_lengths, sent_perm_idx = sent_lengths.sort(dim=0, descending=True)
        sents = sents[sent_perm_idx]

        sents = self.embeddings(sents.cuda())

        packed_words = pack_padded_sequence(sents, lengths=sent_lengths.tolist(), batch_first=True)


        valid_bsz = packed_words.batch_sizes


        packed_words, _ = self.gru(packed_words)

        if self.use_layer_norm:
            normed_words = self.layer_norm(packed_words.data)
        else:
            normed_words = packed_words

        # Word Attenton
        att = torch.tanh(self.attention(normed_words.data))
        att = self.context_vector(att).squeeze(1)

        val = att.max() # (u_w : context vector)
        att = torch.exp(att - val) # att.size: (n_words)


        att, _ = pad_packed_sequence(PackedSequence(att, valid_bsz), batch_first=True)

        att_weights = att / torch.sum(att, dim=1, keepdim=True)

        # Restore as sentences by repadding sents为(h_it)
        sents, _ = pad_packed_sequence(packed_words, batch_first=True)

        # Compute sentence vectors 聚合操作
        sents = sents * att_weights.unsqueeze(2)
        sents = sents.sum(dim=1)

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
        self.conv3 = SAGEConv(hid_dim, hid_dim)
        self.conv4 = SAGEConv(hid_dim, hid_dim)
        self.conv5 = SAGEConv(hid_dim, hid_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv3(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv4(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv5(x, adj)
        return F.relu(x)