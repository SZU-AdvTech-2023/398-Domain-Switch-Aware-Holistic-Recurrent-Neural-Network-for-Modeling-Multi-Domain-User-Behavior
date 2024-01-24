# -*- coding: utf-8 -*-
import torch
from torch import nn

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs

class GRU4Rec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(GRU4Rec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.forward_layers = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
        self.forward_layernorms = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.gru = torch.nn.GRU(input_size=args.hidden_units, hidden_size=args.hidden_units, num_layers=args.num_blocks)
        self.mse = nn.MSELoss(reduction='sum')  # define mse for construction loss


    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs = self.emb_dropout(seqs)

        #[B,L,D]
        seqs = torch.transpose(seqs, 0, 1)
        #[L,B,D]
        seqs, hidden = self.gru(seqs)
        seqs = torch.transpose(seqs, 0, 1)

        seqs = self.forward_layernorms(seqs)
        seqs = self.forward_layers(seqs)

        return seqs

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, pos_domain_switch):
        log_feats = self.log2feats(log_seqs)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        pos_domain_switch_embs = self.item_emb(torch.LongTensor(pos_domain_switch).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        pos_domain_switch_logits = (log_feats * pos_domain_switch_embs).sum(dim=-1)
        return pos_logits, neg_logits, pos_domain_switch_logits

    def behavior_regularizer(self, domain_switch_behavior, next_behavior):
        domain_switch_behavior_embs = self.item_emb(torch.LongTensor(domain_switch_behavior).to(self.dev)) # [B,num_dom, L] -> [B,num_dom,L,D]
        next_behavior_embs = self.item_emb(torch.LongTensor(next_behavior).to(self.dev)) # [B,num_dom, L] -> [B,num_dom,L,D]
        regularizer_loss = self.mse(domain_switch_behavior_embs, next_behavior_embs)

        return regularizer_loss

    def predict(self, user_ids, log_seqs, item_ids):
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet
        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste
        item_embs = self.item_emb(torch.LongTensor(item_ids).to(self.dev))  # (U, I, C)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # preds # (U, I)