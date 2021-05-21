import itertools

import torch
import torch.nn as nn

from .ids_util import get_converter


def insert(lol, *arrays):
    for x in zip(lol, *arrays):
        l, elements = x[0], x[1:]
        l.extend(elements)


class HierarchicalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, char2index):
        super(HierarchicalEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.set_token_dict(char2index)

    def set_token_dict(self, corpus_dict):
        conv = get_converter()

        self.subchar_dict = {}
        new_size = len(conv)
        for k, v in corpus_dict.items():
            if k in conv:
                self.subchar_dict[v] = conv.to_seq_triplet(k)
            else:
                new_id = new_size
                self.subchar_dict[v] = [new_id], [-1], [-1]
                conv.rules[k] = k  # dummy rule
                new_size += 1

        self.sub_char = nn.Embedding(new_size, 5*self.embedding_dim)

        self.Wlh = nn.Linear(self.embedding_dim, 5*self.embedding_dim)
        self.Wrh = nn.Linear(self.embedding_dim, 5*self.embedding_dim)

        self.lut = {}

    def _proc_leaf(self, nodes):
        # inconsequential here because the insert function will skip the last element
        if len(nodes) == 0:
            return [[], []]

        sum_ = torch.cat([n[3].view(1, -1) for n in nodes])
        ig, lfg, rfg, og, ug = torch.chunk(sum_, 5, dim=1)

        c = torch.sigmoid(ig) * torch.tanh(ug)
        h = torch.sigmoid(og) * torch.tanh(c)
        return [h, c]

    def _proc_int(self, nodes, ln, rn):
        # inconsequential here because the insert function will skip the last element
        if len(nodes) == 0:
            return [[], []]

        sum_ = torch.cat([n[3].view(1, -1) for n in nodes]) + \
               self.Wlh(torch.cat([n[4].view(1, -1) for n in ln])) + \
               self.Wrh(torch.cat([n[4].view(1, -1) for n in rn]))

        ig, lfg, rfg, og, ug = torch.chunk(sum_, 5, dim=1)

        c = torch.sigmoid(lfg) * torch.cat([n[5].view(1, -1) for n in ln]) + \
            torch.sigmoid(rfg) * torch.cat([n[5].view(1, -1) for n in rn]) + \
            torch.sigmoid(ig) * torch.tanh(ug)
        h = torch.sigmoid(og) * torch.tanh(c)
        return [h, c]

    def refresh(self, input):
        key_list = []
        proc_list = []

        for k in torch.unique(input).cpu().numpy():
            if not self.training and k in self.lut:
                continue
            key_list.append(k)
            proc_list.append([list(a) for a in zip(*self.subchar_dict[k])])

        for nodes in itertools.zip_longest(*proc_list):
            nodes_list = []
            idx_list = []
            for i, n in enumerate(nodes):
                if n is not None:
                    nodes_list.append(n)
                    idx_list.append(i)

            in_block = input.new([n[0] for n in nodes_list])
            insert(nodes_list, self.sub_char(in_block))

            leaf_list = []
            int_list = []
            left_children = []
            right_children = []
            for i, n in zip(idx_list, nodes_list):
                if n[1] == n[2] == -1:
                    leaf_list.append(n)
                else:
                    int_list.append(n)
                    left_children.append(proc_list[i][n[1]])
                    right_children.append(proc_list[i][n[2]])

            insert(leaf_list, *self._proc_leaf(leaf_list))
            insert(int_list, *self._proc_int(int_list, left_children, right_children))

        for k, nodes in zip(key_list, proc_list):
            self.lut[k] = nodes[-1][4].view(1, -1)

    def forward(self, input):
        self.refresh(input)
        to_emb = torch.cat([self.lut[k] for k in input.view(input.numel()).cpu().numpy()])

        return to_emb.view(input.shape + (self.embedding_dim,))
