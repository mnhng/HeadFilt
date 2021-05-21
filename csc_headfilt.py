#!/usr/bin/env python3
import argparse
import itertools
import json
import pathlib

import pandas as pd
import torch
from torch.utils.data import DataLoader
import transformers as tfm

from nnblk import HierarchicalEmbedding
import util


def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--batch_size', type=int, default=16)
    argparser.add_argument('--config', required=True)
    argparser.add_argument('--checkpoint', '-c', default='bert-base-chinese')
    argparser.add_argument('--emb_weights', '-e', required=True)
    argparser.add_argument('--dim', '-d', type=int, default=512)
    argparser.add_argument('--margin', '-m', type=float, required=True)
    argparser.add_argument('--factor', '-f', type=float, required=True)
    argparser.add_argument('--outpath', '-o', default='out/csc_hf')
    return argparser.parse_args()


def _infer(net, batches, vocab, hier_emb, margin, factor):
    net.eval()
    hier_emb.eval()
    location = []
    correction = []
    dev = next(net.parameters()).device
    with torch.no_grad():
        factor_mat = hier_emb(torch.tensor(range(len(vocab)), device=dev))

        for in_ in batches:
            input_ = torch.nn.utils.rnn.pad_sequence(in_, batch_first=True)
            in_mask = util.pad_mask(in_, batch_first=True).to(input_.device)
            prob = net(input_, attention_mask=in_mask)[0]
            util.emb_filter(in_, prob, factor_mat, margin, factor)
            pred_cor = torch.argmax(prob, dim=2)

            pred_loc = input_ != pred_cor
            lengths = [len(i) for i in in_]
            location.extend([yh[:l].int().cpu().numpy()[1:-1] for yh, l in zip(pred_loc, lengths)])
            correction.extend([vocab.convert_ids_to_tokens(yh[:l].cpu().numpy()[1:-1]) for yh, l in zip(pred_cor, lengths)])

    return location, correction


def main(args):
    dev = torch.device('cuda')

    with open(args.config) as fh:
        config = json.load(fh)
    outpath = pathlib.Path(args.outpath)
    outpath.mkdir(parents=True, exist_ok=True)

    vocab = tfm.BertTokenizer.from_pretrained('bert-base-chinese-ext')
    model = tfm.BertForMaskedLM.from_pretrained(args.checkpoint).to(dev)
    util.resize_bert(model, len(vocab))

    ch2idx = {k: i for i, k in enumerate(vocab.convert_ids_to_tokens(range(len(vocab))))}
    new_emb = HierarchicalEmbedding(len(vocab), args.dim, ch2idx)
    new_emb.load_state_dict(torch.load(args.emb_weights))
    new_emb.to(dev)

    if config.get('sep_pred_corr'):
        m_batches = []
        for d in config['test']:
            input_ = util.to_tensors(vocab, dev, util.read(d['input']))
            m_batches.append(DataLoader(input_, args.batch_size, collate_fn=lambda x: x))

        output = [_infer(model, batches, vocab, new_emb, args.margin, args.factor) for batches in m_batches]
        output = list(itertools.chain(*output))
        filt_res = util.eval_csc13(outpath, 'filt', *output, *config['test'])
    else:
        input_ = util.to_tensors(vocab, dev, util.read(config['test']['input']))
        batches = DataLoader(input_, args.batch_size, collate_fn=lambda x: x)

        output = _infer(model, batches, vocab, new_emb, args.margin, args.factor)
        filt_res = util.correction(outpath, 'filt', *output, **config['test'])

    print(pd.DataFrame([filt_res]))


if __name__ == '__main__':
    main(get_args())
