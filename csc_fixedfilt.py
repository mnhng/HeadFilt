#!/usr/bin/env python3
import argparse
import itertools
import json
import pathlib

import pandas as pd
import torch
from torch.utils.data import DataLoader
import transformers as tfm

import util


def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--batch_size', type=int, default=16)
    argparser.add_argument('--config', required=True)
    argparser.add_argument('--checkpoint', '-c', default='bert-base-chinese')
    argparser.add_argument('--outpath', '-o', default='out/csc_ff')
    return argparser.parse_args()


def filter_set(in_, pred_prob, confusion_set):
    for i, row in enumerate(in_):
        for j, char in enumerate(row.cpu().numpy()):
            k = confusion_set.get(char, None)
            if k is not None:
                mask = torch.ones_like(pred_prob[i, j], dtype=torch.bool)
                mask[k] = 0
                pred_prob[i, j].masked_fill_(mask, float('-inf'))


def _infer(net, batches, vocab, confusion_set):
    net.eval()
    location = []
    correction = []
    with torch.no_grad():
        for in_ in batches:
            input_ = torch.nn.utils.rnn.pad_sequence(in_, batch_first=True)
            in_mask = util.pad_mask(in_, batch_first=True).to(input_.device)
            prob = net(input_, attention_mask=in_mask)[0]
            filter_set(in_, prob, confusion_set)
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
    cfs = util.load_confusion_set(simplified=config['simplified'])
    outpath = pathlib.Path(args.outpath)
    outpath.mkdir(parents=True, exist_ok=True)

    vocab = tfm.BertTokenizer.from_pretrained('bert-base-chinese-ext')
    model = tfm.BertForMaskedLM.from_pretrained(args.checkpoint).to(dev)

    cfs = {vocab.encode(k)[0]: vocab.encode(v) for k, v in cfs.items()}
    cfs = {k: torch.tensor(v, device=dev) for k, v in cfs.items()}

    if config.get('sep_pred_corr'):
        m_batches = []
        for d in config['test']:
            input_ = util.to_tensors(vocab, dev, util.read(d['input']))
            m_batches.append(DataLoader(input_, args.batch_size, collate_fn=lambda x: x))

        output = [_infer(model, batches, vocab, cfs) for batches in m_batches]
        output = list(itertools.chain(*output))
        filt_res = util.eval_csc13(outpath, 'filt', *output, *config['test'])
    else:
        input_ = util.to_tensors(vocab, dev, util.read(config['test']['input']))
        batches = DataLoader(input_, args.batch_size, collate_fn=lambda x: x)

        output = _infer(model, batches, vocab, cfs)
        filt_res = util.correction(outpath, 'filt', *output, **config['test'])

    print(pd.DataFrame([filt_res]))


if __name__ == '__main__':
    main(get_args())
