#!/usr/bin/env python3
import argparse
import itertools
import json
import pathlib

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import transformers as tfm

import util


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--batch_size', type=int, default=16)
    argparser.add_argument('--config', required=True)
    argparser.add_argument('--max_steps', '-t', type=int, default=2500000)
    argparser.add_argument('--max_train_epoch', '-e', type=int, default=20)
    argparser.add_argument('--lr', type=float, default=3e-5)
    argparser.add_argument('--seed', '-s', type=int, default=1234)
    argparser.add_argument('--checkpoint', '-c', default='bert-base-chinese')
    argparser.add_argument('--outpath', '-o', required=True)
    return argparser.parse_args()


def collate(x):
    return zip(*x)


def mean(x):
    return sum(x) / len(x)


def _train(net, batches, opt, sched, progress):
    net.train()

    while progress.should_continue():
        losses = []
        for in_, ref_cor in batches:
            opt.zero_grad()
            input_ = torch.nn.utils.rnn.pad_sequence(in_, batch_first=True)
            in_mask = util.pad_mask(in_, batch_first=True).to(input_.device)
            ref_cor = torch.nn.utils.rnn.pad_sequence(ref_cor, batch_first=True)
            prob = net(input_, attention_mask=in_mask)[0]

            in_mask = in_mask.bool()
            loss = torch.nn.functional.cross_entropy(prob[in_mask], ref_cor[in_mask])
            loss.backward()
            opt.step()
            sched.step()

            losses.append(loss.item())
            progress.finish_one_step()
            if not progress.should_continue():
                break
        progress.finish_one_epoch()
        print(f'Epoch {progress.epoch()}: Loss {mean(losses):.4f}')


def _infer(net, batches, vocab):
    net.eval()
    location = []
    correction = []
    with torch.no_grad():
        for in_ in batches:
            input_ = torch.nn.utils.rnn.pad_sequence(in_, batch_first=True)
            in_mask = util.pad_mask(in_, batch_first=True).to(input_.device)
            prob = net(input_, attention_mask=in_mask)[0]
            pred_cor = torch.argmax(prob, dim=2)

            pred_loc = input_ != pred_cor
            lengths = [len(i) for i in in_]
            location.extend([yh[:l].int().cpu().numpy()[1:-1] for yh, l in zip(pred_loc, lengths)])
            correction.extend([vocab.convert_ids_to_tokens(yh[:l].cpu().numpy()[1:-1]) for yh, l in zip(pred_cor, lengths)])

    return location, correction


def main(args):
    torch.manual_seed(args.seed)
    dev = torch.device('cuda')

    with open(args.config) as fh:
        config = json.load(fh)
    outpath = pathlib.Path(args.outpath)
    outpath.mkdir(parents=True, exist_ok=True)

    vocab = tfm.BertTokenizer.from_pretrained('bert-base-chinese-ext')
    t_input, t_output = util.training_data(config['train'])
    # create model
    model = tfm.BertForMaskedLM.from_pretrained(args.checkpoint).to(dev)
    util.resize_bert(model, len(vocab))
    opt = tfm.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, correct_bias=True)

    batches = DataLoader(
            list(zip(util.to_tensors(vocab, dev, t_input), util.to_tensors(vocab, dev, t_output))),
            args.batch_size,
            collate_fn=collate)
    training_steps = len(batches) * args.max_train_epoch
    sched = get_linear_schedule_with_warmup(opt, training_steps // 10, training_steps)

    progress = util.Progress(args.max_steps, args.max_train_epoch)
    _train(model, batches, opt, sched, progress)

    if config.get('sep_pred_corr'):
        m_batches = []
        for d in config['test']:
            input_ = util.to_tensors(vocab, dev, util.read(d['input']))
            m_batches.append(DataLoader(input_, args.batch_size, collate_fn=lambda x: x))

        output = [_infer(model, batches, vocab) for batches in m_batches]
        output = list(itertools.chain(*output))
        res = util.eval_csc13(outpath, 'base', *output, *config['test'])

    else:
        input_ = util.to_tensors(vocab, dev, util.read(config['test']['input']))
        batches = DataLoader(input_, args.batch_size, collate_fn=lambda x: x)

        output = _infer(model, batches, vocab)
        res = util.correction(outpath, 'base', *output, **config['test'])

    print(pd.DataFrame([res]))
    model.save_pretrained(outpath)


if __name__ == '__main__':
    main(get_args())
