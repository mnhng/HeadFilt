#!/usr/bin/env python3
import argparse
import logging
import itertools
import transformers
import torch
import numpy as np

from nnblk import HierarchicalEmbedding
import util


def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--steps', '-t', type=int, default=500000)
    argparser.add_argument('--lr', type=float, default=1e-4)
    argparser.add_argument('--seed', '-s', type=int, default=1234)
    argparser.add_argument('--dim', '-d', type=int, default=512)
    argparser.add_argument('--simplified', action='store_true')
    argparser.add_argument('--margin', '-m', type=float, default=.4)
    argparser.add_argument('--out', '-o', required=True)
    return argparser.parse_args()


def _acc(pred, true): return (true == pred).sum()


def sim_fn(x):
    x = x / torch.norm(x, dim=1, keepdim=True)
    # https://github.com/pytorch/pytorch/issues/27209
    x = x.unsqueeze(0)
    z = torch.cdist(x, x, p=2)
    return z.squeeze(0)


def contrastive_loss(dist, label, margin):
    pos = (dist - margin).clamp(min=0)
    neg = (margin - dist).clamp(min=0)
    return torch.where(label == 1, pos, neg).sum()


def eval_model(emb, tokenizer, characters, margin):
    emb.eval()
    input_ = tokenizer.convert_tokens_to_ids(characters)
    input_ = torch.tensor(input_).cuda()
    with torch.no_grad():
        pred = sim_fn(emb(input_))
    return pred < margin


def imitate(emb, tokenizer, characters, cfs, args):
    opt = torch.optim.Adam(emb.parameters(), args.lr)

    losses = []
    base_acc = 0
    batch_size = min(args.batch_size, len(characters))
    for i in range(args.steps):
        samples = np.random.choice(characters, size=batch_size, replace=False)
        input_ = torch.tensor(tokenizer.convert_tokens_to_ids(samples)).cuda()
        true = util.cfs2mat(cfs, samples, True).float().cuda()

        emb.train()
        opt.zero_grad()
        pred = sim_fn(emb(input_))
        loss = contrastive_loss(pred, true, args.margin)
        loss.backward()
        losses.append(loss.item())

        if (i+1) % 1000 == 0:
            tpred = eval_model(emb, tokenizer, characters, args.margin)
            ttrue = util.cfs2mat(cfs, characters, True).cuda()
            acc = _acc(tpred, ttrue.bool()).item()
            total = len(characters)**2
            print(i+1, np.mean(losses), total - acc, acc / total)
            acc /= total
            if acc > base_acc:
                torch.save(emb.state_dict(), args.out)
                base_acc = acc
                print(f'save checkpoint with acc={acc}')
            if acc == 1:
                print('success')
                break
            losses = []
            emb.train()
        opt.step()
    print('best score', base_acc)


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfs = util.load_confusion_set(symmetric=True, simplified=args.simplified)
    all_chars = list(cfs.keys()) + list(itertools.chain(*cfs.values()))
    train_chars = sorted(list(set(all_chars)))

    vocab = transformers.BertTokenizer.from_pretrained('bert-base-chinese-ext')

    ch2idx = {k: i for i, k in enumerate(vocab.convert_ids_to_tokens(range(len(vocab))))}
    new_emb = HierarchicalEmbedding(len(vocab), args.dim, ch2idx)
    new_emb.cuda()
    print(new_emb)

    test_chars = np.random.choice(train_chars, size=min(10, len(train_chars)), replace=False)
    imitate(new_emb, vocab, train_chars, cfs, args)

    pred_mat = eval_model(new_emb, vocab, test_chars, args.margin).cpu().numpy().astype(int)
    true_mat = util.cfs2mat(cfs, test_chars).astype(int)
    print('predicted')
    print(pred_mat)
    print('ground truth')
    print(true_mat)
    print(_acc(pred_mat, true_mat) / pred_mat.size, *test_chars)


if __name__ == '__main__':
    logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
    main(get_args())
