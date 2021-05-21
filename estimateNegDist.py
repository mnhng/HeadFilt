#!/usr/bin/env python3
import argparse
import logging
import itertools
import transformers
import torch

from nnblk import HierarchicalEmbedding
import util


def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dim', '-d', type=int, default=512)
    argparser.add_argument('--simplified', action='store_true')
    argparser.add_argument('--checkpoint', '-c', required=True)
    argparser.add_argument('--margin', '-m', type=float, required=True)
    return argparser.parse_args()


def dist_fn(x):
    x = x / torch.norm(x, dim=1, keepdim=True)
    # https://github.com/pytorch/pytorch/issues/27209
    x = x.unsqueeze(0)
    z = torch.cdist(x, x, p=2)
    return z.squeeze(0)


def dist_matrix(embedder, tokenizer, characters):
    embedder.eval()
    input_ = tokenizer.convert_tokens_to_ids(characters)
    input_ = torch.tensor(input_).cuda()
    with torch.no_grad():
        return dist_fn(embedder(input_))


def estimate(emb, tokenizer, cfs, characters, args):
    tpred = dist_matrix(emb, tokenizer, characters)
    ttrue = util.cfs2mat(cfs, characters, True).cuda()
    avg_dist = tpred[ttrue == 0].mean().item()
    print('avg dist between dissimilar points', avg_dist)
    print('dist factor', avg_dist / args.margin - 1)


def main(args):
    cfs = util.load_confusion_set(True, simplified=args.simplified)
    all_chars = list(cfs.keys()) + list(itertools.chain(*cfs.values()))
    all_chars = sorted(list(set(all_chars)))
    print(len(all_chars))

    vocab = transformers.BertTokenizer.from_pretrained('bert-base-chinese-ext')

    ch2idx = {k: i for i, k in enumerate(vocab.convert_ids_to_tokens(range(len(vocab))))}
    new_emb = HierarchicalEmbedding(len(vocab), args.dim, ch2idx)
    new_emb.load_state_dict(torch.load(args.checkpoint))
    new_emb.cuda()
    print(new_emb)

    estimate(new_emb, vocab, cfs, all_chars, args)


if __name__ == '__main__':
    logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
    main(get_args())
