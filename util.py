import itertools
import json
import subprocess

from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def _flatten(list_of_list):
    return list(itertools.chain(*list_of_list))


def to_tensors(tokenizer, device, lines):
    lines = [tokenizer.encode(l, add_special_tokens=True, return_tensors='pt') for l in lines]
    lines = [l.squeeze(0).to(device) for l in lines]
    return lines


def pad_mask(sequences, batch_first=False):
    ret = [torch.ones(len(s)) for s in sequences]
    return pad_sequence(ret, batch_first=batch_first)


def training_data(path_sets):
    t_input = _flatten([read(p['input']) for p in path_sets])
    t_output = _flatten([read(p['correction']) for p in path_sets])

    assert len(t_input) == len(t_output)
    return t_input, t_output


def resize_bert(model, new_num_tokens):
    old_decoder = model.cls.predictions.decoder
    old_num_tokens, in_features = old_decoder.weight.shape
    old_bias = model.cls.predictions.bias

    assert old_decoder.bias is None
    new_decoder = torch.nn.Linear(in_features, new_num_tokens, bias=False)
    new_decoder.to(old_decoder.weight.device)

    new_bias = torch.Tensor(new_num_tokens)
    torch.nn.init.uniform_(new_bias, -.1, .1)

    # Copy token embeddings from the previous weights
    num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
    new_decoder.weight.data[:num_tokens_to_copy, :] = old_decoder.weight.data[:num_tokens_to_copy, :]
    new_bias[:num_tokens_to_copy] = old_bias[:num_tokens_to_copy]

    model.cls.predictions.decoder = new_decoder
    model.cls.predictions.bias = torch.nn.parameter.Parameter(new_bias.to(old_bias.device))
    model.resize_token_embeddings(new_num_tokens)


def sim_fn(x, y):
    x = x / torch.norm(x, dim=1, keepdim=True)
    y = y / torch.norm(y, dim=1, keepdim=True)
    # https://github.com/pytorch/pytorch/issues/27209
    z = torch.cdist(x.unsqueeze(0), y.unsqueeze(0), p=2)
    return z.squeeze(0)


def emb_filter(in_, pred_prob, low_dim, margin, scale):
    # in_ is a list of vectors
    # pred_prob is a tensor of dim [batch_size, max_len, D]
    factor = np.log(low_dim.shape[0] - 1) / (scale*margin)
    for i, row in enumerate(in_):
        delta = sim_fn(low_dim[row], low_dim) - margin
        pred_prob[i, :len(row)] -= torch.log1p((delta * factor).exp())


def read(fpath):
    with open(fpath) as fh:
        return [line.strip('\n') for line in fh]


def load_confusion_set(symmetric=False, simplified=False):
    if simplified:
        f1, f2 = 'data/Bakeoff2013_CharacterSet_SimilarShape.smp', 'data/Bakeoff2013_CharacterSet_SimilarPronunciation.smp'
    else:
        f1, f2 = 'data/Bakeoff2013_CharacterSet_SimilarShape.txt', 'data/Bakeoff2013_CharacterSet_SimilarPronunciation.txt'

    confusion_set = {}
    for line in read(f1):
        c, cset = line.split(',')
        confusion_set[c] = c + cset + confusion_set.get(c, '')

    for line in read(f2)[1:]:
        col = line.split('\t')
        c = col[0]
        confusion_set[c] = confusion_set.get(c, '') + ''.join(col[1:])
    confusion_set = {k: list(set(v)) for k, v in confusion_set.items()}

    if symmetric:
        all_chars = list(confusion_set.keys()) + list(itertools.chain(*confusion_set.values()))
        for ch in set(all_chars):
            for cch in confusion_set.get(ch, []):
                confusion_set[cch] = list(set(confusion_set.get(cch, []) + [ch]))
            confusion_set[ch] = list(set(confusion_set.get(ch, []) + [ch]))

    return confusion_set


def cfs2mat(confusion_set, characters, tensor=False):
    assert len(characters) == len(set(characters))
    N = len(characters)
    ret = torch.zeros((N, N)) if tensor else np.zeros((N, N))
    ch2id = {c: i for i, c in enumerate(characters)}
    for i, c in enumerate(characters):
        idx = [ch2id[d] for d in confusion_set.get(c, []) if d in ch2id]
        ret[i, idx] = 1
    return ret


def report_csc13(pred_paths, true_paths, rep_paths):
    for i, (pfp, tfp, rfp) in enumerate(zip(pred_paths, true_paths, rep_paths)):
        cmd = f'java -jar tools/sighan7csc.jar -s {i+1} -i {pfp} -t {tfp} -o {rfp}'
        subprocess.call(cmd.split())

    with open(rep_paths[0]) as fh:
        lines = [l.split() for l in fh]
    lines = [l for l in lines if '=' in l]
    cols = ['FA', 'DA', 'DP', 'DR', 'DF1']
    ret = {c: float(l[l.index('=')+1]) for c, l in zip(cols, lines)}

    with open(rep_paths[1]) as fh:
        lines = [l.split() for l in fh]
    lines = [l for l in lines if '=' in l][1:]
    cols = ['CA', 'CP']
    ret.update({c: float(l[l.index('=')+1]) for c, l in zip(cols, lines)})

    return ret


def eval_csc13(out_dir, fname, pred_loc, pred_chr, corr_loc, corr_chr, *args):
    out_paths = f'{out_dir}/{fname}.pre.hyp', f'{out_dir}/{fname}.cor.hyp'
    rep_paths = f'{out_dir}/{fname}.pre.rep', f'{out_dir}/{fname}.cor.rep'
    with open(out_paths[0], 'w') as fh:
        for t, l in zip(read(args[0]['tag']), pred_loc):
            no_errors = l.sum()
            out = [t]
            if no_errors:
                for i in np.where(l)[0]:
                    out.append(i + 1)
            else:
                out.append(no_errors)
            print(*out, sep=', ', file=fh)

    with open(out_paths[1], 'w') as fh:
        for t, l, c in zip(read(args[1]['tag']), corr_loc, corr_chr):
            if l.sum():
                out = [t]
                for i in np.where(l)[0]:
                    out.append(i + 1)
                    out.append(c[i])
                print(*out, sep=', ', file=fh)
    ref_paths = [dataset['ref'] for dataset in args]
    return report_csc13(out_paths, ref_paths, rep_paths)


def eval_csc15(out_path, ref_path, report_path):
    cmd = f'java -jar tools/sighan15csc.jar -i {out_path} -t {ref_path} -o {report_path}'
    subprocess.call(cmd.split())
    with open(report_path) as fh:
        lines = [l.strip().split() for l in fh]
    lines = [l for l in lines if '=' in l]
    cols = ['FA', 'DA', 'DP', 'DR', 'DF1', 'CA', 'CP', 'CR', 'CF1']
    return {c: float(l[l.index('=')+1]) for c, l in zip(cols, lines)}


def correction(out_dir, fname, location, correction, **kwargs):
    hyp_path = f'{out_dir}/{fname}.hyp'
    with open(hyp_path, 'w') as fh:
        for t, l, c in zip(read(kwargs['tag']), location, correction):
            no_errors = l.sum()
            out = [t]
            if no_errors:
                for i in np.where(l)[0]:
                    out.append(i + 1)
                    out.append(c[i])
            else:
                out.append(0)
            print(*out, sep=', ', file=fh)
    return eval_csc15(hyp_path, kwargs['ref'], f'{out_dir}/{fname}.rep')


class Progress:
    def __init__(self, max_steps, max_epochs):
        self.step = 1
        self._epoch = 1
        self.max_steps = max_steps
        self.max_epochs = max_epochs

    def should_continue(self):
        return self.step <= self.max_steps and self._epoch <= self.max_epochs

    def finish_one_step(self):
        self.step += 1

    def finish_one_epoch(self):
        self._epoch += 1

    def extend(self, steps=None, epochs=None):
        if epochs:
            self.max_epochs += epochs
        if steps:
            self.max_steps += steps

    def epoch(self):
        return self._epoch - 1
