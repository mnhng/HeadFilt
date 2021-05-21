#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from .tree import Tree
from .evaluator import StackEvaluator

from .inventory import SIMPLE_COMPONENTS
from .inventory import OPT_COMPONENTS
from .inventory import OPT_COMPONENTS_COMPAT

IDC = {'⿰': 2, '⿱': 2, '⿲': 3, '⿳': 3,
       '⿴': 2, '⿵': 2, '⿶': 2, '⿷': 2,
       '⿸': 2, '⿹': 2, '⿺': 2, '⿻': 2}


def parse_ids_rules(path):
    lines = [l.strip().split('\t') for l in open(path)]
    return {l[1]: l[2] for l in lines if len(l) > 1 and l[1] != l[2]}


def make_bnode(node, *children):
    replace = {'⿲': '⿰', '⿳': '⿱'}
    replacement = replace.get(node.operand)
    if replacement:
        aux_node = Tree(operator='', operand=replacement)
        aux_node.add_child(children[1])
        aux_node.add_child(children[2])
        node.operand = replacement
        node.add_child(children[0])
        node.add_child(aux_node)
    else:
        node.add_child(children[0])
        node.add_child(children[1])

    return node


def var_node(x):
    return Tree(operator='', operand=x)


def unwrap_var(n):
    return n.operand


class IdsConverter(object):
    def __init__(self, rule_files, max_len, basic_units):
        self.rules = {}
        for fp in rule_files:
            self.rules.update(parse_ids_rules(fp))

        self.MAX_SEQ_LEN = max_len
        self.basic_units = basic_units
        self.btree_builder = StackEvaluator(IDC, var_node, var_node, unwrap_var, make_bnode)

    def _decompose(self, char, verbose=False):
        if verbose:
            print(char)
        if char in self.basic_units:
            return [char]
        if char in '[GHUAJKTXV]':
            return []

        components = self.rules[char]

        ret = []
        for part in components:
            ret += self._decompose(part, verbose)

        return ret

    def to_seq(self, ch):
        ret = self._decompose(ch)
        return ret

    def to_binary_tree(self, ch):
        return self.btree_builder.evaluate(self.to_seq(ch))

    def to_seq_triplet(self, ch):
        tmp = self.to_binary_tree(ch)
        for i, n in enumerate(tmp.visit()):
            setattr(n, 'idx', i)
        values = [self.basic_units[n.operand] for n in tmp.visit()]
        left_indices = [n.children[0].idx if not n.is_leaf() else -1 for n in tmp.visit()]
        right_indices = [n.children[1].idx if not n.is_leaf() else -1 for n in tmp.visit()]
        return values, left_indices, right_indices

    def __contains__(self, value):
        return value in self.rules or value in self.basic_units

    def __len__(self):
        return len(self.basic_units)


def get_converter(legacy=False, fine_decomp=False):
    root = Path(__file__).parent
    patch_path = root/'ids_update.txt'
    if legacy:
        main_path = root/'ids_emnlp.txt'
        max_len = 30
        basic_units = OPT_COMPONENTS
    elif fine_decomp:
        main_path = root/'ids_latest.txt'
        max_len = 60
        basic_units = SIMPLE_COMPONENTS
    else:
        main_path = root/'ids_latest.txt'
        max_len = 30
        basic_units = OPT_COMPONENTS_COMPAT

    return IdsConverter([main_path, patch_path], max_len, basic_units)
