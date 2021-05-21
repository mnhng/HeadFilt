import itertools

BLANK = ''


class Tree(object):
    def __init__(self, operator, operand):
        self.operator = operator
        self.operand = operand
        self.parent = None
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def visit(self):
        for c in self.children:
            yield from c.visit()
        yield self

    def __len__(self):
        count = 1
        for child in self.children:
            count += len(child)
        return count

    def depth(self):
        count = 1
        for child in self.children:
            count = max(count, 1 + child.depth())

        return count

    def pad(self, depth):
        if depth < 1:
            raise ValueError()
        if depth == 1:
            assert self.is_leaf()
            return

        if self.is_leaf():
            self.add_child(Tree(BLANK, BLANK))
            self.add_child(Tree(BLANK, BLANK))

        for c in self.children:
            c.pad(depth - 1)

        return self

    def prune(self):
        for node in self.visit():
            if node.operator == BLANK and node.operand == BLANK:
                parent = node.parent
                parent.children = [c for c in parent.children if c is not node]

    def is_null(self):
        return len(self.operator) == 0 and len(self.operand) == 0

    def is_leaf(self):
        return all([c.is_null() for c in self.children])

    def __str__(self):
        rep = '%s:%s' % (self.operator, self.operand)
        if len(self.children) == 0:
            return rep

        return '(' + ', '.join([rep] + [c.__str__() for c in self.children]) + ')'


BLANK_NODE = Tree(BLANK, BLANK)
def merge_tree(trees):
    ret = Tree(operator=[n.operator for n in trees], operand=[n.operand for n in trees])

    for subtrees in itertools.zip_longest(*[n.children for n in trees], fillvalue=BLANK_NODE):
        ret.add_child(merge_tree(subtrees))

    return ret
