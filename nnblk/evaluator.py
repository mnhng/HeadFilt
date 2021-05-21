class StackEvaluator:
    def __init__(self, op_dict, build_op, build_var, get_op, compute):
        self.op_dict = op_dict
        self.build_op = build_op
        self.build_var = build_var
        self.get_op = get_op
        self.compute = compute

    def evaluate(self, sequence):
        operator_stack = []
        operand_stack = []
        size_stack = [0]
        for unit in sequence:
            if unit in self.op_dict:
                operator_stack.append(self.build_op(unit))
                size_stack.append(0)
            else:
                operand_stack.append(self.build_var(unit))
                size_stack[-1] += 1

            while True:
                if len(operator_stack) == 0:
                    break
                op = operator_stack[-1]
                nb_operands = self.op_dict[self.get_op(op)]
                if size_stack[-1] < nb_operands:
                    break

                operator_stack.pop()
                variables = []
                for i in range(-nb_operands, 0):
                    variables.append(operand_stack.pop(i))

                operand_stack.append(self.compute(op, *variables))

                size_stack.pop()
                size_stack[-1] += 1

        assert len(operator_stack) == 0
        assert len(operand_stack) == 1
        assert len(size_stack) == 1
        assert size_stack[0] == 1

        return operand_stack[0]
