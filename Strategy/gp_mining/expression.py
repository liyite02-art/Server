from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Iterable, Optional


TYPE_BASIC = "basic"
TYPE_DELAY = "const_delay"
TYPE_QUAN = "const_quan"
TYPE_EXP = "const_exp"
CONST_TYPES = {TYPE_DELAY, TYPE_QUAN, TYPE_EXP}


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    func: Any
    arg_types: tuple[str, ...]
    return_type: str = TYPE_BASIC


@dataclass(frozen=True)
class Node:
    kind: str
    value: Any
    children: tuple["Node", ...] = ()
    arg_types: tuple[str, ...] = ()

    def __str__(self) -> str:
        if self.kind == "terminal":
            return str(self.value)
        if self.kind == "const":
            return repr(self.value)
        args = ", ".join(str(child) for child in self.children)
        return f"{self.value}({args})"

    @property
    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(child.depth for child in self.children)

    @property
    def length(self) -> int:
        return 1 + sum(child.length for child in self.children)

    def iter_paths(self, prefix: tuple[int, ...] = ()) -> Iterable[tuple[tuple[int, ...], "Node"]]:
        yield prefix, self
        for i, child in enumerate(self.children):
            yield from child.iter_paths(prefix + (i,))

    def subtree(self, path: tuple[int, ...]) -> "Node":
        node = self
        for idx in path:
            node = node.children[idx]
        return node

    def replace(self, path: tuple[int, ...], new_node: "Node") -> "Node":
        if not path:
            return new_node
        idx = path[0]
        children = list(self.children)
        children[idx] = children[idx].replace(path[1:], new_node)
        return Node(self.kind, self.value, tuple(children), self.arg_types)


class ExpressionFactory:
    def __init__(
        self,
        function_specs: dict[str, FunctionSpec],
        terminal_names: list[str],
        delay_list: tuple[int, ...],
        quan_list: tuple[float, ...],
        exp_list: tuple[float, ...],
        rng: random.Random,
    ) -> None:
        self.function_specs = function_specs
        self.terminal_names = list(terminal_names)
        self.const_values = {
            TYPE_DELAY: tuple(delay_list),
            TYPE_QUAN: tuple(quan_list),
            TYPE_EXP: tuple(exp_list),
        }
        self.rng = rng
        self.basic_functions = [spec for spec in function_specs.values() if spec.return_type == TYPE_BASIC]
        if not self.terminal_names:
            raise ValueError("ExpressionFactory requires at least one terminal")
        if not self.basic_functions:
            raise ValueError("ExpressionFactory requires at least one basic function")

    def terminal(self, desired_type: str = TYPE_BASIC) -> Node:
        if desired_type == TYPE_BASIC:
            return Node("terminal", self.rng.choice(self.terminal_names))
        if desired_type in CONST_TYPES:
            return Node("const", self.rng.choice(self.const_values[desired_type]))
        raise ValueError(f"Unsupported terminal type: {desired_type}")

    def random_tree(self, max_depth: int, desired_type: str = TYPE_BASIC, min_depth: int = 1) -> Node:
        if desired_type != TYPE_BASIC:
            return self.terminal(desired_type)
        if max_depth <= 1 or (min_depth <= 1 and self.rng.random() < 0.35):
            return self.terminal(TYPE_BASIC)

        spec = self.rng.choice(self.basic_functions)
        children = tuple(
            self.random_tree(max_depth - 1, arg_type, max(min_depth - 1, 1))
            for arg_type in spec.arg_types
        )
        return Node("function", spec.name, children, spec.arg_types)

    def mutate_subtree(self, tree: Node, max_depth: int) -> Node:
        paths = [path for path, _ in tree.iter_paths()]
        path = self.rng.choice(paths)
        old = tree.subtree(path)
        desired_type = TYPE_BASIC
        if path:
            parent = tree.subtree(path[:-1])
            desired_type = parent.arg_types[path[-1]]
        elif old.kind == "const":
            desired_type = TYPE_EXP
        new_subtree = self.random_tree(max_depth=max_depth, desired_type=desired_type)
        return tree.replace(path, new_subtree)

    def crossover(self, left: Node, right: Node) -> tuple[Node, Node]:
        left_paths = [(path, node) for path, node in left.iter_paths() if node.kind != "const"]
        right_paths = [(path, node) for path, node in right.iter_paths() if node.kind != "const"]
        left_path, _ = self.rng.choice(left_paths)
        left_type = TYPE_BASIC
        if left_path:
            left_parent = left.subtree(left_path[:-1])
            left_type = left_parent.arg_types[left_path[-1]]
        compatible = []
        for path, node in right_paths:
            right_type = TYPE_BASIC
            if path:
                right_parent = right.subtree(path[:-1])
                right_type = right_parent.arg_types[path[-1]]
            if right_type == left_type:
                compatible.append((path, node))
        if not compatible:
            return left, right
        right_path, _ = self.rng.choice(compatible)
        left_sub = left.subtree(left_path)
        right_sub = right.subtree(right_path)
        return left.replace(left_path, right_sub), right.replace(right_path, left_sub)


def build_function_specs(func_map_dict: dict[str, tuple[Any, list[str]]], exclude: Optional[set[str]] = None) -> dict[str, FunctionSpec]:
    exclude = exclude or set()
    specs = {}
    for name, (func, arg_types) in func_map_dict.items():
        if name in exclude:
            continue
        specs[name] = FunctionSpec(name=name, func=func, arg_types=tuple(arg_types))
    return specs
