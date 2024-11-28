from pprint import pformat

import equinox as eqx
import jax
import rich
from jax.tree_util import PyTreeDef
from jaxtyping import PyTree
from rich.tree import Tree

from utilpy import StringBuilder

UNIT: str = "+\t"


def format_pytreedef(treedef: PyTreeDef) -> str:
    def format_pytreedef_inner(node: PyTreeDef, sb: StringBuilder, depth: int, leaf_name: str = "") -> StringBuilder:
        children = node.children()
        node_data = node.node_data()
        flattend_data: eqx._module._FlattenedData | None = None
        if node_data is None:
            display_data = "(Leaf)"
            if leaf_name != "":
                display_data = f"{display_data}: {leaf_name}"
        else:
            display_data = node_data[0]
            if isinstance(node_data[1], eqx._module._FlattenedData):
                flattend_data = node_data[1]

        if depth == 0:
            sb.Append(f"{display_data}\n")
        else:
            sb.Append(f"{UNIT*depth}+--> {display_data} \n")
        if flattend_data is not None:
            for name, value in zip(flattend_data.static_field_names, flattend_data.static_field_values, strict=False):
                sb.Append(f"{UNIT*(depth+1)}(static): {name} : {value} \n")
            for child, name in zip(children, flattend_data.dynamic_field_names, strict=False):
                sb = format_pytreedef_inner(child, sb, depth + 1, name)
        else:
            for child in children:
                sb = format_pytreedef_inner(child, sb, depth + 1)
        return sb

    return f"{format_pytreedef_inner(treedef, StringBuilder(), 0)}"


def print_pytree(
    pytree: PyTree,
    highlight: bool = False,
    max_length: int = 5,
    is_hide_big_node: bool = True,
    show_static: bool = True,
) -> None:
    tree_param, treedef = jax.tree.flatten(pytree)
    root_tree = Tree(
        label="pytree",
        hide_root=True,
        highlight=highlight,
    )

    def print_pytree_inner(
        node: PyTreeDef,
        tree: Tree,
        index: int,
        leaf_name: str = "",
    ) -> int:
        children = node.children()
        node_data = node.node_data()
        flattend_data: eqx._module._FlattenedData | None = None
        if node_data is None:
            display_data = pformat(tree_param[index])
            if len(display_data.split("\n")) > max_length and is_hide_big_node:
                if isinstance(display_data, jax.numpy.ndarray):
                    display_data = f"[orange1][bold]jax.numpy.ndarray ({display_data.shape})[/][/]"
                else:
                    display_data = f"[orange1][bold]{type(tree_param[index])}[/][/]"
            index += 1

        else:
            display_data = f"[orange1][bold]{node_data[0]}[/][/]"
            if node_data[1] is not None and isinstance(node_data[1], eqx._module._FlattenedData):
                flattend_data = node_data[1]

        if leaf_name != "":
            display_data = f"[blue][bold]{leaf_name}[/][/]: {display_data}"
        branch = tree.add(f"{display_data}")
        if flattend_data is not None:
            if show_static:  # staticプロパティを表示するなら
                for name, value in zip(
                    flattend_data.static_field_names, flattend_data.static_field_values, strict=False
                ):
                    branch.add(f"[yellow][bold](static)[/][/]: [blue][bold]{name}[/][/] : {value}")
            for child, name in zip(children, flattend_data.dynamic_field_names, strict=False):
                index = print_pytree_inner(child, branch, index, name)
        else:
            for child in children:
                index = print_pytree_inner(
                    child,
                    branch,
                    index,
                )

        return index

    print_pytree_inner(treedef, root_tree, 0)
    rich.print(root_tree)
