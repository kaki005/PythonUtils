from pprint import pformat
from typing import cast

import equinox as eqx
import jax
import rich
from jax.tree_util import PyTreeDef
from jaxtyping import PyTree
from rich.tree import Tree

PROP_NAME_COLOR = "blue"
"""color of property name"""
STATIC_COLOR = "yellow"
"""color of static attribute"""
TYPE_COLOR = "orange1"
"""color of property type"""


def print_pytreedef(
    treedef: PyTreeDef,
    highlight: bool = False,
    show_static: bool = True,
) -> None:
    root_tree = Tree(
        label="pytree",
        hide_root=True,
        highlight=highlight,
    )

    def print_pytreedef_inner(node: PyTreeDef, tree: Tree, prop_name: str = ""):
        children = node.children()
        node_data = node.node_data()
        if node_data is None:
            display_data = "(Leaf)"
            if prop_name != "":  # プロパティ名があれば
                display_data = f"[{PROP_NAME_COLOR}][bold]{prop_name}[/][/]: {display_data}"
        else:
            display_data = f"{node_data[0]}"
        branch = tree.add(display_data)
        if (
            node_data is not None and node_data[1] is not None and isinstance(node_data[1], eqx._module._FlattenedData)
        ):  # FlattenedDataなら
            flattend_data = cast(eqx._module._FlattenedData, node_data[1])  # cast
            if show_static:
                for name, value in zip(
                    flattend_data.static_field_names, flattend_data.static_field_values, strict=False
                ):
                    branch.add(f"[{STATIC_COLOR}][bold](static)[/][/]: [{PROP_NAME_COLOR}][bold]{name}[/][/] : {value}")
            for child, name in zip(children, flattend_data.dynamic_field_names, strict=False):
                print_pytreedef_inner(child, branch, name)
        else:
            for child in children:
                print_pytreedef_inner(child, branch)

    print_pytreedef_inner(treedef, root_tree)
    rich.print(root_tree)


def print_pytree(
    pytree: PyTree,
    highlight: bool = False,
    max_length: int = 5,
    is_hide_big_node: bool = True,
    show_static: bool = True,
) -> None:
    """Decompose a pytree and print its tree structure and values."""
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
        prop_name: str = "",
    ) -> int:
        children = node.children()
        node_data = node.node_data()
        if node_data is None:  # leaf 　なら
            display_data = pformat(tree_param[index])
            if len(display_data.split("\n")) > max_length and is_hide_big_node:
                display_data = f"[{TYPE_COLOR}][bold]{type(tree_param[index])}[/][/]"
                if "shape" in dir(tree_param[index]):
                    display_data += f" {tree_param[index].shape}"
            index += 1

        else:  # ノードデータがあれば
            display_data = f"[{TYPE_COLOR}][bold]{node_data[0]}[/][/]"  # 型

        if prop_name != "":  # プロパティ名があれば　先頭につける
            display_data = f"[{PROP_NAME_COLOR}][bold]{prop_name}[/][/]: {display_data}"
        branch = tree.add(display_data)  # 描画
        if (
            node_data is not None and node_data[1] is not None and isinstance(node_data[1], eqx._module._FlattenedData)
        ):  # FlattenedDataなら
            flattend_data = cast(eqx._module._FlattenedData, node_data[1])  # cast
            if show_static:  # staticプロパティを表示するなら
                for name, value in zip(
                    flattend_data.static_field_names, flattend_data.static_field_values, strict=False
                ):
                    branch.add(f"[{STATIC_COLOR}][bold](static)[/][/]: [{PROP_NAME_COLOR}][bold]{name}[/][/] : {value}")
            for child, name in zip(children, flattend_data.dynamic_field_names, strict=False):  # 各子どもを
                index = print_pytree_inner(child, branch, index, name)  # プロパティ名つきで描画
        else:  # FlattenedDataでなければ
            for child in children:  # 各子どもを
                index = print_pytree_inner(  # プロパティ名なしで描画
                    child,
                    branch,
                    index,
                )

        return index

    print_pytree_inner(treedef, root_tree, 0)
    rich.print(root_tree)
