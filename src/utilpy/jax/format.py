from pprint import pformat
from typing import Any, Optional

import equinox as eqx
import jax
from jax.tree_util import PyTreeDef

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


def format_pytree(tree: Any) -> str:
    tree_param, treedef = jax.tree.flatten(tree)

    def format_pytree_inner(
        node: PyTreeDef, sb: StringBuilder, index: int, depth: int, leaf_name: str = ""
    ) -> tuple[StringBuilder, int]:
        children = node.children()
        node_data = node.node_data()
        flattend_data: eqx._module._FlattenedData | None = None
        if node_data is None:
            leaf = pformat(tree_param[index])

            display_data = ""
            if "\n" in leaf:
                lines = leaf.split("\n")
                unit = "\t" if index == len(tree_param) - 1 else UNIT
                for i, line in enumerate(lines):
                    display_data += f"{"(Leaf) " if i == 0 else unit*(depth+1)}{line}"
                    if i != len(lines) - 1:
                        display_data += "\n"
            else:
                display_data = f"(Leaf) {leaf}"
            index += 1

        else:
            display_data = node_data[0]
            if node_data[1] is not None and isinstance(node_data[1], eqx._module._FlattenedData):
                flattend_data = node_data[1]

        if depth == 0:
            sb.Append(f"{display_data}\n")
        else:
            sb.Append(f"{UNIT*depth}+--> {display_data} \n")
        if flattend_data is not None:
            for name, value in zip(flattend_data.static_field_names, flattend_data.static_field_values, strict=False):
                sb.Append(f"{"+\t"*(depth+1)}(static): {name} : {value} \n")
            for child, name in zip(children, flattend_data.dynamic_field_names, strict=False):
                sb, index = format_pytree_inner(child, sb, index, depth + 1, name)
        else:
            for child in children:
                sb, index = format_pytree_inner(child, sb, index, depth + 1)
        return sb, index

    return f"{format_pytree_inner(treedef, StringBuilder(), 0, 0)[0]}"
