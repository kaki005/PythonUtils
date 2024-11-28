import pathlib
from pathlib import Path

from rich.filesize import decimal
from rich.markup import escape
from rich.style import StyleType
from rich.text import Text
from rich.tree import Tree


def format_directory(directory: Path, style: StyleType = "bold bright_blue") -> Tree:
    tree = Tree(
        f":open_file_folder: [link file://{directory}]{directory}",
        guide_style=style,
    )

    def format_directory_inner(directory: pathlib.Path, tree: Tree) -> Tree:
        """Recursively build a Tree with directory contents."""
        # Sort dirs first then by filename
        paths = sorted(
            Path(directory).iterdir(),
            key=lambda path: (path.is_file(), path.name.lower()),
        )
        for path in paths:
            # Remove hidden files
            if path.name.startswith("."):
                continue
            if path.is_dir():
                style = "dim" if path.name.startswith("__") else ""
                branch = tree.add(
                    f"[bold magenta]:open_file_folder: [link file://{path}]{escape(path.name)}",
                    style=style,
                    guide_style=style,
                )
                format_directory_inner(path, branch)
            else:
                text_filename = Text(path.name, "green")
                text_filename.highlight_regex(r"\..*$", "bold red")
                text_filename.stylize(f"link file://{path}")
                file_size = path.stat().st_size
                text_filename.append(f" ({decimal(file_size)})", "blue")
                icon = "🐍 " if path.suffix == ".py" else "📄 "
                tree.add(Text(icon) + text_filename)
        return tree

    return format_directory_inner(directory, tree)
