from __future__ import annotations

from tree_sitter import Language, Parser, Query, QueryCursor
import tree_sitter_python

FUNCTION_SCOPE_QUERY = """
(function_definition
  name: (identifier) @function.name
) @function.scope
""".strip()

PYTHON_LANGUAGE = Language(tree_sitter_python.language())
FUNCTION_QUERY = Query(PYTHON_LANGUAGE, FUNCTION_SCOPE_QUERY)


def slice_function(source_code_string: str, target_lineno: int) -> str:
    """Return the full function definition that contains the target line number."""

    if target_lineno < 1:
        raise ValueError("target_lineno must be a 1-based line number")

    source_bytes = source_code_string.encode("utf-8")
    parser = Parser()
    parser.language = PYTHON_LANGUAGE
    tree = parser.parse(source_bytes)
    cursor = QueryCursor(FUNCTION_QUERY)
    target_line_index = target_lineno - 1

    for _, captures in cursor.matches(tree.root_node):
        function_nodes = captures.get("function.scope", [])
        if not function_nodes:
            continue

        function_node = function_nodes[0]
        start_line = function_node.start_point[0]
        end_line = function_node.end_point[0]

        if start_line <= target_line_index <= end_line:
            return source_bytes[function_node.start_byte : function_node.end_byte].decode(
                "utf-8"
            )

    raise ValueError(f"No function definition found for line {target_lineno}")


class Surgeon:
    """Compatibility wrapper around the Tree-sitter slicing function."""

    @staticmethod
    def slice_function(source_code_string: str, target_lineno: int) -> str:
        return slice_function(source_code_string, target_lineno)
