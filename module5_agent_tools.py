"""Helper tools for the AI Study Assistant Agent.

These tools are deliberately simple for learning purposes. Tasks and notes are
stored in memory, so they reset when the Python process restarts.
"""

import ast
import operator
from typing import Any


# Small editable knowledge base used by search_notes().
study_notes = {
    "Java": (
        "Java is an object-oriented, class-based programming language. "
        "Common topics include classes, objects, inheritance, polymorphism, "
        "encapsulation, abstraction, exceptions, and JDBC."
    ),
    "Java Collections Framework": (
        "The Java Collections Framework provides interfaces and classes for "
        "groups of objects. List keeps ordered elements and allows duplicates; "
        "Set stores unique elements; Queue is commonly used to process items "
        "in order; Map stores key-value pairs. ArrayList is a resizable array, "
        "HashMap stores key-value pairs, TreeSet stores sorted unique values, "
        "TreeMap stores keys in sorted order, and PriorityQueue retrieves the "
        "highest-priority element according to its ordering."
    ),
    "RAG": (
        "Retrieval-Augmented Generation (RAG) retrieves relevant information "
        "from a knowledge source and gives it to a language model as context "
        "before generating an answer. A typical pipeline splits documents into "
        "chunks, creates embeddings, stores them in a vector database, retrieves "
        "matching chunks, and answers using the retrieved context."
    ),
    "Embeddings and Vector Databases": (
        "An embedding is a numerical vector representing the meaning of text. "
        "Similar meanings tend to have nearby vectors. Vector databases store "
        "embeddings and help retrieve semantically similar chunks."
    ),
    "AI Agents and Function Calling": (
        "An AI agent can decide when to use tools to complete a task. Function "
        "calling lets a model request a named function with structured arguments. "
        "The application runs the function and returns its result to the model. "
        "Examples of tools include a calculator, a notes search, and a task manager."
    ),
}

# This list is intentionally in-memory for this beginner project.
tasks: list[dict[str, Any]] = []


# Supported calculator operators. Python eval() is intentionally not used.
_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate_math_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate_math_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_math_node(node.left)
        right = _evaluate_math_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("The exponent is too large (maximum absolute value is 100).")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_math_node(node.operand))
    raise ValueError("Use only numbers, parentheses, and +, -, *, /, //, %, ** operators.")


def calculate(expression: str) -> str:
    """Calculate a basic arithmetic expression, for example '125 * 8'."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate_math_node(tree)
        return f"Result: {result}"
    except ZeroDivisionError:
        return "Error: division by zero is not allowed."
    except (SyntaxError, ValueError, TypeError, OverflowError) as exc:
        return f"Could not calculate that expression: {exc}"


def search_notes(query: str) -> str:
    """Search the built-in study notes using words from the query."""
    query_words = {word.lower().strip(".,!?():;\"'") for word in query.split()}
    query_words.discard("")
    if not query_words:
        return "Please provide a topic to search for."

    scored_results = []
    for title, content in study_notes.items():
        searchable = f"{title} {content}".lower()
        score = sum(1 for word in query_words if word in searchable)
        if score:
            scored_results.append((score, title, content))

    if not scored_results:
        available = ", ".join(study_notes.keys())
        return f"No matching notes were found. Available topics: {available}"

    scored_results.sort(key=lambda item: item[0], reverse=True)
    return "\n\n".join(f"{title}: {content}" for _, title, content in scored_results[:3])


def manage_tasks(action: str, task: str | None = None) -> str:
    """Manage study tasks. Actions: add, list, or complete. Complete accepts an ID or task text."""
    normal_action = action.lower().strip()

    if normal_action == "add":
        if not task or not task.strip():
            return "Please provide a task description to add."
        new_id = max((item["id"] for item in tasks), default=0) + 1
        tasks.append({"id": new_id, "task": task.strip(), "completed": False})
        return f"Added task #{new_id}: {task.strip()}"

    if normal_action == "list":
        if not tasks:
            return "There are no study tasks yet. Ask me to add one."
        rows = []
        for item in tasks:
            status = "Completed" if item["completed"] else "Pending"
            rows.append(f"#{item['id']} — {item['task']} [{status}]")
        return "Your study tasks:\n" + "\n".join(rows)

    if normal_action == "complete":
        if not task or not task.strip():
            return "Provide the task ID or task description to mark as completed."
        requested = task.strip()
        found = None
        if requested.startswith("#"):
            requested = requested[1:]
        if requested.isdigit():
            found = next((item for item in tasks if item["id"] == int(requested)), None)
        if found is None:
            found = next((item for item in tasks if item["task"].lower() == requested.lower()), None)
        if found is None:
            return f"Task '{task}' was not found. Ask me to list your tasks."
        found["completed"] = True
        return f"Marked task #{found['id']} as completed: {found['task']}"

    return "Unknown action. Use 'add', 'list', or 'complete'."
