# Import the `ast` module for abstract syntax tree parsing
import ast

from datetime import datetime

# Imports solely for annotation
from openai import Client
from typing import Generator

from transformers import PreTrainedTokenizer

from ..components import (
    write_docstrings,
    write_arguments_annotations,
    write_comments,
    potentially_add_class_to_typing_import,
)
from ..connection import submit_record
from ..utils.utils import (
    _get_nodes_dict_with_functions_classes_methods,
    _check_no_duplicating_methods,
    _load_tokenizer,
)


def document_python_code(
    code: str,
    client: Client,
    modify_existing_documentation: bool = False,
    do_write_arguments_annotation: bool = True,
    do_write_docstrings: bool = True,
    do_write_comments: bool = True,
    model_checkpoint: str | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    in_time: datetime | None = None,
) -> Generator[str, None, None]:
    """
    Generates and adds docstrings to Python code using an OpenAI client.
    The function processes the code to annotate arguments and add docstrings to functions, classes, and methods.
    It also ensures that necessary typing imports are included.

    Args:
        code (`str`):
            The Python code as a string to which docstrings and argument annotations will be added.
        client (`Client`):
            An instance of the OpenAI client used to generate docstrings and annotations.
        modify_existing_annotation (`bool`, *optional*, defaults to `False`):
            If `True`, existing annotations and docstrings will be modified.
            If `False`, existing annotations and docstrings will be preserved unless they are empty.
        model_checkpoint (`str`, *optional*, defaults to `'Qwen/Qwen2.5-Coder-32B-Instruct-fast'`):
            The model checkpoint to use for generating docstrings and annotations.

    Yields:
        `str`: The updated code with added or modified docstrings and argument annotations.

    Raises:
        `SyntaxError`: If the generated code is not valid Python syntax.
    """
    # Save the initial time for recording purposes
    if in_time is None:
        in_time = datetime.now()
    # Read the code as a raw string to avoid AST parsing errors
    code = rf"{code}"
    # Make copy of the initial code
    in_code = str(code)
    # Parse the code into an AST
    tree = ast.parse(code)

    # Check that there are no duplicate methods, classes, or functions
    _check_no_duplicating_methods(tree.body)
    # Get a dictionary of nodes that need to be annotated or documented
    target_nodes_dict = _get_nodes_dict_with_functions_classes_methods(tree.body)

    # Load tokenizer to track the number of input tokens
    if tokenizer is None:
        tokenizer = _load_tokenizer(model_checkpoint)

    output = None
    # Set default values
    annotations_response_data = {}
    docstrings_response_data = {}
    comments_response_data = {}
    if do_write_arguments_annotation:
        # Annotate the arguments and returns of functions, classes, and methods
        for output in write_arguments_annotations(
            target_nodes_dict=target_nodes_dict,
            code=code,
            client=client,
            tokenizer=tokenizer,
            modify_existing_documentation=modify_existing_documentation,
            model_checkpoint=model_checkpoint,
        ):
            if isinstance(output, str):
                yield output
        # Get the required imports from the `typing` package that will need to be added in the end
        if output is not None:
            code, required_typing_imports, annotations_response_data = output
            annotations_response_data["required_imports"] = required_typing_imports
            # If there are classes from the `typing` package that were used for annotation but not imported,
            # add them to the imports
            for typing_class in required_typing_imports:
                code = potentially_add_class_to_typing_import(code, tree, typing_class)
                yield code
            # Replace spaces with tabs for consistent formatting
            code = code.replace(" " * 4, "\t")
            # Lines may have changed, so it's easier to rerun `ast.parse` which takes < 1ms than track
            # this throughout the code
            tree = ast.parse(code)
            # Get dictionary with target nodes with the updated AST code
            target_nodes_dict = _get_nodes_dict_with_functions_classes_methods(
                tree.body
            )

    if do_write_docstrings:
        # Add docstrings to functions, classes, and methods
        for output in write_docstrings(
            target_nodes_dict=target_nodes_dict,
            code=code,
            client=client,
            tokenizer=tokenizer,
            modify_existing_documentation=modify_existing_documentation,
            model_checkpoint=model_checkpoint,
        ):
            if isinstance(output, str):
                yield output
        if output is not None:
            code, docstrings_response_data = output

    if do_write_comments:
        # Add comments to the code where necessary
        for output in write_comments(
            code=code,
            client=client,
            tokenizer=tokenizer,
            modify_existing_documentation=modify_existing_documentation,
            model_checkpoint=model_checkpoint,
        ):
            if isinstance(output, str):
                yield output
        code, comments_response_data = output
    # Replace spaces with tabs for consistent formatting
    code = code.replace(" " * 4, "\t")
    # Make sure the generated code has valid Python syntax
    ast.parse(code)
    # Save to database
    submit_record(
        table="responses",
        in_code=in_code,
        out_code=code,
        in_time=in_time,
        out_time=datetime.now(),
        **{"annotations_" + k: v for k, v in annotations_response_data.items()},
        **{"docstrings_" + k: v for k, v in docstrings_response_data.items()},
        **{"comments_" + k: v for k, v in comments_response_data.items()},
    )
    yield code
