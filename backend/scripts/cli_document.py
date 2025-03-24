#!/usr/bin/env python3
import os
import sys
import argparse
from datetime import datetime
import tempfile

from pydocass.core.document_python_code import document_python_code
from pydocass.connection import submit_record
from pydocass.utils.utils import format_code_with_black, get_client
from pydocass.utils.constants import DEFAULT_MODEL_CHECKPOINT


DEFAULT_USE_STREAMING = os.getenv("USE_STREAMING", "false").lower() == "true"


def main():
    """Run the documentation assistant from the command line."""
    parser = argparse.ArgumentParser(
        description="PyDocAss - Python Documentation Assistant"
    )
    
    parser.add_argument(
        "input_file", 
        help="Path to the Python file to document. Use '-' to read from stdin."
    )
    
    parser.add_argument(
        "-o", "--output", 
        help="Path to write the documented code. If not provided, will print to stdout."
    )
    
    parser.add_argument(
        "--no-modify-existing", 
        action="store_false", 
        dest="modify_existing_documentation",
        help="Don't modify existing documentation."
    )
    
    parser.add_argument(
        "--no-arguments-annotations", 
        action="store_false", 
        dest="do_write_arguments_annotations",
        help="Don't write argument annotations."
    )
    
    parser.add_argument(
        "--no-docstrings", 
        action="store_false", 
        dest="do_write_docstrings",
        help="Don't write docstrings."
    )
    
    parser.add_argument(
        "--no-comments", 
        action="store_false", 
        dest="do_write_comments",
        help="Don't write comments."
    )
    
    parser.add_argument(
        "--use-streaming", 
        action="store_true",
        default=DEFAULT_USE_STREAMING,
        dest="use_streaming",
        help="Use streaming for the documentation process."
    )
    
    parser.add_argument(
        "--model", 
        default=None,
        dest="model_checkpoint",
        help=f"Model checkpoint to use. Default: {DEFAULT_MODEL_CHECKPOINT}"
    )
    
    parser.add_argument(
        "--api-key",
        help="API key for Nebius AI Studio or OpenAI. Can also be set via NEBIUS_API_KEY or OPENAI_API_KEY environment variables."
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show progress updates during documentation process."
    )
    
    args = parser.parse_args()
    # Read the input code
    if args.input_file == '-':
        code = sys.stdin.read()
    else:
        try:
            with open(args.input_file, 'r') as f:
                code = f.read()
        except FileNotFoundError:
            print(f"Error: Input file '{args.input_file}' not found", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading input file: {str(e)}", file=sys.stderr)
            sys.exit(1)

    
    try:
        # Record input for tracking
        in_time = datetime.now()
        submit_record(table="inputs", in_time=in_time, in_code=code)
        
        # Get the OpenAI/Nebius client
        client = get_client({"api_key": args.api_key})
        
        # Process the code
        documented_code = None
        
        if args.verbose:
            print("Starting documentation process...", file=sys.stderr)
        
        for chunk in document_python_code(
            code=code,
            client=client,
            modify_existing_documentation=args.modify_existing_documentation,
            do_write_arguments_annotation=args.do_write_arguments_annotations,
            do_write_docstrings=args.do_write_docstrings,
            do_write_comments=args.do_write_comments,
            use_streaming=args.use_streaming,
            in_time=in_time,
            model_checkpoint=args.model_checkpoint,
        ):
            documented_code = chunk
            if args.verbose:
                print(".", end="", file=sys.stderr, flush=True)
        
        if args.verbose:
            print("\nFormatting code with Black...", file=sys.stderr)
        
        # Format the final code with Black
        documented_code = format_code_with_black(documented_code)
        
        # Output the documented code
        if args.output:
            with open(args.output, 'w') as f:
                f.write(documented_code)
            if args.verbose:
                print(f"Documented code written to {args.output}", file=sys.stderr)
        else:
            print(documented_code)
            
    except Exception as e:
        import sys, pdb
        print(f"Error: {str(e)}", file=sys.stderr)
        exc_type, exc_value, exc_traceback = sys.exc_info()
        pdb.post_mortem(exc_traceback)
        # sys.exit(1)


if __name__ == "__main__":
    main() 