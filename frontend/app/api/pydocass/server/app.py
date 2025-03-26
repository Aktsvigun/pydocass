import os
from datetime import datetime
from argparse import ArgumentParser

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

from pydocass import document_python_code
from pydocass.connection import submit_record
from pydocass.utils.utils import format_code_with_black, get_client


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/document", methods=["POST"])
def document_code():
    data = request.json
    code = rf"{data.get('code', '')}"
    in_time = datetime.now()
    submit_record(table="inputs", in_time=in_time, in_code=code)

    client = get_client(data)

    def generate():
        try:
            chunk: str = code
            for chunk in document_python_code(
                code=code,
                client=client,
                modify_existing_documentation=data["modify_existing_documentation"],
                do_write_arguments_annotation=data["do_write_arguments_annotations"],
                do_write_docstrings=data["do_write_docstrings"],
                do_write_comments=data["do_write_comments"],
                in_time=in_time,
                model_checkpoint=data["model_checkpoint"],
            ):
                yield chunk
            yield format_code_with_black(chunk)

        except Exception as e:
            import pdb
            import sys

            exc_type, exc_value, exc_traceback = sys.exc_info()
            pdb.post_mortem(exc_traceback)
            yield f"Error: {str(e)}"

    return Response(stream_with_context(generate()), mimetype="text/plain")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--port", default="4000", type=str, required=False)
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)
