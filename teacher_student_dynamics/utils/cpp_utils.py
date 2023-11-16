import json
from typing import Any, Dict


def params_to_json(params: Dict[str, Any], output_path: str) -> None:
    """Method to write parameters to a json file.

    Args:
        params: python dictionary of parameters
        output_path: path to save json file
    """
    with open(output_path, "w") as json_file:
        json.dump(params, json_file)


def params_to_txt(params: Dict[str, Any], output_path: str) -> None:
    """Method to write parameters to a .txt file.

    Args:
        params: python dictionary of parameters
        output_path: path to save txt file
    """
    with open(output_path, "w") as txt_file:
        for i, (k, v) in enumerate(params.items()):
            # note: bool is subclass of int so must be handled first to
            # be handled separately.
            if isinstance(v, bool):
                v = int(v)
                value_type = "bool"
            elif isinstance(v, int):
                value_type = "int"
            elif isinstance(v, str):
                value_type = "str"
            elif isinstance(v, float):
                value_type = "float"
            elif isinstance(v, list):
                if not v: # empty list
                    value_type = "it_empty"
                    v = ","
                elif all([isinstance(vi, float) for vi in v]):
                    value_type = "it_float"
                    v = ",".join([str(vi) for vi in v])
                elif all([isinstance(vi, int) for vi in v]):
                    value_type = "it_int"
                    v = ",".join([str(vi) for vi in v])
                elif all([isinstance(vi, str) for vi in v]):
                    value_type = "it_str"
                    v = ",".join([str(vi) for vi in v])
                else:
                    raise ValueError(
                        f"lists with this datatype not yet handled in params_to_txt."
                    )
            else:
                raise ValueError(f"type {type(v)} not yet handled in params_to_txt.")
            if i == len(params) - 1:
                line_end = "\n"
            else:
                line_end = "\n"
            txt_file.write(f"{value_type};{k};{v}{line_end}")
