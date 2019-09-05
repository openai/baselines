import base64
from collections import OrderedDict
import io
import json
import pickle

import cloudpickle
import numpy as np


def is_json_serializable(item):
    """
    Test if an object is serializable into JSON

    :param item: (object) The object to be tested for JSON serialization.
    :return: (bool) True if object is JSON serializable, false otherwise.
    """
    # Try with try-except struct.
    json_serializable = True
    try:
        _ = json.dumps(item)
    except TypeError:
        json_serializable = False
    return json_serializable


def data_to_json(data):
    """
    Turn data (class parameters) into a JSON string for storing

    :param data: (Dict) Dictionary of class parameters to be
        stored. Items that are not JSON serializable will be
        pickled with Cloudpickle and stored as bytearray in
        the JSON file
    :return: (str) JSON string of the data serialized.
    """
    # First, check what elements can not be JSONfied,
    # and turn them into byte-strings
    serializable_data = {}
    for data_key, data_item in data.items():
        # See if object is JSON serializable
        if is_json_serializable(data_item):
            # All good, store as it is
            serializable_data[data_key] = data_item
        else:
            # Not serializable, cloudpickle it into
            # bytes and convert to base64 string for storing.
            # Also store type of the class for consumption
            # from other languages/humans, so we have an
            # idea what was being stored.
            base64_encoded = base64.b64encode(
                cloudpickle.dumps(data_item)
            ).decode()

            # Use ":" to make sure we do
            # not override these keys
            # when we include variables of the object later
            cloudpickle_serialization = {
                ":type:": str(type(data_item)),
                ":serialized:": base64_encoded
            }

            # Add first-level JSON-serializable items of the
            # object for further details (but not deeper than this to
            # avoid deep nesting).
            # First we check that object has attributes (not all do,
            # e.g. numpy scalars)
            if hasattr(data_item, "__dict__") or isinstance(data_item, dict):
                # Take elements from __dict__ for custom classes
                item_generator = (
                    data_item.items if isinstance(data_item, dict) else data_item.__dict__.items
                )
                for variable_name, variable_item in item_generator():
                    # Check if serializable. If not, just include the
                    # string-representation of the object.
                    if is_json_serializable(variable_item):
                        cloudpickle_serialization[variable_name] = variable_item
                    else:
                        cloudpickle_serialization[variable_name] = str(variable_item)

            serializable_data[data_key] = cloudpickle_serialization
    json_string = json.dumps(serializable_data, indent=4)
    return json_string


def json_to_data(json_string, custom_objects=None):
    """
    Turn JSON serialization of class-parameters back into dictionary.

    :param json_string: (str) JSON serialization of the class-parameters
        that should be loaded.
    :param custom_objects: (dict) Dictionary of objects to replace
        upon loading. If a variable is present in this dictionary as a
        key, it will not be deserialized and the corresponding item
        will be used instead. Similar to custom_objects in
        `keras.models.load_model`. Useful when you have an object in
        file that can not be deserialized.
    :return: (dict) Loaded class parameters.
    """
    if custom_objects is not None and not isinstance(custom_objects, dict):
        raise ValueError("custom_objects argument must be a dict or None")

    json_dict = json.loads(json_string)
    # This will be filled with deserialized data
    return_data = {}
    for data_key, data_item in json_dict.items():
        if custom_objects is not None and data_key in custom_objects.keys():
            # If item is provided in custom_objects, replace
            # the one from JSON with the one in custom_objects
            return_data[data_key] = custom_objects[data_key]
        elif isinstance(data_item, dict) and ":serialized:" in data_item.keys():
            # If item is dictionary with ":serialized:"
            # key, this means it is serialized with cloudpickle.
            serialization = data_item[":serialized:"]
            # Try-except deserialization in case we run into
            # errors. If so, we can tell bit more information to
            # user.
            try:
                deserialized_object = cloudpickle.loads(
                    base64.b64decode(serialization.encode())
                )
            except pickle.UnpicklingError:
                raise RuntimeError(
                    "Could not deserialize object {}. ".format(data_key) +
                    "Consider using `custom_objects` argument to replace " +
                    "this object."
                )
            return_data[data_key] = deserialized_object
        else:
            # Read as it is
            return_data[data_key] = data_item
    return return_data


def params_to_bytes(params):
    """
    Turn params (OrderedDict of variable name -> ndarray) into
    serialized bytes for storing.

    Note: `numpy.savez` does not save the ordering.

    :param params: (OrderedDict) Dictionary mapping variable
        names to numpy arrays of the current parameters of the
        model.
    :return: (bytes) Bytes object of the serialized content.
    """
    # Create byte-buffer and save params with
    # savez function, and return the bytes.
    byte_file = io.BytesIO()
    np.savez(byte_file, **params)
    serialized_params = byte_file.getvalue()
    return serialized_params


def bytes_to_params(serialized_params, param_list):
    """
    Turn serialized parameters (bytes) back into OrderedDictionary.

    :param serialized_params: (byte) Serialized parameters
        with `numpy.savez`.
    :param param_list: (list) List of strings, representing
        the order of parameters in which they should be returned
    :return: (OrderedDict) Dictionary mapping variable name to
        numpy array of the parameters.
    """
    byte_file = io.BytesIO(serialized_params)
    params = np.load(byte_file)
    return_dictionary = OrderedDict()
    # Assign parameters to return_dictionary
    # in the order specified by param_list
    for param_name in param_list:
        return_dictionary[param_name] = params[param_name]
    return return_dictionary
