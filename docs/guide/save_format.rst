.. _save_format:


On saving and loading
=====================

Stable baselines stores both neural network parameters and algorithm-related parameters such as
exploration schedule, number of environments and observation/action space. This allows continual learning and easy
use of trained agents without training, but it is not without its issues. Following describes two formats
used to save agents in stable baselines, their pros and shortcomings.

Terminology used in this page:

-  *parameters* refer to neural network parameters (also called "weights"). This is a dictionary
   mapping Tensorflow variable name to a NumPy array.
-  *data* refers to RL algorithm parameters, e.g. learning rate, exploration schedule, action/observation space.
   These depend on the algorithm used. This is a dictionary mapping classes variable names their values.


Cloudpickle (stable-baselines<=2.7.0)
-------------------------------------

Original stable baselines save format. Data and parameters are bundled up into a tuple ``(data, parameters)`` 
and then serialized with ``cloudpickle`` library (essentially the same as ``pickle``).

This save format is still available via an argument in model save function in stable-baselines versions above
v2.7.0 for backwards compatibility reasons, but its usage is discouraged.

Pros:

-  Easy to implement and use.
-  Works with almost any type of Python object, including functions.


Cons:

-  Pickle/Cloudpickle is not designed for long-term storage or sharing between Python version.
-  If one object in file is not readable (e.g. wrong library version), then reading the rest of the
   file is difficult.
-  Python-specific format, hard to read stored files from other languages.


If part of a saved model becomes unreadable for any reason (e.g. different Tensorflow versions), then
it may be tricky to restore any of the model. For this reason another save format was designed.


Zip-archive (stable-baselines>2.7.0)
-------------------------------------

A zip-archived JSON dump and NumPy zip archive of the arrays. The data dictionary (class parameters)
is stored as a JSON file, model parameters are serialized with ``numpy.savez`` function and these two files
are stored under a single .zip archive.

Any objects that are not JSON serializable are serialized with cloudpickle and stored as base64-encoded
string in the JSON file, along with some information that was stored in the serialization. This allows
inspecting stored objects without deserializing the object itself.

This format allows skipping elements in the file, i.e. we can skip deserializing objects that are
broken/non-serializable. This can be done via ``custom_objects`` argument to load functions.

This is the default save format in stable baselines versions after v2.7.0.

File structure:

::

  saved_model.zip/
  ├── data              JSON file of class-parameters (dictionary)
  ├── parameter_list    JSON file of model parameters and their ordering (list)
  ├── parameters        Bytes from numpy.savez (a zip file of the numpy arrays). ...
      ├── ...           Being a zip-archive itself, this object can also be opened ...
          ├── ...       as a zip-archive and browsed.


Pros:


-  More robust to unserializable objects (one bad object does not break everything).
-  Saved file can be inspected/extracted with zip-archive explorers and by other
   languages.


Cons:

-  More complex implementation.
-  Still relies partly on cloudpickle for complex objects (e.g. custom functions).
