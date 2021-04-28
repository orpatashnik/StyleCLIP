# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Helper for managing networks."""

import types
import inspect
import re
import uuid
import sys
import copy
import numpy as np
import tensorflow as tf

from collections import OrderedDict
from typing import Any, List, Tuple, Union, Callable

from . import tfutil
from .. import util

from .tfutil import TfExpression, TfExpressionEx

# pylint: disable=protected-access
# pylint: disable=attribute-defined-outside-init
# pylint: disable=too-many-public-methods

_import_handlers = []  # Custom import handlers for dealing with legacy data in pickle import.
_import_module_src = dict()  # Source code for temporary modules created during pickle import.


def import_handler(handler_func):
    """Function decorator for declaring custom import handlers."""
    _import_handlers.append(handler_func)
    return handler_func


class Network:
    """Generic network abstraction.

    Acts as a convenience wrapper for a parameterized network construction
    function, providing several utility methods and convenient access to
    the inputs/outputs/weights.

    Network objects can be safely pickled and unpickled for long-term
    archival purposes. The pickling works reliably as long as the underlying
    network construction function is defined in a standalone Python module
    that has no side effects or application-specific imports.

    Args:
        name: Network name. Used to select TensorFlow name and variable scopes. Defaults to build func name if None.
        func_name: Fully qualified name of the underlying network construction function, or a top-level function object.
        static_kwargs: Keyword arguments to be passed in to the network construction function.
    """

    def __init__(self, name: str = None, func_name: Any = None, **static_kwargs):
        # Locate the user-specified build function.
        assert isinstance(func_name, str) or util.is_top_level_function(func_name)
        if util.is_top_level_function(func_name):
            func_name = util.get_top_level_function_name(func_name)
        module, func_name = util.get_module_from_obj_name(func_name)
        func = util.get_obj_from_module(module, func_name)

        # Dig up source code for the module containing the build function.
        module_src = _import_module_src.get(module, None)
        if module_src is None:
            module_src = inspect.getsource(module)

        # Initialize fields.
        self._init_fields(name=(name or func_name), static_kwargs=static_kwargs, build_func=func, build_func_name=func_name, build_module_src=module_src)

    def _init_fields(self, name: str, static_kwargs: dict, build_func: Callable, build_func_name: str, build_module_src: str) -> None:
        tfutil.assert_tf_initialized()
        assert isinstance(name, str)
        assert len(name) >= 1
        assert re.fullmatch(r"[A-Za-z0-9_.\\-]*", name)
        assert isinstance(static_kwargs, dict)
        assert util.is_pickleable(static_kwargs)
        assert callable(build_func)
        assert isinstance(build_func_name, str)
        assert isinstance(build_module_src, str)

        # Choose TensorFlow name scope.
        with tf.name_scope(None):
            scope = tf.get_default_graph().unique_name(name, mark_as_used=True)

        # Query current TensorFlow device.
        with tfutil.absolute_name_scope(scope), tf.control_dependencies(None):
            device = tf.no_op(name="_QueryDevice").device

        # Immutable state.
        self._name                  = name
        self._scope                 = scope
        self._device                = device
        self._static_kwargs         = util.EasyDict(copy.deepcopy(static_kwargs))
        self._build_func            = build_func
        self._build_func_name       = build_func_name
        self._build_module_src      = build_module_src

        # State before _init_graph().
        self._var_inits             = dict()    # var_name => initial_value, set to None by _init_graph()
        self._all_inits_known       = False     # Do we know for sure that _var_inits covers all the variables?
        self._components            = None      # subnet_name => Network, None if the components are not known yet

        # Initialized by _init_graph().
        self._input_templates       = None
        self._output_templates      = None
        self._own_vars              = None

        # Cached values initialized the respective methods.
        self._input_shapes          = None
        self._output_shapes         = None
        self._input_names           = None
        self._output_names          = None
        self._vars                  = None
        self._trainables            = None
        self._var_global_to_local   = None
        self._run_cache             = dict()

    def _init_graph(self) -> None:
        assert self._var_inits is not None
        assert self._input_templates is None
        assert self._output_templates is None
        assert self._own_vars is None

        # Initialize components.
        if self._components is None:
            self._components = util.EasyDict()

        # Choose build func kwargs.
        build_kwargs = dict(self.static_kwargs)
        build_kwargs["is_template_graph"] = True
        build_kwargs["components"] = self._components

        # Override scope and device, and ignore surrounding control dependencies.
        with tfutil.absolute_variable_scope(self.scope, reuse=False), tfutil.absolute_name_scope(self.scope), tf.device(self.device), tf.control_dependencies(None):
            assert tf.get_variable_scope().name == self.scope
            assert tf.get_default_graph().get_name_scope() == self.scope

            # Create input templates.
            self._input_templates = []
            for param in inspect.signature(self._build_func).parameters.values():
                if param.kind == param.POSITIONAL_OR_KEYWORD and param.default is param.empty:
                    self._input_templates.append(tf.placeholder(tf.float32, name=param.name))

            # Call build func.
            out_expr = self._build_func(*self._input_templates, **build_kwargs)

        # Collect output templates and variables.
        assert tfutil.is_tf_expression(out_expr) or isinstance(out_expr, tuple)
        self._output_templates = [out_expr] if tfutil.is_tf_expression(out_expr) else list(out_expr)
        self._own_vars = OrderedDict((var.name[len(self.scope) + 1:].split(":")[0], var) for var in tf.global_variables(self.scope + "/"))

        # Check for errors.
        if len(self._input_templates) == 0:
            raise ValueError("Network build func did not list any inputs.")
        if len(self._output_templates) == 0:
            raise ValueError("Network build func did not return any outputs.")
        if any(not tfutil.is_tf_expression(t) for t in self._output_templates):
            raise ValueError("Network outputs must be TensorFlow expressions.")
        if any(t.shape.ndims is None for t in self._input_templates):
            raise ValueError("Network input shapes not defined. Please call x.set_shape() for each input.")
        if any(t.shape.ndims is None for t in self._output_templates):
            raise ValueError("Network output shapes not defined. Please call x.set_shape() where applicable.")
        if any(not isinstance(comp, Network) for comp in self._components.values()):
            raise ValueError("Components of a Network must be Networks themselves.")
        if len(self._components) != len(set(comp.name for comp in self._components.values())):
            raise ValueError("Components of a Network must have unique names.")

        # Initialize variables.
        if len(self._var_inits):
            tfutil.set_vars({self._get_vars()[name]: value for name, value in self._var_inits.items() if name in self._get_vars()})
        remaining_inits = [var.initializer for name, var in self._own_vars.items() if name not in self._var_inits]
        if self._all_inits_known:
            assert len(remaining_inits) == 0
        else:
            tfutil.run(remaining_inits)
        self._var_inits = None

    @property
    def name(self):
        """User-specified name string."""
        return self._name

    @property
    def scope(self):
        """Unique TensorFlow scope containing template graph and variables, derived from the user-specified name."""
        return self._scope

    @property
    def device(self):
        """Name of the TensorFlow device that the weights of this network reside on. Determined by the current device at construction time."""
        return self._device

    @property
    def static_kwargs(self):
        """EasyDict of arguments passed to the user-supplied build func."""
        return copy.deepcopy(self._static_kwargs)

    @property
    def components(self):
        """EasyDict of sub-networks created by the build func."""
        return copy.copy(self._get_components())

    def _get_components(self):
        if self._components is None:
            self._init_graph()
            assert self._components is not None
        return self._components

    @property
    def input_shapes(self):
        """List of input tensor shapes, including minibatch dimension."""
        if self._input_shapes is None:
            self._input_shapes = [t.shape.as_list() for t in self.input_templates]
        return copy.deepcopy(self._input_shapes)

    @property
    def output_shapes(self):
        """List of output tensor shapes, including minibatch dimension."""
        if self._output_shapes is None:
            self._output_shapes = [t.shape.as_list() for t in self.output_templates]
        return copy.deepcopy(self._output_shapes)

    @property
    def input_shape(self):
        """Short-hand for input_shapes[0]."""
        return self.input_shapes[0]

    @property
    def output_shape(self):
        """Short-hand for output_shapes[0]."""
        return self.output_shapes[0]

    @property
    def num_inputs(self):
        """Number of input tensors."""
        return len(self.input_shapes)

    @property
    def num_outputs(self):
        """Number of output tensors."""
        return len(self.output_shapes)

    @property
    def input_names(self):
        """Name string for each input."""
        if self._input_names is None:
            self._input_names = [t.name.split("/")[-1].split(":")[0] for t in self.input_templates]
        return copy.copy(self._input_names)

    @property
    def output_names(self):
        """Name string for each output."""
        if self._output_names is None:
            self._output_names = [t.name.split("/")[-1].split(":")[0] for t in self.output_templates]
        return copy.copy(self._output_names)

    @property
    def input_templates(self):
        """Input placeholders in the template graph."""
        if self._input_templates is None:
            self._init_graph()
            assert self._input_templates is not None
        return copy.copy(self._input_templates)

    @property
    def output_templates(self):
        """Output tensors in the template graph."""
        if self._output_templates is None:
            self._init_graph()
            assert self._output_templates is not None
        return copy.copy(self._output_templates)

    @property
    def own_vars(self):
        """Variables defined by this network (local_name => var), excluding sub-networks."""
        return copy.copy(self._get_own_vars())

    def _get_own_vars(self):
        if self._own_vars is None:
            self._init_graph()
            assert self._own_vars is not None
        return self._own_vars

    @property
    def vars(self):
        """All variables (local_name => var)."""
        return copy.copy(self._get_vars())

    def _get_vars(self):
        if self._vars is None:
            self._vars = OrderedDict(self._get_own_vars())
            for comp in self._get_components().values():
                self._vars.update((comp.name + "/" + name, var) for name, var in comp._get_vars().items())
        return self._vars

    @property
    def trainables(self):
        """All trainable variables (local_name => var)."""
        return copy.copy(self._get_trainables())

    def _get_trainables(self):
        if self._trainables is None:
            self._trainables = OrderedDict((name, var) for name, var in self.vars.items() if var.trainable)
        return self._trainables

    @property
    def var_global_to_local(self):
        """Mapping from variable global names to local names."""
        return copy.copy(self._get_var_global_to_local())

    def _get_var_global_to_local(self):
        if self._var_global_to_local is None:
            self._var_global_to_local = OrderedDict((var.name.split(":")[0], name) for name, var in self.vars.items())
        return self._var_global_to_local

    def reset_own_vars(self) -> None:
        """Re-initialize all variables of this network, excluding sub-networks."""
        if self._var_inits is None or self._components is None:
            tfutil.run([var.initializer for var in self._get_own_vars().values()])
        else:
            self._var_inits.clear()
            self._all_inits_known = False

    def reset_vars(self) -> None:
        """Re-initialize all variables of this network, including sub-networks."""
        if self._var_inits is None:
            tfutil.run([var.initializer for var in self._get_vars().values()])
        else:
            self._var_inits.clear()
            self._all_inits_known = False
            if self._components is not None:
                for comp in self._components.values():
                    comp.reset_vars()

    def reset_trainables(self) -> None:
        """Re-initialize all trainable variables of this network, including sub-networks."""
        tfutil.run([var.initializer for var in self._get_trainables().values()])

    def get_output_for(self, *in_expr: TfExpression, return_as_list: bool = False, **dynamic_kwargs) -> Union[TfExpression, List[TfExpression]]:
        """Construct TensorFlow expression(s) for the output(s) of this network, given the input expression(s).
        The graph is placed on the current TensorFlow device."""
        assert len(in_expr) == self.num_inputs
        assert not all(expr is None for expr in in_expr)
        self._get_vars()  # ensure that all variables have been created

        # Choose build func kwargs.
        build_kwargs = dict(self.static_kwargs)
        build_kwargs.update(dynamic_kwargs)
        build_kwargs["is_template_graph"] = False
        build_kwargs["components"] = self._components

        # Build TensorFlow graph to evaluate the network.
        with tfutil.absolute_variable_scope(self.scope, reuse=True), tf.name_scope(self.name):
            assert tf.get_variable_scope().name == self.scope
            valid_inputs = [expr for expr in in_expr if expr is not None]
            final_inputs = []
            for expr, name, shape in zip(in_expr, self.input_names, self.input_shapes):
                if expr is not None:
                    expr = tf.identity(expr, name=name)
                else:
                    expr = tf.zeros([tf.shape(valid_inputs[0])[0]] + shape[1:], name=name)
                final_inputs.append(expr)
            out_expr = self._build_func(*final_inputs, **build_kwargs)

        # Propagate input shapes back to the user-specified expressions.
        for expr, final in zip(in_expr, final_inputs):
            if isinstance(expr, tf.Tensor):
                expr.set_shape(final.shape)

        # Express outputs in the desired format.
        assert tfutil.is_tf_expression(out_expr) or isinstance(out_expr, tuple)
        if return_as_list:
            out_expr = [out_expr] if tfutil.is_tf_expression(out_expr) else list(out_expr)
        return out_expr

    def get_var_local_name(self, var_or_global_name: Union[TfExpression, str]) -> str:
        """Get the local name of a given variable, without any surrounding name scopes."""
        assert tfutil.is_tf_expression(var_or_global_name) or isinstance(var_or_global_name, str)
        global_name = var_or_global_name if isinstance(var_or_global_name, str) else var_or_global_name.name
        return self._get_var_global_to_local()[global_name]

    def find_var(self, var_or_local_name: Union[TfExpression, str]) -> TfExpression:
        """Find variable by local or global name."""
        assert tfutil.is_tf_expression(var_or_local_name) or isinstance(var_or_local_name, str)
        return self._get_vars()[var_or_local_name] if isinstance(var_or_local_name, str) else var_or_local_name

    def get_var(self, var_or_local_name: Union[TfExpression, str]) -> np.ndarray:
        """Get the value of a given variable as NumPy array.
        Note: This method is very inefficient -- prefer to use tflib.run(list_of_vars) whenever possible."""
        return self.find_var(var_or_local_name).eval()

    def set_var(self, var_or_local_name: Union[TfExpression, str], new_value: Union[int, float, np.ndarray]) -> None:
        """Set the value of a given variable based on the given NumPy array.
        Note: This method is very inefficient -- prefer to use tflib.set_vars() whenever possible."""
        tfutil.set_vars({self.find_var(var_or_local_name): new_value})

    def __getstate__(self) -> dict:
        """Pickle export."""
        state = dict()
        state["version"]            = 5
        state["name"]               = self.name
        state["static_kwargs"]      = dict(self.static_kwargs)
        state["components"]         = dict(self.components)
        state["build_module_src"]   = self._build_module_src
        state["build_func_name"]    = self._build_func_name
        state["variables"]          = list(zip(self._get_own_vars().keys(), tfutil.run(list(self._get_own_vars().values()))))
        state["input_shapes"]       = self.input_shapes
        state["output_shapes"]      = self.output_shapes
        state["input_names"]        = self.input_names
        state["output_names"]       = self.output_names
        return state

    def __setstate__(self, state: dict) -> None:
        """Pickle import."""

        # Execute custom import handlers.
        for handler in _import_handlers:
            state = handler(state)

        # Get basic fields.
        assert state["version"] in [2, 3, 4, 5]
        name = state["name"]
        static_kwargs = state["static_kwargs"]
        build_module_src = state["build_module_src"]
        build_func_name = state["build_func_name"]

        # Create temporary module from the imported source code.
        module_name = "_tflib_network_import_" + uuid.uuid4().hex
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module
        _import_module_src[module] = build_module_src
        exec(build_module_src, module.__dict__) # pylint: disable=exec-used
        build_func = util.get_obj_from_module(module, build_func_name)

        # Initialize fields.
        self._init_fields(name=name, static_kwargs=static_kwargs, build_func=build_func, build_func_name=build_func_name, build_module_src=build_module_src)
        self._var_inits.update(copy.deepcopy(state["variables"]))
        self._all_inits_known   = True
        self._components        = util.EasyDict(state.get("components", {}))
        self._input_shapes      = copy.deepcopy(state.get("input_shapes", None))
        self._output_shapes     = copy.deepcopy(state.get("output_shapes", None))
        self._input_names       = copy.deepcopy(state.get("input_names", None))
        self._output_names      = copy.deepcopy(state.get("output_names", None))

    def clone(self, name: str = None, **new_static_kwargs) -> "Network":
        """Create a clone of this network with its own copy of the variables."""
        static_kwargs = dict(self.static_kwargs)
        static_kwargs.update(new_static_kwargs)
        net = object.__new__(Network)
        net._init_fields(name=(name or self.name), static_kwargs=static_kwargs, build_func=self._build_func, build_func_name=self._build_func_name, build_module_src=self._build_module_src)
        net.copy_vars_from(self)
        return net

    def copy_own_vars_from(self, src_net: "Network") -> None:
        """Copy the values of all variables from the given network, excluding sub-networks."""

        # Source has unknown variables or unknown components => init now.
        if (src_net._var_inits is not None and not src_net._all_inits_known) or src_net._components is None:
            src_net._get_vars()

       # Both networks are inited => copy directly.
        if src_net._var_inits is None and self._var_inits is None:
            names = [name for name in self._get_own_vars().keys() if name in src_net._get_own_vars()]
            tfutil.set_vars(tfutil.run({self._get_vars()[name]: src_net._get_vars()[name] for name in names}))
            return

        # Read from source.
        if src_net._var_inits is None:
            value_dict = tfutil.run(src_net._get_own_vars())
        else:
            value_dict = src_net._var_inits

        # Write to destination.
        if self._var_inits is None:
            tfutil.set_vars({self._get_vars()[name]: value for name, value in value_dict.items() if name in self._get_vars()})
        else:
            self._var_inits.update(value_dict)

    def copy_vars_from(self, src_net: "Network") -> None:
        """Copy the values of all variables from the given network, including sub-networks."""

        # Source has unknown variables or unknown components => init now.
        if (src_net._var_inits is not None and not src_net._all_inits_known) or src_net._components is None:
            src_net._get_vars()

        # Source is inited, but destination components have not been created yet => set as initial values.
        if src_net._var_inits is None and self._components is None:
            self._var_inits.update(tfutil.run(src_net._get_vars()))
            return

        # Destination has unknown components => init now.
        if self._components is None:
            self._get_vars()

        # Both networks are inited => copy directly.
        if src_net._var_inits is None and self._var_inits is None:
            names = [name for name in self._get_vars().keys() if name in src_net._get_vars()]
            tfutil.set_vars(tfutil.run({self._get_vars()[name]: src_net._get_vars()[name] for name in names}))
            return

        # Copy recursively, component by component.
        self.copy_own_vars_from(src_net)
        for name, src_comp in src_net._components.items():
            if name in self._components:
                self._components[name].copy_vars_from(src_comp)

    def copy_trainables_from(self, src_net: "Network") -> None:
        """Copy the values of all trainable variables from the given network, including sub-networks."""
        names = [name for name in self._get_trainables().keys() if name in src_net._get_trainables()]
        tfutil.set_vars(tfutil.run({self._get_vars()[name]: src_net._get_vars()[name] for name in names}))

    def convert(self, new_func_name: str, new_name: str = None, **new_static_kwargs) -> "Network":
        """Create new network with the given parameters, and copy all variables from this network."""
        if new_name is None:
            new_name = self.name
        static_kwargs = dict(self.static_kwargs)
        static_kwargs.update(new_static_kwargs)
        net = Network(name=new_name, func_name=new_func_name, **static_kwargs)
        net.copy_vars_from(self)
        return net

    def setup_as_moving_average_of(self, src_net: "Network", beta: TfExpressionEx = 0.99, beta_nontrainable: TfExpressionEx = 0.0) -> tf.Operation:
        """Construct a TensorFlow op that updates the variables of this network
        to be slightly closer to those of the given network."""
        with tfutil.absolute_name_scope(self.scope + "/_MovingAvg"):
            ops = []
            for name, var in self._get_vars().items():
                if name in src_net._get_vars():
                    cur_beta = beta if var.trainable else beta_nontrainable
                    new_value = tfutil.lerp(src_net._get_vars()[name], var, cur_beta)
                    ops.append(var.assign(new_value))
            return tf.group(*ops)

    def run(self,
            *in_arrays: Tuple[Union[np.ndarray, None], ...],
            input_transform: dict = None,
            output_transform: dict = None,
            return_as_list: bool = False,
            print_progress: bool = False,
            minibatch_size: int = None,
            num_gpus: int = 1,
            assume_frozen: bool = False,
            **dynamic_kwargs) -> Union[np.ndarray, Tuple[np.ndarray, ...], List[np.ndarray]]:
        """Run this network for the given NumPy array(s), and return the output(s) as NumPy array(s).

        Args:
            input_transform:    A dict specifying a custom transformation to be applied to the input tensor(s) before evaluating the network.
                                The dict must contain a 'func' field that points to a top-level function. The function is called with the input
                                TensorFlow expression(s) as positional arguments. Any remaining fields of the dict will be passed in as kwargs.
            output_transform:   A dict specifying a custom transformation to be applied to the output tensor(s) after evaluating the network.
                                The dict must contain a 'func' field that points to a top-level function. The function is called with the output
                                TensorFlow expression(s) as positional arguments. Any remaining fields of the dict will be passed in as kwargs.
            return_as_list:     True = return a list of NumPy arrays, False = return a single NumPy array, or a tuple if there are multiple outputs.
            print_progress:     Print progress to the console? Useful for very large input arrays.
            minibatch_size:     Maximum minibatch size to use, None = disable batching.
            num_gpus:           Number of GPUs to use.
            assume_frozen:      Improve multi-GPU performance by assuming that the trainable parameters will remain changed between calls.
            dynamic_kwargs:     Additional keyword arguments to be passed into the network build function.
        """
        assert len(in_arrays) == self.num_inputs
        assert not all(arr is None for arr in in_arrays)
        assert input_transform is None or util.is_top_level_function(input_transform["func"])
        assert output_transform is None or util.is_top_level_function(output_transform["func"])
        output_transform, dynamic_kwargs = _handle_legacy_output_transforms(output_transform, dynamic_kwargs)
        num_items = in_arrays[0].shape[0]
        if minibatch_size is None:
            minibatch_size = num_items

        # Construct unique hash key from all arguments that affect the TensorFlow graph.
        key = dict(input_transform=input_transform, output_transform=output_transform, num_gpus=num_gpus, assume_frozen=assume_frozen, dynamic_kwargs=dynamic_kwargs)
        def unwind_key(obj):
            if isinstance(obj, dict):
                return [(key, unwind_key(value)) for key, value in sorted(obj.items())]
            if callable(obj):
                return util.get_top_level_function_name(obj)
            return obj
        key = repr(unwind_key(key))

        # Build graph.
        if key not in self._run_cache:
            with tfutil.absolute_name_scope(self.scope + "/_Run"), tf.control_dependencies(None):
                with tf.device("/cpu:0"):
                    in_expr = [tf.placeholder(tf.float32, name=name) for name in self.input_names]
                    in_split = list(zip(*[tf.split(x, num_gpus) for x in in_expr]))

                out_split = []
                for gpu in range(num_gpus):
                    with tf.device(self.device if num_gpus == 1 else "/gpu:%d" % gpu):
                        net_gpu = self.clone() if assume_frozen else self
                        in_gpu = in_split[gpu]

                        if input_transform is not None:
                            in_kwargs = dict(input_transform)
                            in_gpu = in_kwargs.pop("func")(*in_gpu, **in_kwargs)
                            in_gpu = [in_gpu] if tfutil.is_tf_expression(in_gpu) else list(in_gpu)

                        assert len(in_gpu) == self.num_inputs
                        out_gpu = net_gpu.get_output_for(*in_gpu, return_as_list=True, **dynamic_kwargs)

                        if output_transform is not None:
                            out_kwargs = dict(output_transform)
                            out_gpu = out_kwargs.pop("func")(*out_gpu, **out_kwargs)
                            out_gpu = [out_gpu] if tfutil.is_tf_expression(out_gpu) else list(out_gpu)

                        assert len(out_gpu) == self.num_outputs
                        out_split.append(out_gpu)

                with tf.device("/cpu:0"):
                    out_expr = [tf.concat(outputs, axis=0) for outputs in zip(*out_split)]
                    self._run_cache[key] = in_expr, out_expr

        # Run minibatches.
        in_expr, out_expr = self._run_cache[key]
        out_arrays = [np.empty([num_items] + expr.shape.as_list()[1:], expr.dtype.name) for expr in out_expr]

        for mb_begin in range(0, num_items, minibatch_size):
            if print_progress:
                print("\r%d / %d" % (mb_begin, num_items), end="")

            mb_end = min(mb_begin + minibatch_size, num_items)
            mb_num = mb_end - mb_begin
            mb_in = [src[mb_begin : mb_end] if src is not None else np.zeros([mb_num] + shape[1:]) for src, shape in zip(in_arrays, self.input_shapes)]
            mb_out = tf.get_default_session().run(out_expr, dict(zip(in_expr, mb_in)))

            for dst, src in zip(out_arrays, mb_out):
                dst[mb_begin: mb_end] = src

        # Done.
        if print_progress:
            print("\r%d / %d" % (num_items, num_items))

        if not return_as_list:
            out_arrays = out_arrays[0] if len(out_arrays) == 1 else tuple(out_arrays)
        return out_arrays

    def list_ops(self) -> List[TfExpression]:
        _ = self.output_templates  # ensure that the template graph has been created
        include_prefix = self.scope + "/"
        exclude_prefix = include_prefix + "_"
        ops = tf.get_default_graph().get_operations()
        ops = [op for op in ops if op.name.startswith(include_prefix)]
        ops = [op for op in ops if not op.name.startswith(exclude_prefix)]
        return ops

    def list_layers(self) -> List[Tuple[str, TfExpression, List[TfExpression]]]:
        """Returns a list of (layer_name, output_expr, trainable_vars) tuples corresponding to
        individual layers of the network. Mainly intended to be used for reporting."""
        layers = []

        def recurse(scope, parent_ops, parent_vars, level):
            if len(parent_ops) == 0 and len(parent_vars) == 0:
                return

            # Ignore specific patterns.
            if any(p in scope for p in ["/Shape", "/strided_slice", "/Cast", "/concat", "/Assign"]):
                return

            # Filter ops and vars by scope.
            global_prefix = scope + "/"
            local_prefix = global_prefix[len(self.scope) + 1:]
            cur_ops = [op for op in parent_ops if op.name.startswith(global_prefix) or op.name == global_prefix[:-1]]
            cur_vars = [(name, var) for name, var in parent_vars if name.startswith(local_prefix) or name == local_prefix[:-1]]
            if not cur_ops and not cur_vars:
                return

            # Filter out all ops related to variables.
            for var in [op for op in cur_ops if op.type.startswith("Variable")]:
                var_prefix = var.name + "/"
                cur_ops = [op for op in cur_ops if not op.name.startswith(var_prefix)]

            # Scope does not contain ops as immediate children => recurse deeper.
            contains_direct_ops = any("/" not in op.name[len(global_prefix):] and op.type not in ["Identity", "Cast", "Transpose"] for op in cur_ops)
            if (level == 0 or not contains_direct_ops) and (len(cur_ops) != 0 or len(cur_vars) != 0):
                visited = set()
                for rel_name in [op.name[len(global_prefix):] for op in cur_ops] + [name[len(local_prefix):] for name, _var in cur_vars]:
                    token = rel_name.split("/")[0]
                    if token not in visited:
                        recurse(global_prefix + token, cur_ops, cur_vars, level + 1)
                        visited.add(token)
                return

            # Report layer.
            layer_name = scope[len(self.scope) + 1:]
            layer_output = cur_ops[-1].outputs[0] if cur_ops else cur_vars[-1][1]
            layer_trainables = [var for _name, var in cur_vars if var.trainable]
            layers.append((layer_name, layer_output, layer_trainables))

        recurse(self.scope, self.list_ops(), list(self._get_vars().items()), 0)
        return layers

    def print_layers(self, title: str = None, hide_layers_with_no_params: bool = False) -> None:
        """Print a summary table of the network structure."""
        rows = [[title if title is not None else self.name, "Params", "OutputShape", "WeightShape"]]
        rows += [["---"] * 4]
        total_params = 0

        for layer_name, layer_output, layer_trainables in self.list_layers():
            num_params = sum(int(np.prod(var.shape.as_list())) for var in layer_trainables)
            weights = [var for var in layer_trainables if var.name.endswith("/weight:0")]
            weights.sort(key=lambda x: len(x.name))
            if len(weights) == 0 and len(layer_trainables) == 1:
                weights = layer_trainables
            total_params += num_params

            if not hide_layers_with_no_params or num_params != 0:
                num_params_str = str(num_params) if num_params > 0 else "-"
                output_shape_str = str(layer_output.shape)
                weight_shape_str = str(weights[0].shape) if len(weights) >= 1 else "-"
                rows += [[layer_name, num_params_str, output_shape_str, weight_shape_str]]

        rows += [["---"] * 4]
        rows += [["Total", str(total_params), "", ""]]

        widths = [max(len(cell) for cell in column) for column in zip(*rows)]
        print()
        for row in rows:
            print("  ".join(cell + " " * (width - len(cell)) for cell, width in zip(row, widths)))
        print()

    def setup_weight_histograms(self, title: str = None) -> None:
        """Construct summary ops to include histograms of all trainable parameters in TensorBoard."""
        if title is None:
            title = self.name

        with tf.name_scope(None), tf.device(None), tf.control_dependencies(None):
            for local_name, var in self._get_trainables().items():
                if "/" in local_name:
                    p = local_name.split("/")
                    name = title + "_" + p[-1] + "/" + "_".join(p[:-1])
                else:
                    name = title + "_toplevel/" + local_name

                tf.summary.histogram(name, var)

#----------------------------------------------------------------------------
# Backwards-compatible emulation of legacy output transformation in Network.run().

_print_legacy_warning = True

def _handle_legacy_output_transforms(output_transform, dynamic_kwargs):
    global _print_legacy_warning
    legacy_kwargs = ["out_mul", "out_add", "out_shrink", "out_dtype"]
    if not any(kwarg in dynamic_kwargs for kwarg in legacy_kwargs):
        return output_transform, dynamic_kwargs

    if _print_legacy_warning:
        _print_legacy_warning = False
        print()
        print("WARNING: Old-style output transformations in Network.run() are deprecated.")
        print("Consider using 'output_transform=dict(func=tflib.convert_images_to_uint8)'")
        print("instead of 'out_mul=127.5, out_add=127.5, out_dtype=np.uint8'.")
        print()
    assert output_transform is None

    new_kwargs = dict(dynamic_kwargs)
    new_transform = {kwarg: new_kwargs.pop(kwarg) for kwarg in legacy_kwargs if kwarg in dynamic_kwargs}
    new_transform["func"] = _legacy_output_transform_func
    return new_transform, new_kwargs

def _legacy_output_transform_func(*expr, out_mul=1.0, out_add=0.0, out_shrink=1, out_dtype=None):
    if out_mul != 1.0:
        expr = [x * out_mul for x in expr]

    if out_add != 0.0:
        expr = [x + out_add for x in expr]

    if out_shrink > 1:
        ksize = [1, 1, out_shrink, out_shrink]
        expr = [tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding="VALID", data_format="NCHW") for x in expr]

    if out_dtype is not None:
        if tf.as_dtype(out_dtype).is_integer:
            expr = [tf.round(x) for x in expr]
        expr = [tf.saturate_cast(x, out_dtype) for x in expr]
    return expr
