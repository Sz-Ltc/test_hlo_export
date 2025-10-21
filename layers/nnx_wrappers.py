# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NNX <> Linen interoperability."""

from functools import partial
import typing as tp
from typing import Any

from flax import linen
from flax import nnx
from flax.core import FrozenDict
from flax.core import meta
from flax.nnx import variablelib
from flax.nnx.module import Module
import jax
from jax import tree_util as jtu

from layers.initializers import shard_variables

M = tp.TypeVar("M", bound=Module)


def is_vanilla_variable(vs: variablelib.VariableState) -> bool:
  """Check if variable state is vanilla.

  A variable state is vanilla if its metadata is essentially blank.
  Returns False only if it has non-empty hooks or any non-built-in attribute.

  Args:
      vs: Variable state object

  Returns:
      True if vanilla variable, False otherwise
  """
  for key, value in vs.get_metadata().items():
    if key.endswith("_hooks"):
      if value != ():
        return False
    else:
      return False
  return True


def to_linen_var(vs: variablelib.VariableState) -> meta.AxisMetadata:
  """Convert NNX variable state to Linen variable.

  Args:
      vs: NNX variable state object

  Returns:
      Corresponding Linen variable or metadata object
  """
  metadata = vs.get_metadata()
  if "linen_meta_type" in metadata:
    linen_type = metadata["linen_meta_type"]
    if hasattr(linen_type, "from_nnx_metadata"):
      return linen_type.from_nnx_metadata({"value": vs.value, **metadata})
    return linen_type(vs.value, **metadata)
  if is_vanilla_variable(vs):
    return vs.value
  return nnx.bridge.NNXMeta(type(vs), vs.value, metadata)


def linen_rngs_dict(linen_module: linen.Module, add_default: bool = False):
  """Given a module, split out one of its every active RNG key collections."""
  assert linen_module.scope is not None, (
    "linen_rngs_dict() must be called inside a Linen module."
  )
  rngs: dict[str, tp.Any] = {
    name: linen_module.make_rng(name) for name in linen_module.scope.rngs.keys()
  }
  if add_default and "default" not in rngs:
    rngs["default"] = 0
  return rngs


def _get_module_method(module, method: tp.Callable[..., Any] | str | None):
  """Get a callable method from the module, or raise TypeError."""
  if method is None:
    method = "__call__"

  if isinstance(method, str):
    attribute_name = method
    method = getattr(type(module), attribute_name)
    if not callable(method):
      class_name = type(module).__name__
      raise TypeError(
        f"'{class_name}.{attribute_name}' must be a callable, got {type(method)}."
      )
  if not callable(method):
    class_name = type(module).__name__
    raise TypeError(f"'{method}' must be a callable, got {type(method)}.")

  return method


def _enable_linen_module_paths(module: Module):
  """Ensure that linen module_path is correct when in NNX."""

  def wrap(call_fn, name: str):
    def wrapped(*args, **kwargs):
      if not linen.module._context.module_stack:  # pylint: disable=W0212
        return call_fn(*args, **kwargs)
      nn_module = linen.module._context.module_stack[-1]  # pylint: disable=W0212
      old_path = nn_module.path
      # We modify the path of the current nn module in place. This is a litte
      # bit hacky but should be good as a temporary solution.
      nn_module.scope.path += (name,)
      try:
        return call_fn(*args, **kwargs)
      finally:
        nn_module.scope.path = old_path

    return wrapped

  for path, node in nnx.iter_graph(module):
    # Only enable it on non-root nnx modules.
    if path and isinstance(node, nnx.Module):
      node.__class__ = type(
        node.__class__.__name__,
        (node.__class__,),
        {
          "__call__": wrap(node.__class__.__call__, str(path[-1])),
        },
      )


class ToLinen(linen.Module):
  """A wrapper to turn any NNX module into a Linen module.

  The result Linen module can be used standalone with all Linen APIs, or as a
  submodule of
  another Linen module.

  Since NNX modules are stateful and owns the state, we only create it once
  during init
  time, and will track its state and static data as separate variables.

  Example::

    >>> from flax import linen as nn, nnx
    >>> import jax
    >>> model = nnx.bridge.ToLinen(nnx.Linear, args=(32, 64))
    >>> x = jax.numpy.ones((1, 32))
    >>> y, variables = model.init_with_output(jax.random.key(0), x)
    >>> y.shape
    (1, 64)
    >>> variables['params']['kernel'].shape
    (32, 64)
    >>> # The static GraphDef of the underlying NNX module
    >>> variables.keys()
    dict_keys(['params'])

  Args:
    nnx_class: The NNX Module class (not instance!).
    args: The arguments that normally would be passed in to create the NNX
      module.
    kwargs: The keyword arguments that normally would be passed in to create the
      NNX module.
    skip_rng: True if this NNX module doesn't need `rngs` arg during
      initialization (not common).

  Returns:
    A stateful NNX module that behaves the same as the wrapped Linen module.
  """

  nnx_class: tp.Callable[..., Module]
  args: tp.Sequence = ()
  kwargs: tp.Mapping[str, tp.Any] = FrozenDict({})
  skip_rng: bool = False
  metadata_fn: tp.Callable[[variablelib.VariableState], tp.Any] | None = to_linen_var

  @linen.compact
  def __call__(
    self, *args, nnx_method: tp.Callable[..., Any] | str | None = None, **kwargs
  ):
    module_kwargs = dict(self.kwargs)
    maybe_add_default = not self.is_initializing()

    def _module_kwargs():
      if not self.skip_rng:
        module_kwargs["rngs"] = nnx.Rngs(
          **linen_rngs_dict(self, add_default=maybe_add_default)
        )
      return module_kwargs

    # init codepath
    if self.is_initializing():
      module = self.nnx_class(*self.args, **_module_kwargs())
      _enable_linen_module_paths(module)
      # TODO: add lazy_init here in case there's an `ToNNX` submodule under `module`.
      # update linen variables before call module to save initial state
      self._update_variables(module)
      method_fn = _get_module_method(module, nnx_method)
      out = method_fn(module, *args, **kwargs)
      return out

    # create the nnx module
    module = self.nnx_class(*self.args, **_module_kwargs())
    _enable_linen_module_paths(module)

    # update nnx module from linen variables
    def maybe_unbox(x):
      if isinstance(x, meta.AxisMetadata):
        return x.unbox()
      return x

    states = jtu.tree_map(
      maybe_unbox,
      list(self.variables.values()),
      is_leaf=lambda x: isinstance(x, meta.AxisMetadata),
    )
    if not states:
      states = ({},)
    nnx.update(module, *states)

    method_fn = _get_module_method(module, nnx_method)
    out = method_fn(module, *args, **kwargs)
    self._update_variables(module)
    return out

  def __getattr__(self, name: str):
    if hasattr(super(), name):
      return super().__getattribute__(name)
    maybe_method = getattr(self.nnx_class, name, None)
    if callable(maybe_method):
      method = partial(self.__call__, nnx_method=maybe_method)
      setattr(method, "__self__", self)
      return method
    return super().__getattribute__(name)

  def _update_variables(self, module):
    """Store the NNX module's graph def and state inside Linen module variables."""
    state = nnx.state(module, nnx.Not(nnx.RngState))

    collection_flat_state: dict[str, list[tuple[tuple[str, ...], tp.Any]]] = {}

    # group state by collection
    for path, leaf in nnx.to_flat_state(state):
      type_ = type(leaf)
      collection = variablelib.variable_name_from_type(type_, allow_register=True)
      if collection not in collection_flat_state:
        collection_flat_state[collection] = []
      collection_flat_state[collection].append((path, leaf))

    # update linen variables
    for collection, flat_state in collection_flat_state.items():
      if self.is_mutable_collection(collection):

        def _to_linen_var(x):
          if isinstance(x, nnx.VariableState):
            if self.metadata_fn is not None:
              return self.metadata_fn(x)  # pylint: disable=too-many-function-args
            else:
              return x.value
          return x

        collection_state = nnx.traversals.unflatten_mapping(flat_state)
        collection_state = jax.tree.map(
          _to_linen_var,
          collection_state,
          is_leaf=lambda x: isinstance(x, nnx.VariableState),
        )
        for k, v in collection_state.items():
          self.put_variable(collection, k, v)

  def init(self, *args, **kwargs):
    mesh = kwargs.pop("mesh", None)
    variables = super().init(*args, **kwargs)
    if mesh is not None:
      variables = shard_variables(variables, mesh)
    return variables

  def init_with_output(self, *args, **kwargs):
    mesh = kwargs.pop("mesh", None)
    outputs, variables = super().init_with_output(*args, **kwargs)
    if mesh is not None:
      variables = shard_variables(variables, mesh)
    return outputs, variables


def to_linen(
  nnx_class: tp.Callable[..., Module],
  *args,
  metadata_fn: (tp.Callable[[variablelib.VariableState], tp.Any] | None) = to_linen_var,
  name: str | None = None,
  skip_rng: bool = False,
  abstract_init: bool = True,
  **kwargs,
):
  """Shortcut of `nnx.bridge.ToLinen`.
  Used if user is not changing any of its default fields.
  """
  return ToLinen(
    nnx_class,
    args=args,
    kwargs=FrozenDict(kwargs),
    metadata_fn=metadata_fn,
    skip_rng=skip_rng,
    name=name,
  )
