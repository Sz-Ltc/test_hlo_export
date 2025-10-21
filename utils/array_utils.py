from collections.abc import Iterable


def normalize_axes(axes: Iterable[int], ndim: int) -> tuple[int, ...]:
  """Normalize axis indices to positive values.

  Args:
      axes: Iterable of axis indices (can be negative)
      ndim: Number of dimensions

  Returns:
      Tuple of normalized positive axis indices
  """
  return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def canonicalize_tuple(x):
  """Convert input to a tuple.

  Args:
      x: Input value or iterable

  Returns:
      Tuple representation of the input
  """
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)
