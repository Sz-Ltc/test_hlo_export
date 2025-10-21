from utils.array_utils import normalize_axes, canonicalize_tuple


def test_normalize_axes():
  # Test 1: Positive axes only
  assert normalize_axes([0, 1, 2], 3) == (0, 1, 2)

  # Test 2: Negative axes, should normalize them by adding ndim
  assert normalize_axes([-1, -2], 3) == (2, 1)

  # Test 3: Combination of positive and negative axes
  assert normalize_axes([0, -1, 2], 3) == (0, 2, 2)

  # Test 4: Empty list of axes
  assert normalize_axes([], 3) == ()

  # Test 5: Large negative values (e.g., -4 with ndim=3 should normalize to -1)
  assert normalize_axes([-4], 3) == (-1,)

  # Test 6: Negative axes with ndim being small (e.g., -1 with ndim=1)
  assert normalize_axes([-1], 1) == (0,)

  # Test 7: Test case with ndim=5 and both positive and negative axes
  assert normalize_axes([0, -1, -3], 5) == (0, 4, 2)


def test_canonicalize_tuple():
  # Test 1: Input is a list (iterable)
  assert canonicalize_tuple([1, 2, 3]) == (1, 2, 3)

  # Test 2: Input is a string (iterable)
  assert canonicalize_tuple("abc") == ("a", "b", "c")

  # Test 3: Input is an integer (non-iterable)
  assert canonicalize_tuple(5) == (5,)

  # Test 4: Input is a float (non-iterable)
  assert canonicalize_tuple(3.14) == (3.14,)

  # Test 5: Input is an empty list (iterable)
  assert canonicalize_tuple([]) == ()

  # Test 6: Input is an empty string (iterable)
  assert canonicalize_tuple("") == ()

  # Test 7: Input is None (non-iterable)
  assert canonicalize_tuple(None) == (None,)

  # Test 8: Input is a tuple (iterable)
  assert canonicalize_tuple((1, 2, 3)) == (1, 2, 3)

  # Test 9: Input is a dictionary (iterable, but should convert to a tuple of keys)
  assert canonicalize_tuple({"a": 1, "b": 2}) == ("a", "b")
