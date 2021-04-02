# python 3.7
"""Utility functions for image editing from latent space."""

import os.path
import numpy as np

__all__ = [
    'parse_indices', 'interpolate', 'mix_style',
    'get_layerwise_manipulation_strength', 'manipulate', 'parse_boundary_list'
]


def parse_indices(obj, min_val=None, max_val=None):
  """Parses indices.

  If the input is a list or tuple, this function has no effect.

  The input can also be a string, which is either a comma separated list of
  numbers 'a, b, c', or a dash separated range 'a - c'. Space in the string will
  be ignored.

  Args:
    obj: The input object to parse indices from.
    min_val: If not `None`, this function will check that all indices are equal
      to or larger than this value. (default: None)
    max_val: If not `None`, this function will check that all indices are equal
      to or smaller than this field. (default: None)

  Returns:
    A list of integers.

  Raises:
    If the input is invalid, i.e., neither a list or tuple, nor a string.
  """
  if obj is None or obj == '':
    indices = []
  elif isinstance(obj, int):
    indices = [obj]
  elif isinstance(obj, (list, tuple, np.ndarray)):
    indices = list(obj)
  elif isinstance(obj, str):
    indices = []
    splits = obj.replace(' ', '').split(',')
    for split in splits:
      numbers = list(map(int, split.split('-')))
      if len(numbers) == 1:
        indices.append(numbers[0])
      elif len(numbers) == 2:
        indices.extend(list(range(numbers[0], numbers[1] + 1)))
  else:
    raise ValueError(f'Invalid type of input: {type(obj)}!')

  assert isinstance(indices, list)
  indices = sorted(list(set(indices)))
  for idx in indices:
    assert isinstance(idx, int)
    if min_val is not None:
      assert idx >= min_val, f'{idx} is smaller than min val `{min_val}`!'
    if max_val is not None:
      assert idx <= max_val, f'{idx} is larger than max val `{max_val}`!'

  return indices


def interpolate(src_codes, dst_codes, step=5):
  """Interpolates two sets of latent codes linearly.

  Args:
    src_codes: Source codes, with shape [num, *code_shape].
    dst_codes: Target codes, with shape [num, *code_shape].
    step: Number of interplolation steps, with source and target included. For
      example, if `step = 5`, three more samples will be inserted. (default: 5)

  Returns:
    Interpolated codes, with shape [num, step, *code_shape].

  Raises:
    ValueError: If the input two sets of latent codes are with different shapes.
  """
  if not (src_codes.ndim >= 2 and src_codes.shape == dst_codes.shape):
    raise ValueError(f'Shapes of source codes and target codes should both be '
                     f'[num, *code_shape], but {src_codes.shape} and '
                     f'{dst_codes.shape} are received!')
  num = src_codes.shape[0]
  code_shape = src_codes.shape[1:]

  a = src_codes[:, np.newaxis]
  b = dst_codes[:, np.newaxis]
  l = np.linspace(0.0, 1.0, step).reshape(
      [step if axis == 1 else 1 for axis in range(a.ndim)])
  results = a + l * (b - a)
  assert results.shape == (num, step, *code_shape)

  return results


def mix_style(style_codes,
              content_codes,
              num_layers=1,
              mix_layers=None,
              is_style_layerwise=True,
              is_content_layerwise=True):
  """Mixes styles from style codes to those of content codes.

  Each style code or content code consists of `num_layers` codes, each of which
  is typically fed into a particular layer of the generator. This function mixes
  styles by partially replacing the codes of `content_codes` from some certain
  layers with those of `style_codes`.

  For example, if both style code and content code are with shape [10, 512],
  meaning to have 10 layers and each employs a 512-dimensional latent code. And
  the 1st, 2nd, and 3rd layers are the target layers to perform style mixing.
  Then the top half of the content code (with shape [3, 512]) will be replaced
  by the top half of the style code (also with shape [3, 512]).

  NOTE: This function also supports taking single-layer latent codes as inputs,
  i.e., setting `is_style_layerwise` or `is_content_layerwise` as False. In this
  case, the corresponding code will be first repeated for `num_layers` before
  performing style mixing.

  Args:
    style_codes: Style codes, with shape [num_styles, *code_shape] or
      [num_styles, num_layers, *code_shape].
    content_codes: Content codes, with shape [num_contents, *code_shape] or
      [num_contents, num_layers, *code_shape].
    num_layers: Total number of layers in the generative model. (default: 1)
    mix_layers: Indices of the layers to perform style mixing. `None` means to
      replace all layers, in which case the content code will be completely
      replaced by style code. (default: None)
    is_style_layerwise: Indicating whether the input `style_codes` are
      layer-wise codes. (default: True)
    is_content_layerwise: Indicating whether the input `content_codes` are
      layer-wise codes. (default: True)
    num_layers

  Returns:
    Codes after style mixing, with shape [num_styles, num_contents, num_layers,
      *code_shape].

  Raises:
    ValueError: If input `content_codes` or `style_codes` is with invalid shape.
  """
  if not is_style_layerwise:
    style_codes = style_codes[:, np.newaxis]
    style_codes = np.tile(
        style_codes,
        [num_layers if axis == 1 else 1 for axis in range(style_codes.ndim)])
  if not is_content_layerwise:
    content_codes = content_codes[:, np.newaxis]
    content_codes = np.tile(
        content_codes,
        [num_layers if axis == 1 else 1 for axis in range(content_codes.ndim)])

  if not (style_codes.ndim >= 3 and style_codes.shape[1] == num_layers and
          style_codes.shape[1:] == content_codes.shape[1:]):
    raise ValueError(f'Shapes of style codes and content codes should be '
                     f'[num_styles, num_layers, *code_shape] and '
                     f'[num_contents, num_layers, *code_shape] respectively, '
                     f'but {style_codes.shape} and {content_codes.shape} are '
                     f'received!')

  layer_indices = parse_indices(mix_layers, min_val=0, max_val=num_layers - 1)
  if not layer_indices:
    layer_indices = list(range(num_layers))

  num_styles = style_codes.shape[0]
  num_contents = content_codes.shape[0]
  code_shape = content_codes.shape[2:]

  s = style_codes[:, np.newaxis]
  s = np.tile(s, [num_contents if axis == 1 else 1 for axis in range(s.ndim)])
  c = content_codes[np.newaxis]
  c = np.tile(c, [num_styles if axis == 0 else 1 for axis in range(c.ndim)])

  from_style = np.zeros(s.shape, dtype=bool)
  from_style[:, :, layer_indices] = True
  results = np.where(from_style, s, c)
  assert results.shape == (num_styles, num_contents, num_layers, *code_shape)

  return results


def get_layerwise_manipulation_strength(num_layers,
                                        truncation_psi,
                                        truncation_layers):
  """Gets layer-wise strength for manipulation.

  Recall the truncation trick played on layer [0, truncation_layers):

  w = truncation_psi * w + (1 - truncation_psi) * w_avg

  So, when using the same boundary to manipulate different layers, layer
  [0, truncation_layers) and layer [truncation_layers, num_layers) should use
  different strength to eliminate the effect from the truncation trick. More
  concretely, the strength for layer [0, truncation_layers) is set as
  `truncation_psi`, while that for other layers are set as 1.
  """
  strength = [1.0 for _ in range(num_layers)]
  if truncation_layers > 0:
    for layer_idx in range(0, truncation_layers):
      strength[layer_idx] = truncation_psi
  return strength


def manipulate(latent_codes,
               boundary,
               start_distance=-5.0,
               end_distance=5.0,
               step=21,
               layerwise_manipulation=False,
               num_layers=1,
               manipulate_layers=None,
               is_code_layerwise=False,
               is_boundary_layerwise=False,
               layerwise_manipulation_strength=1.0):
  """Manipulates the given latent codes with respect to a particular boundary.

  Basically, this function takes a set of latent codes and a boundary as inputs,
  and outputs a collection of manipulated latent codes.

  For example, let `step` to be 10, `latent_codes` to be with shape [num,
  *code_shape], and `boundary` to be with shape [1, *code_shape] and unit norm.
  Then the output will be with shape [num, 10, *code_shape]. For each 10-element
  manipulated codes, the first code is `start_distance` away from the original
  code (i.e., the input) along the `boundary` direction, while the last code is
  `end_distance` away. Remaining codes are linearly interpolated. Here,
  `distance` is sign sensitive.

  NOTE: This function also supports layer-wise manipulation, in which case the
  generator should be able to take layer-wise latent codes as inputs. For
  example, if the generator has 18 convolutional layers in total, and each of
  which takes an independent latent code as input. It is possible, sometimes
  with even better performance, to only partially manipulate these latent codes
  corresponding to some certain layers yet keeping others untouched.

  NOTE: Boundary is assumed to be normalized to unit norm already.

  Args:
    latent_codes: The input latent codes for manipulation, with shape
      [num, *code_shape] or [num, num_layers, *code_shape].
    boundary: The semantic boundary as reference, with shape [1, *code_shape] or
      [1, num_layers, *code_shape].
    start_distance: Start point for manipulation. (default: -5.0)
    end_distance: End point for manipulation. (default: 5.0)
    step: Number of manipulation steps. (default: 21)
    layerwise_manipulation: Whether to perform layer-wise manipulation.
      (default: False)
    num_layers: Number of layers. Only active when `layerwise_manipulation` is
      set as `True`. Should be a positive integer. (default: 1)
    manipulate_layers: Indices of the layers to perform manipulation. `None`
      means to manipulate latent codes from all layers. (default: None)
    is_code_layerwise: Whether the input latent codes are layer-wise. If set as
      `False`, the function will first repeat the input codes for `num_layers`
      times before perform manipulation. (default: False)
    is_boundary_layerwise: Whether the input boundary is layer-wise. If set as
      `False`, the function will first repeat boundary for `num_layers` times
      before perform manipulation. (default: False)
    layerwise_manipulation_strength: Manipulation strength for each layer. Only
      active when `layerwise_manipulation` is set as `True`. This field can be
      used to resolve the strength discrepancy across layers when truncation
      trick is on. See function `get_layerwise_manipulation_strength()` for
      details. A tuple, list, or `numpy.ndarray` is expected. If set as a single
      number, this strength will be used for all layers. (default: 1.0)

  Returns:
    Manipulated codes, with shape [num, step, *code_shape] if
      `layerwise_manipulation` is set as `False`, or shape [num, step,
      num_layers, *code_shape] if `layerwise_manipulation` is set as `True`.

  Raises:
    ValueError: If the input latent codes, boundary, or strength are with
      invalid shape.
  """
  if not (boundary.ndim >= 2 and boundary.shape[0] == 1):
    raise ValueError(f'Boundary should be with shape [1, *code_shape] or '
                     f'[1, num_layers, *code_shape], but '
                     f'{boundary.shape} is received!')

  if not layerwise_manipulation:
    assert not is_code_layerwise
    assert not is_boundary_layerwise
    num_layers = 1
    manipulate_layers = None
    layerwise_manipulation_strength = 1.0

  # Preprocessing for layer-wise manipulation.
  # Parse indices of manipulation layers.
  layer_indices = parse_indices(
      manipulate_layers, min_val=0, max_val=num_layers - 1)
  if not layer_indices:
    layer_indices = list(range(num_layers))
  # Make latent codes layer-wise if needed.
  assert num_layers > 0
  if not is_code_layerwise:
    x = latent_codes[:, np.newaxis]
    x = np.tile(x, [num_layers if axis == 1 else 1 for axis in range(x.ndim)])
  else:
    x = latent_codes
    if x.shape[1] != num_layers:
      raise ValueError(f'Latent codes should be with shape [num, num_layers, '
                       f'*code_shape], where `num_layers` equals to '
                       f'{num_layers}, but {x.shape} is received!')
  # Make boundary layer-wise if needed.
  if not is_boundary_layerwise:
    b = boundary
    b = np.tile(b, [num_layers if axis == 0 else 1 for axis in range(b.ndim)])
  else:
    b = boundary[0]
    if b.shape[0] != num_layers:
      raise ValueError(f'Boundary should be with shape [num_layers, '
                       f'*code_shape], where `num_layers` equals to '
                       f'{num_layers}, but {b.shape} is received!')
  # Get layer-wise manipulation strength.
  if isinstance(layerwise_manipulation_strength, (int, float)):
    s = [float(layerwise_manipulation_strength) for _ in range(num_layers)]
  elif isinstance(layerwise_manipulation_strength, (list, tuple)):
    s = layerwise_manipulation_strength
    if len(s) != num_layers:
      raise ValueError(f'Shape of layer-wise manipulation strength `{len(s)}` '
                       f'mismatches number of layers `{num_layers}`!')
  elif isinstance(layerwise_manipulation_strength, np.ndarray):
    s = layerwise_manipulation_strength
    if s.size != num_layers:
      raise ValueError(f'Shape of layer-wise manipulation strength `{s.size}` '
                       f'mismatches number of layers `{num_layers}`!')
  else:
    raise ValueError(f'Unsupported type of `layerwise_manipulation_strength`!')
  s = np.array(s).reshape(
      [num_layers if axis == 0 else 1 for axis in range(b.ndim)])
  b = b * s

  if x.shape[1:] != b.shape:
    raise ValueError(f'Latent code shape {x.shape} and boundary shape '
                     f'{b.shape} mismatch!')
  num = x.shape[0]
  code_shape = x.shape[2:]

  x = x[:, np.newaxis]
  b = b[np.newaxis, np.newaxis, :]
  l = np.linspace(start_distance, end_distance, step).reshape(
      [step if axis == 1 else 1 for axis in range(x.ndim)])
  results = np.tile(x, [step if axis == 1 else 1 for axis in range(x.ndim)])
  is_manipulatable = np.zeros(results.shape, dtype=bool)
  is_manipulatable[:, :, layer_indices] = True
  results = np.where(is_manipulatable, x + l * b, results)
  assert results.shape == (num, step, num_layers, *code_shape)

  return results if layerwise_manipulation else results[:, :, 0]


def manipulate2(latent_codes,
               proj,
               mindex,
               start_distance=-5.0,
               end_distance=5.0,
               step=21,
               layerwise_manipulation=False,
               num_layers=1,
               manipulate_layers=None,
               is_code_layerwise=False,
               layerwise_manipulation_strength=1.0):


  if not layerwise_manipulation:
    assert not is_code_layerwise
#    assert not is_boundary_layerwise
    num_layers = 1
    manipulate_layers = None
    layerwise_manipulation_strength = 1.0

  # Preprocessing for layer-wise manipulation.
  # Parse indices of manipulation layers.
  layer_indices = parse_indices(
      manipulate_layers, min_val=0, max_val=num_layers - 1)
  if not layer_indices:
    layer_indices = list(range(num_layers))
  # Make latent codes layer-wise if needed.
  assert num_layers > 0
  if not is_code_layerwise:
    x = latent_codes[:, np.newaxis]
    x = np.tile(x, [num_layers if axis == 1 else 1 for axis in range(x.ndim)])
  else:
    x = latent_codes
    if x.shape[1] != num_layers:
      raise ValueError(f'Latent codes should be with shape [num, num_layers, '
                       f'*code_shape], where `num_layers` equals to '
                       f'{num_layers}, but {x.shape} is received!')
  # Make boundary layer-wise if needed.
#  if not is_boundary_layerwise:
#    b = boundary
#    b = np.tile(b, [num_layers if axis == 0 else 1 for axis in range(b.ndim)])
#  else:
#    b = boundary[0]
#    if b.shape[0] != num_layers:
#      raise ValueError(f'Boundary should be with shape [num_layers, '
#                       f'*code_shape], where `num_layers` equals to '
#                       f'{num_layers}, but {b.shape} is received!')
  # Get layer-wise manipulation strength.
  if isinstance(layerwise_manipulation_strength, (int, float)):
    s = [float(layerwise_manipulation_strength) for _ in range(num_layers)]
  elif isinstance(layerwise_manipulation_strength, (list, tuple)):
    s = layerwise_manipulation_strength
    if len(s) != num_layers:
      raise ValueError(f'Shape of layer-wise manipulation strength `{len(s)}` '
                       f'mismatches number of layers `{num_layers}`!')
  elif isinstance(layerwise_manipulation_strength, np.ndarray):
    s = layerwise_manipulation_strength
    if s.size != num_layers:
      raise ValueError(f'Shape of layer-wise manipulation strength `{s.size}` '
                       f'mismatches number of layers `{num_layers}`!')
  else:
    raise ValueError(f'Unsupported type of `layerwise_manipulation_strength`!')
#  s = np.array(s).reshape(
#      [num_layers if axis == 0 else 1 for axis in range(b.ndim)])
#  b = b * s

#  if x.shape[1:] != b.shape:
#    raise ValueError(f'Latent code shape {x.shape} and boundary shape '
#                     f'{b.shape} mismatch!')
  num = x.shape[0]
  code_shape = x.shape[2:]

  x = x[:, np.newaxis]
#  b = b[np.newaxis, np.newaxis, :]
#  l = np.linspace(start_distance, end_distance, step).reshape(
#      [step if axis == 1 else 1 for axis in range(x.ndim)])
  results = np.tile(x, [step if axis == 1 else 1 for axis in range(x.ndim)])
  is_manipulatable = np.zeros(results.shape, dtype=bool)
  is_manipulatable[:, :, layer_indices] = True
  
  tmp=MPC(proj,x,mindex,start_distance,end_distance,step)
  tmp = tmp[:, :,np.newaxis]
  tmp1 = np.tile(tmp, [num_layers if axis == 2 else 1 for axis in range(tmp.ndim)])
  
  
  results = np.where(is_manipulatable, tmp1, results)
#  print(results.shape)
  assert results.shape == (num, step, num_layers, *code_shape)
  return results if layerwise_manipulation else results[:, :, 0]

def MPC(proj,x,mindex,start_distance,end_distance,step):
    # x shape (batch_size,1,num_layers,feature)
#    print(x.shape)
    x1=proj.transform(x[:,0,0,:]) #/np.sqrt(proj.explained_variance_) # (batch_size,num_pc)
    
    x1 = x1[:, np.newaxis]
    x1 = np.tile(x1, [step if axis == 1 else 1 for axis in range(x1.ndim)])
    
    
    l = np.linspace(start_distance, end_distance, step)[None,:]
    x1[:,:,mindex]+=l
    
    tmp=x1.reshape((-1,x1.shape[-1])) #*np.sqrt(proj.explained_variance_)
#    print('xxx')
    x2=proj.inverse_transform(tmp)
    x2=x2.reshape((x1.shape[0],x1.shape[1],-1))
    
#    x1 = x1[:, np.newaxis]
#    x1 = np.tile(x1, [step if axis == 1 else 1 for axis in range(x1.ndim)])
    
    return x2
    



def parse_boundary_list(boundary_list_path):
  """Parses boundary list.

  Sometimes, a text file containing a list of boundaries will significantly
  simplify image manipulation with a large amount of boundaries. This function
  is used to parse boundary information from such list file.

  Basically, each item in the list should be with format
  `($NAME, $SPACE_TYPE): $PATH`. `DISABLE` at the beginning of the line can
  disable a particular boundary.

  Sample:

  (age, z): $AGE_BOUNDARY_PATH
  (gender, w): $GENDER_BOUNDARY_PATH
  DISABLE(pose, wp): $POSE_BOUNDARY_PATH

  Args:
    boundary_list_path: Path to the boundary list.

  Returns:
    A dictionary, whose key is a two-element tuple (boundary_name, space_type)
      and value is the corresponding boundary path.

  Raise:
    ValueError: If the given boundary list does not exist.
  """
  if not os.path.isfile(boundary_list_path):
    raise ValueError(f'Boundary list `boundary_list_path` does not exist!')

  boundaries = {}
  with open(boundary_list_path, 'r') as f:
    for line in f:
      if line[:len('DISABLE')] == 'DISABLE':
        continue
      boundary_info, boundary_path = line.strip().split(':')
      boundary_name, space_type = boundary_info.strip()[1:-1].split(',')
      boundary_name = boundary_name.strip()
      space_type = space_type.strip().lower()
      boundary_path = boundary_path.strip()
      boundaries[(boundary_name, space_type)] = boundary_path
  return boundaries
