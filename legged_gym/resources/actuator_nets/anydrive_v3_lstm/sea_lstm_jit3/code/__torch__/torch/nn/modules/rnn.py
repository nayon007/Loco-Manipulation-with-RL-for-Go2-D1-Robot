class LSTM(Module):
  __parameters__ = ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0", "weight_ih_l1", "weight_hh_l1", "bias_ih_l1", "bias_hh_l1", ]
  __buffers__ = []
  weight_ih_l0 : Tensor
  weight_hh_l0 : Tensor
  bias_ih_l0 : Tensor
  bias_hh_l0 : Tensor
  weight_ih_l1 : Tensor
  weight_hh_l1 : Tensor
  bias_ih_l1 : Tensor
  bias_hh_l1 : Tensor
  training : bool
  _is_full_backward_hook : None
  _flat_weights_names : List[str]
  _all_weights : List[List[str]]
  _flat_weights : List[Tensor]
  proj_size : Final[int] = 0
  bidirectional : Final[bool] = False
  batch_first : Final[bool] = True
  input_size : Final[int] = 2
  dropout : Final[float] = 0.
  mode : Final[str] = "LSTM"
  hidden_size : Final[int] = 8
  num_layers : Final[int] = 2
  bias : Final[bool] = True
  def forward__0(self: __torch__.torch.nn.modules.rnn.LSTM,
    input: Tensor,
    hx: Optional[Tuple[Tensor, Tensor]]=None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    max_batch_size = torch.size(input, 0)
    if torch.__is__(hx, None):
      _0 = ops.prim.dtype(input)
      _1 = ops.prim.device(input)
      h_zeros = torch.zeros([2, max_batch_size, 8], dtype=_0, layout=None, device=_1, pin_memory=None)
      _2 = ops.prim.dtype(input)
      _3 = ops.prim.device(input)
      c_zeros = torch.zeros([2, max_batch_size, 8], dtype=_2, layout=None, device=_3, pin_memory=None)
      hx0 = (h_zeros, c_zeros)
    else:
      hx1 = unchecked_cast(Tuple[Tensor, Tensor], hx)
      hx2 = (self).permute_hidden(hx1, None, )
      hx0 = hx2
    _4 = (self).check_forward_args(input, hx0, None, )
    _5 = self._flat_weights
    _6 = self.training
    _7, _8, = hx0
    _9, _10, _11 = torch.lstm(input, [_7, _8], _5, True, 2, 0., _6, False, True)
    hidden = (_10, _11)
    _12 = (self).permute_hidden(hidden, None, )
    return (_9, _12)
  def forward__1(self: __torch__.torch.nn.modules.rnn.LSTM,
    input: __torch__.torch.nn.utils.rnn.PackedSequence,
    hx: Optional[Tuple[Tensor, Tensor]]=None) -> Tuple[__torch__.torch.nn.utils.rnn.PackedSequence, Tuple[Tensor, Tensor]]:
    input0, batch_sizes, sorted_indices, unsorted_indices, = input
    max_batch_size = torch.select(batch_sizes, 0, 0)
    max_batch_size0 = int(max_batch_size)
    if torch.__is__(hx, None):
      _13 = ops.prim.dtype(input0)
      _14 = ops.prim.device(input0)
      h_zeros = torch.zeros([2, max_batch_size0, 8], dtype=_13, layout=None, device=_14, pin_memory=None)
      _15 = ops.prim.dtype(input0)
      _16 = ops.prim.device(input0)
      c_zeros = torch.zeros([2, max_batch_size0, 8], dtype=_15, layout=None, device=_16, pin_memory=None)
      hx3 = (h_zeros, c_zeros)
    else:
      hx4 = unchecked_cast(Tuple[Tensor, Tensor], hx)
      hx5 = (self).permute_hidden(hx4, sorted_indices, )
      hx3 = hx5
    _17 = (self).check_forward_args(input0, hx3, batch_sizes, )
    _18 = self._flat_weights
    _19 = self.training
    _20, _21, = hx3
    _22, _23, _24 = torch.lstm(input0, batch_sizes, [_20, _21], _18, True, 2, 0., _19, False)
    hidden = (_23, _24)
    output_packed = __torch__.torch.nn.utils.rnn.PackedSequence(_22, batch_sizes, sorted_indices, unsorted_indices)
    _25 = (self).permute_hidden(hidden, unsorted_indices, )
    return (output_packed, _25)
  def permute_hidden(self: __torch__.torch.nn.modules.rnn.LSTM,
    hx: Tuple[Tensor, Tensor],
    permutation: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
    _26 = __torch__.torch.nn.modules.rnn.apply_permutation
    if torch.__is__(permutation, None):
      _27 = hx
    else:
      permutation0 = unchecked_cast(Tensor, permutation)
      _28 = (_26((hx)[0], permutation0, 1, ), _26((hx)[1], permutation0, 1, ))
      _27 = _28
    return _27
  def check_forward_args(self: __torch__.torch.nn.modules.rnn.LSTM,
    input: Tensor,
    hidden: Tuple[Tensor, Tensor],
    batch_sizes: Optional[Tensor]) -> None:
    _29 = "Expected hidden[0] size {}, got {}"
    _30 = "Expected hidden[1] size {}, got {}"
    _31 = (self).check_input(input, batch_sizes, )
    _32 = (hidden)[0]
    _33 = (self).get_expected_hidden_size(input, batch_sizes, )
    _34 = (self).check_hidden_size(_32, _33, _29, )
    _35 = (hidden)[1]
    _36 = (self).get_expected_cell_size(input, batch_sizes, )
    _37 = (self).check_hidden_size(_35, _36, _30, )
    return None
  def check_input(self: __torch__.torch.nn.modules.rnn.LSTM,
    input: Tensor,
    batch_sizes: Optional[Tensor]) -> None:
    _38 = "input must have {} dimensions, got {}"
    _39 = "input.size(-1) must be equal to input_size. Expected {}, got {}"
    if torch.__isnot__(batch_sizes, None):
      expected_input_dim = 2
    else:
      expected_input_dim = 3
    _40 = torch.ne(torch.dim(input), expected_input_dim)
    if _40:
      _41 = torch.format(_38, expected_input_dim, torch.dim(input))
      ops.prim.RaiseException(_41)
    else:
      pass
    if torch.ne(2, torch.size(input, -1)):
      _42 = torch.format(_39, 2, torch.size(input, -1))
      ops.prim.RaiseException(_42)
    else:
      pass
    return None
  def check_hidden_size(self: __torch__.torch.nn.modules.rnn.LSTM,
    hx: Tensor,
    expected_hidden_size: Tuple[int, int, int],
    msg: str="Expected hidden size {}, got {}") -> None:
    _43 = torch.size(hx)
    _44, _45, _46, = expected_hidden_size
    if torch.ne(_43, [_44, _45, _46]):
      _47 = torch.format(msg, expected_hidden_size, torch.list(torch.size(hx)))
      ops.prim.RaiseException(_47)
    else:
      pass
    return None
  def get_expected_hidden_size(self: __torch__.torch.nn.modules.rnn.LSTM,
    input: Tensor,
    batch_sizes: Optional[Tensor]) -> Tuple[int, int, int]:
    if torch.__isnot__(batch_sizes, None):
      batch_sizes0 = unchecked_cast(Tensor, batch_sizes)
      mini_batch0 = int(torch.select(batch_sizes0, 0, 0))
      mini_batch = mini_batch0
    else:
      mini_batch = torch.size(input, 0)
    return (2, mini_batch, 8)
  def get_expected_cell_size(self: __torch__.torch.nn.modules.rnn.LSTM,
    input: Tensor,
    batch_sizes: Optional[Tensor]) -> Tuple[int, int, int]:
    if torch.__isnot__(batch_sizes, None):
      batch_sizes1 = unchecked_cast(Tensor, batch_sizes)
      mini_batch1 = int(torch.select(batch_sizes1, 0, 0))
      mini_batch = mini_batch1
    else:
      mini_batch = torch.size(input, 0)
    return (2, mini_batch, 8)
def apply_permutation(tensor: Tensor,
    permutation: Tensor,
    dim: int=1) -> Tensor:
  _48 = torch.index_select(tensor, dim, permutation)
  return _48
