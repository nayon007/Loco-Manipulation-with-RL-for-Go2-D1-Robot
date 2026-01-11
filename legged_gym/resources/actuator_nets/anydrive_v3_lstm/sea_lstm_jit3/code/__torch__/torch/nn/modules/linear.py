class Linear(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Tensor
  training : bool
  _is_full_backward_hook : None
  out_features : Final[int] = 1
  in_features : Final[int] = 8
  def forward(self: __torch__.torch.nn.modules.linear.Linear,
    input: Tensor) -> Tensor:
    _0 = __torch__.torch.nn.functional.linear
    return _0(input, self.weight, self.bias, )
