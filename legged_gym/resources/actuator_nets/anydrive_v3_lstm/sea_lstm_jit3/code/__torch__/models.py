class LSTMsea(Module):
  __parameters__ = []
  __buffers__ = ["in_scale", "out_scale", ]
  in_scale : Tensor
  out_scale : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  hidden_dim : int
  lstm : __torch__.torch.nn.modules.rnn.LSTM
  linear : __torch__.torch.nn.modules.linear.Linear
  def forward(self: __torch__.models.LSTMsea,
    x: Tensor,
    hc0: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    x0 = torch.mul(x, self.in_scale)
    x1, hcn, = (self.lstm).forward__0(x0, hc0, )
    _0 = self.out_scale
    _1 = torch.squeeze((self.linear).forward(x1, ))
    return (torch.mul(_0, _1), hcn)
