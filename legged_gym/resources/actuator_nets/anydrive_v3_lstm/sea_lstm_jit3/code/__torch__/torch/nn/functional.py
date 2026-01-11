def linear(input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor]=None) -> Tensor:
  return torch.linear(input, weight, bias)
