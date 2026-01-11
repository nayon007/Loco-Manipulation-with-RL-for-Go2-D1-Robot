class PackedSequence(NamedTuple):
  data : Tensor
  batch_sizes : Tensor
  sorted_indices : Optional[Tensor]
  unsorted_indices : Optional[Tensor]
