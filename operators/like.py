import torch

# Solve col LIKE %word1%word2%

def like_contains(a, p1):

  n1 = p1.shape[0]
  n, m = a.shape
  # print(f"MaskL: Memory allocated: {torch.cuda.memory_allocated() / 2**30:.4f} GiB")

  appear = torch.zeros(n, dtype=torch.bool, device=a.device)
  for pos in range(m - n1 + 1):
    appear.logical_or_((a[:, pos:pos+n1] == p1).all(dim=-1))

  # print(f"MaskR: Memory allocated: {torch.cuda.memory_allocated() / 2**30:.4f} GiB")
  return appear

def like_mask(a, p1, p2):
  assert a.dim() == 2 and p1.dim() == 1 and p2.dim() == 1
  n1, n2 = p1.shape[0], p2.shape[0]
  n, m = a.shape
  # print(f"MaskL: Memory allocated: {torch.cuda.memory_allocated() / 2**30:.4f} GiB")

  appear = torch.zeros(n, dtype=torch.bool, device=a.device)
  result = torch.zeros(n, dtype=torch.bool, device=a.device)
  for pos in range(m - n1 - n2 + 1):
    appear.logical_or_((a[:, pos:pos+n1] == p1).all(dim=-1))
    result.logical_or_(torch.logical_and(appear, (a[:, pos+n1:pos+n1+n2] == p2).all(dim=-1)))

  # print(f"MaskR: Memory allocated: {torch.cuda.memory_allocated() / 2**30:.4f} GiB")
  return result

def test():
  a = torch.tensor(
    [
      [1, 2, 3, 3, 5, 6, 3, 4, 5, 10],
      [1, 2, 3, 3, 4, 5, 7, 8, 9, 10],
      [0, 0, 3, 4, 5, 0, 0, 2, 3, 3],
      [3, 4, 5, 2, 3, 3, 3, 4, 5, 0],
      [0, 0, 0, 1, 0, 0, 0, 1, 0, 1]
    ]
  )

  p1 = torch.tensor([2, 3, 3])
  p2 = torch.tensor([3, 4, 5])

  print( like_mask(a, p1, p2) )

  b = torch.tensor(
    [
      [1, 2, 3, 3, 5, 6, 3, 4, 5, 10],
      [1, 2, 3, 3, 4, 5, 7, 8, 9, 10],
      [0, 0, 3, 4, 5, 0, 0, 2, 3, 3],
      [3, 4, 5, 2, 3, 3, 3, 4, 5, 0],
      [0, 0, 0, 1, 0, 0, 0, 2, 3, 1]
    ]
  )

  p = torch.tensor([2, 3, 3])

  print( like_contains(a, p) )