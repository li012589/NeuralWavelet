import torch

lineSize = 4
batch = 2

o = torch.randn(batch, 3, lineSize)
e = torch.randn(batch, 3, lineSize)

v = torch.cat([o, e], -1).reshape(batch, 3, 2, lineSize).permute([0, 1, 3, 2]).reshape(batch, 3, 2 * lineSize)

transMatrix = torch.tensor([[3/4, 1/2, -1/4, 0, 0, 0, 0, 0],
                            [-1/8, 1/4, 3/4, 1/4, -1/8, 0, 0, 0],
                            [0, 0, -1/8, 1/4, 3/4, 1/4, -1/8, 0],
                            [0, 0, 0, 0, -1/8, 1/4, 5/8, 1/4],
                            [-1/2, 1, -1/2, 0, 0, 0, 0, 0],
                            [0, 0, -1/2, 1, -1/2, 0, 0, 0],
                            [0, 0, 0, 0, -1/2, 1, -1/2, 0],
                            [0, 0, 0, 0, 0, 0, -1, 1]])

trans_v = torch.matmul(v, transMatrix.t())

l1 = torch.nn.Conv1d(3, 3, 2)

weight = torch.tensor([[[0.5, 0.5], [0, 0], [0, 0]], [[0, 0], [0.5, 0.5], [0, 0]], [[0, 0], [0, 0], [0.5, 0.5]]])

l1.weight.data = weight
l1.bias.data.zero_()

tmpd = l1(torch.nn.functional.pad(o, (0, 1), "replicate"))

d = e - tmpd

l2 = torch.nn.Conv1d(3, 3, 2)

weight = torch.tensor([[[0.25, 0.25], [0, 0], [0, 0]], [[0, 0], [0.25, 0.25], [0, 0]], [[0, 0], [0, 0], [0.25, 0.25]]])

l2.weight.data = weight
l2.bias.data.zero_()

tmps = l2(torch.nn.functional.pad(d, (1, 0), "replicate"))

s = o + tmps

vp = torch.cat([s, d], -1)

assert vp.allclose(trans_v)


