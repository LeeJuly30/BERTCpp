import torch
from model_pb2 import Model

def flatten_tensor(tensor, trans=False):
    if trans:
        tensor = tensor.t().contiguous().view(1, -1)
    else:
        tensor = tensor.contiguous().view(1, -1)
    return tensor.numpy().tolist()[0]
pytorch_model = torch.load('../model/pytorch_model.bin', map_location='cpu')
model = Model()
pre_name = None
pre_layer = None
for name, param in pytorch_model.items():
    p = Model.Paramter()
    dim = param.dim()
    p.n_dim = dim
    if 'embeddings' not in name and 'weight' in name:
        for i in range(dim-1, -1, -1):
            p.dim.append(param.size(i))
        p.data[:] = flatten_tensor(param, True)
    else:
        for i in range(dim):
            p.dim.append(param.size(i))
        p.data[:] = flatten_tensor(param)
    p.name = name
    model.param.append(p)
with open('../model/model.proto', 'wb') as writer:
    writer.write(model.SerializeToString())
            