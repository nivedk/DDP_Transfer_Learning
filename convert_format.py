from torch.autograd import Variable

import torch.onnx
import torchvision
import torch

dummy_input = Variable(torch.randn(1, 3, 256, 256))
state_dict = torch.load('./Models_Forward/1_forward_Model.pth')
model.load_state_dict(state_dict)
torch.onnx.export(model, dummy_input, "moment-in-time.onnx")