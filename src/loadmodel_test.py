import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from src.model import *
from src.util import *
model_path1 = rootPath + "/checkpoint/pretrained_malconv.pth"
model_path = rootPath + "/checkpoint/*"#/malconv.pt"
filename = get_filename(model_path)

model1 = PreMalConv()
model1.load_state_dict(torch.load(model_path1))
model1.eval()

model = MalConv()
model.load_state_dict(torch.load(model_path))
model.eval()

get_filename(model_path)

print(model1)
print(model)

