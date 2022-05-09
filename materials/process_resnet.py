import json
import torch
import torch.nn.functional as F
from torchvision import models

# p = torch.load('resnet50-raw.pth')
# w = p['fc.weight'].data
# b = p['fc.bias'].data

# p.pop('fc.weight')
# p.pop('fc.bias')
# torch.save(p, 'resnet50-base.pth')

model = models.resnet101(pretrained=True)

w = model.fc.weight.data
b = model.fc.bias.data

del model.fc
torch.save(model.state_dict(), 'resnet101-base.pth')


v = torch.cat([w, b.unsqueeze(1)], dim=1).tolist()
wnids = json.load(open('imagenet-split.json', 'r'))['train']
wnids = sorted(wnids)
obj = []
for i in range(len(wnids)):
    obj.append((wnids[i], v[i]))
json.dump(obj, open('fc-weights.json', 'w'))

