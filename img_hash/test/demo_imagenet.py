import argparse
import json

import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import nd, image
from mxnet.gluon.data.vision import transforms

from mxnet.gluon.model_zoo.vision import get_model

# parser = argparse.ArgumentParser(description='Predict ImageNet classes from a given image')
# parser.add_argument('--model', type=str, required=True,
#                     help='name of the model to use')
# parser.add_argument('--saved-params', type=str, default='',
#                     help='path to the saved model parameters')
# parser.add_argument('--input-pic', type=str, required=True,
#                     help='path to the input picture')
# opt = parser.parse_args()
from mxnet.visualization import plot_network

classes = 1000

# with open('imagenet_labels.txt', 'r') as f:
#     class_names = [l.strip('\n') for l in f.readlines()]
class_names = json.load(open('image_net_labels.json', 'r'))

# Load Model
model_name = "mobilenet1.0"
pretrained = True
kwargs = {'classes': classes, 'pretrained': pretrained}
net = get_model(model_name, **kwargs)

# Load Images
img = image.imread("/Users/wang/workspace/XX-ImageLabel/img_hash/data/imgs_data/08e0vvDseXl8lzDNo6JU4oodjy4LJ.jpg")

# Transform
transform_fn = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

img = transform_fn(img)
# net.hybridize()
pred = net(img.expand_dims(0))
# net.export('mobilenet1.0')

sym = mx.sym.load('mobilenet1.0-symbol.json')
plot_network(sym).view()  # 画图

topK = 5
ind = nd.topk(pred, k=topK)[0].astype('int')
print('The input picture is classified to be')
for i in range(topK):
    print('\t[%s], with probability %.3f.' %
          (class_names[ind[i].asscalar()], nd.softmax(pred)[0][ind[i]].asscalar()))
