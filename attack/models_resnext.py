import torch.nn as nn
from resnext101_64x4d_features import resnext101_64x4d_features


__all__ = ['resnext101_64x4d']


pretrained_settings = {
    'resnext101_64x4d': {
        'imagenet': {
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    }
}


class ResNeXt101_64x4d(nn.Module):

    def __init__(self, nb_classes=1000):
        super(ResNeXt101_64x4d, self).__init__()
        self.features = resnext101_64x4d_features
        self.avgpool = nn.AvgPool2d((7, 7), (1, 1))
        self.fc = nn.Linear(2048, nb_classes)

    def forward(self, input):
        x = self.features(input)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnext101_64x4d(num_classes=1000, pretrained='imagenet'):
    model = ResNeXt101_64x4d()
    if pretrained:
        settings = pretrained_settings['resnext101_64x4d'][pretrained]
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    return model
