import torch
import torch.nn as nn
import torchvision.models

class ContentLoss(nn.Module):
    def __init__(self, loss):
        super(ContentLoss, self).__init__()
        self.criterion = loss
        self.net = self.content_model()

    def get_loss(self, pred, target):
        pred_f = self.net(pred)
        target_f = self.net(target)
        loss = self.criterion(pred_f, target_f)

        return loss

    def content_model(self):
        self.cnn = torchvision.models.vgg19(pretrained=True).features
        self.cnn.cuda()
        # Content loss 계산을 위한 레이어 선택
        content_layers = ["relu_8"]

        model = nn.Sequential()
        i = 0
        for layer in self.cnn.children():
            # Content loss 계산을 위한 모델 추출
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = "conv_{}".format(i)
            elif isinstance(layer, nn.ReLU):
                name = "relu_{}".format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = "pool_{}".format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = "bn_{}".format(i)
            else:
                raise RuntimeError("Unrecognized layer: {}".format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                break

        return model