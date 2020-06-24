from image2TrainAndTest import image2TrainAndTest
from image2TrainAndTest import getValueDataFromPath
from image2TrainAndTest import getValueDataFromImg

import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import sys

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.serializers
from chainer.datasets import tuple_dataset
from chainer import Chain, Variable, optimizers
from chainer import training
from chainer.training import extensions


class AlexLike(chainer.Chain):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 224

    def __init__(self):
        super(AlexLike, self).__init__(
            conv1=L.Convolution2D(None,  96, 11, stride=4),
            conv2=L.Convolution2D(3, 256,  5, pad=2),
            conv3=L.Convolution2D(3, 384,  3, pad=1),
            conv4=L.Convolution2D(3, 384,  3, pad=1),
            conv5=L.Convolution2D(3, 256,  3, pad=1),
            fc6=L.Linear(None, 4096),
            fc7=L.Linear(4096, 1024),
            fc8=L.Linear(1024, 2),
        )
        self.train = True

    def __call__(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        h = self.fc8(h)
        return h

class GoogLeNet(chainer.Chain):

    insize = 224

    def __init__(self):
        super(GoogLeNet, self).__init__(
            conv1=L.Convolution2D(3,  64, 7, stride=2, pad=3),
            conv2_reduce=L.Convolution2D(64,  64, 1),
            conv2=L.Convolution2D(64, 192, 3, stride=1, pad=1),
            inc3a=L.Inception(192,  64,  96, 128, 16,  32,  32),
            inc3b=L.Inception(256, 128, 128, 192, 32,  96,  64),
            inc4a=L.Inception(480, 192,  96, 208, 16,  48,  64),
            inc4b=L.Inception(512, 160, 112, 224, 24,  64,  64),
            inc4c=L.Inception(512, 128, 128, 256, 24,  64,  64),
            inc4d=L.Inception(512, 112, 144, 288, 32,  64,  64),
            inc4e=L.Inception(528, 256, 160, 320, 32, 128, 128),
            inc5a=L.Inception(832, 256, 160, 320, 32, 128, 128),
            inc5b=L.Inception(832, 384, 192, 384, 48, 128, 128),
            loss3_fc=L.Linear(1024, 2),

            loss1_conv=L.Convolution2D(512, 128, 1),
            loss1_fc1=L.Linear(2048, 1024),
            loss1_fc2=L.Linear(1024, 2),

            loss2_conv=L.Convolution2D(528, 128, 1),
            loss2_fc1=L.Linear(2048, 1024),
            loss2_fc2=L.Linear(1024, 2)
        )
        self.train = True

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.local_response_normalization(
            F.max_pooling_2d(h, 3, stride=2), n=5, k=1, alpha=2e-05)
        h = F.relu(self.conv2_reduce(h))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(
            F.local_response_normalization(h, n=5, k=1, alpha=2e-05), 3, stride=2)

        h = self.inc3a(h)
        h = self.inc3b(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.inc4a(h)

        l = F.average_pooling_2d(h, 5, stride=3)
        l = F.relu(self.loss1_conv(l))
        l = F.relu(self.loss1_fc1(l))
        l = self.loss1_fc2(l)

        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)

        l = F.average_pooling_2d(h, 5, stride=3)
        l = F.relu(self.loss2_conv(l))
        l = F.relu(self.loss2_fc1(l))
        l = self.loss2_fc2(l)

        h = self.inc4e(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.inc5a(h)
        h = self.inc5b(h)

        h = F.average_pooling_2d(h, 7, stride=1)
        y = self.loss3_fc(F.dropout(h, 0.4))

        return y
        
class MLP(chainer.Chain):

    def __init__(self, n_mid_units=100, n_out=2):
        super().__init__()

        with self.init_scope():
            self.fc1 = L.Linear(None, n_mid_units)
            self.fc2 = L.Linear(n_mid_units, n_mid_units)
            self.fc3 = L.Linear(n_mid_units, n_out)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        return h

def predict(model, valData):
    x = Variable(valData)
    print(x.shape)
    print(x.data[0])
    y = F.softmax(model.predictor(x.data[0]))
    return y.data[0]

def faceDetectionFromPath(path, size):
    img = Image.open(path)
    print(img)
    img = img.resize((size, size))
    r,g,b = img.split()
    rImgData = np.asarray(np.float32(r)/255.0)
    gImgData = np.asarray(np.float32(g)/255.0)
    bImgData = np.asarray(np.float32(b)/255.0)
    imgData = np.asarray([rImgData, gImgData, bImgData])
    print(imgData.shape)
        
    return imgData

def main():
    parse = argparse.ArgumentParser(description='face detection')
    parse.add_argument('--batchsize', '-b', type=int, default=100)
    parse.add_argument('--gpu', '-g', type=int, default=-1)
    parse.add_argument('--model','-m', default='GoogLeNet_my_output_2.model')
    parse.add_argument('--size', '-s', type=int, default=224)
    parse.add_argument('--channel', '-c', default=3)
    parse.add_argument('--testpath', '-p', default="./data/test/2.jpg")
    args = parse.parse_args()

    if args.model == '':
        sys.stderr.write("Tom's Error occurred! ")
        sys.stderr.write("You have to designate the path to model")
        return

    outNumStr = args.model.split(".")[0].split("_")
    outnum = int(outNumStr[ len(outNumStr)-1 ])

    # ネットワークを作成
    predictor = GoogLeNet()
    print(args.model)

    chainer.serializers.load_npz('GoogLeNet_my_output_2.model', predictor, strict=False)

    ident = [""] * outnum
    for line in open("test.txt", "r"):
        dirname = line.split(",")[0]
        label = line.split(",")[1]
        ident[int(label)] = dirname

    # fetch value data to predict who is he/she
    faceImgs = faceDetectionFromPath(args.testpath, args.size)

    # pred = predict(predictor, faceImgs)
    # print(pred)
    # predR = np.round(pred)
    # for pre_i in np.arange(len(predR)):
    #     if predR[pre_i] == 1:
    #         print("This is a {}".format(ident[pre_i]))
            
            
            
    # # どんな画像か表示してみます
    # plt.imshow(x.reshape(224, 224,), cmap='gray')
    # plt.show()

    # ミニバッチの形にする（複数の画像をまとめて推論に使いたい場合は、サイズnのミニバッチにしてまとめればよい）
    print('元の形：', faceImgs.shape, end=' -> ')

    faceImgs = faceImgs[None, ...]

    print('ミニバッチの形にしたあと：', faceImgs.shape)

    # ネットワークと同じデバイス上にデータを送る
    x = predictor.xp.asarray(faceImgs)

    # モデルのforward関数に渡す
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y = predictor(faceImgs)

    # Variable形式で出てくるので中身を取り出す
    y = y.array
    print(y)
    # 結果をCPUに送る
    # y = to_cpu(y)

    # 予測確率の最大値のインデックスを見る
    pred_label = y.argmax(axis=1)

    print('ネットワークの予測:', pred_label[0])


if __name__ == '__main__':
    main()
