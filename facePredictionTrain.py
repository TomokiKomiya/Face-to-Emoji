# -*- coding: utf-8 -*-

from image2TrainAndTest import image2TrainAndTest

import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob

import pickle
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.serializers
from chainer.datasets import tuple_dataset
from chainer import Chain, Variable, optimizers
from chainer import training
from chainer.training import extensions
from chainer import iterators
import re
import os

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

def main():
    parse = argparse.ArgumentParser(description='face detection train')
    parse.add_argument('--epoch','-e',type=int, default=20,
                       help='Number of sweeps over the dataset to train')
    parse.add_argument('--gpu','-g',type=int, default=-1,
                       help='GPU ID(negative value indicates CPU')
    parse.add_argument('--path','-p', default='')

    args = parse.parse_args()

    pathsAndLabels = []
    label_i = 0
    data_list = glob.glob(args.path + "*")
    datatxt = open("test.txt","w")
    for datafinderName in data_list:
        pathsAndLabels.append(np.asarray([datafinderName+"/", label_i]))
        pattern = r".*/(.*)"
        matchOB = re.finditer(pattern, datafinderName)
        directoryname = ""
        if matchOB:
            for a in matchOB:
                directoryname += a.groups()[0]
        datatxt.write(directoryname + "," + str(label_i) + "\n")
        label_i = label_i + 1
    datatxt.close()

    train, valid = image2TrainAndTest(pathsAndLabels,channels=3)
    print('Training dataset size:', len(train))
    print('Validation dataset size:', len(valid))

    # ミニバッチ
    bathchsize = 16
    train_iter = iterators.SerialIterator(train, bathchsize)
    valid_iter = iterators.SerialIterator(valid, bathchsize, shuffle=False, repeat=False)

    # ネットワークを作成
    predictor = MLP()

    # L.Classifier でラップし、損失の計算などをモデルに含める
    net = L.Classifier(predictor)

    # 最適化手法を選択してオプティマイザを作成し、最適化対象のネットワークを持たせる
    optimizer = optimizers.MomentumSGD(lr=0.1).setup(net)

    # アップデータにイテレータとオプティマイザを渡す
    updater = training.StandardUpdater(train_iter, optimizer, device=-1) # device=-1でCPUでの計算実行を指定
    
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='result')
    
    trainer.extend(extensions.LogReport(trigger=(1, 'epoch'), log_name='log'))
    # trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.Evaluator(valid_iter, net, device=-1), name='val')
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))

    trainer.extend(extensions.ProgressBar())
    
    trainer.run()
    
    outputname = "output_" + str(len(pathsAndLabels))
    modelOutName = outputname + ".model"
    OptimOutName = outputname + ".state"
    
    # net.save('test.model')
    chainer.serializers.save_npz(modelOutName, net)
    chainer.serializers.save_npz(OptimOutName, optimizer)

if __name__ == '__main__':
    main()