# -*- coding: utf-8 -*-

from image2TrainAndTest import image2TrainAndTest
import network

import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import sys

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

def main():
    parse = argparse.ArgumentParser(description='face detection train')
    parse.add_argument('--epoch','-e',type=int, default=20,
                       help='Number of sweeps over the dataset to train')
    parse.add_argument('--gpu','-g',type=int, default=-1,
                       help='GPU ID(negative value indicates CPU')
    parse.add_argument('--path','-p', default='')
    parse.add_argument('--network','-n',type=str, default="GoogLeNet",
                       help='choise network')

    args = parse.parse_args()

    pathsAndLabels = []
    label_i = 0
    data_list = glob.glob(args.path + "*")
    datatxt = open("face.txt","w")
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
    print(type(args.network))
    if args.network == "GoogLeNet":
        print("使用するネットワーク : {}".format(args.network))
        predictor = network.GoogLeNet(n_out=len(pathsAndLabels))
    elif args.network == "MLP":
        print("使用するネットワーク : {}".format(args.network))
        predictor = network.MLP(n_out=len(pathsAndLabels))
    else:
        print("***** Please choise network !!! *****")
        sys.exit()

    # GPUかどうか
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # L.Classifier でラップし、損失の計算などをモデルに含める
    net = L.Classifier(predictor)

    # 最適化手法を選択してオプティマイザを作成し、最適化対象のネットワークを持たせる
    optimizer = optimizers.MomentumSGD(lr=0.1).setup(net)

    # アップデータにイテレータとオプティマイザを渡す
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu) # device=-1でCPUでの計算実行を指定

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='result_face_10')

    trainer.extend(extensions.LogReport(trigger=(1, 'epoch'), log_name='log'))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.Evaluator(valid_iter, net, device=args.gpu), name='val')
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))

    trainer.extend(extensions.ProgressBar())

    trainer.run()

    outputname = args.network + "_output_10_" + str(len(pathsAndLabels))
    modelOutName = outputname + ".model"
    OptimOutName = outputname + ".state"

    # net.save('test.model')
    chainer.serializers.save_npz(modelOutName, net)
    chainer.serializers.save_npz(OptimOutName, optimizer)

if __name__ == '__main__':
    main()