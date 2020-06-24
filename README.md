# Face to Emoji

## TrainingOriginalModel
学習コマンド
```
python facePredictionTrain.py -p ./images/ -g 0
```
予測コマンド
```
python facePrediction.py
```
学習させたい画像を`image`ディレクトリに格納.  
呼び出すには`-p`で指定する.  
例 
```
./images/  
　　├ smile/*  
　　├ angry/*  
　　├ sad/*  
　　└ joy/*  
```

既存のネットワークは下記の三種類である.  
デフォルトとして`GoogLeNet`が設定されている.  
`-n`で指定する.  
```
- GoogLeNet
- AlexLike
- MLP
```

GPUを使用する場合は`-g`を`0`に指定する

## ImageToObjectPrediction
RTC上で深層学習をもちいて画像を認識する.  
以下のコマンドで実行する.  
```
python ImageToObjectPrediction.py
```

## Requirement packages
### chainer
`pip install chainer`

### opencv
`pip install opencv-python`