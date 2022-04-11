# python and c++ tflite inference

# source model

source model is 「age-gender-model that is tensorflow model .

this time, it used as tflite after convert from ```h5``` or ```pb``` model 


<img src="https://user-images.githubusercontent.com/48679574/162663355-8d294318-4d79-4783-b22b-1fb7ed538b8a.png" width="800px">


# python inference latency

## latency
```
# tf-keras
Inference Latency (milliseconds) is 0.16689300537109375 [ms]

# pruning TFlite model
Inference Latency (milliseconds) is 0.16045570373535156 [ms]
```

## how much memory used
```zsh
baseline Keras model:             241477953.00 bytes
pruned Keras model:                25858023.00 bytes
pruned TFlite model:               25829461.00 bytes
pruned and quantize TFlite model:   6463904.00 bytes
```

# c++ inference latency
```zsh
>>>
image size: 299 x 299

=== Run TFLite inference ===
pred age: 33  and pred gender: M
TFLite C++ Inference Latency: 0.031 ms
```
