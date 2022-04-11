# py_tflite_inference

tflite model inference and pruning

# versions
```zsh
python==3.7.0
tensorflow==2.3.0
keras==2.3.1
```

# target image

![36_0_0_face](https://user-images.githubusercontent.com/48679574/115800189-0def7f00-a415-11eb-9892-e9b2b9bf25f8.jpg)

# python tflite inference

## keras to tflite inference
```zsh
start calculation .....
tflite gender M and age 33
Inference Latency (milliseconds) is 0.16689300537109375 [ms]
used memory info
pmem(rss=2354262016, vms=34936909824, shared=413888512, text=3018752, lib=0, data=32560934912, dirty=0)
```
## pruned and quantize TFlite model inference 
```zsh
tflite gender M and age 35
Inference Latency (milliseconds) is 0.16045570373535156 [ms]
used memory info
pmem(rss=343609344, vms=3279093760, shared=185520128, text=3018752, lib=0, data=2039975936, dirty=0)
```


# pruning model accuracy and size (keras API)

## model accuracy
```zsh
keras base_model:           age mse: 0.03089246153831482, gender accuracy: 0.9829999828338623
keras model for pruning :   age mse: 0.03180079782009125, gender accuracy: 0.9829999828338623
Pruned and quantized TFLite age mse: 0.02427875685595266, gender accuracy: 0.9820792079207921
```

## model size
```zsh
baseline Keras model:             241477953.00 bytes
pruned Keras model:                25858023.00 bytes
pruned TFlite model:               25829461.00 bytes
pruned and quantize TFlite model:   6463904.00 bytes
```
