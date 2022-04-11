# cpp_tflite_inference

cpp tflite model inference that converted tf.keras age-gender-model model

# tflite inference structure for Cmake project
```zsh
├── CMakeLists.txt
├── Docker
│   ├── bashrc
│   ├── devel-cpu.Dockerfile
│   └── tflite-android.Dockerfile
├── Makefile
├── inference.cpp
├── resource
│   ├── 36_0_0_face.jpg
│   └── age_gender_model.tflite
├── run.sh
├── whole_setup.sh
├── tensorflow_utils
   ├── tensorflow <=== git clone --depth 1 https://github.com/tensorflow/tensorflow.git -b v2.3.0-rc2
   └── tf_prebuild_linux-cpu-x86_64
       └── libtensorflowlite.so (cp ../tensorflow/bazel-bin/tensorflow/lite/libtensorflowlite.so .)
```

# setup to run inference 

<b>build and run Docker image</b>
```zsh
$ make run1
$ make run2
$ docker run --rm -it -v $PWD:/mnt tflite-builder:latest bash
```

<b>setup structure and inference in Docker</b>
```zsh
# in Docker
cd mnt
# setup structure above
./whole_setup.sh

# run inference
./run.sh
```

# target image

![36_0_0_face](https://user-images.githubusercontent.com/48679574/141508876-732be631-5f3e-4689-ae5b-9faf85d9026b.jpg)

# cpp TFLite inference Latency [m/s]
```zsh
./inference

>>>
image size: 299 x 299

=== Run TFLite inference ===
pred age: 33  and pred gender: M
TFLite C++ Inference Latency: 0.031 ms
```

# python tflite inference (age-gender-model)

- [python inference](https://github.com/madara-tribe/py_tflite_inference)


