# !/bin/sh
cd tensorflow_utils
rm -rf tensorflow
mv /tmp/tensorflow .
cd tensorflow
bazel build '//tensorflow/lite:libtensorflowlite.so'
./tensorflow/lite/tools/make/download_dependencies.sh

cp bazel-bin/tensorflow/lite/libtensorflowlite.so ../tf_prebuild_linux-cpu-x86_64/
