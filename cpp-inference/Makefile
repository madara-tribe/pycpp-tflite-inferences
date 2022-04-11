run1:
	docker build -t tensorflow/tensorflow:devel -f Docker/devel-cpu.Dockerfile Docker	
run2:
	docker build -t tflite-builder -f Docker/tflite-android.Dockerfile Docker
in:
	docker run --rm -it -v $PWD:/mnt tflite-builder:latest bash
