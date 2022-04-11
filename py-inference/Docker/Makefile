IMAGE=tflite-py
run:
	docker build -t ${IMAGE} .
in:
	NV_GPU=1 nvidia-docker run --runtime=nvidia -v /home/hagi:/home/app/sig --rm -it ${IMAGE}:latest bash

in2:
	 NV_GPU=2 nvidia-docker run --runtime=nvidia -v /home/hagi:/home/app/sig --rm -it ${IMAGE}:latest bash
