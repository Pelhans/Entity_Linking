sudo docker run --runtime=nvidia -it --rm -e CUDA_VISIBLE_DEVICES=0 -p 6009:6006 -v `pwd`:`pwd` 192.168.31.103:5000/gpu_cn:cuda9-v3-tensorboardx
