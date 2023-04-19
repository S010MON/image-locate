# Running on GPU

## Jupyter:
```
docker run -it --gpus all --rm -v $(realpath ~/git/picture-locate):/tf/notebooks -p 8888:8888 tensorflow/tensorflow:latest-gpu-jupyter
```

## Python3
```
docker run -it --gpus all --rm -v $(realpath ~/git/picture-locate):/tf/notebooks tensorflow/tensorflow:latest-gpu-py3 bash
```
