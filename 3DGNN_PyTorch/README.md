#  3D Graph Neural Networks for RGBD Semantic Segmentation(ICCV 2017)

###  I. Architecture

<img src='net.png'>

### II. Usage

```shell
$ python3 3DGNN.py
```

- The input for the **3DGNN model** is defined in the code. You can understand it through the simulation data.
- How to encode feature from the 2D image and how to do prediction is up to you. There are too many methods to encode the feature such as DeepLab V1, V2, V3, V3+, RefineNet, and etc.

### III. Others

- Paper: [3D Graph Neural Networks for RGBD Semantic Segmentation](http://www.cs.toronto.edu/~rjliao/papers/iccv_2017_3DGNN.pdf)


- [Author's implementation (Python + Caffee)](https://github.com/xjqicuhk/3DGNN)