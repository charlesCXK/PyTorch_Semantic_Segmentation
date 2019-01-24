#  PointNet (CVPR 2017)

###  I. Architecture

<img src='net.png'>

### II. Usage

```python
$ python3 pointnet.py
```

The input for both ***PointNetClassify*** and ***PointNetSegmentation*** is a teansor with a shape of ***(32,3,2500)***. **32** means batch size = 32,  **3** means 3-d feature(x-y-z coordinate), **2500** means 2500 points per batch. The output size is shown in the code.

### III. Others

- Paper: [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)


- [Project page](http://stanford.edu/~rqi/pointnet/)
- [Author's implementation](https://github.com/charlesq34/pointnet)