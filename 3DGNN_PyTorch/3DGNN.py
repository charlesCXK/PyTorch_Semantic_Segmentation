import os
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
3DGNN
Input:
    (N, C, H, W) encoded feature and (N, 3, H, W) x-y-z position
Output:
    (N, 2*C, H, W) feature
'''
class ThreeDGNNModule(nn.Module):
    def __init__(self, gnniteration=3, k=64, in_channel=256, MLPnum=1, gpu=True):
        super().__init__()
        self.gpu = gpu
        self.iteration_num = gnniteration       # number of iterations
        self.k = k      # k neighbours
        self.relu = nn.ReLU()

        self.neighborMLP = nn.ModuleList([nn.Linear(in_channel, in_channel) for l in range(MLPnum)])
        self.RNNUpdate = nn.Linear(in_channel * 2, in_channel)

    '''
    Given two arguments:
        batch_mat: (N, H*W, 3)
            N is batch_size,
            H*W is the size of the picture,
            3 represents x-y-z positions.
        k is the number of its neighbours.
    Return value:
        batch_indices: (N, H*W, k)
        for each point in each batch, return the index of top k nearest neighbours.
    
    Tips: the distance between two points is: 
    sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2) 
    = sqrt(x1^2 + x2^2 + y1^2 + y2^2 + Z1^2 + Z2^2 - 2X1X2 - 2Y1Y2 - 2Z1Z2)
    So use matrix will accelerate it.
    '''
    def get_knn_indices(self, batch_mat, k):
        # bmm:  batch matrix multiply
        # (N, H*W, 3) and (N, 3, H*W)
        r = torch.bmm(batch_mat, batch_mat.permute(0, 2, 1))
        N = r.size()[0]
        HW = r.size()[1]

        if self.gpu:
            batch_indices = torch.zeros((N, HW, k)).cuda()
        else:
            batch_indices = torch.zeros((N, HW, k))

        for idx, val in enumerate(r):
            '''
            val[i, j] = matrix[i, :] dot matrix'[:, j] = matrix[i, :] dot matrix[j, :]
            In fact, the result is XiXj + YiYj + ZiZj
            '''
            diag = val.diag().unsqueeze(0)  # diag[i, i] is Xi*Xi+Yi*Yi+Zi*Zi
            diag = diag.expand_as(val)

            # compute the distance matrix
            D = (diag + diag.t() - 2 * val).sqrt()
            topk, indices = torch.topk(D, k=k, largest=False)
            batch_indices[idx] = indices.data
        return batch_indices

    '''
    cnn_feature: (N, C, H, W)
    points: (N, 3, H, W)
    '''
    def forward(self, cnn_feature, points):
        N, C, H, W = cnn_feature.shape
        points = points.view(N, 3, H*W).permute(0, 2, 1).contiguous()  # (N, H*W, 3)

        '''
        ******************* 
        Graph Construction 
        *******************
        '''
        # get k nearest neighbors
        kneighbours = self.get_knn_indices(points, k=self.k)    # (N, H*W, K)
        kneighbours = kneighbours.view(N, H*W*self.k).long()    # (N, H*W*K)

        '''
        ******************* 
        Propagation Model
        *******************
        '''
        h = cnn_feature.clone()     # feature encoded from 2D images

        h = h.permute(0, 2, 3, 1).contiguous().view(N, (H*W), C)      # (N, C, H, W) -> (N, H, W, C) -> (N, H*W, C)
        message = h.clone()

        for iter in range(self.iteration_num):
            for batch_index in range(N):
                neighbor_features = torch.index_select(h[batch_index], 0, kneighbours[batch_index])    # (H*W*K, C), feature of its neighbours
                neighbor_features = neighbor_features.view(H*W, self.k, C)      # (H*W, K, C)

                # perform a MLP on each neighbour's feature
                # 'g is a multi-layer perceptron, (MLP). Unless otherwise speciﬁed, all instances of MLP that we employ have one layer with ReLU [19] as the nonlinearity'
                for idx, layer in enumerate(self.neighborMLP):
                    neighbor_features = self.relu(layer(neighbor_features))  # (H*W, K, C)

                message[batch_index] = torch.mean(neighbor_features, dim=1)     # (H*W, C), sum neighbors' features

            ''' Vanilla RNN Update '''
            next_h = torch.cat((h, message), 2)     # (N, H*W, C) -> (N, H*W, 2C)
            h = self.relu(self.RNNUpdate(next_h))

        # Reshape the tensor, making it have the same size with that of the original input CNN feature.
        h = h.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()     # h: (N, C, H, W)
        # concatenate h and the original CNN feature
        output_concatenate = torch.cat((cnn_feature, h), 1)     # output: (N, 2C, H, W)

        return output_concatenate

if __name__ == '__main__':
    CNN_channel = 256        # channel num of CNN feature
    H, W = 64, 64     # size of feature map
    useGPU = False       # use GPU or CPU

    # define the net
    net = ThreeDGNNModule(gnniteration=3, k=64, in_channel=CNN_channel, MLPnum=1, gpu=useGPU)
    cnn_feature = torch.randn(2, CNN_channel, H, W)
    points = torch.randn(2, 3, H, W)        # point cloud data

    if useGPU:
        net, cnn_feature, points = net.cuda(), cnn_feature.cuda(), points.cuda()

    out_feature = net(cnn_feature, points)
    print(out_feature.shape)