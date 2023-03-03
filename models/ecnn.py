'''
@Author: PangAY
@Date: 2023-03-13 15:45
@Description: ECNN, use multi-channels to extract infos
@LastEditTime: 2023-03-3 16:10
'''
import gym
import numpy as np
from gym import spaces

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ECNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 64):
        """特征提取网络
        """
        super().__init__(observation_space, features_dim)
        net_shape = observation_space.shape # 每个 movement 的特征数量, 8 个 movement, 就是 (N, 8, K)
        # 这里 N 表示由 N 个 frame 堆叠而成

        self.view_conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(1, net_shape[-1]), padding=0), # N*8*K -> 128*8*1
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(8,1), padding=0), # 128*8*1 -> 256*1*1
            nn.ReLU(),
        )
        view_out_size = self._get_conv_out(net_shape)

        self.fc = nn.Sequential(
            nn.Linear(view_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim)
        )


    def _get_conv_out(self, shape):
        o = self.view_conv(torch.zeros(net_shape[0], *shape))#view
        return int(np.prod(o.size()))


    def forward(self, observations):
        batch_size = observations.size()[0] # (BatchSize, N, 8, K)
        observations = observations.reshape(-1,net_shape[0],net_shape[1:])
        #observations.unsqueeze_(1) # 从 (N, 8, K) --> (N, 1, 8, K) 第二维增加一个维度
        conv_out=torch.randn(batch_size,net_shape[0],256)# 输出维度
        temp_observatons=torch.randn(net_shape[0],batch_size,net_shape[1:])
        ### 将数据格式变为[N,n,8,k],N是特征数量，n是同时开启仿真的数量，8个movemet
        for i in range(0,obs_num):
            for j in range(0,batch_size):
                temp_observatons[i][j]=observations[j][i] #对obs_num 和 bathc_size 的维度进行转换
        observations=temp_observatons
        for i in range(0,obs_num):
            #print('temp_conv_out',temp_conv_out.size())
            observation = observations[i]
            observation = observation.reshape(-1, 1, net_shape[1:])
            temp_conv_out = self.view_conv(observation).view(batch_size, -1)
            for j in range(0,batch_size):
                conv_out[j][i]=temp_conv_out[j]
        conv_out=torch.reshape(conv_out,(batch_size,-1))
        return self.fc(conv_out)