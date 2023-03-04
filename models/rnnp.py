'''
@Author: PangAY
@Date: 2023-03-4 12:02:58
@Description: Use RNN to predict features and then use CNN
@LastEditTime: 2023-03-4 14:47:22
'''
import gym
import numpy as np

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class RNNP(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 64):
        """利用 RNN 提取信息
        """
        super().__init__(observation_space, features_dim)
        self.net_shape = observation_space.shape # 每个 movement 的特征数量, 8 个 movement, 就是 (N, 8, K)
        # 这里 N 表示由 N 个 frame 堆叠而成

        self.view_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, self.net_shape[-1]), padding=0), # N*1*8*K -> N*64*8*1
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(8,1), padding=0), # N*64*8*1 -> N*128*1*1 (BatchSize, N, 128, 1, 1)
            nn.ReLU(),
        ) # 每一个 junction matrix 提取的特征
        view_out_size = self._get_conv_out(self.net_shape)

        self.predict = nn.LSTM(
                input_size=self.net_shape[1]*self.net_shape[2],
                hidden_size=self.net_shape[1]*self.net_shape[2],
                num_layers=2,
                batch_first=True
            )

        self.fc = nn.Sequential(
            nn.Linear(view_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim)
        )
    def _get_conv_out(self, shape):
        o = self.view_conv(torch.zeros(1, 1, *shape[1:]))
        return int(np.prod(o.size()))


    def forward(self, observations):
        batch_size = observations.size()[0] # (BatchSize, N, 8, K)
        observations=observations.view(batch_size,self.net_shape[0],-1)#(BatchSize, N, 8*k)
        observations_Predict=self.predict(observations)[0]
        observations_Predict=observations_Predict[:,-1]# 取出最后预测的状态
        observations_Predict=observations_Predict.view(batch_size,1,self.net_shape[1],self.net_shape[2])#(BatchSize,8*k)->(BatchSize,1,8,k)
        conv_out = self.view_conv(observations_Predict).view(batch_size, -1) #
        return self.fc(conv_out)