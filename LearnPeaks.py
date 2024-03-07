# -*- coding: utf-8 -*-
"""
Implementation of msiPL (Abdelmoula et al): Identify an informative peak Peaks

    - This function should be called after training the model
    - Briefly: this is a backpropagated-based threshold analysis on the neural network weight hyper-parameter (see, Equation#4)
      This analysis is to identify m/z contributed strongly to the learned non-manifold (encoded structures).
"""
import numpy as np
from scipy.signal import argrelextrema


def LearnPeaks(All_mz, W_enc, std_spectra, latent_dim,Beta,meanSpec_Orig):
    W1 = W_enc[0] # Connected with the input layer
    W2 = W_enc[6] # z_mean Layer，inputlayer不算，没参数；
                #每层包含weight和bias，所以0和1同属一层，2,3同层
    for EncID in range(latent_dim):
        W2_EncFeat1 = W2[:,EncID]          
        Act_Neuron_W2 = np.argsort(-W2_EncFeat1) #Note: -ve is used to sort descending
        W2_EncFeat1[Act_Neuron_W2[0]]
        Neuron_W1 = W1[:,Act_Neuron_W2[0]]
        Weights_norm_W1 = std_spectra*Neuron_W1#size:7671*1
        ij =  np.argsort(Weights_norm_W1)[::-1]
        Weights_norm_W1 = np.sort(Weights_norm_W1)[::-1]
        Weights_norm_W1[0]
        
    # ======== Threshold Weights mean + Beta*std:
        T = np.mean(Weights_norm_W1) + Beta*np.std(Weights_norm_W1)
        PeakID = ij[np.argwhere(Weights_norm_W1 >= T)]; PeakID = PeakID[:,0] #Ranked indices
        #PeakID表示原始Weights_norm_w1中大于阈值的下标序列（按数值递减）
    #举例
    #    Weights_norm_W1=[2,3,14,8,15],ij=[4,2,3,1,0],更新后 weights_norm_w1=[15,14,8,3,2],假设T=5,PeakID=[ij[0],ij[1],ij[2]],即为PeakID=[4,2,3]
    # All_mz     
    # ======== Get union list of m/z from all encFetaures ========
        Enc_mz = [All_mz[i] for i in PeakID]
        if EncID==0:
            Learned_mzBins = []
            Common_PeakID = []
        #print('test: ',Enc_mz,Learned_mzBins)
        Learned_mzBins = list(set().union(Enc_mz , Learned_mzBins))
        Common_PeakID = list(set().union(PeakID , Common_PeakID))
        
        if EncID==latent_dim-1:
            Learned_mzBins = np.sort(Learned_mzBins) #大于阈值的那些mz数值
            Common_PeakID = np.sort(Common_PeakID) #大于阈值的mz数值下标
        
    LocalMax = np.squeeze(np.transpose(argrelextrema(meanSpec_Orig, np.greater)))
    #squeeze,降维，不加参数的话，宽度为1的维度都消失 
    #找均值谱强度极大值（局部），输出下标
    mz_LocalMax = [All_mz[i] for i in LocalMax] #均值谱强度极大值下标对应的mz值
    Nearest_Peakindx = [np.argmin(np.abs(mz_LocalMax[:] - Learned_mzBins[i])) for i in  range(len(Learned_mzBins))]
    #给Learned_mzBins中的每一个mz数值 在均值谱强度极大值对应mz数组中  找一个mz数值相近的，输出它们在mz_LocalMax中的下标
    Peak_Indx = np.unique(Nearest_Peakindx)
    Learned_mzPeaks = [mz_LocalMax[i] for i in Peak_Indx] #存储选出峰的mz值（这些mz对应的强度对latent_z影响大，且是均值谱的极大值）
    Learned_mzPeaks = np.asarray(Learned_mzPeaks)
    #本意是找既大于阈值，又是极值点那些值(Learned_mzPeaks)和下标(Peak_Index)
    Real_PeakIdx = [np.argmin(np.abs(All_mz[:] - Learned_mzPeaks[i])) for i in  range(len(Learned_mzPeaks))]
    #把选出的mz值对应到All_mz的下标

    return Learned_mzBins, Learned_mzPeaks, Common_PeakID,Real_PeakIdx 

