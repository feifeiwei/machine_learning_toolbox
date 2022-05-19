#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def batch_softmax(x):
    
    return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(3,1)



def cross_entropy_loss(logits, labels):
    '''
    logits:[B，N]
    labels:[B，N]
    
    '''
    logits = batch_softmax(logits)
    
    loss = -labels * np.log(logits)
    
    return loss.sum() / logits.shape[0]



if __name__=="__main__":
    
    x = np.random.randn(3,4)
    y = np.zeros_like(x)
    
    y[0][0] = 1
    y[1][1] = 1
    y[2][2] = 1
    loss = cross_entropy_loss(x, y)
    print(loss)
