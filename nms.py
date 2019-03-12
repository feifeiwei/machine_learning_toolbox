# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 09:53:49 2018

@author: 60236
"""
import numpy as np

class NMS(object):
    '''
    input : array [[x,y,xmax,ymax,conf, label],
                  [x,y,xmax,ymax,conf, label],
                  [x,y,xmax,ymax,conf, label],
                  ....
                  ...
                   ]
    
    return: array [[x,y,xmax,ymax,conf, label],
                  [x,y,xmax,ymax,conf, label],
                  [x,y,xmax,ymax,conf, label],
                  ....
                  ...]
    '''
    def __init__(self, conf_thres=0.5, nms_thres=0.2):
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
    
    def nms(self, boxes, conf_thres=0.5, nms_thres=0.4):
        conf_mask = boxes[:,4]> self.conf_thres
        boxes = boxes[conf_mask]
        
        if not boxes.shape[0]:
            return 0
        
        unique_labels = np.unique(boxes[:,-1])
        res = []
        for c in unique_labels:
            c_boxes = boxes[boxes[:,-1]==c]
            c_boxes = c_boxes[np.argsort(-c_boxes[:,-2])]
            
            while c_boxes.shape[0]:
                res.append(c_boxes[0])
                
                if len(c_boxes)==1:
                    break
                ious = self.bbox_iou( res[-1].reshape(1,-1), c_boxes[1:])
                c_boxes = c_boxes[1:][ious<self.nms_thres]
            
        return np.array(res)

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        s = box2.shape[0]
      
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
            
        # get the corrdinates of the intersection rectangle
        
    
        # get the corrdinates of the intersection rectangle
        inter_rect_x1 =  np.max(np.concatenate((b1_x1.repeat(s),b2_x1)).reshape(-1,s), 0)
        inter_rect_y1 =  np.max(np.concatenate((b1_y1.repeat(s),b2_y1)).reshape(-1,s), 0)
        inter_rect_x2 =  np.min(np.concatenate((b1_x2.repeat(s),b2_x2)).reshape(-1,s), 0)
        inter_rect_y2 =  np.min(np.concatenate((b1_y2.repeat(s),b2_y2)).reshape(-1,s), 0)
            
        if s == 1:
            inter_rect_x1 = np.max(inter_rect_x1)
            inter_rect_y1 = np.max(inter_rect_y1)
            inter_rect_x2 = np.min(inter_rect_x2)
            inter_rect_y2 = np.min(inter_rect_y2)
    
        # Intersection area
        inter_area =    np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, float('inf')) * \
                        np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, float('inf'))
    
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
        return iou
    def __call__(self, boxes):
        return self.nms(boxes, self.conf_thres, self.nms_thres)

    

if __name__=="__main__":
    
    boxes = [[10,10,100,100,0.6, 0],
             [50,50,150,150,0.6, 0],
             [11,11,99,99,0.9,1],
             [11,11,99,99,0.7,1],
             [100,100,150,150,0.8, 0]]
    
    boxes = np.array(boxes)    
    
    nms = NMS()
    res = nms(boxes)
    print('\n boxes:',boxes)
    print('\nres:',res)
    
    
            
            

    
