import numpy as np
from PIL import Image

def R_distance(R1,R2):
    dist = np.arccos(0.5*(np.trace(R1.dot(R2.T))-1))
    return dist

def grasp_nms(pos,R,joint,score,tax):
    order = np.argsort(score)
    pruned_pos, pruned_R, pruned_joint, pruned_tax=[],[],[],[]
    while order.size > 0:
        index = order[-1]
        cur_pos= pos[index]
        cur_R = R[index]
        cur_joint = joint[index]
        cur_tax = tax[index]
        pruned_pos.append(cur_pos)
        pruned_R.append(cur_R)
        pruned_joint.append(cur_joint)
        pruned_tax.append(cur_tax)
        dist = [np.linalg.norm(cur_pos-pos[idx]) for idx in order[:-1]]
        dist = np.asarray(dist)
        angle = [R_distance(cur_R,R[idx]) for idx in order[:-1]]
        angle = np.asarray(angle)
        left1 = np.where(dist > 0.03)[0]
        left2 = np.where(angle >30/180.*np.pi)[0]
        left = np.intersect1d(left1,left2)
        order = order[left]

    return np.asarray(pruned_pos),np.asarray(pruned_R),np.asarray(pruned_joint),np.asarray(pruned_tax)
