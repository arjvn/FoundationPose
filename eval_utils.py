import numpy as np
import math


def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 10000*1e-6

def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    # return np.array([x, y, z])
    return np.rad2deg([x, y, z])


def trans_rot_error(pose_pred, pose_targets):
    gt_pose_rot = pose_targets[:3,:3]
    gt_pose_trans = pose_targets[:3,-1]
    pred_pose_rot = pose_pred[:3,:3]
    pred_pose_trans = pose_pred[:3,-1]
    try:
        b_rot_angle = rotationMatrixToEulerAngles(gt_pose_rot)
        reproject_b_rot_angle = rotationMatrixToEulerAngles(pred_pose_rot)
    except:
        print("rotation is not orthogonal")
    trans_error = np.linalg.norm(gt_pose_trans-pred_pose_trans)
    rot_error = np.absolute(b_rot_angle-reproject_b_rot_angle)
    return trans_error, rot_error