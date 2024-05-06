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
    # trans_error = np.linalg.norm(gt_pose_trans-pred_pose_trans,axis=0)
    trans_error = np.absolute(np.array(gt_pose_trans) - np.array(pred_pose_trans))
    rot_error = np.absolute(b_rot_angle-reproject_b_rot_angle)
    return trans_error, rot_error

def plot_error(array1,array2):
    import matplotlib.pyplot as plt
    import numpy as np

    # Create subplots
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    array1 = 100 *array1

    axs[0, 0].plot(array1[:, 0], label='Trans Err(cm)')
    axs[0, 0].set_title('Trans Err on X-axis')
    axs[1,0].plot(array1[:, 1], label='Trans Err(cm)')
    axs[1,0].set_title('Trans Err on Y-axis')
    axs[2, 0].plot(array1[:, 2], label='Trans Err(cm)')
    axs[2, 0].set_title('Trans Err on Z-axis')

    axs[0, 1].plot(array2[:, 0], label='Rot Err(degree)')
    axs[0, 1].set_title('Rot Err on X-axis')
    axs[1, 1].plot(array2[:, 1], label='Rot Err(degree)')
    axs[1, 1].set_title('Rot Err on Y-axis')
    axs[2, 1].plot(array2[:, 2], label='Rot Err(degree)')
    axs[2, 1].set_title('Rot Err on Z-axis')

    # Add legend
    axs[0, 0].legend()
    axs[0, 1].legend()

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()
