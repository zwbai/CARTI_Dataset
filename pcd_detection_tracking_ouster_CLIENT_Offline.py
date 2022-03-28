# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import glob
from mmdet3d.apis import inference_detector, init_model, show_result_meshlab
import torch
import time
import numpy as np
from deep_sort import build_tracker
import cv2
from PreprocessCSV import rotatey
from ouster import client
from contextlib import closing
from more_itertools import nth
import numpy as np
import time
import cv2
import open3d as o3d
import matplotlib.pyplot as plt  # type: ignore
import pyproj
import warnings
warnings.filterwarnings('ignore')
import json
import socket
import datetime
from numpy.linalg import inv

def plot_xyz_points(hostname: str, lidar_port: int = 7502) -> None:
    """Display range from a single scan as 3D points
    Args:
        hostname: hostname of the sensor
        lidar_port: UDP port to listen on for lidar data
    """
    import matplotlib.pyplot as plt  # type: ignore

    # get single scan
    metadata, sample = client.Scans.sample(hostname, 1, lidar_port)
    scan = next(sample)[0]

    # set up figure
    plt.figure()
    ax = plt.axes(projection='3d')
    r = 3
    ax.set_xlim3d([-r, r])
    ax.set_ylim3d([-r, r])
    ax.set_zlim3d([-r, r])

    plt.title("3D Points from {}".format(hostname))

    # [doc-stag-plot-xyz-points]
    # transform data to 3d points and graph
    xyzlut = client.XYZLut(metadata)
    xyz = xyzlut(scan)

    [x, y, z] = [c.flatten() for c in np.dsplit(xyz, 3)]
    ax.scatter(x, y, z, c=z / max(z), s=0.2)
    # [doc-etag-plot-xyz-points]
    plt.show()

def rotatey(data_with_pitch):
    # p_zhensong = np.array([[0.8938, -0.0117, 0.4483, 0, 0],
    #              [-0.0117, 0.9987, 0.0493, 0, 0],
    #              [-0.4483, -0.0493, 0.8925, 0, 0],
    #              [0, 0, 0, 1, 0],
    #              [0, 0, 0, 0, 1]])
    # z_zhensong = -4.916

    # p_saswat = np.array([[0.161657854141595, -0.226681987904914, 0.960459272719990, 0],
    #             [-0.612475536820247, 0.740083946643111, 0.277757931872174, 0],
    #             [-0.773783209297724, -0.633159559890423, -0.0191967895305751, 0],
    #             [-2431842.97910488, -4703515.94883458, 3544377.74979997, 1]])

    # Based on Zhensong's calibration matrix for 9_23_Iowa_univ data
    p = np.array([[0.995560055160684, -0.00223877651973211, 0.0941018833400790, 0, 0],
                    [-0.00223877651973211, 0.998871130050779, 0.0474494829347500, 0, 0],
                    [-0.0941018833400790, - 0.0474494829347500, 0.994431185211463, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1]])
    z = -2.74

    lidarData = np.dot(p, data_with_pitch.T)
    lidarData = lidarData.T
    lidarData[:, 2] = lidarData[:, 2] - z
    return lidarData


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def main():
    record_time = time.time()
    parser = ArgumentParser()
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo/9_23_iowa_univ', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()

    path = './deep_sort/deep/checkpoint/ckpt.t7'
    # Initialize deepsort
    '''
    max_dist=0.2,
    min_confidence=0.3, 
    nms_max_overlap=0.5,
    max_iou_distance=0.7, 
    max_age=70, 
    n_init=3,
    nn_budget=100,
    use_cuda=True
    '''
    deepsort = build_tracker(path,
                             0.2,
                             0.3,
                             0.7,
                             70,
                             30,
                             3,
                             100,
                             True)

    # -----------------------1: pointpillars: --------------------------------------
    config = '../configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py'
    checkpoint = '../checkpoints/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405-2fa62f3d.pth'


    # build the model from a config file and a checkpoint file
    model = init_model(config, checkpoint, device=args.device)
    # test a single image
    FILE_NAME = '../../DATASET/_out_1LiDAR_roadside'

    TestData = sorted(glob.glob(FILE_NAME+'/*.bin'))
    ori_img = cv2.imread('./demo/pcap_out_000001/pcap_out_000001_online.png')

    from mmdet3d.core.visualizer.open3d_vis import Visualizer



    for i in range(len(TestData)):
        frame = TestData[i].split('.')[-1]
        print('frame: ', frame)

        t1 = time.time()
        t = time.time()

    # for i in range(len(TestData)):
        t2 = time.time()
        # pcd = "./20220302_ousterdata/"+str(t)+".bin"
        # pcd = 'data/ouster_Iowa_uv_rotated/pcap_out_000005.bin'
        #

        result, data = inference_detector(model, TestData[i])
        # print('results: {}'.format(result))
        # print('data:{}'.format(data))

        # model.show_results(data, result, out_dir='results')
        # show the results
        arg_show = False
        # if i % 5 == 0:
        #     arg_show = True
        if 'pts_bbox' in result[0].keys():
            pred_bboxes = result[0]['pts_bbox']['boxes_3d'].tensor.numpy()
            pred_scores = result[0]['pts_bbox']['scores_3d'].numpy()
        else:
            pred_bboxes = result[0]['boxes_3d'].tensor.numpy()
            pred_scores = result[0]['scores_3d'].numpy()

        # filter out low score bboxes for visualization
        if args.score_thr > 0:
            inds = pred_scores > args.score_thr
            pred_bboxes = pred_bboxes[inds]

        pred_bboxes = 100*pred_bboxes[:, 0:7]
        # print(pred_bboxes)
        t3 = time.time()
        
        pred_bboxes = np.concatenate((pred_bboxes, np.zeros([pred_bboxes.shape[0], 1], 'float32')), axis=1)
        # print("results.len {}".format(len(result)))
        # print("result['pts_bbox'].shape {}".format(result[0]['pts_bbox'].shape))
        # print("pred_bboxes {}".format(pred_bboxes))
        bboxes_tracking = np.concatenate((pred_bboxes[:, 0:2], pred_bboxes[:, 3:5]), axis=1)
        # print('bboxes_tracking {}'.format(bboxes_tracking))
        conf_tracking = np.ones([bboxes_tracking.shape[0], 1], 'int32')
        # [x, y, z, x_size, y_size, z_size, yaw]
        # print('out_dir {}, file_name {}'.format(out_dir, file_name))

        if bboxes_tracking.size != 0:
            # bboxes_tracking[:, 2:] *= 1.05
            outputs = deepsort.update(bboxes_tracking, conf_tracking, ori_img)
            # print("outputs {}".format(outputs))

        t4 = time.time()
        cmm_det_tra_data = []
        for i_pred in range(pred_bboxes.shape[0]):
            for i_tracking in range(len(outputs)):
                if np.abs(pred_bboxes[i_pred, 0] - outputs[i_tracking][0]) + np.abs(pred_bboxes[i_pred, 1] - outputs[i_tracking][1]) < 100:
                    pred_bboxes[i_pred, -1] = outputs[i_tracking][-1]
                    continue

            x = -1*pred_bboxes[i_pred, 0]/100
            y = -1*pred_bboxes[i_pred, 1]/100
            # z = (pred_bboxes[i_pred, 2]-274 - pred_bboxes[i_pred, 5]/2)/100
            z = (pred_bboxes[i_pred, 2] - 274)/100
            point = np.array([x, y, z, 1])

            t5 = time.time()
            cmm_det_tra_data.append({'frame': frame,'id': int(pred_bboxes[i_pred, -1]), 'x':x,
                                        'y': y, 'z': z,
                                        'x_size': round(pred_bboxes[i_pred, 3], 2), 'y_size': round(pred_bboxes[i_pred, 4], 2),
                                        'z_size': round(pred_bboxes[i_pred, 5], 2), 'yaw': round(pred_bboxes[i_pred, 6], 2), 
                                        'time (ms)': (t5 - t1) * 1000, 'FPS': 1 / (t5 - t1)})

        # print("pred_bbox with id {}".format(pred_bboxes))
        print("cmm_det_tra_data {}".format(cmm_det_tra_data))
        # t2 = time_synchronized()
        
        send_data = (str(cmm_det_tra_data)+"\n")#json.dumps
        # s.send(send_data.encode('utf-8'))#
        
        f = open('./logs/logdata-CARTI-'+str(record_time)+'.txt', "a")
        f.write(send_data)
        f.close()
        # print(t2)
        print('\tDone testing the {}th sample, time: {:.1f}ms, speed {:.2f}FPS'.format(i, (t5 - t1) * 1000,
                                                                                        1 / (t5 - t1)))


if __name__ == '__main__':
    main()
