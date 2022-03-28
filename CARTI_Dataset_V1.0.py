#!/usr/bin/env python

# Author: Zhengwei Bai
# WebPage: zwbai@github.io
#



"""
Version Description of V1.0
Finished Date: 03/16/2022, 20:00, at UC, Riverside
Author: Zhengwei Bai

Summary:
"""

import glob
import os
import sys
from queue import Queue
from queue import Empty

import argparse
import time
from datetime import datetime
import math
import random
import numpy as np
from matplotlib import cm
import open3d as o3d
from carla_data_descriptor import CarlaDataDescriptor
import CMM_CARLA_Config as CFG
from math import pi
import logging
from numpy.linalg import pinv, inv

from CMM_CARLA_Config import *

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

""" OUTPUT FOLDER GENERATION """
PHASE = "training"
OUTPUT_FOLDER = os.path.join("../../DATASET/_out_1LiDAR_roadside", PHASE)
folders = ['calib', 'image_2', 'label_2', 'velodyne', 'ImageSets']


def maybe_create_dir(path):
    if not os.path.exists(directory):
        os.makedirs(directory)


for folder in folders:
    directory = os.path.join(OUTPUT_FOLDER, folder)
    maybe_create_dir(directory)

""" DATA SAVE PATHS """
INDEX_PATH = os.path.join(OUTPUT_FOLDER, 'ImageSets')
LIDAR01_PATH = os.path.join(OUTPUT_FOLDER, 'velodyne/{0:06}.bin')
LIDAR01_PATH_PLY = os.path.join(OUTPUT_FOLDER, 'velodyne/{0:06}.ply')
LABEL01_PATH = os.path.join(OUTPUT_FOLDER, 'label_2/{0:06}.txt')
IMAGE01_PATH = os.path.join(OUTPUT_FOLDER, 'image_2/{0:06}.png')
CALIBRATION01_PATH = os.path.join(OUTPUT_FOLDER, 'calib/{0:06}.txt')


CP_DISTANCE = 51.2


def isInRoadsideRange(location):

    locationX = location.x
    locationY = location.y
    minX = CFG.Lidar['LocationX'] - 51.2
    maxX = CFG.Lidar['LocationX'] + 51.2
    minY = CFG.Lidar['LocationY'] - 51.2
    maxY = CFG.Lidar['LocationY'] + 51.2

    if locationX >= minX and locationX<= maxX and locationY >= minY and locationY <= maxY:
        inRangeFlag = True
    else:
        inRangeFlag = False

    return inRangeFlag

def isInRange(location, sensor_location):

    locationX = location.x
    locationY = location.y
    minX = sensor_location.x - 51.2
    maxX = sensor_location.x + 51.2
    minY = sensor_location.y - 51.2
    maxY = sensor_location.y + 51.2

    if locationX >= minX and locationX<= maxX and locationY >= minY and locationY <= maxY:
        inRangeFlag = True
    else:
        inRangeFlag = False

    return inRangeFlag


def dis2sensor(actor_location):
    dis = math.sqrt(np.square(actor_location.x - CFG.Lidar['LocationX']) + np.square(actor_location.y - CFG.Lidar['LocationY']))

    return round(dis, 2)


def creat_kitti_datapoint(actor, sensor, actor_type):
    if actor:
        datapoint = CarlaDataDescriptor()
        bbox_2d = [0, 0, 0, 0] # no camera so far

        rotation_y = get_relative_rotation_y(actor.get_transform().rotation.yaw, sensor.rotation.yaw)
        # id = actor.id
        datapoint.set_bbox(bbox_2d)
        datapoint.set_3d_object_dimensions(actor.bounding_box.extent)
        datapoint.set_type(actor_type)
        datapoint.set_3d_object_location(actor, sensor, actor_type)
        datapoint.set_rotation_y(rotation_y)
        datapoint.set_id(actor.id)
        # print(actor)
        # print(datapoint)
        return datapoint
    else:
        return None


def get_relative_rotation_y(actor_yaw, sensor_yaw):
    """ Returns the relative rotation of the agent to the camera in yaw
    The relative rotation is the difference between the camera rotation (on car) and the agent rotation"""

    rot_actor = degrees_to_radians(actor_yaw) # -180 ~ 180 wst clockwise
    # rot_actor = -1 * rot_actor # 180 ~ -180 wst clockwise
    rot_sensor = degrees_to_radians(sensor_yaw)
    rot_y_lidar = rot_actor - rot_sensor # rt wrt lidar coordinate
    rot_y_camera = rot_y_lidar - 0.5*pi
    if rot_y_camera < -1.0*pi:
        rot_y_camera = rot_y_camera + 2.0 * pi
    if rot_y_camera > 1.0*pi:
        rot_y_camera -= 2.0*pi
    # print('vehicle rotation :{}'.format(rot_y_camera))
    # the difference of the x-axis direction between carla and kitti
    kitti_ry = round(rot_y_camera, 2)

    return kitti_ry


def degrees_to_radians(degrees):
    return np.round(degrees * math.pi / 180.0, 2)


def save_kitti_label_data(filename, datapoints):
    with open(filename, 'w') as f:
        out_str = "\n".join([str(point) for point in datapoints if point])
        f.write(out_str)
    # logging.info("Wrote kitti label data to %s", filename)



def proj_to_camera(pos_vector):
    # transform the points to camera
    TR_velodyne = np.array([[0, -1, 0],
                            [0, 0, -1],
                            [1, 0, 0]])
    # Add translation vector from velo to camera. This is 0 because the position of camera and lidar is equal in our configuration.

    transformed_3d_pos = np.dot(TR_velodyne, pos_vector)
    return transformed_3d_pos


def save_lidar_data(filename, lidar_measurement, format="bin"):
    """ Saves lidar data to given filename, according to the lidar data format.
        bin is used for KITTI-data format, while .ply is the regular point cloud format
        In Unreal, the coordinate system of the engine is defined as, which is the same as the lidar points
        z
        ^   ^ x
        |  /
        | /
        |/____> y
        This is a left-handed coordinate system, with x being forward, y to the right and z up
        See also https://github.com/carla-simulator/carla/issues/498
        However, the lidar coordinate system from KITTI is defined as
              z
              ^   ^ x
              |  /
              | /
        y<____|/
        Which is a right handed coordinate sylstem
        Therefore, we need to flip the y axis of the lidar in order to get the correct lidar format for kitti.

        This corresponds to the following changes from Carla to Kitti
            Carla: X   Y   Z
            KITTI: X  -Y   Z
        NOTE: We do not flip the coordinate system when saving to .ply.
    """
    # logging.info("Wrote lidar data to %s", filename)

    if format == "bin":
        data = np.copy(np.frombuffer(lidar_measurement.raw_data, dtype=np.float32))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))

        data[:, 1] = -1 * data[:, 1]
        lidar_array = np.array(data).astype(np.float32)
        # logging.debug("Lidar min/max of x: {} {}".format(
        #               lidar_array[:, 0].min(), lidar_array[:, 0].max()))
        # logging.debug("Lidar min/max of y: {} {}".format(
        #               lidar_array[:, 1].min(), lidar_array[:, 0].max()))
        # logging.debug("Lidar min/max of z: {} {}".format(
        #               lidar_array[:, 2].min(), lidar_array[:, 0].max()))
        lidar_array.tofile(filename)
    else:
        lidar_measurement.save_to_disk(filename)


def save_calibration_matrices(sensor_data, filename):
    """ Saves the calibration matrices to a file.
        AVOD (and KITTI) refers to P as P=K*[R;t], so we will just store P.
        The resulting file will contain:
        3x4    p0-p3      Camera P matrix. Contains extrinsic
                          and intrinsic parameters. (P=K*[R;t])
        3x3    r0_rect    Rectification matrix, required to transform points
                          from velodyne to camera coordinate frame.
        3x4    tr_velodyne_to_cam    Used to transform from velodyne to cam
                                     coordinate frame according to:
                                     Point_Camera = P_cam * R0_rect *
                                                    Tr_velo_to_cam *
                                                    Point_Velodyne.
        3x4    tr_imu_to_velo        Used to transform from imu to velodyne coordinate frame. This is not needed since we do not export
                                     imu data.
    """
    """
        in this carla dataset the calibration matrices are set as follows
        3x4 P0: 1 0 0 0 0 1 0 0 0 0 1 0
        3x4 P1: 1 0 0 0 0 1 0 0 0 0 1 0
        3x4 P2: 1 0 0 0 0 1 0 0 0 0 1 0
        3x4 P3: 1 0 0 0 0 1 0 0 0 0 1 0
        3x3 R0_rect: 1 0 0 0 1 0 0 0 1
        3x4 Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0
        3x4 Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0
    """
    # KITTI format demands that we flatten in row-major order
    ravel_mode = 'C'
    P0 = np.identity(3)
    P0 = np.column_stack((P0, np.array([0, 0, 0])))
    P0 = np.ravel(P0, order=ravel_mode)
    R0 = np.identity(3)
    TR_velodyne = np.array([[0, -1, 0],
                            [0, 0, -1],
                            [1, 0, 0]])
    # Add translation vector from velo to camera. This is 0 because the position of camera and lidar is equal in our configuration.
    TR_velodyne = np.column_stack((TR_velodyne, np.array([0, 0, 0])))
    TR_imu_to_velo = np.identity(3)
    TR_imu_to_velo = np.column_stack((TR_imu_to_velo, np.array([0, 0, 0])))

    """
    A new calibration matrix for cooperative perception:
    sensor location and pose (SLaP)
    [x, y, z, pitch, yaw, roll]
    """
    SLaP = np.array([sensor_data.transform.location.x, sensor_data.transform.location.y, sensor_data.transform.location.z, 
                    sensor_data.transform.rotation.pitch, sensor_data.transform.rotation.yaw, sensor_data.transform.rotation.roll])


    def write_flat(f, name, arr):
        f.write("{}: {}\n".format(name, ' '.join(
            map(str, arr.flatten(ravel_mode).squeeze()))))



    # All matrices are written on a line with spacing
    with open(filename, 'w') as f:
        for i in range(4):  # Avod expects all 4 P-matrices even though we only use the first
            write_flat(f, "P" + str(i), P0)
        write_flat(f, "R0_rect", R0)
        write_flat(f, "Tr_velo_to_cam", TR_velodyne)
        write_flat(f, "TR_imu_to_velo", TR_imu_to_velo)
        write_flat(f, "SLaP", SLaP)


    # logging.info("Wrote all calibration matrices to %s", filename)


def save_index_data(OUTPUT_FOLDER, id):
    """ Appends the id of the given record to the files """
    for name in ['train.txt', 'val.txt', 'trainval.txt']:
        path = os.path.join(OUTPUT_FOLDER, name)
        with open(path, 'a') as f:
            f.write("{0:06}".format(id) + '\n')
        # logging.info("Wrote reference files to %s", path)

# Sensor callback.
# This is where you receive the sensor data and
# process it as you liked and the important part is that,
# at the end, it should include an element into the sensor queue.
def lidar_sensor_callback(sensor_data, world, sensor_queue, sensor_name, vehicle):
    # Do stuff with the sensor_data data like save it to disk
    # Then you just need to add to the queue
    frame = sensor_data.frame
    sensor_location = vehicle.get_location()
    dis2Roadside = dis2sensor(sensor_location)

    if frame%5 == 0 and dis2Roadside <= CP_DISTANCE:
        actor_list = world.get_actors()
        vehicle_list = actor_list.filter('vehicle.*')
        pedestrians_list = actor_list.filter('walker.pedestrian.*')
        # print('vehicle_list length: ', len(actor_list))
        lidar_list = actor_list.filter('sensor.lidar.ray_cast')
        # print('lidar_list: ', lidar.get_location())
        label_data = []
        for actor in vehicle_list:
            if isInRange(actor.get_location(), sensor_location):
                kitti_datapoint = creat_kitti_datapoint(actor, sensor_data.transform, 'Car')
                if kitti_datapoint:
                    label_data.append(kitti_datapoint)

        for actor in pedestrians_list:
            if isInRange(actor.get_location(), sensor_location):
                kitti_datapoint = creat_kitti_datapoint(actor, sensor_data.transform, 'Pedestrian')
                if kitti_datapoint:
                    label_data.append(kitti_datapoint)

        # print('label_data', label_data)
        # print('frame: ', frame)

        if sensor_name == "lidar01":
            label_filename = LABEL01_PATH.format(frame)
            calib_filename = CALIBRATION01_PATH.format(frame)
            if LIDAR_DATA_FORMAT == "bin":
                lidar_filename = LIDAR01_PATH.format(frame)
            else:
                lidar_filename = LIDAR01_PATH_PLY.format(frame)

        if sensor_name == "lidar02":
            label_filename = LABEL02_PATH.format(frame)
            calib_filename = CALIBRATION02_PATH.format(frame)
            if LIDAR_DATA_FORMAT == "bin":
                lidar_filename = LIDAR02_PATH.format(frame)
            else:
                lidar_filename = LIDAR02_PATH_PLY.format(frame)

        # print('label_filename', )
        save_kitti_label_data(label_filename, label_data)
        save_calibration_matrices(sensor_data, calib_filename)
        # save_index_data(INDEX_PATH, frame)
        save_lidar_data(lidar_filename, sensor_data, LIDAR_DATA_FORMAT)

    sensor_queue.put((sensor_data.frame, sensor_name))

def roadside_lidar_sensor_callback(sensor_data, world, sensor_queue, sensor_name, vehicle):
    # Do stuff with the sensor_data data like save it to disk
    # Then you just need to add to the queue
    frame = sensor_data.frame
    sensor_location = vehicle.get_location()
    dis2Roadside = dis2sensor(sensor_location)
    if frame%5 == 0 and dis2Roadside <= CP_DISTANCE:
        actor_list = world.get_actors()
        vehicle_list = actor_list.filter('vehicle.*')
        pedestrians_list = actor_list.filter('walker.pedestrian.*')
        # print('vehicle_list length: ', len(actor_list))
        lidar_list = actor_list.filter('sensor.lidar.ray_cast')
        # print('lidar_list: ', lidar.get_location())
        label_data = []
        for actor in vehicle_list:
            if isInRoadsideRange(actor.get_location()):
                kitti_datapoint = creat_kitti_datapoint(actor, sensor_data.transform, 'Car')
                if kitti_datapoint:
                    label_data.append(kitti_datapoint)

        for actor in pedestrians_list:
            if isInRoadsideRange(actor.get_location()):
                kitti_datapoint = creat_kitti_datapoint(actor, sensor_data.transform, 'Pedestrian')
                if kitti_datapoint:
                    label_data.append(kitti_datapoint)

        # print('label_data', label_data)
        # print('frame: ', frame)

        if sensor_name == "lidar01":
            label_filename = LABEL01_PATH.format(frame)
            calib_filename = CALIBRATION01_PATH.format(frame)
            if LIDAR_DATA_FORMAT == "bin":
                lidar_filename = LIDAR01_PATH.format(frame)
            else:
                lidar_filename = LIDAR01_PATH_PLY.format(frame)

        # print('label_filename', )
        save_kitti_label_data(label_filename, label_data)
        save_calibration_matrices(sensor_data, calib_filename)
        save_index_data(INDEX_PATH, frame)
        save_lidar_data(lidar_filename, sensor_data, LIDAR_DATA_FORMAT)

    sensor_queue.put((sensor_data.frame, sensor_name))

def cam_sensor_callback(sensor_data,  world, sensor_queue, sensor_name, vehicle):
    # Do stuff with the sensor_data data like save it to disk
    # Then you just need to add to the queue
    frame = sensor_data.frame
    sensor_location = vehicle.get_location()
    dis2Roadside = dis2sensor(sensor_location)
    if sensor_name == "cam01":
        img_filename = IMAGE01_PATH.format(frame)

    if sensor_name == "cam02":
        img_filename = IMAGE02_PATH.format(frame)
    if frame%5 == 0 and dis2Roadside <= CP_DISTANCE:
        sensor_data.save_to_disk(img_filename)  
    sensor_queue.put((sensor_data.frame, sensor_name))

def generate_lidar_bp(arg, world, blueprint_library):
    """Generates a CARLA blueprint based on the script parameters"""
    if arg.semantic:
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
    else:
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        if arg.no_noise:
            lidar_bp.set_attribute('dropoff_general_rate', '0.0')
            lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
            lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
        else:

            lidar_bp.set_attribute('noise_stddev', '0.01')
            # lidar_bp.set_attribute('dropoff_general_rate', '0.05')
            # lidar_bp.set_attribute('dropoff_intensity_limit', '0.9')
            # lidar_bp.set_attribute('dropoff_zero_intensity', '0.1')

    lidar_bp.set_attribute('upper_fov', str(arg.upper_fov))
    lidar_bp.set_attribute('lower_fov', str(arg.lower_fov))
    lidar_bp.set_attribute('channels', str(arg.channels))
    # lidar_bp.set_attribute('atmosphere_attenuation_rate', str(0.05))
    lidar_bp.set_attribute('range', str(arg.range))
    lidar_bp.set_attribute('rotation_frequency', str(10))
    lidar_bp.set_attribute('points_per_second', str(arg.points_per_second))
    return lidar_bp



def main(arg):
    # We start creating the client
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()

    try:
        # We need to save the settings to be able to recover them at the end
        # of the script to leave the server in the same state that we found it.
        original_settings = world.get_settings()
        settings = world.get_settings()
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        tm_port = traffic_manager.get_port()

        # We set CARLA syncronous mode
        settings.fixed_delta_seconds = 0.1
        settings.synchronous_mode = True
        world.apply_settings(settings)

        # We create the sensor queue in which we keep track of the information
        # already received. This structure is thread safe and can be
        # accessed by all the sensors callback concurrently without problem.
        sensor_queue = Queue()

        # Bluepints for the sensors
        blueprint_library = world.get_blueprint_library()
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        lidar01_bp = generate_lidar_bp(arg, world, blueprint_library)
        lidar01_bp.set_attribute('upper_fov', str(2.0)) 




        vehicle_bp = blueprint_library.filter(arg.filter)[0]
        # vehicle_bp.set_attribute('color', '[255, 0, 0]')
        vehicle_transform = carla.Transform(carla.Location(x=CFG.Vehicle['LocationX'], y=CFG.Vehicle['LocationY'], z=CFG.Vehicle['LocationZ']),
                                          carla.Rotation(pitch=CFG.Vehicle['Pitch'], yaw=CFG.Vehicle['Yaw'], roll=CFG.Vehicle['Roll']))
        vehicle = world.spawn_actor(vehicle_bp, vehicle_transform)
        vehicle.set_autopilot(True)


        # We create all the sensors and keep them in a list for convenience.
        sensor_list = []


        lidar01_transform = carla.Transform(carla.Location(x=CFG.Lidar['LocationX'], y=CFG.Lidar['LocationY'], z=CFG.Lidar['LocationZ']),
                                          carla.Rotation(pitch=CFG.Lidar['Pitch'], yaw=CFG.Lidar['Yaw'], roll=CFG.Lidar['Roll']))
        lidar01 = world.spawn_actor(lidar01_bp, lidar01_transform)
        lidar01.listen(lambda data: roadside_lidar_sensor_callback(data, world, sensor_queue, "lidar01", vehicle))
        sensor_list.append(lidar01)


        # Main loop
        frame = 0
        dt0 = datetime.now()
        frame_CP = 0
        while True:
            # Tick the server
            world.tick()
            w_frame = world.get_snapshot().frame
            # print("\nWorld's frame: %d" % w_frame)

            # Now, we wait to the sensors data to be received.
            # As the queue is blocking, we will wait in the queue.get() methods
            # until all the information is processed and we continue with the next frame.
            # We include a timeout of 1.0 s (in the get method) and if some information is
            # not received in this time we continue.
            try:
                withInRange = isInRange(vehicle.get_location(), lidar01.get_location())
                
                if withInRange:
                    frame_CP += 1
                    print("\nrecored frame %d, CP at World's frame: %d" %(frame_CP/5, w_frame))
                    for _ in range(len(sensor_list)):
                        s_frame = sensor_queue.get(True, 1.0)
                        print("    Frame: %d   Sensor: %s" % (s_frame[0], s_frame[1]))

            except Empty:
                print("    Some of the sensor information is missed")
            # time.sleep(0.1)
            process_time = datetime.now() - dt0
            sys.stdout.write('\r' + 'FPS: ' + str(1.0 / process_time.total_seconds()))
            sys.stdout.flush()
            dt0 = datetime.now()

    finally:
        world.apply_settings(original_settings)
        for sensor in sensor_list:
            sensor.destroy()

        vehicle.destroy()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port of CARLA Simulator (default: 2000)')
    argparser.add_argument(
        '--no-rendering',
        action='store_true',
        help='use the no-rendering mode which will provide some extra'
        ' performance but you will lose the articulated objects in the'
        ' lidar, such as pedestrians')
    argparser.add_argument(
        '--semantic',
        action='store_true',
        help='use the semantic lidar instead, which provides ground truth'
        ' information')
    argparser.add_argument(
        '--no-noise',
        action='store_true',
        help='remove the drop off and noise from the normal (non-semantic) lidar')
    argparser.add_argument(
        '--no-autopilot',
        action='store_false',
        help='disables the autopilot so the vehicle will remain stopped')
    argparser.add_argument(
        '--show-axis',
        action='store_true',
        help='show the cartesian coordinates axis')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='model3',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--upper-fov',
        default=15.0,
        type=float,
        help='lidar\'s upper field of view in degrees (default: 15.0)')
    argparser.add_argument(
        '--lower-fov',
        default=-25.0,
        type=float,
        help='lidar\'s lower field of view in degrees (default: -25.0)')
    argparser.add_argument(
        '--channels',
        default=64.0,
        type=float,
        help='lidar\'s channel count (default: 64)')
    argparser.add_argument(
        '--range',
        default=100.0,
        type=float,
        help='lidar\'s maximum range in meters (default: 100.0)')
    argparser.add_argument(
        '--points-per-second',
        default=1000000,
        type=int,
        help='lidar\'s points per second (default: 500000)')
    argparser.add_argument(
        '-x',
        default=0.0,
        type=float,
        help='offset in the sensor position in the X-axis in meters (default: 0.0)')
    argparser.add_argument(
        '-y',
        default=0.0,
        type=float,
        help='offset in the sensor position in the Y-axis in meters (default: 0.0)')
    argparser.add_argument(
        '-z',
        default=0.0,
        type=float,
        help='offset in the sensor position in the Z-axis in meters (default: 0.0)')
    args = argparser.parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')
