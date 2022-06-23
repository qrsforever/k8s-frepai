#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file pre.py
# @brief
# @author QRS
# @version 1.0
# @date 2022-03-16 20:40


import cv2
import numpy as np
import math
import time
import os
import json
import pickle
import tempfile

from sklearn.decomposition import PCA
from sklearn import preprocessing
from frepai.utils.easydict import DotDict
from frepai.utils.draw import get_rect_points
from frepai.utils.logger import EasyLogger as logger
from frepai.utils.errcodes import HandlerError
from frepai.utils import mkdir_p, rmdir_p, easy_wget

# 331 32 56
# 352 61 70

lower_red_1 = np.array([0, 43, 46])
upper_red_1 = np.array([10, 255, 255])
lower_red_2 = np.array([156, 43, 46])
upper_red_2 = np.array([180, 255, 255])

lower_orange = np.array([11, 43, 46])
upper_orange = np.array([25, 255, 255])

lower_yellow = np.array([26, 43, 46])
upper_yellow = np.array([34, 255, 255])

lower_green = np.array([35, 43, 46])
upper_green = np.array([77, 255, 255])

lower_cyan = np.array([78, 43, 46])
upper_cyan = np.array([99, 255, 255])

lower_blue = np.array([100, 43, 46])
upper_blue = np.array([124, 255, 255])

lower_purple = np.array([125, 43, 46])
upper_purple = np.array([155, 255, 255])

lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 46])

lower_white = np.array([0, 0, 221])
upper_white = np.array([180, 30, 255])

lower_gray = np.array([0, 0, 46])
upper_gray = np.array([180, 43, 220])

INPUT_WIDTH = 112
INPUT_HEIGHT = 112
SMALL_AREA_THRESH = 50 * 50
MIN_AREA_THRESH = 8 * 8


def shuffle_tile_input(image):
    indexes = [10, 3, 9, 7, 5, 11, 0, 2, 15, 1, 4, 14, 8, 12, 6, 13]
    tiled_array = image.reshape(4, 28, 4, 28, 3)
    tiled_array = tiled_array.swapaxes(1, 2).reshape((-1, 28, 28, 3))
    tiled_array = np.take(tiled_array, indexes, axis=0)
    vs = []
    for i in range(4):
        vs.append(np.hstack(tiled_array[i * 4: (i + 1) * 4]))
    return np.vstack(vs)


def _pre_kstest(args, resdata, progress_cb):
    feat_list = []
    progress_cb(30)
    with tempfile.TemporaryDirectory() as tmp_dir:
        for item in args.pcaks:
            epath = easy_wget(item['ef_url'], tmp_dir)
            if os.path.exists(epath):
                efnpy = np.load(epath)
                if efnpy.shape[0] > max(item['slices']):
                    feat_list.append(efnpy[item['slices']])
                else:
                    logger.error(f'efnpy.shape[0] ({efnpy.shape}) < max_slices ({max(item["slices"])})')

    progress_cb(60)
    feat_np = np.concatenate(feat_list, axis=0).reshape((-1, 512))
    logger.info(f'feat_np.shape: {feat_np.shape}')

    nc = args.n_components
    sc = args.scaler
    pca = PCA(n_components=nc)

    if sc == 'Normalizer':
        scaler = preprocessing.Normalizer()
    elif sc == 'Standard':
        scaler = preprocessing.StandardScaler()
    elif sc == 'MinMax':
        scaler = preprocessing.MinMaxScaler()
    elif sc == 'Robust':
        scaler = preprocessing.RobustScaler(quantile_range=(25., 75.))
    else:
        scaler = preprocessing.Normalizer()

    progress_cb(70)
    scaler.fit(feat_np)
    data_out = pca.fit_transform(scaler.transform(feat_np))

    pcaks = {
        'pca': pca,
        'scaler': scaler,
        'ecdfs': data_out
    }

    kstest_ecdfs_path = f'{resdata["cache_path"]}/kstest_ecdfs.npy'
    with open(kstest_ecdfs_path, 'wb') as fw:
        pickle.dump(pcaks, fw)

    resdata['kstest_coss3_path'] = os.path.join('/', *args.resdata['out_path'][8:].split('/')[1:])
    resdata['kstest_ecdfs_path'] = kstest_ecdfs_path
    progress_cb(100)
    return resdata


def video_preprocess(args, progress_cb=None):
    if 'dev_args' in args and len(args['dev_args']) > 0:
        args.update(json.loads(args['dev_args']))

    args = DotDict(args)

    devmode = (args.save_video or args.best_stride_video)

    # logger.info(args)

    resdata = {'errno': 0, 'pigeon': args.pigeon, 'devmode': devmode, 'task': 'pre', 'upload_files': []}

    def _send_progress(x):
        if progress_cb:
            resdata['progress'] = round(0.4 * x, 2)
            progress_cb(resdata)
            logger.info(f"{round(x, 2)} {resdata['progress']}")

    if 'https://' in args.video:
        segs = args.video[8:].split('/')
        vname = segs[-1].split('.')[0]
        coss3_path = os.path.join('/', *segs[1:-2], 'outputs', vname, 'repnet_tf')
    else:
        vname = 'unknow'
        coss3_path = ''

    video_path = args.video
    logger.info(f'from: {video_path}')

    cache_path = f'/data/cache/{int(time.time() * 1000)}/{vname}'
    mkdir_p(cache_path)
    resdata['cache_path'] = cache_path

    if 'pcaks' in args:
        return _pre_kstest(args, resdata, _send_progress)

    if not os.path.isfile(video_path):
        try:
            video_path = easy_wget(video_path, f'{cache_path}/source.mp4')
        except Exception as err:
            raise HandlerError(80001, f'wget video[{args.video}] fail [{err}]!')

    logger.info(f'to: {video_path}')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HandlerError(80002, f'open video[{args.video}] [{video_path}] fail!')

    fps = round(cap.get(cv2.CAP_PROP_FPS))
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    black_box = get_rect_points(width, height, args.black_box)
    if black_box is not None:
        black_x1, black_y1, black_x2, black_y2 = black_box
    focus_box = get_rect_points(width, height, args.focus_box)
    if focus_box is not None:
        focus_x1, focus_y1, focus_x2, focus_y2 = focus_box
        w = focus_x2 - focus_x1
        h = focus_y2 - focus_y1

    logger.info(f'width[{width} vs {w}] height[{height} vs {h}] framerate[{fps}] count[{cnt}]')

    area, frames_invalid = w * h, False
    if w < 0 or h < 0 or area < MIN_AREA_THRESH:
        raise HandlerError(80003, f'invalid focus box[{args.focus_box}]!')

    if args.rmstill_frame_enable:
        area_rate_thres = args.get('rmstill_rate_threshold', 0.001)
        rmstill_bin_threshold = args.get('rmstill_bin_threshold', 20)
        rmstill_brightness_norm = args.get('rmstill_brightness_norm', False)
        rmstill_area_mode = args.get('rmstill_area_mode', 0)
        rmstill_noise_level = args.get('rmstill_noise_level', 1)
        rmstill_area_thres = math.ceil(area_rate_thres * area)
        rmstill_filter_kernel = args.get('rmstill_filter_kernel', 3)
        rmstill_noise_kernel = np.ones((rmstill_filter_kernel, rmstill_filter_kernel), np.uint8)

        if area < SMALL_AREA_THRESH:
            rmstill_white_thres = int(args.get('rmstill_white_rate', 0) * area)
            frames_invalid = True

        logger.info(f'rmstill: ({area}, {rmstill_area_thres}, {rmstill_bin_threshold}, {rmstill_noise_level})')

    elif args.color_tracker_enable:
        color_pre_count = 0
        color_select = args.get('color_select', 8)
        color_rate_threshold = args.get('color_rate_threshold', 0.9)
        color_buffer_size = args.get('color_buffer_size', 12)
        color_lower_rate = args.get('color_lower_rate', 0.2)
        color_upper_rate = args.get('color_upper_rate', 0.8)
        color_track_direction = args.get('color_track_direction', 0)
        color_buffer = np.zeros((color_buffer_size, ))
        if color_track_direction > 0:
            color_direction_buffer = np.zeros_like(color_buffer)
        color_area_thres = math.ceil(color_rate_threshold * area)
        color_lower_value = int(color_buffer_size * color_lower_rate)
        color_upper_value = int(color_buffer_size * color_upper_rate)
        logger.info(f'color_tracker: ({color_area_thres}, {color_lower_value}, {color_upper_value})')

    elif args.stdwave_tracker_enable:
        stdwave_feature_select = args.get('stdwave_feature_select', 'std')
        stdwave_sigma_count = args.get('stdwave_sigma_count', -5.0)
        stdwave_window_size = args.get('stdwave_window_size', 50)
        stdwave_distance_size = args.get('stdwave_distance_size', 150)
        stdwave_minstd_thresh = args.get('stdwave_minstd_thresh', 0.08)
        stdwave_blur_type = args.get('stdwave_blur_type', 'none')
        stdwave_filter_kernel = args.get('stdwave_filter_kernel', 3)
        stdwave_noise_kernel = np.ones((stdwave_filter_kernel, stdwave_filter_kernel), np.uint8)
        resdata['stdwave_sigma_count'] = stdwave_sigma_count
        resdata['stdwave_window_size'] = stdwave_window_size
        resdata['stdwave_distance_size'] = stdwave_distance_size
        resdata['stdwave_minstd_thresh'] = stdwave_minstd_thresh
        logger.info(f'stdwave_tracker: ({stdwave_sigma_count}, {stdwave_window_size}, {stdwave_distance_size})')

    stdwave = []
    keepframe, keepidxes = [], []
    if devmode:
        binframes, binpoints = [], []
    idx, frame_tmp = 0, np.zeros((h, w), dtype=np.uint8)
    pre_frame_gray = np.zeros((h, w, 1), dtype=np.uint8)

    ret, frame_bgr = cap.read()
    while ret:
        keep_flag = False
        if black_box is not None:
            frame_bgr[black_y1:black_y2, black_x1:black_x2, :] = 0
        if focus_box is not None:
            frame_bgr = frame_bgr[focus_y1:focus_y2, focus_x1:focus_x2, :]

        # frame_bgr = cv2.fastNlMeansDenoisingColored(frame_bgr,None, 10, 10, 7, 21)

        if args.rmstill_frame_enable:
            if rmstill_brightness_norm:
                h, s, v = cv2.split(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV))
                v = np.array((v - np.mean(v)) / np.std(v) * 32 + 127, dtype=np.uint8)
                frame_bgr = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            frame_tmp = cv2.absdiff(frame_gray, pre_frame_gray)
            frame_tmp = cv2.threshold(frame_tmp, rmstill_bin_threshold, 255, cv2.THRESH_BINARY)[1]
            if rmstill_noise_level > 0:
                # Opening
                frame_tmp = cv2.erode(frame_tmp, rmstill_noise_kernel, iterations=rmstill_noise_level)
                frame_tmp = cv2.dilate(frame_tmp, rmstill_noise_kernel, iterations=rmstill_noise_level)
            val = np.sum(frame_tmp == 255)
            if rmstill_area_mode == 0:
                if val > rmstill_area_thres:
                    keep_flag = True
                    if devmode:
                        frame_tmp = cv2.cvtColor(frame_tmp, cv2.COLOR_GRAY2RGB)
                        binframes.append(cv2.resize(frame_tmp, (INPUT_WIDTH, INPUT_HEIGHT)))
                        binpoints.append(np.round(val / rmstill_area_thres, 2))
            else:
                contours, _ = cv2.findContours(frame_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if len(contours) > 0:
                    contours = sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True)
                    if cv2.contourArea(contours[0]) > rmstill_area_thres:
                        keep_flag = True
                        if devmode:
                            frame_tmp = cv2.cvtColor(frame_tmp, cv2.COLOR_GRAY2RGB)
                            cv2.drawContours(frame_tmp, [contours[0]], 0, (0, 0, 255), 3)
                            binframes.append(cv2.resize(frame_tmp, (INPUT_WIDTH, INPUT_HEIGHT)))

            if frames_invalid and keep_flag:
                if val > rmstill_white_thres:
                    logger.info(f'val[{val}] vs rmstill_white_thres[{rmstill_white_thres}]')
                    frames_invalid = False
            pre_frame_gray = frame_gray

        elif args.color_tracker_enable:
            frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            if color_select == 0:
                mask_red_1 = cv2.inRange(frame_hsv, lower_red_1, upper_red_1)
                mask_red_2 = cv2.inRange(frame_hsv, lower_red_2, upper_red_2)
                color_mask = cv2.bitwise_or(mask_red_1, mask_red_2)
            elif color_select == 1:
                color_mask = cv2.inRange(frame_hsv, lower_orange, upper_orange)
            elif color_select == 2:
                color_mask = cv2.inRange(frame_hsv, lower_yellow, upper_yellow)
            elif color_select == 3:
                color_mask = cv2.inRange(frame_hsv, lower_green, upper_green)
            elif color_select == 4:
                color_mask = cv2.inRange(frame_hsv, lower_cyan, upper_cyan)
            elif color_select == 5:
                color_mask = cv2.inRange(frame_hsv, lower_blue, upper_blue)
            elif color_select == 6:
                color_mask = cv2.inRange(frame_hsv, lower_purple, upper_purple)
            elif color_select == 7:
                color_mask = cv2.inRange(frame_hsv, lower_black, upper_black)
            elif color_select == 8:
                color_mask = cv2.inRange(frame_hsv, lower_white, upper_white)
            elif color_select == 9:
                color_mask = cv2.inRange(frame_hsv, lower_gray, upper_gray)
            val = np.sum(color_mask == 255)
            if val > color_area_thres:
                color_buffer[-1] = 1
            else:
                color_buffer[-1] = 0
            color_buffer = np.roll(color_buffer, shift=-1, axis=0)
            color_count = np.sum(color_buffer, axis=0)
            if color_track_direction == 0:
                val = color_count
            else:
                if color_track_direction == 1: # +
                    color_direction_buffer[-1] = 1 if color_count > color_pre_count else 0
                elif color_track_direction == 2: # -
                    color_direction_buffer[-1] = 1 if color_count < color_pre_count else 0
                color_direction_buffer = np.roll(color_direction_buffer, shift=-1, axis=0)
                val = np.sum(color_direction_buffer, axis=0)

            color_pre_count = color_count

            if color_lower_value < val < color_upper_value:
                keep_flag = True
                if devmode:
                    frame_tmp = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2RGB)
                    binframes.append(cv2.resize(frame_tmp, (INPUT_WIDTH, INPUT_HEIGHT)))

        elif args.stdwave_tracker_enable:
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            if stdwave_blur_type == 'averaging':
                frame_gray = cv2.blur(frame_gray, (stdwave_filter_kernel, stdwave_filter_kernel))
            elif stdwave_blur_type == 'median':
                frame_gray = cv2.medianBlur(frame_gray, stdwave_filter_kernel)
            elif stdwave_blur_type == 'gaussian':
                frame_gray = cv2.GaussianBlur(frame_gray, (stdwave_filter_kernel, stdwave_filter_kernel), 0)
            elif stdwave_blur_type == 'open':
                frame_gray = cv2.morphologyEx(frame_gray, cv2.MORPH_OPEN, stdwave_noise_kernel, iterations=3)

            if stdwave_feature_select == 'std':
                stdwave.append(np.std(frame_gray))
            elif stdwave_feature_select == 'mean':
                stdwave.append(np.mean(frame_gray))
            else:
                raise HandlerError(80004, f'unkown stdwave_feature_select args [{stdwave_feature_select}]')

        if keep_flag:
            if focus_box is not None:
                if args.focus_box_repnum > 1:
                    frame_bgr = np.hstack([frame_bgr] * args.focus_box_repnum)
                    frame_bgr = np.vstack([frame_bgr] * args.focus_box_repnum)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (INPUT_WIDTH, INPUT_HEIGHT))
            if args.angle:
                M = cv2.getRotationMatrix2D(center=(int(INPUT_WIDTH / 2), int(INPUT_HEIGHT / 2)), angle=args.angle, scale=1.0)
                frame_rgb = cv2.warpAffine(frame_rgb, M, (INPUT_WIDTH, INPUT_HEIGHT))
            keepframe.append(frame_rgb)
            keepidxes.append(idx)

        if idx % 888 == 0:
            _send_progress(100 * idx / cnt)

        idx += 1
        ret, frame_bgr = cap.read()
    cap.release()

    if args.stdwave_tracker_enable:
        np.save(f'{cache_path}/stdwave_data.npy', np.asarray(stdwave))
        resdata['upload_files'].append('stdwave_data.npy')
    else:
        logger.info(f'valid frames count: [{len(keepframe)}] cache_path[{cache_path}]')

        if frames_invalid or len(keepframe) < 32:
            logger.warning(f'invalid[{frames_invalid}] or valid frames count: {len(keepframe)} <= 32')
            resdata['progress'] = 100
            if progress_cb:
                progress_cb(resdata)
            rmdir_p(os.path.dirname(cache_path))
            return None

    if len(keepframe) > 0:
        np.savez_compressed(f'{cache_path}/keepframe.npz', x=np.asarray(keepframe))
        np.save(f'{cache_path}/keepidxes.npy', np.asarray(keepidxes))

    if devmode:
        if len(binpoints) > 0:
            np.save(f'{cache_path}/binpoints.npy', np.asarray(binpoints))
        if len(binframes) > 0:
            np.savez_compressed(f'{cache_path}/binframes.npz', x=np.asarray(binframes))

    with open(f'{cache_path}/config.json', 'w') as f:
        f.write(json.dumps(dict(args)))

    resdata['upload_files'].append('config.json')
    resdata['cache_path'] = cache_path
    resdata['coss3_path'] = coss3_path
    resdata['frame_count'] = cnt
    resdata['frame_rate'] = fps

    _send_progress(100)
    return resdata
