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
from frepai.utils.oss import coss3_delete
from scipy import stats

ffmpeg_args = '-preset ultrafast -vcodec libx264 -pix_fmt yuv420p'

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
SMALL_AREA_THRESH = 150 * 150
MIN_AREA_THRESH = 8 * 8


def input_tile_shuffle(image):
    indexes = [10, 3, 9, 7, 5, 11, 0, 2, 15, 1, 4, 14, 8, 12, 6, 13]
    tiled_array = image.reshape(4, 28, 4, 28, 3)
    tiled_array = tiled_array.swapaxes(1, 2).reshape((-1, 28, 28, 3))
    tiled_array = np.take(tiled_array, indexes, axis=0)
    vs = []
    vs.append(np.hstack([tiled_array[0], np.flip(tiled_array[1], axis=0), np.flip(tiled_array[2], axis=1), 255 - tiled_array[3]]))
    vs.append(np.hstack([tiled_array[4], np.flip(tiled_array[5], axis=0), 255 - np.flip(tiled_array[6], axis=1), tiled_array[7]]))
    vs.append(np.hstack([tiled_array[8], 255 - np.flip(tiled_array[9], axis=0), np.flip(tiled_array[10], axis=1), tiled_array[11]]))
    vs.append(np.hstack([255 - tiled_array[12], np.flip(tiled_array[13], axis=0), np.flip(tiled_array[14], axis=1), tiled_array[15]]))
    # for i in range(4):
    #     vs.append(np.hstack(tiled_array[i * 4: (i + 1) * 4]))
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


def gray_image_blur(frame_gray, mode, kernel):
    if mode == 'averaging':
        frame_gray = cv2.blur(frame_gray, (kernel, kernel))
    elif mode == 'median':
        frame_gray = cv2.medianBlur(frame_gray, kernel)
    elif mode == 'gaussian':
        frame_gray = cv2.GaussianBlur(frame_gray, (kernel, kernel), 0)
    return frame_gray


g_switch_names = [
    "rmstill_frame_enable", "color_tracker_enable",
    "direction_tracker_enable", "diffimpulse_tracker_enable",
    "featpeak_tracker_enbale", "stdwave_tracker_enable"]


def video_preprocess(args, progress_cb=None):
    if 'dev_args' in args and len(args['dev_args']) > 0:
        logger.info(args['dev_args'])
        args.update({key: False for key in g_switch_names})
        args.update(json.loads(args['dev_args']))

        check_enable_ok = False
        for sw in g_switch_names:
            if sw in args and args[sw]:
                check_enable_ok = True
                break
        if not check_enable_ok:
            logger.warning('error, no enable')
            args['stdwave_tracker_enable'] = True

    args = DotDict(args)

    devmode = args.best_stride_video

    # logger.info(args)
    debug_data = []

    resdata = {'errno': 0, 'pigeon': args.pigeon, 'devmode': devmode, 'task': 'pre', 'sumcnt': 0, 'upload_files': []}

    def _send_progress(x, fix=False):
        if progress_cb:
            resdata['progress'] = x if fix else round(0.4 * x, 2)
            progress_cb(resdata)
            logger.info(f"{round(x, 2)} {resdata['progress']}")

    video_path = args.video
    logger.info(f'from: {video_path}')

    if 'https://' in video_path:
        segs = video_path[8:].split('/')
        vname = segs[-1].split('.')[0]
        coss3_path = os.path.join('/', *segs[1:-2], 'outputs', vname, 'repnet_tf')
        coss3_delete(coss3_path[1:], logger)
    else:
        vname = 'unknow'
        coss3_path = ''

    cache_path = f'/data/cache/{int(time.time() * 1000)}/{vname}'
    mkdir_p(cache_path)
    resdata['cache_path'] = cache_path

    if 'pcaks' in args:
        return _pre_kstest(args, resdata, _send_progress)

    if not os.path.isfile(video_path):
        try:
            video_path = easy_wget(video_path, f'{cache_path}/source.mp4')
        except Exception as err:
            raise HandlerError(80001, f'wget video[{video_path}] fail [{err}]!')

    logger.info(f'to: {video_path}')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HandlerError(80002, f'open video[{args.video}] [{video_path}] fail!')

    fps = round(cap.get(cv2.CAP_PROP_FPS))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    all_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    black_box = get_rect_points(width, height, args.black_box)
    if black_box is not None:
        black_x1, black_y1, black_x2, black_y2 = black_box
    focus_box = get_rect_points(width, height, args.focus_box)
    if focus_box is not None:
        focus_x1, focus_y1, focus_x2, focus_y2 = focus_box
        w = focus_x2 - focus_x1
        h = focus_y2 - focus_y1
    check_box = args.get('check_box', None)
    if check_box is not None:
        check_x1, check_y1, check_x2, check_y2 = get_rect_points(width, height, check_box)

    def _get_box_frame(img):
        vbox, cbox = None, None
        if black_box is not None:
            img[black_y1:black_y2, black_x1:black_x2, :] = 0
        if focus_box is not None:
            vbox = img[focus_y1:focus_y2, focus_x1:focus_x2, :]
        if check_box is not None:
            cbox = img[check_y1:check_y2, check_x1:check_x2, :]
        return vbox, cbox

    area, frames_invalid = w * h, False
    if w < 0 or h < 0 or area < MIN_AREA_THRESH:
        cap.release()
        raise HandlerError(80003, f'invalid focus box[{args.focus_box}]!')

    debug_write_video = False
    if 'debug_pre_write_video' in args:
        debug_write_video = args['debug_pre_write_video']
    if debug_write_video:
        writer = cv2.VideoWriter(f'{cache_path}/_pre_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    global_gray_frame, global_gray_check = None, None
    global_remove_shadow = args.get('global_remove_shadow', None)
    grap_speed = args.get('global_grap_speed', -1)
    if grap_speed > 0:
        global_grap_step = grap_speed
    else:
        global_grap_step = int(fps * args.get('global_grap_interval', -1))
    global_blur_type = args.get('global_blur_type', 'none')
    global_filter_kernel = args.get('global_filter_kernel', 3)
    global_feature_select = args.get('global_feature_select', 'mean')
    global_feature_minval = args.get('global_feature_minval', 10)
    global_feature_minnum = args.get('global_feature_minnum', 50)
    global_hdiff_rate = args.get('global_hdiff_rate', 1.0)
    global_bg_finding = args.get('global_bg_finding', False)

    # find bg
    resdata['global_bg_focus'] = 0
    resdata['global_bg_check'] = 0
    if global_bg_finding:
        np.random.seed(all_cnt)
        sample_size = 200 if all_cnt > 200 else all_cnt
        indexes = np.random.choice(range(all_cnt), size=sample_size, replace=False)
        focus_feats, focus_feat2idx = [], {}
        check_feats, check_feat2idx = [], {}

        def _box_feat(img, feats, feat2idx):
            gray = cv2.cvtColor(frame_focus, cv2.COLOR_BGR2GRAY)
            gray = cv2.erode(gray, np.ones((3, 3), np.uint8), iterations=1)
            gray = cv2.dilate(gray, np.ones((3, 3), np.uint8), iterations=1)
            real_feat = int(np.mean(gray))
            feat = round(real_feat / 10) * 10
            feats.append(feat)
            if feat not in feat2idx:
                feat2idx[feat] = {}
            if real_feat not in feat2idx[feat]:
                feat2idx[feat][real_feat] = []
            feat2idx[feat][real_feat].append(i)

        def _mode_frame(feats, feat2idx):
            mode = stats.mode(feats)[0][0]
            max_cnt, vid_idx = -1, 0 
            for value in feat2idx[mode].values():
                n = len(value)
                if n > max_cnt:
                    max_cnt = n
                    vid_idx = value[0]
            return int(vid_idx)

        for i in sorted(indexes):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                logger.warning(f'read {i} frame error.')
                continue
            frame_focus, frame_check = _get_box_frame(frame)
            _box_feat(frame_focus, focus_feats, focus_feat2idx)
            if check_box is not None:
                _box_feat(frame_check, check_feats, check_feat2idx)

        resdata['global_bg_focus'] = _mode_frame(focus_feats, focus_feat2idx)
        if check_box is not None:
            resdata['global_bg_check'] = _mode_frame(check_feats, check_feat2idx)

    cap.set(cv2.CAP_PROP_POS_FRAMES, resdata['global_bg_focus'])
    _, frame = cap.read()
    global_gray_frame = cv2.cvtColor(_get_box_frame(frame)[0], cv2.COLOR_BGR2GRAY)
    if check_box is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, resdata['global_bg_check'])
        _, frame = cap.read()
        global_gray_check = cv2.cvtColor(_get_box_frame(frame)[1], cv2.COLOR_BGR2GRAY)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if global_grap_step > 0:
        resdata['global_grap_step'] = global_grap_step
        s, video_path = 0, f'{cache_path}/source_lite.mp4'
        lite_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if s % global_grap_step == 0:
                lite_writer.write(frame)
            s += 1
        lite_writer.release()
        cap.release()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HandlerError(80002, f'open video[{args.video}] [{video_path}] fail!')
        cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        cnt = all_cnt

    logger.info(f'width[{width} vs {w}] height[{height} vs {h}] framerate[{fps}] count[{all_cnt} vs {cnt}]')

    resdata['video_path'] = video_path

    # global_bg_frame_ = [None, None]
    # global_bg_window = args.get('global_bg_window', 0)
    # global_bg_atonce = args.get('global_bg_atonce', True)
    # if global_bg_window > 0:
    #     global_bgw_buffer = [(0, None)] * global_bg_window
    # def _find_bg(idx, img, feat, feats):
    #     if global_bg_frame_[1] is None:
    #         global_bg_frame_[1] = img
    #     if global_bg_window > 0:
    #         if global_bg_frame_[0] is not None and global_bg_atonce:
    #             return global_bg_frame_[0]
    #         global_bgw_buffer[idx % global_bg_window] = (feat, img)
    #         if idx >= global_bg_window:
    #             mode = stats.mode(feats[-1 * global_bg_window:])[0][0]
    #             for feat, frame in global_bgw_buffer:
    #                 if feat == mode:
    #                     global_bg_frame_[0] = frame
    #                     return frame
    #     return global_bg_frame_[1]

    def _calc_feat(img, dtype=1):
        feat = np.sort(img.ravel())
        if isinstance(global_hdiff_rate, (tuple, list)):
            low, hight = int(global_hdiff_rate[0] * len(feat)), int(global_hdiff_rate[1] * len(feat))
            feat = feat[-hight: -low]
        else:
            feat = feat[int(-1 * global_hdiff_rate * len(feat)):]
        if global_feature_select == 'mean':
            feat = np.mean(feat)
        elif global_feature_select == 'std':
            feat = np.std(feat)
        else:
            raise HandlerError(80004, f'unkown global_feature_select [{global_feature_select}]')
        if dtype == 1: # int
            feat = int(feat)
        return 0.001 if feat == 0 else feat

    def _remove_shadow(img, dksize=3, bksize=5, normalize=True):
        result_planes = []
        bgr_planes = cv2.split(img)
        for plane in bgr_planes:
            img = cv2.dilate(plane, np.ones((dksize,dksize), np.uint8))
            img = cv2.medianBlur(img, bksize)
            img = 255 - cv2.absdiff(plane, img)
            if normalize:
                img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(img)
        return cv2.merge(result_planes)

    if args.rmstill_frame_enable:
        area_rate_thres = args.get('rmstill_rate_threshold', 0.001)
        if area_rate_thres < 0:
            area_rate_thres = float(1) / (w * h)
        rmstill_bin_threshold = args.get('rmstill_bin_threshold', 20)
        rmstill_brightness_norm = args.get('rmstill_brightness_norm', False)
        rmstill_area_mode = args.get('rmstill_area_mode', 0)
        rmstill_noise_level = args.get('rmstill_noise_level', 1)
        rmstill_area_thres = math.ceil(area_rate_thres * area)
        rmstill_filter_kernel = args.get('rmstill_filter_kernel', 3)
        rmstill_noise_kernel = np.ones((rmstill_filter_kernel, rmstill_filter_kernel), np.uint8)
        if area < SMALL_AREA_THRESH:
            rmstill_white_thres = int(args.get('rmstill_white_rate', 0.1) * area)
            rmstill_white_window = args.get('rmstill_white_window', 10)
            rmstill_white_buffer = np.zeros((rmstill_white_window, ))
            frames_invalid = True
        resdata['rmstill_area_thres'] = rmstill_area_thres
        resdata['within_period_threshold'] = args.get('within_period_threshold', 0.5)

        logger.info(f'rmstill: ({area}, {rmstill_area_thres}, {rmstill_bin_threshold}, {rmstill_noise_level})')

    elif args.color_tracker_enable:
        color_pre_count = 0
        color_select = args.get('color_select', 8)
        color_enhance_blur = args.get('color_enhance_blur', 0)
        color_enhance_dilate = args.get('color_enhance_dilate', None)
        if color_enhance_dilate is not None:
            kernel, iterc = color_enhance_dilate
            color_enhance_dilate = (np.ones((kernel, kernel), np.uint8), iterc)
        color_enhance_erode = args.get('color_enhance_erode', None)
        if color_enhance_erode is not None:
            kernel, iterc = color_enhance_erode
            color_enhance_erode = (np.ones((kernel, kernel), np.uint8), iterc)
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
        resdata['within_period_threshold'] = args.get('within_period_threshold', 0.5)
        resdata['color_lower_value'] = color_lower_value 
        resdata['color_upper_value'] = color_upper_value 

    elif args.featpeak_tracker_enbale:
        _height = args.get('featpeak_height_minmax', (-1, -1))
        _width = args.get('featpeak_width_minmax', (-1, -1))
        _prominence = args.get('featpeak_prominence_minmax', (10, -1))

        resdata['featpeak_detect_trough'] = args.get('featpeak_detect_trough', False)
        resdata['featpeak_data_normal'] = args.get('featpeak_data_normal', True)
        resdata['featpeak_window_size'] = args.get('featpeak_window_size', 10)
        resdata['featpeak_min_threshold'] = args.get('featpeak_min_threshold', -1)
        resdata['featpeak_distance_size'] = args.get('featpeak_distance_size', 10)
        resdata['featpeak_relative_height'] = args.get('featpeak_relative_height', 0.8)
        resdata['featpeak_height_minmax'] = (_height, -1) if isinstance(_height, int) else _height
        resdata['featpeak_width_minmax'] = (_width, -1) if isinstance(_width, int) else _width
        resdata['featpeak_prominence_minmax'] = (_prominence, -1) if isinstance(_prominence, int) else _prominence

    elif args.direction_tracker_enable:
        direction_arrow = args.get('direction_arrow', 'lr') # lr, rl, tb, bt
        direction_agg_axis = 0 if direction_arrow in ('rl', 'lr') else 1
        resdata['direction_inverse'] = True if direction_arrow in ('rl', 'bt') else False
        resdata['direction_scale_threshold'] = args.get('direction_scale_threshold', 2.0)
        resdata['direction_window_size'] = args.get('direction_window_size', 10)

    elif args.stdwave_tracker_enable:
        stdwave_sub_average = args.get('stdwave_sub_average', True)
        stdwave_sigma_count = args.get('stdwave_sigma_count', 3.0)
        seconds = args.get('stdwave_window_secs', 0)
        stdwave_window_size = int(fps * seconds) if seconds > 0 else args.get('stdwave_window_size', 50)
        seconds = args.get('stdwave_distance_secs', 0)
        stdwave_distance_size = int(fps * seconds) if seconds > 0 else args.get('stdwave_distance_size', 150)
        stdwave_minstd_thresh = args.get('stdwave_minstd_thresh', 0.5)
        resdata['stdwave_sigma_count'] = stdwave_sigma_count
        resdata['stdwave_sub_average'] = stdwave_sub_average
        resdata['stdwave_window_size'] = stdwave_window_size
        resdata['stdwave_distance_size'] = stdwave_distance_size
        resdata['stdwave_minstd_thresh'] = stdwave_minstd_thresh
        logger.info(f'stdwave_tracker: ({stdwave_sigma_count}, {stdwave_window_size}, {stdwave_distance_size})')

    elif args.diffimpulse_tracker_enable:
        diffimpulse_one_threshold = int(args.get('diffimpulse_rate_threshold', 0.02) * area)
        diffimpulse_bin_threshold = args.get('diffimpulse_bin_threshold', 20)
        diffimpulse_window_size = args.get('diffimpulse_window_size', [7, 5])
        resdata['diffimpulse_one_threshold'] = diffimpulse_one_threshold
        resdata['diffimpulse_bin_threshold'] = diffimpulse_bin_threshold
        resdata['diffimpulse_window_size'] = diffimpulse_window_size
        logger.info(f'diff_impulse: ({diffimpulse_one_threshold}, {diffimpulse_bin_threshold}, {diffimpulse_window_size})')

    tile_shuffle = args.get('input_tile_shuffle', False)

    stdwave, diffimpulse, featpeak, direction = [], [], [], []
    keepframe, keepidxes, half_focus_width = [], [], -1
    if devmode:
        binframes, binpoints, colorvals = [], [], []
    idx, frame_tmp = 0, np.zeros((h, w), dtype=np.uint8)
    pre_frame_gray, pre_check_gray = global_gray_frame, global_gray_check
    ret, frame_raw = cap.read()
    progress_step = int(cnt / 5)
    while ret:
        keep_flag = False
        frame_bgr, check_bgr = _get_box_frame(frame_raw)

        if global_remove_shadow is not None and isinstance(global_remove_shadow, (tuple, list)):
            frame_bgr = _remove_shadow(frame_bgr, *global_remove_shadow)

        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if check_box is not None:
            check_gray = cv2.cvtColor(check_bgr, cv2.COLOR_BGR2GRAY)

        if global_blur_type != "none":
            frame_gray = gray_image_blur(frame_gray, global_blur_type, global_filter_kernel)

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
                        binpoints.append(val)
            elif rmstill_area_mode == 1:
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
                rmstill_white_buffer[-1] = val
                rmstill_white_buffer = np.roll(rmstill_white_buffer, shift=-1, axis=0)
                wpoint_mean = np.mean(rmstill_white_buffer)
                if wpoint_mean > rmstill_white_thres:
                    logger.info(f'wpoint_mean[{wpoint_mean}] vs rmstill_white_thres[{rmstill_white_thres}], {rmstill_white_window}')
                    frames_invalid = False

        elif args.color_tracker_enable:
            frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            if color_enhance_blur > 0:
                frame_hsv = cv2.medianBlur(frame_hsv, color_enhance_blur)
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
            if color_enhance_dilate is not None:
                color_mask = cv2.dilate(color_mask, color_enhance_dilate[0], iterations=color_enhance_dilate[1])
            if color_enhance_erode is not None:
                color_mask = cv2.erode(color_mask, color_enhance_erode[0], iterations=color_enhance_erode[1])
            colorval = np.sum(color_mask == 255)
            if colorval > color_area_thres:
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
                    colorvals.append([colorval , val])
                    frame_tmp = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2RGB)
                    binframes.append(cv2.resize(frame_tmp, (INPUT_WIDTH, INPUT_HEIGHT)))

        elif args.featpeak_tracker_enbale:
            frame_tmp = cv2.absdiff(frame_gray, pre_frame_gray)
            feat = _calc_feat(frame_tmp)
            feat = int(np.mean(feat))
            featpeak.append(feat)

        elif args.direction_tracker_enable:
            frame_tmp = cv2.absdiff(frame_gray, pre_frame_gray)
            if half_focus_width < 0:
                if direction_agg_axis == 0:
                    half_focus_width = int(0.5 * frame_tmp.shape[1])
                else:
                    half_focus_width = int(0.5 * frame_tmp.shape[0])
            data = np.mean(frame_tmp, axis=direction_agg_axis)
            lfeat = _calc_feat(data[:half_focus_width])
            rfeat = _calc_feat(data[half_focus_width:])
            mfeat = abs(lfeat - rfeat)
            direction.append([mfeat if mfeat != 0 else 0.01, lfeat, rfeat])
            if debug_write_video:
                if global_remove_shadow is not None:
                    frame_raw[focus_y1:focus_y2, focus_x1:focus_x2, :] = frame_bgr
                else:
                    frame_raw[focus_y1:focus_y2, focus_x1:focus_x2, :] = 0
                cv2.putText(
                        frame_raw,
                        '%d %.2f %.2f %.2f' % (idx, mfeat, lfeat, rfeat),
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 2)
                writer.write(frame_raw)

        elif args.stdwave_tracker_enable:
            feat = _calc_feat(cv2.absdiff(frame_gray, pre_frame_gray))
            if pre_check_gray is not None:
                feat = max(0, feat - _calc_feat(cv2.absdiff(check_gray, pre_check_gray)))
            stdwave.append(feat)

            if debug_write_video: # debug
                if len(stdwave) < 500:
                    writer.write(frame_raw)
                if len(stdwave) == 500:
                    logger.info(f'{frame_bgr.shape} {frame_raw.shape}')
                    writer.release()
                    logger.info(f'{np.array(stdwave, np.int16)}')
                    os.system(f'ffmpeg -an -i {cache_path}/_pre_video.mp4 {ffmpeg_args} {cache_path}/pre-video.mp4 2>/dev/null')
                    resdata['upload_files'].append('pre-video.mp4')

        elif args.diffimpulse_tracker_enable:
            frame_tmp = cv2.absdiff(frame_gray, pre_frame_gray)
            cc = np.sum(frame_tmp > diffimpulse_bin_threshold)
            if devmode:
                debug_data.append(cc)
            if cc > diffimpulse_one_threshold:
                diffimpulse.append(1)
            else:
                diffimpulse.append(0)

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
            if tile_shuffle:
                frame_rgb = input_tile_shuffle(frame_rgb)
            keepframe.append(frame_rgb)
            keepidxes.append(idx)

        if idx % progress_step == 0:
            _send_progress(100 * idx / cnt)

        if not global_bg_finding:
            pre_frame_gray = frame_gray
        if check_box is not None:
            pre_check_gray = check_gray

        idx += 1
        ret, frame_raw = cap.read()
    cap.release()

    if devmode and len(debug_data) > 0:
        logger.info(debug_data)

    if args.stdwave_tracker_enable:
        stdwave = np.asarray(stdwave)
        if len(stdwave[stdwave > global_feature_minval]) < global_feature_minnum:
            _send_progress(100, True)
            rmdir_p(os.path.dirname(cache_path))
            return None
        np.save(f'{cache_path}/stdwave_data.npy', stdwave)
        resdata['upload_files'].append('stdwave_data.npy')
    elif args.direction_tracker_enable:
        direction = np.asarray(direction)
        features = direction[:, 0]
        if len(features[features > global_feature_minval]) < global_feature_minnum:
            _send_progress(100, True)
            rmdir_p(os.path.dirname(cache_path))
            return None
        np.save(f'/data/direction_data.npy', direction)
        if debug_write_video:
            writer.release()
            os.system(f'ffmpeg -an -i {cache_path}/_pre_video.mp4 {ffmpeg_args} /data/pre-video.mp4 2>/dev/null')
        _send_progress(100, True)
        return None
    elif args.featpeak_tracker_enbale:
        if devmode:
            logger.info(featpeak)
        if resdata['featpeak_detect_trough']:
            featpeak = np.asarray(featpeak)
            featpeak = -1 * featpeak + np.max(featpeak)
        np.save(f'{cache_path}/featpeak_data.npy', featpeak)
        resdata['upload_files'].append('featpeak_data.npy')
    elif args.diffimpulse_tracker_enable:
        if devmode:
            logger.info(diffimpulse)
        np.save(f'{cache_path}/diffimpulse_data.npy', np.asarray(diffimpulse))
        resdata['upload_files'].append('diffimpulse_data.npy')
    else:
        # repnet
        logger.info(f'valid frames count: [{len(keepframe)}] cache_path[{cache_path}]')
        if frames_invalid or len(keepframe) < 32:
            logger.warning(f'invalid[{frames_invalid}] or valid frames count: {len(keepframe)} <= 32')
            _send_progress(100, True)
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
        if len(colorvals) > 0:
            np.save(f'{cache_path}/colorvals.npy', np.asarray(colorvals))

    with open(f'{cache_path}/config.json', 'w') as f:
        f.write(json.dumps(dict(args)))

    resdata['upload_files'].append('config.json')
    resdata['cache_path'] = cache_path
    resdata['coss3_path'] = coss3_path
    resdata['frame_count'] = all_cnt
    resdata['frame_rate'] = fps

    _send_progress(100)
    return resdata
