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
import functools

from sklearn.decomposition import PCA
from sklearn import preprocessing
from frepai.utils.easydict import DotDict
from frepai.utils.draw import get_rect_points, get_ploy_points
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
NUM_FRAMES = 64
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

    out_path = args.pigeon['out_path']
    resdata['kstest_coss3_path'] = os.path.join('/', *out_path[8:].split('/')[1:])
    resdata['kstest_ecdfs_path'] = kstest_ecdfs_path
    resdata['pcaks'] = out_path
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

    if video_path is not None and 'https://' in video_path:
        segs = video_path[8:].split('/')
        vname = segs[-1].split('.')[0]
        coss3_path = os.path.join('/', *segs[1:3], 'outputs', vname, 'repnet_tf')
        coss3_delete(coss3_path[1:], logger)
    else:
        vname = 'unknow'
        coss3_path = ''

    cache_path = f'/data/cache/{int(time.time() * 1000)}/{vname}'
    mkdir_p(cache_path)
    resdata['cache_path'] = cache_path
    resdata['coss3_path'] = ''

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
    focus_pts = args.get('focus_pts', None)
    if focus_pts is not None:
        pts = get_ploy_points(width, height, focus_pts)
        img_bg = np.zeros((height, width, 3), dtype=np.uint8)
        img_bg = cv2.polylines(img_bg, [pts], True, (0, 255, 0), 1)
        focus_pts_mask = cv2.fillPoly(img_bg, [pts], (255, 255, 255))
        focus_box = *np.min(pts, axis=0), *np.max(pts, axis=0)
    else:
        focus_box = get_rect_points(width, height, args.focus_box)

    if focus_box is not None:
        focus_x1, focus_y1, focus_x2, focus_y2 = focus_box
        w = focus_x2 - focus_x1
        h = focus_y2 - focus_y1

    check_box = args.get('check_box', None)
    if check_box is not None:
        check_x1, check_y1, check_x2, check_y2 = get_rect_points(width, height, check_box)

    logger.info(f'{focus_box}, {black_box}, {check_box}, {focus_pts}')

    def _get_box_frame(img):# {{{
        vbox, cbox = None, None
        if focus_pts is not None:
            img = cv2.bitwise_and(img, focus_pts_mask)
        if black_box is not None:
            img[black_y1:black_y2, black_x1:black_x2, :] = 0
        if focus_box is not None:
            vbox = img[focus_y1:focus_y2, focus_x1:focus_x2, :]
        if check_box is not None:
            cbox = img[check_y1:check_y2, check_x1:check_x2, :]
        return vbox, cbox# }}}

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
    global_enhance_funcs = []
    global_remove_shadow = args.get('global_remove_shadow', None)
    grap_speed = args.get('global_grap_speed', -1)
    if grap_speed > 1:
        global_grap_step = grap_speed
    else:
        global_grap_step = int(fps * args.get('global_grap_interval', -1))
    global_blur_type = args.get('global_blur_type', 'none')
    global_filter_kernel = args.get('global_filter_kernel', 3)
    global_repnet_smooth = args.get('global_repnet_smooth', False)
    global_lowest_bright = args.get('global_lowest_bright', None)
    global_feature_select = args.get('global_feature_select', 'mean')
    global_feature_minval = args.get('global_feature_minval', 10)
    global_feature_minnum = args.get('global_feature_minnum', 50)
    global_hdiff_rate = args.get('global_hdiff_rate', 1.0)
    global_bg_finding = args.get('global_bg_finding', False)
    global_mask_enhance = args.get('global_mask_enhance', None)
    if global_mask_enhance is not None and isinstance(global_mask_enhance, dict):
        for key, params in global_mask_enhance.items():
            if key == 'dilate':
                kernel = np.ones((params[0], params[0]), np.uint8)
                global_enhance_funcs.append(functools.partial(cv2.dilate, kernel=kernel, iterations=params[1]))
            elif key == 'erode':
                kernel = np.ones((params[0], params[0]), np.uint8)
                global_enhance_funcs.append(functools.partial(cv2.erode, kernel=kernel, iterations=params[1]))

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

    resdata['global_grap_step'] = global_grap_step
    resdata['video_path'] = video_path
    resdata['frame_count'] = cnt
    resdata['frame_count_all'] = all_cnt

    # global_bg_frame_ = [None, None]{{{
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
    #     return global_bg_frame_[1]}}}

    def _calc_feat(img, dtype=1):# {{{
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
        return 0.001 if feat == 0 else feat# }}}

    def _remove_shadow(img, dksize=3, bksize=5, normalize=True):# {{{
        result_planes = []
        bgr_planes = cv2.split(img)
        for plane in bgr_planes:
            img = cv2.dilate(plane, np.ones((dksize,dksize), np.uint8))
            img = cv2.medianBlur(img, bksize)
            img = 255 - cv2.absdiff(plane, img)
            if normalize:
                img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(img)
        return cv2.merge(result_planes)# }}}

    def _parse_color_range(color_range):# {{{
        hsv_range = []
        for item in color_range:
            h_, s_, v_ = [0, 255], [0, 255], [0, 255]
            for key, val in item.items():
                if key == 'h':
                    h_ = val
                elif key == 's':
                    s_ = val
                elif key == 'v':
                    v_ = val
            lo = np.array([h_[0], s_[0], v_[0]])
            hi = np.array([h_[1], s_[1], v_[1]])
            hsv_range.append((lo, hi))
        return hsv_range# }}}

    if args.rmstill_frame_enable:# {{{
        rmstill_rate_range = args.get('rmstill_rate_range', None)
        if rmstill_rate_range is None:
            area_rate_thres = args.get('rmstill_rate_threshold', 0.001)
            if area_rate_thres < 0:
                area_rate_thres = float(1) / (w * h)
            rmstill_rate_range = [area_rate_thres, 1.0]
            args['rmstill_rate_range'] = rmstill_rate_range
        rmstill_bin_threshold = args.get('rmstill_bin_threshold', 20)
        rmstill_brightness_norm = args.get('rmstill_brightness_norm', False)
        rmstill_area_mode = args.get('rmstill_area_mode', 0)
        rmstill_area_range = [math.ceil(rmstill_rate_range[0] * area), math.ceil(rmstill_rate_range[1] * area)]

        if area < SMALL_AREA_THRESH:
            rmstill_white_thres = int(args.get('rmstill_white_rate', 0.1) * area)
            rmstill_white_window = args.get('rmstill_white_window', 10)
            rmstill_white_buffer = np.zeros((rmstill_white_window, ))
            frames_invalid = True
        resdata['rmstill_area_range'] = rmstill_area_range
        resdata['avg_pred_score'] = args.get('avg_pred_score', 0.2)
        resdata['within_period_threshold'] = args.get('within_period_threshold', 0.5)

        logger.info(f'rmstill: ({area}, {rmstill_area_range}, {rmstill_bin_threshold})')
# }}}
    elif args.color_tracker_enable:# {{{
        color_pre_count = 0
        color_hsv_range = []
        color_select = args.get('color_select', 8)
        color_select_range = args.get('color_select_range', [])
        if color_select < 0 or len(color_select_range) > 0:
            for item in color_select_range:
                h_, s_, v_ = [0, 255], [0, 255], [0, 255]
                for key, val in item.items():
                    if key == 'h':
                        h_ = val
                    elif key == 's':
                        s_ = val
                    elif key == 'v':
                        v_ = val
                lo = np.array([h_[0], s_[0], v_[0]])
                hi = np.array([h_[1], s_[1], v_[1]])
                color_hsv_range.append((lo, hi))
        else:
            if color_select == 0:
                color_hsv_range.append((lower_red_1, upper_red_1))
                color_hsv_range.append((lower_red_2, upper_red_2))
            elif color_select == 1:
                color_hsv_range.append((lower_orange, upper_orange))
            elif color_select == 2:
                color_hsv_range.append((lower_yellow, upper_yellow))
            elif color_select == 3:
                color_hsv_range.append((lower_green, upper_green))
            elif color_select == 4:
                color_hsv_range.append((lower_cyan, upper_cyan))
            elif color_select == 5:
                color_hsv_range.append((lower_blue, upper_blue))
            elif color_select == 6:
                color_hsv_range.append((lower_purple, upper_purple))
            elif color_select == 7:
                color_hsv_range.append((lower_black, upper_black))
            elif color_select == 8:
                color_hsv_range.append((lower_white, upper_white))
            elif color_select == 9:
                color_hsv_range.append((lower_gray, upper_gray))
        logger.info(f'{color_hsv_range}')
        assert len(color_hsv_range) > 0

        color_enhance_blur = args.get('color_enhance_blur', 0)
        color_rate_range = args.get('color_rate_range', None)
        if color_rate_range is None:
            color_rate_threshold = args.get('color_rate_threshold', 0.9)
            color_rate_range = [color_rate_threshold, 1.0]
            args['color_rate_range'] = color_rate_range
        color_area_range = [math.ceil(color_rate_range[0] * area), math.ceil(color_rate_range[1] * area)]

        color_buffer_size = args.get('color_buffer_size', 12)
        color_lower_rate = args.get('color_lower_rate', 0.2)
        color_upper_rate = args.get('color_upper_rate', 0.8)
        color_track_direction = args.get('color_track_direction', 0)
        color_buffer = np.zeros((color_buffer_size, ))
        if color_track_direction > 0:
            color_direction_buffer = np.zeros_like(color_buffer)
        color_lower_value = int(color_buffer_size * color_lower_rate)
        color_upper_value = int(color_buffer_size * color_upper_rate)
        logger.info(f'color_tracker: ({color_area_range}, {color_lower_value}, {color_upper_value})')
        resdata['color_area_range'] = color_area_range
        resdata['avg_pred_score'] = args.get('avg_pred_score', 0.2)
        resdata['within_period_threshold'] = args.get('within_period_threshold', 0.5)
        resdata['color_lower_value'] = color_lower_value
        resdata['color_upper_value'] = color_upper_value
# }}}
    elif args.stdwave_tracker_enable:# {{{
        stdwave_sub_average = args.get('stdwave_sub_average', True)
        stdwave_sigma_count = args.get('stdwave_sigma_count', 3.0)
        seconds = args.get('stdwave_window_secs', 0)
        stdwave_window_size = int(fps * seconds) if seconds > 0 else args.get('stdwave_window_size', 50)
        seconds = args.get('stdwave_distance_secs', 0)
        stdwave_distance_size = int(fps * seconds) if seconds > 0 else args.get('stdwave_distance_size', 150)
        stdwave_minstd_thresh = args.get('stdwave_minstd_thresh', 0.5)
        if grap_speed > 0:
            stdwave_window_size = max(1, int(stdwave_window_size / grap_speed))
            stdwave_distance_size = max(1, int(stdwave_distance_size / grap_speed))
        stdwave_hsv_range = _parse_color_range(args.get('stdwave_color_select', []))
        stdwave_hsv_rate = args.get('stdwave_hsv_rate', 0.3)

        resdata['stdwave_sigma_count'] = stdwave_sigma_count
        resdata['stdwave_sub_average'] = stdwave_sub_average
        resdata['stdwave_window_size'] = stdwave_window_size
        resdata['stdwave_distance_size'] = stdwave_distance_size
        resdata['stdwave_minstd_thresh'] = stdwave_minstd_thresh
        logger.info(f'stdwave_tracker: ({stdwave_sigma_count}, {stdwave_window_size}, {stdwave_distance_size})')
# }}}
    elif args.featpeak_tracker_enbale:# {{{
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
# }}}
    elif args.direction_tracker_enable:# {{{
        direction_arrow = args.get('direction_arrow', 'lr') # lr, rl, tb, bt
        direction_agg_axis = 0 if direction_arrow in ('rl', 'lr') else 1
        resdata['direction_inverse'] = True if direction_arrow in ('rl', 'bt') else False
        resdata['direction_scale_threshold'] = args.get('direction_scale_threshold', 2.0)
        resdata['direction_window_size'] = args.get('direction_window_size', 10)
# }}}
    elif args.diffimpulse_tracker_enable:# {{{
        diffimpulse_one_threshold = int(args.get('diffimpulse_rate_threshold', 0.02) * area)
        diffimpulse_bin_threshold = args.get('diffimpulse_bin_threshold', 20)
        diffimpulse_window_size = args.get('diffimpulse_window_size', [7, 5])
        resdata['diffimpulse_one_threshold'] = diffimpulse_one_threshold
        resdata['diffimpulse_bin_threshold'] = diffimpulse_bin_threshold
        resdata['diffimpulse_window_size'] = diffimpulse_window_size
        logger.info(f'diff_impulse: ({diffimpulse_one_threshold}, {diffimpulse_bin_threshold}, {diffimpulse_window_size})')
# }}}
    tile_shuffle = args.get('input_tile_shuffle', False)

    brightvals, featdata, diffimpulse, featpeak = [], [], [], []
    keepframe, keepidxes, half_focus_width = [], [], -1
    if devmode:
        binframes, binpoints, contareas, colorvals, waverates = [], [], [], [], []
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

        frame_hsv = None
        if global_lowest_bright is not None:
            frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            _, _, v = cv2.split(frame_hsv)
            brightvals.append(np.max(v))

        if args.rmstill_frame_enable:# {{{
            if rmstill_brightness_norm:
                if frame_hsv is None:
                    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(frame_hsv)
                v = np.array((v - np.mean(v)) / np.std(v) * 32 + 127, dtype=np.uint8)
                frame_bgr = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
                frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            frame_tmp = cv2.absdiff(frame_gray, pre_frame_gray)
            frame_tmp = cv2.threshold(frame_tmp, rmstill_bin_threshold, 255, cv2.THRESH_BINARY)[1]
            for func in global_enhance_funcs:
                frame_tmp = func(frame_tmp)
            val = np.sum(frame_tmp == 255)
            if rmstill_area_mode == 0:
                if rmstill_area_range[0] < val < rmstill_area_range[1]:
                    keep_flag = True
                    if devmode:
                        frame_tmp = cv2.cvtColor(frame_tmp, cv2.COLOR_GRAY2RGB)
                        binframes.append(cv2.resize(frame_tmp, (INPUT_WIDTH, INPUT_HEIGHT)))
                        binpoints.append(val)
            elif rmstill_area_mode == 1:
                contours, _ = cv2.findContours(frame_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if len(contours) > 0:
                    contours = sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True)
                    area = cv2.contourArea(contours[0])
                    if rmstill_area_range[0] < area < rmstill_area_range[1]:
                        keep_flag = True
                        if devmode:
                            frame_tmp = cv2.cvtColor(frame_tmp, cv2.COLOR_GRAY2RGB)
                            cv2.drawContours(frame_tmp, [contours[0]], 0, (0, 0, 255), 3)
                            binframes.append(cv2.resize(frame_tmp, (INPUT_WIDTH, INPUT_HEIGHT)))
                            contareas.append(area)

            if frames_invalid and keep_flag:
                rmstill_white_buffer[-1] = val
                rmstill_white_buffer = np.roll(rmstill_white_buffer, shift=-1, axis=0)
                wpoint_mean = np.mean(rmstill_white_buffer)
                if wpoint_mean > rmstill_white_thres:
                    logger.info(f'wpoint_mean[{wpoint_mean}] vs rmstill_white_thres[{rmstill_white_thres}], {rmstill_white_window}')
                    frames_invalid = False
# }}}
        elif args.color_tracker_enable:# {{{
            color_mask = None
            if frame_hsv is None:
                frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            if color_enhance_blur > 0:
                frame_hsv = cv2.medianBlur(frame_hsv, color_enhance_blur)

            for lo, hi in color_hsv_range:
                mask = cv2.inRange(frame_hsv, lo, hi)
                color_mask = mask if color_mask is None else cv2.bitwise_or(color_mask, mask)

            for func in global_enhance_funcs:
                color_mask = func(color_mask)

            colorval = np.sum(color_mask == 255)
            if color_area_range[0] < colorval < color_area_range[1]:
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
                featdata.append(round(colorval / area, 3))
                if devmode:
                    colorvals.append([featdata[-1], val])
                    frame_tmp = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2RGB)
                    binframes.append(cv2.resize(frame_tmp, (INPUT_WIDTH, INPUT_HEIGHT)))
# }}}
        elif args.stdwave_tracker_enable:# {{{
            frame_tmp = cv2.absdiff(frame_gray, pre_frame_gray)
            feat = _calc_feat(frame_tmp)
            if pre_check_gray is not None:
                feat = max(0, feat - _calc_feat(cv2.absdiff(check_gray, pre_check_gray)))
            rate_ = 0
            if len(stdwave_hsv_range) > 0 and feat > global_feature_minval:
                if frame_hsv is None:
                    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
                ys, xs = np.where((frame_tmp > global_feature_minval))
                count = 0
                for y, x in zip(ys, xs):
                    hsv = frame_hsv[y, x]
                    for lo, hi in stdwave_hsv_range:
                        if (lo[0] < hsv[0] < hi[0]) and (lo[1] < hsv[1] < hi[1]) and \
                                (lo[2] < hsv[2] < hi[2]):
                            count += 1
                            break
                rate_ = round(count / len(xs), 3)
                if rate_ < stdwave_hsv_rate:
                    feat = 0
            if devmode:
                waverates.append(rate_)
            featdata.append(feat)
# }}}
        elif args.featpeak_tracker_enbale:# {{{
            frame_tmp = cv2.absdiff(frame_gray, pre_frame_gray)
            feat = _calc_feat(frame_tmp)
            feat = int(np.mean(feat))
            featpeak.append(feat)
# }}}
        elif args.direction_tracker_enable:# {{{
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
            featdata.append([mfeat if mfeat != 0 else 0.01, lfeat, rfeat])
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
# }}}
        elif args.diffimpulse_tracker_enable:# {{{
            frame_tmp = cv2.absdiff(frame_gray, pre_frame_gray)
            cc = np.sum(frame_tmp > diffimpulse_bin_threshold)
            if devmode:
                debug_data.append(cc)
            if cc > diffimpulse_one_threshold:
                diffimpulse.append(1)
            else:
                diffimpulse.append(0)
# }}}
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

    if global_lowest_bright is not None:
        if np.mean(brightvals) < global_lowest_bright:
            logger.warning(f'global_lowest_bright: {global_lowest_bright}')
            _send_progress(100, True)
            rmdir_p(os.path.dirname(cache_path))
            return None

    if args.stdwave_tracker_enable:
        featdata = np.asarray(featdata)
        if len(featdata[featdata > global_feature_minval]) < global_feature_minnum:
            logger.warning(f'global_feature_minval: {global_feature_minval}, {global_feature_minnum}')
            _send_progress(100, True)
            rmdir_p(os.path.dirname(cache_path))
            return None
        np.save(f'{cache_path}/stdwave_data.npy', featdata)
        if devmode:
            np.save(f'{cache_path}/stdwave_rates.npy', waverates)
    elif args.color_tracker_enable and global_feature_minval < 0:
        featdata = np.asarray(featdata)
        if len(featdata[featdata > global_feature_minval]) < global_feature_minnum:
            logger.warning(f'global_feature_minval: {global_feature_minval}, {global_feature_minnum}')
            _send_progress(100, True)
            rmdir_p(os.path.dirname(cache_path))
            return None
    elif args.direction_tracker_enable:
        featdata = np.asarray(featdata)
        features = featdata[:, 0]
        if len(features[features > global_feature_minval]) < global_feature_minnum:
            _send_progress(100, True)
            rmdir_p(os.path.dirname(cache_path))
            return None
        np.save(f'/data/direction_data.npy', featdata)
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

    keep_frame_count, fill_frame_count, fill_frame_idxes = len(keepframe), 0, None
    if keep_frame_count > 0:
        keepframes, keepidxes = np.asarray(keepframe), np.asarray(keepidxes)
        if global_repnet_smooth:
            fill_frame_count = NUM_FRAMES - keep_frame_count % NUM_FRAMES
            if fill_frame_count > 10:
                fill_frame_step = int(keep_frame_count / fill_frame_count)
                fill_frame_idxes = np.arange(fill_frame_step, keep_frame_count, fill_frame_step)
                fill_frame_idxes = fill_frame_idxes[:fill_frame_count]
                fill_diff_idxes = np.diff([0] + fill_frame_idxes.tolist() + [keep_frame_count])
                keepidxes = keepidxes + np.hstack([[i] * j for i, j in enumerate(fill_diff_idxes)])
                keepidxes = np.insert(keepidxes, fill_frame_idxes, keepidxes[fill_frame_idxes] - 1)
                keepframes = np.insert(keepframes, fill_frame_idxes, keepframes[fill_frame_idxes], axis=0)
                resdata['fill_frame_count'] = len(fill_frame_idxes)
                logger.info(f'frames: {keep_frame_count} {fill_frame_count}')
                np.save(f'{cache_path}/fillidxes.npy', fill_frame_idxes + np.arange(0, len(fill_frame_idxes)))
        np.savez_compressed(f'{cache_path}/keepframe.npz', x=keepframes)
        np.save(f'{cache_path}/keepidxes.npy', keepidxes)

    if devmode:
        if len(brightvals) > 0:
            brightvals = np.asarray(brightvals)
            if fill_frame_idxes is not None:
                brightvals = np.insert(brightvals, fill_frame_idxes, brightvals[fill_frame_idxes])
            np.save(f'{cache_path}/brightvals.npy', brightvals)
        if len(binpoints) > 0:
            binpoints = np.asarray(binpoints)
            if fill_frame_idxes is not None:
                binpoints = np.insert(binpoints, fill_frame_idxes, binpoints[fill_frame_idxes])
            np.save(f'{cache_path}/binpoints.npy', binpoints)
        if len(binframes) > 0:
            binframes = np.asarray(binframes)
            if fill_frame_idxes is not None:
                binframes = np.insert(binframes, fill_frame_idxes, binframes[fill_frame_idxes], axis=0)
            np.savez_compressed(f'{cache_path}/binframes.npz', x=binframes)
        if len(contareas) > 0:
            contareas = np.asarray(contareas)
            if fill_frame_idxes is not None:
                contareas = np.insert(contareas, fill_frame_idxes, contareas[fill_frame_idxes])
            np.save(f'{cache_path}/contareas.npy', contareas)
        if len(colorvals) > 0:
            colorvals = np.asarray(colorvals)
            if fill_frame_idxes is not None:
                colorvals = np.insert(colorvals, fill_frame_idxes, colorvals[fill_frame_idxes])
            np.save(f'{cache_path}/colorvals.npy', colorvals)

    with open(f'{cache_path}/config.json', 'w') as f:
        f.write(json.dumps(dict(args)))

    resdata['upload_files'].append('config.json')
    resdata['cache_path'] = cache_path
    resdata['coss3_path'] = coss3_path
    resdata['frame_rate'] = fps
    _send_progress(100)
    logger.info('preprocess end')
    return resdata
