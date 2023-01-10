#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file post.py
# @brief
# @author QRS
# @version 1.0
# @date 2022-03-16 20:41


import cv2
import numpy as np
import os
import io
import json # noqa
import pickle
import traceback
import matplotlib.pyplot as plt
from frepai.utils.easydict import DotDict
from frepai.utils.logger import EasyLogger as logger
from frepai.utils.errcodes import HandlerError
from frepai.utils.draw import (
        get_rect_points, get_ploy_points,
        draw_osd_sim, draw_hist_density)
from frepai.utils.oss import coss3_put, coss3_domain
from frepai.utils import rmdir_p


INPUT_WIDTH = 112
INPUT_HEIGHT = 112
NUM_FRAMES = 64

ffmpeg_args = '-preset ultrafast -vcodec libx264 -pix_fmt yuv420p'


def _get_videos_samples(video_path, focus_box, black_box, sumcnt, size):# {{{
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HandlerError(83011, f'open video [{video_path}] err!')
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    focus_box = get_rect_points(width, height, focus_box)
    black_box = get_rect_points(width, height, black_box)
    if black_box is not None:
        bx1, by1, bx2, by2 = black_box
    if focus_box is not None:
        fx1, fy1, fx2, fy2 = focus_box

    if size < sumcnt:
        frames_indexes = np.sort(np.random.choice(sumcnt, size, replace=False))
    frames = []
    cap_index = -1
    while len(frames) < size:
        success, frame_bgr = cap.read()
        if not success:
            break
        cap_index += 1
        if size < sumcnt:
            if cap_index != frames_indexes[len(frames)]:
                continue
        if black_box is not None:
            frame_bgr[by1:by2, bx1:bx2, :] = 0
        if focus_box is not None:
            frame_bgr = frame_bgr[fy1:fy2, fx1:fx2, :]
        frames.append(cv2.resize(frame_bgr, (INPUT_WIDTH, INPUT_HEIGHT)))
    cap.release()

    return frames, width, height, fps# }}}


def _post_kstest(pigeon, progress_cb):# {{{
    progress_cb(50)
    kstest_ecdfs_path = pigeon['kstest_ecdfs_path']
    kstest_coss3_path = pigeon['kstest_coss3_path']

    prefix_map = [kstest_ecdfs_path, kstest_coss3_path]
    coss3_put(kstest_ecdfs_path, prefix_map)
    progress_cb(100)
    return None# }}}


def _post_featpeak(pigeon, args, progress_cb):# {{{
    devmode, cache_path, coss3_path = pigeon['devmode'], pigeon['cache_path'], pigeon['coss3_path']
    video_path = pigeon['video_path']
    progress_cb(10)
    spf = 1 / pigeon['frame_rate']
    all_frames_count = pigeon['frame_count_all']
    featpeak_indexes = np.load(f'{cache_path}/featpeak_indexes.npy')

    SLEN, c = len(featpeak_indexes), 0
    progress_cb(30)
    json_result = {}
    json_result['num_frames'] = all_frames_count
    json_result['fps'] = 1
    frames_info = []
    if SLEN > 0:
        for i in range(all_frames_count):
            if i % pigeon['frame_rate'] == 0:
                if c < SLEN and i > featpeak_indexes[c]:
                    c += 1
                frames_info.append({
                    'image_id': '%d.jpg' % i,
                    'at_time': round((i + 1) * spf, 3),
                    'cum_counts': c * args.reg_factor})
    else:
        frames_info = [{'image_id': '0.jpg', 'at_time': 0, 'cum_counts': 0}]

    json_result['frames_period'] = frames_info
    pigeon['sumcnt'] = frames_info[-1]['cum_counts']

    progress_cb(30)

    if devmode and SLEN > 0:
        featpeak_window_size = pigeon['featpeak_window_size']
        featpeak_data_normal = pigeon['featpeak_data_normal']
        featpeak_distance_size = pigeon['featpeak_distance_size']
        featpeak_min_threshold = pigeon['featpeak_min_threshold']
        featpeak_relative_height = pigeon['featpeak_relative_height']
        featpeak_height_minmax = pigeon['featpeak_height_minmax']
        featpeak_width_minmax = pigeon['featpeak_width_minmax']
        featpeak_prominence_minmax = pigeon['featpeak_prominence_minmax']

        featpeak_data = np.load(f'{cache_path}/featpeak_data.npy')
        if featpeak_data_normal:
            featpeak_post = np.load(f'{cache_path}/featpeak_post.npy')
        else:
            featpeak_post = featpeak_data
        with open(f'{cache_path}/featpeak_props.pkl', 'rb') as fr:
            featpeak_props = pickle.load(fr)

        progress_cb(35)

        widths, prominences, half_wlen, ss = None, None, 0, featpeak_props['ss']
        if featpeak_window_size > 2:
            half_wlen = featpeak_window_size // 2
        if 'prominences' in featpeak_props:
            prominences = featpeak_props['prominences']
        if 'widths' in featpeak_props:
            widths = featpeak_props['widths']
            width_heights = featpeak_props['width_heights']
            left_ips = featpeak_props['left_ips']
            right_ips = featpeak_props['right_ips']
        M = 1500
        featpeak_data = np.hstack([featpeak_data, [featpeak_data[-1]] * (M - len(featpeak_data) % M)])
        if featpeak_data_normal:
            featpeak_post = np.hstack([featpeak_post, [featpeak_post[-1]] * (M - len(featpeak_post) % M)])
        N = len(featpeak_post)
        images = []
        for i in range(0, N, M):
            ys = featpeak_post[i:i + M]
            mm = i + M # len(ys)
            xs = range(i, mm)
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(120, 8), sharex=True, tight_layout=False)
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
            plt.xlim(i, mm)
            axes[0].scatter(xs, ys)
            axes[1].plot(xs, featpeak_data[i:mm])

            indices = np.where((i < featpeak_indexes) & (featpeak_indexes < mm))[0]
            peaks = featpeak_indexes[indices]
            axes[0].plot(peaks, featpeak_post[peaks], "x")
            if prominences is not None:
                axes[0].vlines(x=peaks, ymin=featpeak_post[peaks] - featpeak_prominence_minmax[0], ymax=featpeak_post[peaks], color='r', linewidth=2)
                axes[0].vlines(x=peaks, ymin=featpeak_post[peaks] - prominences[indices], ymax=featpeak_post[peaks], color='g', linewidth=1)
            if widths is not None:
                axes[0].hlines(y=width_heights[indices], xmin=left_ips[indices], xmax=right_ips[indices], color='y', linewidth=2)
            if featpeak_distance_size > 0:
                axes[0].hlines(y=featpeak_post[peaks], xmin=peaks, xmax=peaks + featpeak_distance_size, color='b')
            if featpeak_window_size > 2:
                axes[0].hlines(y=featpeak_post[peaks], xmin=peaks - half_wlen, xmax=peaks + half_wlen, color='y', linewidth=2)
            if featpeak_min_threshold > 0:
                axes[0].axhline(y=featpeak_min_threshold, color='m', marker='o', linestyle='-', linewidth=1)
            for j in range(peaks.shape[0]):
                peak = peaks[j]
                peak_height = featpeak_post[peak]
                axes[0].text(peak, peak_height, f'{peak},{round(peak_height)}')
                axes[1].text(peak, featpeak_data[peak], f'{peak},{featpeak_data[peak]}')
                if prominences is not None:
                    peak_prominence = prominences[indices][j]
                    axes[0].text(peak + 1, peak_height - 0.5 * peak_prominence, f'{peak_prominence}')
                    axes[0].text(peak + 1, peak_height - 0.5 * peak_prominence, f'{peak_prominence}')
                if widths is not None:
                    peak_width = widths[indices][j]
                    l_x, r_x = left_ips[indices][j], right_ips[indices][j]
                    w_y = width_heights[indices][j]
                    axes[0].text(0.5 * (l_x + r_x) - 3, w_y + 2, f'{round(peak_width)}, {round(w_y)}')
                    axes[0].text(l_x, w_y + 2, f'{int(l_x)}')
                    axes[0].text(r_x, w_y + 2, f'{int(r_x)}')
            with io.BytesIO() as buff:
                fig.savefig(buff, format='raw')
                buff.seek(0)
                image = np.frombuffer(buff.getvalue(), dtype=np.uint8)
            imgw, imgh = fig.canvas.get_width_height()
            imgw, imgh = int(imgw), int(imgh)
            image = image.reshape((imgh, imgw, -1))
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            images.append(image)
            plt.close(fig)

            cv2.imwrite(f'{cache_path}/featpeak_{i}.jpg', image)
            pigeon['upload_files'].append(f'featpeak_{i}.jpg')
            pigeon[f'featpeak_image_{i}'] = f'{coss3_domain}{coss3_path}/featpeak_{i}.jpg'

        bimage = np.hstack(images)
        bwidth = bimage.shape[1]

        progress_cb(65)
        tmp_video_file = f'{cache_path}/_featpeak.mp4'
        frames, width, height, fps = _get_videos_samples(video_path,
                args.focus_box, args.black_box, all_frames_count, bwidth)
        writer = cv2.VideoWriter(tmp_video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        progress_cb(75)

        fontscale, th = 0.6 if height < 500 else 2, int(0.08 * height)
        frames.extend([frames[-1]] * (len(featpeak_post) - len(frames)))
        F, J = len(frames), len(frames_info)
        logger.info(f'{N}, {F}')
        limage = 255 * np.ones((bimage.shape[0], width, bimage.shape[2]), dtype=np.uint8)
        bimage = np.hstack([bimage, limage])
        for i in range(0, bwidth, fps):
            progress = i / bwidth
            img = bimage[:, i: i + width]
            img = cv2.resize(img, (width, height))
            img[height - INPUT_HEIGHT - 5:height - 5, 5:INPUT_WIDTH + 5, :] = frames[int(F * progress)]
            cv2.putText(img,
                    '%dX%d %.1f C:%.1f/%.1f W:%d D:%d R:%.2f H:%s' % (
                        width, height, fps,
                        frames_info[int(J * progress)]['cum_counts'], pigeon['sumcnt'],
                        featpeak_window_size, featpeak_distance_size, featpeak_relative_height,
                        str(featpeak_height_minmax)),
                    (2, int(0.62 * height)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontscale,
                    (0, 0, 0), 1)
            cv2.putText(img,
                    " W:%s P:%s S:%s" % (
                        str(featpeak_width_minmax),
                        str(featpeak_prominence_minmax),
                        str(ss)),
                    (INPUT_WIDTH + 12, height - int(th * 0.35)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontscale,
                    (0, 0, 0), 1)
            writer.write(img)
        writer.release()

        progress_cb(90)
        os.system(f'ffmpeg -an -i {tmp_video_file} {ffmpeg_args} {cache_path}/target-stride.mp4 2>/dev/null')
        pigeon['upload_files'].append('target-stride.mp4')
        pigeon['stride_mp4'] = f'{coss3_domain}{coss3_path}/target-stride.mp4'
        json_result['stride_mp4'] = pigeon['stride_mp4']

    progress_cb(95)
    with open(f'{cache_path}/result.json', 'w') as fw:
        json.dump(json_result, fw, indent=4)
    pigeon['upload_files'].append('result.json')
    pigeon['target_json'] = f'{coss3_domain}{coss3_path}/result.json'

    progress_cb(96)
    return pigeon# }}}


def _post_stdwave(pigeon, args, progress_cb):# {{{
    devmode, cache_path, coss3_path = pigeon['devmode'], pigeon['cache_path'], pigeon['coss3_path']
    video_path = pigeon['video_path']
    progress_cb(10)
    spf = 1 / pigeon['frame_rate']
    all_frames_count = pigeon['frame_count_all']
    stdwave_indexes = np.load(f'{cache_path}/stdwave_indexes.npy').tolist()
    SLEN, c = len(stdwave_indexes), 0
    progress_cb(30)
    json_result = {}
    json_result['num_frames'] = all_frames_count
    json_result['fps'] = 1
    frames_info = []
    grap_step = pigeon.get('global_grap_step', 1)
    if SLEN > 0:
        for i in range(all_frames_count):
            if i % pigeon['frame_rate'] == 0:
                if c < SLEN and i > grap_step * stdwave_indexes[c]:
                    c += 1
                frames_info.append({
                    'image_id': '%d.jpg' % i,
                    'at_time': round((i + 1) * spf, 3),
                    'cum_counts': c * args.reg_factor})
    else:
        frames_info = [{'image_id': '0.jpg', 'at_time': 0, 'cum_counts': 0}]

    json_result['frames_period'] = frames_info
    pigeon['sumcnt'] = frames_info[-1]['cum_counts']

    progress_cb(50)
    if devmode and SLEN > 0:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HandlerError(83011, f'open video [{video_path}] err!')

        brightvals, BV = [], -1
        if os.path.exists(f'{cache_path}/brightvals.npy'):
            brightvals = np.load(f'{cache_path}/brightvals.npy')
            BV = np.mean(brightvals)

        fps = round(cap.get(cv2.CAP_PROP_FPS))
        cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f'{video_path}:{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}')

        bg_focus_secs, bg_check_secs = pigeon['global_bg_focus'] / fps, pigeon['global_bg_check'] / fps
        bg_focus_str = '%02d:%02d' % (int(bg_focus_secs / 60), bg_focus_secs % 60)
        bg_check_str = '%02d:%02d' % (int(bg_check_secs / 60), bg_check_secs % 60)
        stdwave_sigma_count = pigeon['stdwave_sigma_count']
        stdwave_window_size = pigeon['stdwave_window_size']
        stdwave_distance_size = pigeon['stdwave_distance_size']
        mean, std, dd = pigeon['stdwave_mean'], pigeon['stdwave_std'], pigeon['stdwave_dd']

        tmp_video_file = f'{cache_path}/_stdwave.mp4'
        stdwave_data = np.load(f'{cache_path}/stdwave_data.npy')
        stdwave_post = np.load(f'{cache_path}/stdwave_post.npy')
        stdwave_rate = np.load(f'{cache_path}/stdwave_rates.npy')
        N, T = len(stdwave_data), pigeon['stdwave_threshold']
        figwidth = min(120, int(all_frames_count / 100))
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(figwidth, 8), sharex=True, tight_layout=False)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
        plt.xlim(0, N)
        axes[0].scatter(range(N), stdwave_post)
        axes[0].axhline(y=T, color='r', marker='o', linestyle='-', linewidth=5)
        axes[0].axhline(y=mean, color='g', marker='o', linestyle='-', linewidth=5)
        axes[0].axhline(y=std, color='b', marker='o', linestyle='-', linewidth=5)
        for i in range(0, N - stdwave_distance_size, 2 * stdwave_distance_size):
            axes[0].plot((i, i + stdwave_window_size), (T, T), 'bo-', linewidth=10)
            axes[0].plot((i, i + stdwave_distance_size), (mean, mean), 'bo-', linewidth=10)
        axes[1].scatter(range(N), stdwave_data)
        progress_cb(60)

        with io.BytesIO() as buff:
            fig.savefig(buff, format='raw')
            buff.seek(0)
            image = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        imgw, imgh = fig.canvas.get_width_height()
        imgw, imgh = int(imgw), int(imgh)
        image = image.reshape((imgh, imgw, -1))
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        plt.close(fig=fig)

        cv2.imwrite(f'{cache_path}/stdwave.jpg', image)
        pigeon['upload_files'].append('stdwave.jpg')
        pigeon['stdwave_image'] = f'{coss3_domain}{coss3_path}/stdwave.jpg'

        progress_cb(65)

        focus_pts = args.get('focus_pts', None)
        if focus_pts is not None:
            pts = get_ploy_points(width, height, focus_pts)
            focus_pts_mask = cv2.fillPoly(np.zeros((height, width, 3), dtype=np.uint8), [pts], (255, 255, 255))
            focus_box = *np.min(pts, axis=0), *np.max(pts, axis=0)
        else:
            focus_box = get_rect_points(width, height, args.focus_box)

        black_box = get_rect_points(width, height, args.black_box)
        if black_box is not None:
            bx1, by1, bx2, by2 = black_box
        if focus_box is not None:
            fx1, fy1, fx2, fy2 = focus_box

        def _get_box_frame(img):
            if focus_pts is not None:
                img = cv2.bitwise_and(img, focus_pts_mask)
            if black_box is not None:
                img[by1:by2, bx1:bx2, :] = 0
            if focus_box is not None:
                img = img[fy1:fy2, fx1:fx2, :]
            return img

        image = cv2.resize(image, (int(height * imgw / imgh), height), interpolation=cv2.INTER_LINEAR)
        window = width # int(0.3 * image.shape[1])

        fontscale = 0.7 if height < 500 else 2

        frames, sum_counts, frames_indexes = [], [], None
        if image.shape[1] < cnt:
            frames_indexes = np.sort(np.random.choice(cnt, image.shape[1], replace=False))
        else:
            image = cv2.resize(image, (cnt, height), interpolation=cv2.INTER_LINEAR)
        cap_index, cur_cnt = -1, 0
        while len(frames) < image.shape[1]:
            success, frame_bgr = cap.read()
            if not success:
                break
            cap_index += 1
            if frames_indexes is not None:
                if cap_index != frames_indexes[len(frames)]:
                    continue
            frame_bgr = _get_box_frame(frame_bgr)
            if len(stdwave_indexes) > 0 and cap_index >= stdwave_indexes[0]:
                cur_cnt += 1 * args.reg_factor
                stdwave_indexes.pop(0)
            sum_counts.append(cur_cnt)
            frame = cv2.resize(frame_bgr, (INPUT_WIDTH, INPUT_HEIGHT))
            cv2.putText(frame,
                    '%.3f %d' % (stdwave_rate[cap_index], stdwave_data[cap_index]),
                    (2, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontscale,
                    (255, 0, 0), 2)

            if BV > 0:
                cv2.putText(frame,
                        '%d,%d' % (brightvals[cap_index], BV),
                        (2, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontscale,
                        (255, 0, 0), 2)
            frames.append(frame)
        cap.release()

        progress_cb(70)
        wfps = fps
        writer = cv2.VideoWriter(tmp_video_file, cv2.VideoWriter_fourcc(*'mp4v'), wfps, (width, height))

        F = len(frames)
        bimg = 255 * np.ones((height, window, image.shape[2]), dtype=np.uint8)
        cv2.putText(bimg,
                'D: %s' % dd,
                (2, int(0.3 * height)),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontscale,
                (0, 0, 0), 2)

        bimage = np.hstack([image, bimg])
        logger.info(f'bimage shape: {bimage.shape} F: {F}')

        th = int(0.08 * height)
        for i in range(image.shape[1]):
            if (i + 1) % 211 == 0:
                progress_cb(70 + 19 * i / image.shape[1])
            img = bimage[:, i:i + window]
            img = cv2.resize(img, (width, height))
            if i < F:
                img[height - INPUT_HEIGHT - 5:height - 5, 5:INPUT_WIDTH + 5, :] = frames[i] # [:,:,::-1]
                cv2.putText(img,
                        '%dX%d %.1f F: %d C:%.1f/%.1f W:%d D:%d' % (
                            width, height, wfps, F,
                            sum_counts[i], sum_counts[-1],
                            stdwave_window_size,
                            stdwave_distance_size),
                        (2, int(0.06 * height)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontscale,
                        (0, 0, 0), 2)
                cv2.putText(img,
                        "N:%.2f S:%.3f T:%.2f %s %s" % (
                            stdwave_sigma_count, std, T,
                            bg_focus_str, '%s' % bg_check_str if bg_check_secs > 0 else ''),
                        (INPUT_WIDTH + 12, height - int(th * 0.35)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontscale,
                        (0, 0, 0), 2)
            writer.write(img)
        writer.release()
        progress_cb(90)
        os.system(f'ffmpeg -an -i {tmp_video_file} {ffmpeg_args} {cache_path}/target-stride.mp4 2>/dev/null')
        pigeon['upload_files'].append('target-stride.mp4')
        pigeon['stride_mp4'] = f'{coss3_domain}{coss3_path}/target-stride.mp4'
        json_result['stride_mp4'] = pigeon['stride_mp4']

    progress_cb(95)
    with open(f'{cache_path}/result.json', 'w') as fw:
        json.dump(json_result, fw, indent=4)
    pigeon['upload_files'].append('result.json')
    pigeon['target_json'] = f'{coss3_domain}{coss3_path}/result.json'

    progress_cb(96)
    return pigeon# }}}


def _post_diffimpulse(pigeon, args, progress_cb):# {{{
    devmode, cache_path, coss3_path = pigeon['devmode'], pigeon['cache_path'], pigeon['coss3_path']
    video_path = pigeon['video_path']
    progress_cb(10)
    spf = 1 / pigeon['frame_rate']
    all_frames_count = pigeon['frame_count_all']
    diffimpulse_indexes = np.load(f'{cache_path}/diffimpulse_indexes.npy').tolist()
    SLEN, c = len(diffimpulse_indexes), 0
    progress_cb(30)
    json_result = {}
    json_result['num_frames'] = all_frames_count
    json_result['fps'] = 1
    frames_info = []
    if SLEN > 0:
        for i in range(all_frames_count):
            if i % pigeon['frame_rate'] == 0:
                if c < SLEN and i > diffimpulse_indexes[c]:
                    c += 1
                frames_info.append({
                    'image_id': '%d.jpg' % i,
                    'at_time': round((i + 1) * spf, 3),
                    'cum_counts': c * args.reg_factor})
    else:
        frames_info = [{'image_id': '0.jpg', 'at_time': 0, 'cum_counts': 0}]

    json_result['frames_period'] = frames_info
    pigeon['sumcnt'] = frames_info[-1]['cum_counts']

    progress_cb(50)
    if devmode and SLEN > 0:
        diffimpulse_one_threshold = pigeon['diffimpulse_one_threshold']
        diffimpulse_bin_threshold = pigeon['diffimpulse_bin_threshold']
        thresh_zero, thresh_one = pigeon['diffimpulse_window_size']
        diffimpulse_0_1 = np.load(f'{cache_path}/diffimpulse_0_1.npy')
        one_cnt_list = np.load(f'{pigeon["cache_path"]}/diffimpulse_cnt_1.npy').tolist()
        one_cnt_list = [x for x in one_cnt_list if x > 3 and x < 30]

        N = len(diffimpulse_0_1)
        fig = plt.figure(figsize=(48, 8))
        plt.xlim(0, N)
        plt.scatter(range(N), diffimpulse_0_1)
        progress_cb(60)

        with io.BytesIO() as buff:
            fig.savefig(buff, format='raw')
            buff.seek(0)
            image = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        imgw, imgh = fig.canvas.get_width_height()
        imgw, imgh = int(imgw), int(imgh)
        image = image.reshape((imgh, imgw, -1))
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        cv2.imwrite(f'{cache_path}/diffimpulse.jpg', image)
        pigeon['upload_files'].append('diffimpulse.jpg')
        pigeon['diffimpulse_image'] = f'{coss3_domain}{coss3_path}/diffimpulse.jpg'

        tmp_video_file = f'{cache_path}/_diffimpulse.mp4'
        progress_cb(65)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HandlerError(83011, f'open video [{video_path}] err!')

        fps = round(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        focus_box = get_rect_points(width, height, args.focus_box)
        black_box = get_rect_points(width, height, args.black_box)
        if black_box is not None:
            bx1, by1, bx2, by2 = black_box
        if focus_box is not None:
            fx1, fy1, fx2, fy2 = focus_box

        image = cv2.resize(image, (int(height * imgw / imgh), height), interpolation=cv2.INTER_LINEAR)
        window = width # int(0.3 * image.shape[1])

        fontscale = 0.7 if height < 500 else 2

        frames, sum_counts, frames_indexes = [], [], None
        if image.shape[1] < all_frames_count:
            frames_indexes = np.sort(np.random.choice(all_frames_count, image.shape[1], replace=False))
        cap_index, cur_cnt = -1, 0
        while len(frames) < image.shape[1]:
            success, frame_bgr = cap.read()
            if not success:
                break
            cap_index += 1
            if frames_indexes is not None:
                if cap_index != frames_indexes[len(frames)]:
                    continue
            if black_box is not None:
                frame_bgr[by1:by2, bx1:bx2, :] = 0
            if focus_box is not None:
                frame_bgr = frame_bgr[fy1:fy2, fx1:fx2, :]
            if len(diffimpulse_indexes) > 0 and cap_index >= diffimpulse_indexes[0]:
                cur_cnt += 1 * args.reg_factor
                diffimpulse_indexes.pop(0)
            sum_counts.append(cur_cnt)
            frames.append(cv2.resize(frame_bgr, (INPUT_WIDTH, INPUT_HEIGHT)))
        cap.release()

        progress_cb(70)
        wfps = fps
        writer = cv2.VideoWriter(tmp_video_file, cv2.VideoWriter_fourcc(*'mp4v'), wfps, (width, height))

        F = len(frames)
        fig = plt.figure(figsize=(10, 8))
        plt.hist(one_cnt_list)
        with io.BytesIO() as buff:
            fig.savefig(buff, format='raw')
            buff.seek(0)
            bimg = np.frombuffer(buff.getvalue(), dtype=np.uint8)

        bimg = bimg.reshape(fig.canvas.get_width_height()[::-1] + (-1,))
        bimg = cv2.cvtColor(bimg,cv2.COLOR_RGB2BGR)
        bimg = cv2.resize(bimg, (window, height), interpolation=cv2.INTER_LINEAR)
        bimage = np.hstack([image, bimg])
        logger.info(f'bimage shape: {bimage.shape} F: {F}')

        th = int(0.08 * height)
        for i in range(image.shape[1]):
            if (i + 1) % 211 == 0:
                progress_cb(70 + 19 * i / image.shape[1])
            img = bimage[:, i:i + window]
            img = cv2.resize(img, (width, height))
            if i < F:
                img[height - INPUT_HEIGHT - 5:height - 5, 5:INPUT_WIDTH + 5, :] = frames[i] # [:,:,::-1]
                cv2.putText(img,
                        '%dX%d %.1f F: %d C:%.1f/%.1f' % (
                            width, height, wfps, F,
                            sum_counts[i], sum_counts[-1]),
                        (2, int(0.06 * height)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontscale,
                        (0, 0, 0), 2)
                cv2.putText(img,
                        "A:%d B:%d Z:%d O:%d" % (
                            diffimpulse_one_threshold,
                            diffimpulse_bin_threshold,
                            thresh_zero, thresh_one),
                        (INPUT_WIDTH + 12, height - int(th * 0.35)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontscale,
                        (0, 0, 0), 2)
            writer.write(img)
        writer.release()
        progress_cb(90)
        os.system(f'ffmpeg -an -i {tmp_video_file} {ffmpeg_args} {cache_path}/target-stride.mp4 2>/dev/null')
        pigeon['upload_files'].append('target-stride.mp4')
        pigeon['stride_mp4'] = f'{coss3_domain}{coss3_path}/target-stride.mp4'
        json_result['stride_mp4'] = pigeon['stride_mp4']

    progress_cb(95)
    with open(f'{cache_path}/result.json', 'w') as fw:
        json.dump(json_result, fw, indent=4)
    pigeon['upload_files'].append('result.json')
    pigeon['target_json'] = f'{coss3_domain}{coss3_path}/result.json'

    progress_cb(96)
    return pigeon# }}}


def _post_repnet(pigeon, args, progress_cb):# {{{
    devmode, cache_path, coss3_path = pigeon['devmode'], pigeon['cache_path'], pigeon['coss3_path']
    video_path = pigeon['video_path']
    keepidxes = np.load(f'{cache_path}/keepidxes.npy')
    with open(f'{cache_path}/engine.pkl', 'rb') as r:
        engine = pickle.load(r)

    grap_step = pigeon['global_grap_step']
    fill_frame_count = pigeon.get('fill_frame_count', 0)
    all_frames_count = pigeon['frame_count'] + fill_frame_count
    valid_frames_count = len(keepidxes)
    still_frames_count = all_frames_count - valid_frames_count

    is_still_frames = [False] * all_frames_count
    final_within_period = [.0] * all_frames_count
    final_per_frame_counts = [.0] * all_frames_count

    i, j = 0, 0
    for k in range(all_frames_count):
        if i < valid_frames_count and k == keepidxes[i]:
            final_within_period[k] = engine['within_period'][i]
            final_per_frame_counts[k] = engine['per_frame_counts'][i]
            i += 1
        elif j < still_frames_count:
            is_still_frames[k] = True
            j += 1
        else:
            raise HandlerError(83003, 'frames count invalid: %d vs %d vs %d' % (i, j, k))

    within_period = final_within_period
    per_frame_counts = np.asarray(final_per_frame_counts, dtype=np.float)
    sum_counts_dev = np.cumsum(per_frame_counts)
    if args.reg_factor == 1:
        sum_counts = sum_counts_dev
    else:
        per_frame_counts = args.reg_factor * per_frame_counts
        sum_counts = np.cumsum(args.reg_factor * per_frame_counts)
    sum_counts = np.round(sum_counts, 3)
    logger.info(f'count: {sum_counts[-1]:.2} vs {np.sum(engine["per_frame_counts"]):.2}')
    real_val = round(float(sum_counts[-1]), 2)
    int_val = int(sum_counts[-1])
    pigeon['sumcnt'] = int_val if real_val - int_val < 0.1 else int_val + 1

    json_result = {}
    json_result['period'] = engine['pred_period']
    json_result['score'] = engine['pred_score']
    json_result['stride'] = engine['chosen_stride']
    json_result['fps'] = 1
    json_result['num_frames'] = all_frames_count
    frames_info = []
    spf = 1 / pigeon['frame_rate']
    for i, sc in enumerate(sum_counts if grap_step < 0 else np.repeat(sum_counts, grap_step, axis=0)):
        if i % pigeon['frame_rate'] == 0:
            frames_info.append({
                'at_time': round((i + 1) * spf, 3),
                'cum_counts': sc
            })
    else:
        frames_info.append({
            'at_time': round((i + 1) * spf, 3),
            'cum_counts': sum_counts[-1]
        })
    json_result['frames_period'] = frames_info

    if devmode:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HandlerError(83011, f'open video [{video_path}] err!')

        fps = round(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fmt = cv2.VideoWriter_fourcc(*'mp4v')
        if args.save_video:
            tmp4_name = 'target.mp4'
            tmp_tfile = os.path.join(cache_path, f'_{tmp4_name}')
            target_vid = cv2.VideoWriter(tmp_tfile, fmt, fps, (width, height))
        if args.best_stride_video:
            smp4_name = 'target-stride.mp4'
            tmp_sfile = os.path.join(cache_path, f'_{smp4_name}')
            stride_vid = cv2.VideoWriter(tmp_sfile, fmt, fps, (width, height))

        area = 1
        if args.osd_sims:
            embs_sims = np.load(f'{cache_path}/embs_sims.npy')

            focus_pts = args.get('focus_pts', None)
            if focus_pts is not None:
                focus_pts = get_ploy_points(width, height, focus_pts)
                focus_box = *np.min(focus_pts, axis=0), *np.max(focus_pts, axis=0)
            else:
                focus_box = get_rect_points(width, height, args.focus_box)
            black_box = get_rect_points(width, height, args.black_box)
            if black_box is not None:
                bx1, by1, bx2, by2 = black_box
            if focus_box is not None:
                fx1, fy1, fx2, fy2 = focus_box
                area = (fy2 - fy1) * (fx2 - fx1)
            pigeon['upload_files'].append('embs_feat.npy')
            # pigeon['upload_files'].append('embs_sims.npy')
            # pigeon['embs_sims'] = f'{coss3_domain}{coss3_path}/embs_sims.npy'
            pigeon['embs_feat'] = f'{coss3_domain}{coss3_path}/embs_feat.npy'
            pigeon['embs_sims'] = 'noused'
        else:
            black_box = None
            focus_box = None

        tsm_last_thresh = pigeon.get('tsm_last_threshold', 0.5)
        tsm_last_smooth = pigeon.get('tsm_last_smooth', False)
        bg_focus_secs = pigeon['global_bg_focus'] / fps
        if bg_focus_secs > 0:
            bg_focus_str = '%02d:%02d' % (int(bg_focus_secs / 60), bg_focus_secs % 60)
        else:
            bg_focus_str = ''

        bottom_text = '%s %s %s' % (
            bg_focus_str,
            'F:%.3f' % (float(valid_frames_count) / all_frames_count),
            'R:%.1f' % args.angle if args.angle else '',
        )

        if args['rmstill_frame_enable']:
            bottom_text += '%s|%s|%s' % (
                '%.3f,%.3f' % (args['rmstill_rate_range'][0], args['rmstill_rate_range'][1]),
                '%d' % args['rmstill_bin_threshold'],
                'T:%d,%.3f' % (tsm_last_smooth, tsm_last_thresh)
            )
        if args['color_tracker_enable']:
            bottom_text += ' %s|%s|%d|%d' % (
                '%.3f,%.3f' % (args['color_rate_range'][0], args['color_rate_range'][1]),
                '%.2f,%.2f' % (args['color_lower_rate'], args['color_upper_rate']),
                args['color_buffer_size'], args['color_track_direction']
            )

        idx, valid_idx = 0, 0
        th = int(0.08 * height)
        osd, osd_size, alpha = 0, int(width * 0.25), 0.5
        osd_blend, hist_blend = None, None
        keepframe = np.load(f'{cache_path}/keepframe.npz')['x']
        binframes, binpoints, contareas, colorvals, brightvals, fillidxes, BV = [], [], [], [], [], [], -1
        if os.path.exists(f'{cache_path}/binframes.npz'):
            binframes = np.load(f'{cache_path}/binframes.npz')['x']
        if os.path.exists(f'{cache_path}/colorvals.npy'):
            color_lower, color_upper = pigeon['color_lower_value'], pigeon['color_upper_value']
            colorvals = np.load(f'{cache_path}/colorvals.npy')
        if os.path.exists(f'{cache_path}/contareas.npy'):
            contareas = np.load(f'{cache_path}/contareas.npy')
        if os.path.exists(f'{cache_path}/brightvals.npy'):
            brightvals = np.load(f'{cache_path}/brightvals.npy')
            BV = np.mean(brightvals)
        if os.path.exists(f'{cache_path}/binpoints.npy'):
            binpoints = np.load(f'{cache_path}/binpoints.npy')
        if len(binpoints) > 0:
            rmstill_area_range = pigeon['rmstill_area_range']
            hist_blend = draw_hist_density(np.round(binpoints / rmstill_area_range[0], 2), 20, INPUT_WIDTH, INPUT_HEIGHT)
            cv2.putText(hist_blend,
                    '%d' % rmstill_area_range[0],
                    (int(0.1 * INPUT_WIDTH), int(0.5 * INPUT_HEIGHT)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 2)
        if fill_frame_count > 0:
            fillidxes = np.load(f'{cache_path}/fillidxes.npy').tolist()

        chosen_stride = engine['chosen_stride']
        avg_embs_score = engine['avg_embs_score']
        tsm_last_length = engine['tsm_last_length']
        feat_factors = engine['feat_factors']
        grap_speed = args.get('global_grap_speed', -1)
        while True:
            if len(fillidxes) > 0:
                if idx >= fillidxes[0]:
                    fillidxes.pop(0)
                else:
                    success, frame_raw = cap.read()
                    if not success:
                        break
                frame_bgr = frame_raw.copy()
            else:
                success, frame_bgr = cap.read()
                if not success:
                    break
            if black_box is not None:
                if args.black_overlay:
                    frame_bgr[by1:by2, bx1:bx2, :] = 0
                else:
                    cv2.rectangle(frame_bgr, (bx1, by1), (bx2, by2), (0, 0, 0), 2)
            cv2.rectangle(frame_bgr, (0, 0), (int(0.8 * width), th), (0, 0, 0), -1)
            cv2.rectangle(frame_bgr, (0, height - th), (width, height), (0, 0, 0), -1)
            try:
                if args.osd_sims and not is_still_frames[idx] \
                        and valid_idx % (chosen_stride * NUM_FRAMES) == 0:
                    logger.info(f'valid_idx: {valid_idx} idx: {idx} osd: {osd}')
                    osd_blend = draw_osd_sim(embs_sims[osd], osd_size)
                    if tsm_last_smooth and osd == (len(embs_sims) - 1):
                        if tsm_last_length > 0:
                            los = int((1 - tsm_last_length / NUM_FRAMES) * osd_size)
                            logger.info(f'------->{los}')
                            cv2.rectangle(osd_blend, (0, 0), (los, los), (0, 0, 0), 2)
                    if args.ef_is_send:
                        cv2.putText(osd_blend,
                                '%.2f %.2f %.2f' % (args.ef_alpha, args.ef_beta, args.ef_gamma),
                                (int(0.05 * osd_size), int(0.2 * osd_size)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 0, 0), 1)
                        cv2.putText(osd_blend,
                                '%.2f %.2f' % (feat_factors[osd][0], feat_factors[osd][1]),
                                (int(0.05 * osd_size), int(0.85 * osd_size)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 0, 0), 1)
                    cv2.putText(osd_blend,
                            '%d %.3f' % (osd, avg_embs_score[osd]),
                            (int(0.1 * osd_size), int(0.55 * osd_size)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 0, 0), 2)
                    osd += 1
                if osd_blend is not None:
                    frame_bgr[th:osd_size + th, width - osd_size:, :] = osd_blend
            except Exception:
                logger.error(traceback.format_exc(limit=6))
            cv2.putText(frame_bgr,
                    '%d %.1f %d %.1f/%.1f %s %s %s' % (width,
                        fps, chosen_stride, sum_counts_dev[idx], sum_counts_dev[-1],
                        '%d' % fill_frame_count if fill_frame_count > 0 else '',
                        'V:%d' % grap_speed if grap_speed > 0 else '',
                        'P:%.2f/%.2f' % (within_period[idx], engine['pred_score'])),
                    (2, int(0.06 * height)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7 if height < 500 else 2,
                    (255, 255, 255), 2)

            if focus_box is not None:
                cv2.rectangle(frame_bgr, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
            if focus_pts is not None:
                frame_bgr = cv2.polylines(frame_bgr, [focus_pts], True, (255, 0, 0), 2)

            if args.osd_sims and valid_idx < valid_frames_count and valid_idx < len(binframes):
                frame_bgr[height - INPUT_HEIGHT - 10:, :INPUT_WIDTH + 10, :] = 222
                frame_bgr[height - INPUT_HEIGHT - 5:height - 5, 5:INPUT_WIDTH + 5, :] = keepframe[valid_idx][:,:,::-1]
                if hist_blend is None:
                    frame_bgr[th:INPUT_HEIGHT + th, 5:INPUT_WIDTH + 5, :] = binframes[valid_idx]
                    if len(contareas) > 0:
                        cv2.putText(frame_bgr,
                                '%.2f' % round(contareas[valid_idx] / area, 2),
                                (15, th + int(0.4 * INPUT_HEIGHT)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 0, 0), 2)
                    if len(colorvals) > 0:
                        cv2.putText(frame_bgr,
                                '%.2f %.2f' % (
                                    round(colorvals[valid_idx][0], 2),
                                    round(colorvals[valid_idx][1] / args['color_buffer_size'], 2)),
                                (6, th + int(0.4 * INPUT_HEIGHT)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 0, 0), 2)
                        cv2.putText(frame_bgr,
                                '%02d %02d %02d' % (color_lower, colorvals[valid_idx][1], color_upper),
                                (6, th + int(0.8 * INPUT_HEIGHT)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 0, 0), 2)
                else:
                    frame_bgr[th:INPUT_HEIGHT + th, 5:INPUT_WIDTH + 5, :] = cv2.addWeighted(
                            hist_blend, alpha,
                            binframes[valid_idx], 1 - alpha,
                            0)
                    cv2.putText(frame_bgr,
                            '%d' % binpoints[valid_idx],
                            (int(0.1 * INPUT_WIDTH), th + int(0.2 * INPUT_HEIGHT)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 0, 0), 2)
                    cv2.putText(frame_bgr,
                            '%.5f' % (binpoints[valid_idx] / area),
                            (int(0.1 * INPUT_WIDTH), th + int(0.8 * INPUT_HEIGHT)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 0, 0), 2)

            cv2.putText(frame_bgr, bottom_text,
                    (INPUT_WIDTH + 12, height - int(th * 0.35)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7 if height < 500 else 2,
                    (255, 255, 255), 2)

            if BV > 0:
                cv2.putText(frame_bgr,
                        '%d,%d' % (brightvals[idx], BV),
                        (25, height - INPUT_HEIGHT - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0), 2)

            if idx % 281 == 0:
                progress_cb(82 * float(idx) / all_frames_count)

            if args.save_video:
                target_vid.write(frame_bgr)
            if args.best_stride_video and not is_still_frames[idx] \
                    and valid_idx % chosen_stride == 0:
                stride_vid.write(frame_bgr)
            if not is_still_frames[idx]:
                valid_idx += 1
            idx += 1
        cap.release()
        progress_cb(86)
        if args.save_video:
            target_vid.release()
            os.system(f'ffmpeg -an -i {tmp_tfile} {ffmpeg_args} {cache_path}/{tmp4_name} 2>/dev/null')
            pigeon['target_mp4'] = f'{coss3_domain}{coss3_path}/{tmp4_name}'
            json_result['target_mp4'] = pigeon['target_mp4']
            pigeon['upload_files'].append(tmp4_name)
        progress_cb(90)
        if args.best_stride_video:
            stride_vid.release()
            os.system(f'ffmpeg -an -i {tmp_sfile} {ffmpeg_args} {cache_path}/{smp4_name} 2>/dev/null')
            pigeon['stride_mp4'] = f'{coss3_domain}{coss3_path}/{smp4_name}'
            json_result['stride_mp4'] = pigeon['stride_mp4']
            pigeon['upload_files'].append(smp4_name)
        progress_cb(96)

    with open(f'{cache_path}/result.json', 'w') as fw:
        json.dump(json_result, fw, indent=4)
    pigeon['upload_files'].append('result.json')
    pigeon['target_json'] = f'{coss3_domain}{coss3_path}/result.json'

    del within_period, per_frame_counts

    return pigeon# }}}


def video_postprocess(pigeon, progress_cb=None):
    if 'cache_path' not in pigeon:
        raise HandlerError(83001, 'not found cache_path')

    cache_path, coss3_path = pigeon['cache_path'], pigeon['coss3_path']
    pigeon['task'] = 'post'

    if not os.path.isdir(cache_path):
        raise HandlerError(83002, f'cache_path[{cache_path}] cannot open!')

    def _send_progress(x):
        if progress_cb:
            pigeon['progress'] = round(60 + 0.4 * x, 2)
            progress_cb(pigeon)
            logger.info(f"{round(x, 2)} {pigeon['progress']}")

    if 'kstest_ecdfs_path' in pigeon:
        logger.warning(pigeon)
        _post_kstest(pigeon, _send_progress)
        rmdir_p(cache_path)
        return None

    with open(f'{cache_path}/config.json', 'r') as fr:
        args = DotDict(json.load(fr))

    plt.close('all')

    if 'stdwave_window_size' in pigeon:
        pigeon = _post_stdwave(pigeon, args, _send_progress)
    elif 'featpeak_window_size' in pigeon:
        pigeon = _post_featpeak(pigeon, args, _send_progress)
    elif 'diffimpulse_window_size' in pigeon:
        pigeon = _post_diffimpulse(pigeon, args, _send_progress)
    else:
        pigeon = _post_repnet(pigeon, args, _send_progress)

    if coss3_path:
        prefix_map = [cache_path, coss3_path]
        for fn in pigeon['upload_files']:
            coss3_put(f'{cache_path}/{fn}', prefix_map)

    logger.info(pigeon)

    _send_progress(100)

    rmdir_p(cache_path)
    return None
