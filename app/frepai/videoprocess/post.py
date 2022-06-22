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
from frepai.utils.draw import get_rect_points, draw_osd_sim, draw_hist_density
from frepai.utils.oss import coss3_put, coss3_domain
from frepai.utils import rmdir_p


INPUT_WIDTH = 112
INPUT_HEIGHT = 112
NUM_FRAMES = 64

ffmpeg_args = '-preset ultrafast -vcodec libx264 -pix_fmt yuv420p'


def _post_kstest(pigeon, progress_cb):
    progress_cb(50)
    kstest_ecdfs_path = pigeon['kstest_ecdfs_path']
    kstest_coss3_path = pigeon['kstest_coss3_path']

    prefix_map = [kstest_ecdfs_path, kstest_coss3_path]
    coss3_put(kstest_ecdfs_path, prefix_map)
    progress_cb(100)
    return None


def _post_stdwave(pigeon, args, progress_cb):
    devmode, cache_path, coss3_path = pigeon['devmode'], pigeon['cache_path'], pigeon['coss3_path']
    progress_cb(10)
    spf = 1 / pigeon['frame_rate']
    all_frames_count = pigeon['frame_count']
    stdwave_indexes = np.load(f'{cache_path}/stdwave_indexes.npy').tolist()
    progress_cb(30)
    json_result = {}
    json_result['num_frames'] = all_frames_count
    frames_info = [{'image_id': '0.jpg', 'at_time': 0, 'cum_counts': 0}]
    for c, i in enumerate(stdwave_indexes, 1):
        frames_info.append({
            'image_id': '%d.jpg' % i,
            'at_time': round((i + 1) * spf, 3),
            'cum_counts': c * args.reg_factor
        })
    json_result['frames_period'] = frames_info
    pigeon['sumcnt'] = len(stdwave_indexes)

    progress_cb(50)
    if devmode and len(stdwave_indexes) > 0:
        tmp_video_file = f'{cache_path}/_stdwave.mp4'
        stdwave_data = np.load(f'{cache_path}/stdwave_data.npy')
        stdwave_post = np.load(f'{cache_path}/stdwave_post.npy')
        N, T = len(stdwave_data), pigeon['stdwave_threshold']
        mean, std = pigeon['stdwave_mean'], pigeon['stdwave_std'] # noqa
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(48, 8), sharex=True, tight_layout=False)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
        plt.xlim(0, N)
        axes[0].scatter(range(N), stdwave_post)
        axes[0].plot((0, N), (T, T), 'ro-', linewidth=5)
        axes[0].plot((0, N), (mean, mean), 'go-', linewidth=5)
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
        window = int(0.3 * imgw)

        cv2.imwrite(f'{cache_path}/stdwave.jpg', image)
        pigeon['upload_files'].append('stdwave.jpg')
        pigeon['stdwave_image'] = f'{coss3_domain}{coss3_path}/stdwave.jpg'

        progress_cb(65)

        cap = cv2.VideoCapture(f'{cache_path}/source.mp4')
        if not cap.isOpened():
            raise HandlerError(83011, f'open video [{cache_path}/source.mp4] err!')

        fps = round(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        focus_box = get_rect_points(width, height, args.focus_box)
        black_box = get_rect_points(width, height, args.black_box)
        if black_box is not None:
            bx1, by1, bx2, by2 = black_box
        if focus_box is not None:
            fx1, fy1, fx2, fy2 = focus_box

        frames, sum_counts = [], []
        frames_indexes = np.sort(np.random.choice(all_frames_count, imgw, replace=False))
        cap_index, cur_cnt = -1, 0
        while len(frames) < imgw:
            success, frame_bgr = cap.read()
            if not success:
                break
            cap_index += 1
            if cap_index != frames_indexes[len(frames)]:
                continue
            if black_box is not None:
                frame_bgr[by1:by2, bx1:bx2, :] = 0
            if focus_box is not None:
                frame_bgr = frame_bgr[fy1:fy2, fx1:fx2, :]
            if len(stdwave_indexes) > 0 and cap_index >= stdwave_indexes[0]:
                cur_cnt += args.reg_factor
                stdwave_indexes.pop(0)
            sum_counts.append(cur_cnt)
            frames.append(cv2.resize(frame_bgr, (INPUT_WIDTH, INPUT_HEIGHT)))
        cap.release()

        progress_cb(70)

        writer = cv2.VideoWriter(tmp_video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        F = len(frames)
        bimg = np.zeros((imgh, window, image.shape[2]), dtype=np.uint8)

        cv2.putText(bimg,
                'Distances: %s' % pigeon['stdwave_dd'],
                (2, int(0.4 * height)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7 if height < 500 else 2,
                (255, 255, 255), 5)

        image = np.hstack([image, bimg])

        th = int(0.08 * height)
        for i in range(imgw):
            if (i + 1) % 211 == 0:
                progress_cb(70 + 19 * i / imgw)
            img = image[:, i:i + window]
            img = cv2.resize(img, (width, height))
            if i < F:
                img[height - INPUT_HEIGHT - 5:height - 5, 5:INPUT_WIDTH + 5, :] = frames[i] # [:,:,::-1]
                cv2.putText(img,
                        '%dX%d %.1f C:%.1f/%.1f' % (width, height, fps, sum_counts[i], sum_counts[-1]),
                        (2, int(0.06 * height)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7 if height < 500 else 2,
                        (0, 0, 0), 2)
                cv2.putText(img,
                        "S: %d W:%d D: %d" % (
                            pigeon['stdwave_sigma_count'],
                            pigeon['stdwave_window_size'],
                            pigeon['stdwave_distance_size']),
                        (INPUT_WIDTH + 12, height - int(th * 0.35)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7 if height < 500 else 2,
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
    return pigeon


def _post_repnet(pigeon, args, progress_cb):
    devmode, cache_path, coss3_path = pigeon['devmode'], pigeon['cache_path'], pigeon['coss3_path']

    keepidxes = np.load(f'{cache_path}/keepidxes.npy')
    with open(f'{cache_path}/engine.pkl', 'rb') as r:
        engine = pickle.load(r)

    all_frames_count = pigeon['frame_count']
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
    if args.reg_factor:
        per_frame_counts = args.reg_factor * per_frame_counts
    sum_counts = np.cumsum(per_frame_counts)
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
    for i, (in_period, p_count, is_still) in enumerate(zip(within_period, per_frame_counts, is_still_frames)):
        if i % pigeon['frame_rate'] == 0:
            frames_info.append({
                'image_id': '%d.jpg' % i,
                'at_time': round((i + 1) * spf, 3),
                'is_still': is_still,
                'within_period': in_period,
                'pframe_counts': p_count,
                'cum_counts': sum_counts[i]
            })
    else:
        frames_info.append({
            'image_id': '%d.jpg' % i,
            'at_time': round((i + 1) * spf, 3),
            'is_still': is_still,
            'within_period': in_period,
            'pframe_counts': p_count,
            'cum_counts': pigeon['sumcnt']
        })
    json_result['frames_period'] = frames_info

    del within_period, per_frame_counts

    if devmode:
        cap = cv2.VideoCapture(f'{cache_path}/source.mp4')
        if not cap.isOpened():
            raise HandlerError(83011, f'open video [{cache_path}/source.mp4] err!')

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

        if args.osd_sims:
            embs_sims = np.load(f'{cache_path}/embs_sims.npy')
            focus_box = get_rect_points(width, height, args.focus_box)
            black_box = get_rect_points(width, height, args.black_box)
            if black_box is not None:
                bx1, by1, bx2, by2 = black_box
            if focus_box is not None:
                fx1, fy1, fx2, fy2 = focus_box
            pigeon['upload_files'].append('embs_feat.npy')
            pigeon['embs_feat'] = f'{coss3_domain}{coss3_path}/embs_feat.npy'
        else:
            black_box = None
            focus_box = None

        bottom_text = '%s %s' % (
            'F:%.3f' % (float(valid_frames_count) / all_frames_count),
            'R:%.1f' % args.angle if args.angle else '',
        )

        if args['rmstill_frame_enable']:
            bottom_text += ' %s %s %s' % (
                'A:%.4f' % args['rmstill_rate_threshold'],
                'B:%d' % args['rmstill_bin_threshold'],
                'M:%.4f' % (float(1) / ((fy2 - fy1) * (fx2 - fx1)))
            )
        if args['color_tracker_enable']:
            bottom_text += ' %s %s %s %s' % (
                'A:%.2f' % args['color_rate_threshold'],
                'B:%d' % args['color_buffer_size'],
                'C:%.2f,%.2f' % (args['color_lower_rate'], args['color_upper_rate']),
                'D:%d' % args['color_track_direction']
            )

        idx, valid_idx = 0, 0
        th = int(0.08 * height)
        osd, osd_size, alpha = 0, int(width * 0.25), 0.5
        osd_blend, hist_blend = None, None
        keepframe = np.load(f'{cache_path}/keepframe.npz')['x']
        binframes, binpoints = [], []
        if os.path.exists(f'{cache_path}/binframes.npz'):
            binframes = np.load(f'{cache_path}/binframes.npz')['x']
        if os.path.exists(f'{cache_path}/binpoints.npy'):
            binpoints = np.load(f'{cache_path}/binpoints.npy')
        if len(binpoints) > 0:
            hist_blend = draw_hist_density(binpoints, 20, INPUT_WIDTH, INPUT_HEIGHT)

        chosen_stride = engine['chosen_stride']
        feat_factors = engine['feat_factors']

        while True:
            success, frame_bgr = cap.read()
            if not success:
                break
            if black_box is not None:
                if args.black_overlay:
                    frame_bgr[by1:by2, bx1:bx2, :] = 0
                else:
                    cv2.rectangle(frame_bgr, (bx1, by1), (bx2, by2), (0, 0, 0), 2)
                cv2.putText(frame_bgr,
                        '%d,%d' % (bx1, by1),
                        (bx1 + 2, by1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1)
                cv2.putText(frame_bgr,
                        '%d,%d' % (bx2, by2),
                        (bx2 - 65, by2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1)
            cv2.rectangle(frame_bgr, (0, 0), (int(0.7 * width), th), (0, 0, 0), -1)
            cv2.rectangle(frame_bgr, (0, height - th), (width, height), (0, 0, 0), -1)
            try:
                if args.osd_sims and not is_still_frames[idx] \
                        and valid_idx % (chosen_stride * NUM_FRAMES) == 0:
                    logger.info(f'valid_idx: {valid_idx} idx: {idx} osd: {osd}')
                    osd_blend = draw_osd_sim(embs_sims[osd], osd_size)
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
                            '%d' % osd,
                            (int(0.4 * osd_size), int(0.55 * osd_size)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 0), 2)
                    osd += 1
                if osd_blend is not None:
                    frame_bgr[th:osd_size + th, width - osd_size:, :] = osd_blend
            except Exception:
                logger.error(traceback.format_exc(limit=6))
            cv2.putText(frame_bgr,
                    '%dX%d %.1f S:%d C:%.1f/%.1f %s %s' % (width, height,
                        fps, chosen_stride, sum_counts[idx], sum_counts[-1],
                        'L:%.2f' % args.tsm_last_threshold if args.tsm_last_enable else '',
                        'ST' if is_still_frames[idx] else ''),
                    (2, int(0.06 * height)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7 if height < 500 else 2,
                    (255, 255, 255), 2)

            if focus_box is not None:
                cv2.putText(frame_bgr,
                        '%d,%d' % (fx1, fy1),
                        (fx1 + 2, fy1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame_bgr,
                        '%d,%d' % (fx2, fy2),
                        (fx2 - 65, fy2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.rectangle(frame_bgr, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)

            if args.osd_sims and valid_idx < valid_frames_count and valid_idx < len(binframes):
                frame_bgr[height - INPUT_HEIGHT - 10:, :INPUT_WIDTH + 10, :] = 222
                frame_bgr[height - INPUT_HEIGHT - 5:height - 5, 5:INPUT_WIDTH + 5, :] = keepframe[valid_idx][:,:,::-1]
                if hist_blend is None:
                    frame_bgr[th:INPUT_HEIGHT + th, 5:INPUT_WIDTH + 5, :] = binframes[valid_idx]
                else:
                    frame_bgr[th:INPUT_HEIGHT + th, 5:INPUT_WIDTH + 5, :] = cv2.addWeighted(
                            hist_blend, alpha,
                            binframes[valid_idx], 1 - alpha,
                            0)

            cv2.putText(frame_bgr, bottom_text,
                    (INPUT_WIDTH + 12, height - int(th * 0.35)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7 if height < 500 else 2,
                    (255, 255, 255), 2)

            if idx % 181 == 0:
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

    return pigeon


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

    if 'stdwave_sigma_count' in pigeon:
        pigeon = _post_stdwave(pigeon, args, _send_progress)
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
