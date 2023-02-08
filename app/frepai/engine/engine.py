#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file repnet.py
# @brief
# @author QRS
# @version 1.0
# @date 2022-03-22 18:21


import json
import os
import time
import pickle
import numpy as np
import tensorflow_probability as tfp

from frepai.utils.easydict import DotDict
from frepai.utils.logger import EasyLogger as logger
from frepai.utils.errcodes import HandlerError
from frepai.engine.repnet import get_repnet_model, get_counts, get_sims
from frepai.utils import easy_wget

model = None


##################################################################
## common
##################################################################


def _features_sliding_window(data, window_size, method):# {{{
    pad_r_size = (window_size - 1) // 2
    pad_l_size = window_size - 1 - pad_r_size

    # average = np.convolve(stdwave_data, np.ones(stdwave_window_size), 'valid') / stdwave_window_size
    # average = [average[0]] * pad_l_size + average.tolist() + [average[-1]] * pad_r_size

    pdata = np.pad(data, (pad_r_size, pad_l_size), mode='reflect')
    wdata = np.lib.stride_tricks.sliding_window_view(pdata, window_size)
    if method == 'min':
        data = data - np.min(wdata, axis=-1)
    if method == 'minmax':
        minvals = np.min(wdata, axis=-1)
        maxvals = np.max(wdata, axis=-1)
        data = (data - minvals) / (maxvals - minvals)
    elif method == 'mean':
        data = data - np.mean(wdata, axis=-1)
    elif method == 'standard':
        mean = np.mean(wdata, axis=-1)
        std = np.std(wdata, axis=-1)
        data = (data - mean) / std

    return data # }}}


##################################################################
## kstest
##################################################################


def _engine_kstest(pigeon, progress_cb):# {{{
    progress_cb(10)

    kstest_ecdfs_path = pigeon['kstest_ecdfs_path']
    with open(kstest_ecdfs_path, 'rb') as fr:
        pcaks = pickle.load(fr)

    progress_cb(50)
    ecdfs = tfp.distributions.Empirical(pcaks['ecdfs'].T)

    pcaks['ecdfs'] = ecdfs
    progress_cb(80)

    with open(kstest_ecdfs_path, 'wb') as fw:
        pickle.dump(pcaks, fw)
    progress_cb(100)
    return pigeon# }}}


##################################################################
## featpeak
##################################################################


def _local_maxima(x):# {{{
    midpoints, left_edges, right_edges = [], [], []
    i, i_max = 1, x.shape[0] - 1
    while i < i_max:
        if x[i - 1] < x[i]:
            i_ahead = i + 1
            while i_ahead < i_max and x[i_ahead] == x[i]:
                i_ahead += 1
            if x[i_ahead] < x[i]:
                left_edges.append(i)
                right_edges.append(i_ahead - 1)
                midpoints.append((left_edges[-1] + right_edges[-1]) // 2)
                i = i_ahead
        i += 1
    return np.array(midpoints)


def _find_peaks(peaks_values, height=(), distance=-1, prominence=(), width=(), wlen=-1, rel_height=0.5, min_thresh=-1):
    SS = [-1] * 5
    peaks_indices = _local_maxima(peaks_values)
    logger.info(f'123--->{peaks_indices}')
    SS[0] = len(peaks_indices)

    if isinstance(height, (int, float)):
        height = (height, -1)
    if isinstance(prominence, (int, float)):
        prominence = (prominence, -1)
    if isinstance(width, (int, float)):
        width = (width, -1)

    def __filterby__(values, pmin, pmax):
        keep = np.ones(values.size, dtype=np.bool_)
        if pmin > 0:
            keep &= (pmin <= values)
        if pmax > pmin:
            keep &= (values <= pmax)
        return keep

    props = {}
    if len(height) == 2 and height[0] > 1:
        keep = __filterby__(peaks_values[peaks_indices], height[0], height[1])
        peaks_indices = peaks_indices[keep]
        props['peak_heights'] = peaks_values[peaks_indices]
        SS[1] = len(peaks_indices)

    if distance > 0:
        peaks_size = peaks_indices.shape[0]
        keep = np.ones(peaks_size, dtype=np.bool_)
        priority_indices = np.argsort(peaks_values[peaks_indices])
        for i in range(peaks_size - 1, -1, -1):
            j = priority_indices[i]
            if keep[j] == 0:
                continue
            k = j - 1
            while 0 <= k and peaks_indices[j] - peaks_indices[k] < distance:
                keep[k] = False
                k -= 1
            k = j + 1
            while k < peaks_size and peaks_indices[k] - peaks_indices[j] < distance:
                keep[k] = False
                k += 1
        peaks_indices = peaks_indices[keep]
        props = {key: array[keep] for key, array in props.items()}
        SS[2] = len(peaks_indices)

    if len(prominence) == 2 or len(width) == 2:
        peaks_size, values_size = peaks_indices.shape[0], peaks_values.shape[0]
        prominences = np.empty(peaks_size, dtype=np.int32)
        left_bases = np.empty(peaks_size, dtype=np.int32)
        right_bases = np.empty(peaks_size, dtype=np.int32)
        for idx, peak in enumerate(peaks_indices):
            i_min = 0
            i_max = values_size - 1

            if 2 <= wlen:
                i_min = max(peak - wlen // 2, i_min)
                i_max = min(peak + wlen // 2, i_max)

            i = left_bases[idx] = peak
            left_min = peaks_values[peak]
            while i_min <= i and peaks_values[i] <= peaks_values[peak]:
                if peaks_values[i] < left_min:
                    left_min = peaks_values[i]
                    left_bases[idx] = i
                    if left_min < min_thresh:
                        break
                i -= 1
            i = right_bases[idx] = peak
            right_min = peaks_values[peak]
            while i <= i_max and peaks_values[i] <= peaks_values[peak]:
                if peaks_values[i] < right_min:
                    right_min = peaks_values[i]
                    right_bases[idx] = i
                    if right_min < min_thresh:
                        break
                i += 1
            prominences[idx] = peaks_values[peak] - max(left_min, right_min)

        props['prominences'] = prominences
        props['left_bases'] = left_bases
        props['right_bases'] = right_bases

        if len(prominence) == 2 and prominence[0] > 1:
            keep = __filterby__(prominences, prominence[0], prominence[1])
            peaks_indices = peaks_indices[keep]
            props = {key: array[keep] for key, array in props.items()}
            SS[3] = len(peaks_indices)

        if len(width) == 2 and width[0] > 1:
            peaks_size = peaks_indices.shape[0]
            widths = np.empty(peaks_size, dtype=np.float64)
            width_heights = np.empty(peaks_size, dtype=np.float64)
            left_ips = np.empty(peaks_size, dtype=np.float64)
            right_ips = np.empty(peaks_size, dtype=np.float64)
            for idx, peak in enumerate(peaks_indices):
                i_min = props['left_bases'][idx]
                i_max = props['right_bases'][idx]
                height = width_heights[idx] = peaks_values[peak] - props['prominences'][idx] * rel_height

                i = peak
                while i_min < i and height < peaks_values[i]:
                    i -= 1
                left_ip = i
                if peaks_values[i] < height:
                    left_ip += (height - peaks_values[i]) / (peaks_values[i + 1] - peaks_values[i])

                i = peak
                while i < i_max and height < peaks_values[i]:
                    i += 1
                right_ip = i
                if peaks_values[i] < height:
                    right_ip -= (height - peaks_values[i]) / (peaks_values[i - 1] - peaks_values[i])

                widths[idx] = right_ip - left_ip
                left_ips[idx] = left_ip
                right_ips[idx] = right_ip

            props['widths'] = widths
            props['width_heights'] = width_heights
            props['left_ips'] = left_ips
            props['right_ips'] = right_ips

            keep = __filterby__(widths, width[0], width[1])
            peaks_indices = peaks_indices[keep]
            props = {key: array[keep] for key, array in props.items()}
            SS[4] = len(peaks_indices)

    logger.info(f'peaks_indices: {SS}')
    props['ss'] = SS
    return peaks_indices, props


def _engine_featpeak(pigeon, progress_cb):
    devmode = pigeon['devmode']
    featpeak_window_size = pigeon['featpeak_window_size']
    featpeak_data_normal = pigeon['featpeak_data_normal']
    featpeak_distance_size = pigeon['featpeak_distance_size']
    featpeak_min_threshold = pigeon['featpeak_min_threshold']
    featpeak_relative_height = pigeon['featpeak_relative_height']
    featpeak_height_minmax = pigeon['featpeak_height_minmax']
    featpeak_width_minmax = pigeon['featpeak_width_minmax']
    featpeak_prominence_minmax = pigeon['featpeak_prominence_minmax']

    progress_cb(10)
    featpeak_data = np.load(f'{pigeon["cache_path"]}/featpeak_data.npy')
    if featpeak_data_normal:
        featpeak_data = _features_sliding_window(featpeak_data, featpeak_window_size, 'min')
        # featpeak_data = featpeak_data - featpeak_data.min()

    progress_cb(30)
    peaks_indices, properties = _find_peaks(featpeak_data,
            featpeak_height_minmax, featpeak_distance_size, featpeak_prominence_minmax,
            featpeak_width_minmax, featpeak_window_size, featpeak_relative_height, featpeak_min_threshold)

    progress_cb(70)
    if devmode and featpeak_data_normal:
        np.save(f'{pigeon["cache_path"]}/featpeak_post.npy', np.asarray(featpeak_data))
    np.save(f'{pigeon["cache_path"]}/featpeak_indexes.npy', np.asarray(peaks_indices))
    if devmode:
        with open(f'{pigeon["cache_path"]}/featpeak_props.pkl', 'wb') as fw:
            pickle.dump(properties, fw)

    progress_cb(100)
    return pigeon# }}}


##################################################################
## direction
##################################################################


def _engine_direction(pigeon, progress_cb):# {{{
    direction_inverse = pigeon['direction_inverse']
    direction_scale_threshold = pigeon['direction_scale_threshold']
    direction_window_size = pigeon['direction_window_size']

    progress_cb(10)
    direction_data = np.load(f'{pigeon["cache_path"]}/direction_data.npy')
    mfocus, lfocus, rfocus = np.split(direction_data, 3, axis=1)
    mmask = np.where(mfocus > np.mean(mfocus))[0]
    lfocus_scale = lfocus[mmask] / mfocus[mmask]
    rfocus_scale = rfocus[mmask] / mfocus[mmask]
    lr_mask = np.where((lfocus_scale > direction_scale_threshold * rfocus_scale))[0]
    rl_mask = np.where((rfocus_scale > direction_scale_threshold * lfocus_scale))[0]
    progress_cb(20)
    factor = -1 if direction_inverse else 1
    result = np.zeros((len(mfocus)))
    result[mmask[lr_mask]] = factor * direction_window_size
    result[mmask[rl_mask]] = -factor * direction_window_size
    features = np.delete(result, np.argwhere((result == 0)))
    progress_cb(30)
    pad_r_size = (direction_window_size - 1) // 2
    pad_l_size = direction_window_size - 1 - pad_r_size
    features = np.pad(features, (pad_r_size, pad_l_size), constant_values=0)
    logger.info(f'--->: {features}')
    features = np.mean(np.lib.stride_tricks.sliding_window_view(features, direction_window_size), axis=-1)
    progress_cb(40)
    # def _find_peaks(peaks_values, height=(), distance=-1, prominence=(), width=(), wlen=-1, rel_height=0.5, min_thresh=-1):
    peaks_indices, properties = _find_peaks(
            features,
            (0.4, 1.0), 25)
    progress_cb(100, True)
    logger.info(f'{peaks_indices} {properties}')
    return pigeon# }}}


##################################################################
## stdwave
##################################################################


def _engine_stdwave(pigeon, progress_cb):# {{{
    devmode = pigeon['devmode']
    stdwave_sub_average = pigeon['stdwave_sub_average']
    stdwave_sigma_count = pigeon['stdwave_sigma_count']
    stdwave_window_size = pigeon['stdwave_window_size']
    stdwave_distance_size = pigeon['stdwave_distance_size']
    stdwave_minstd_thresh = pigeon['stdwave_minstd_thresh']

    progress_cb(10)
    stdwave_data = np.load(f'{pigeon["cache_path"]}/stdwave_data.npy')
    if stdwave_sub_average:
        stdwave_data = _features_sliding_window(stdwave_data, stdwave_window_size, 'mean')
        if devmode:
            logger.info(np.round(stdwave_data, 3).tolist())
    progress_cb(50)
    mean, std = stdwave_data.mean(), stdwave_data.std()
    logger.info(f'mean: {mean}, std: {std}')
    if std > stdwave_minstd_thresh:
        threshold = mean + stdwave_sigma_count * std
        outliers = np.where(stdwave_data < threshold)[0] if stdwave_sigma_count < 0 else np.where(stdwave_data > threshold)[0]
        pigeon['stdwave_dd'] = sorted(np.diff(outliers).tolist(), reverse=True)[:8]
        pigeon['stdwave_threshold'] = threshold

        progress_cb(80)
        if len(outliers) == 0:
            logger.error(threshold)
            indexes = []
        else:
            # indexes = np.where(np.subtract(outliers, np.roll(outliers, 1)) > stdwave_distance_size)[0]
            # indexes = np.where(dd > stdwave_distance_size)[0]
            # indexes = [outliers[0]] + [outliers[i + 1] for i in indexes]
            cursor = outliers[0]
            indexes = [cursor]
            for t in outliers:
                if (t - cursor) > stdwave_distance_size:
                    indexes.append(t)
                    cursor = t
    else:
        logger.error(f'std[{std}] vs minstd[{stdwave_minstd_thresh}]')
        indexes = []

    if devmode:
        pigeon['stdwave_mean'] = mean
        pigeon['stdwave_std'] = std
        np.save(f'{pigeon["cache_path"]}/stdwave_post.npy', np.asarray(stdwave_data))
    progress_cb(90)
    np.save(f'{pigeon["cache_path"]}/stdwave_indexes.npy', np.asarray(indexes))
    logger.info(pigeon)
    progress_cb(100)
    return pigeon# }}}


##################################################################
## 0-1 impulse
##################################################################


def _engine_diffimpulse(pigeon, progress_cb):# {{{
    devmode = pigeon['devmode']
    diffimpulse_window_size = pigeon['diffimpulse_window_size']
    progress_cb(10)
    diffimpulse_data = np.load(f'{pigeon["cache_path"]}/diffimpulse_data.npy')

    indexes = []
    thresh_zero, thresh_one = diffimpulse_window_size
    rec_zero, rec_one = 0, 0
    if devmode:
        impulse_0_1 = np.array([0] * len(diffimpulse_data))
        if diffimpulse_data[0] == 0:
            zero_cnt_list, one_cnt_list = [0], []
        else:
            zero_cnt_list, one_cnt_list = [], [0]
    for i, d in enumerate(diffimpulse_data):
        if d == 0:
            rec_zero += 1
            if devmode:
                if zero_cnt_list[-1] == 0:
                    one_cnt_list.append(0)
                zero_cnt_list[-1] += 1
            if rec_one > 0:
                if rec_one < thresh_one:
                    rec_one = 0
                else:
                    if rec_zero > thresh_zero:
                        if devmode:
                            impulse_0_1[i - rec_zero - rec_one: i - rec_zero] = 1
                        indexes.append(i - rec_zero)
                        rec_one = 0
        else:
            rec_one += 1
            if devmode:
                if one_cnt_list[-1] == 0:
                    zero_cnt_list.append(0)
                one_cnt_list[-1] += 1
            if rec_zero > 0:
                if rec_zero < thresh_zero:
                    rec_zero = 0
                else:
                    if rec_one > thresh_one:
                        rec_zero = 0

    progress_cb(70)
    np.save(f'{pigeon["cache_path"]}/diffimpulse_indexes.npy', np.asarray(indexes))
    if devmode:
        np.save(f'{pigeon["cache_path"]}/diffimpulse_cnt_0.npy', np.asarray(zero_cnt_list))
        np.save(f'{pigeon["cache_path"]}/diffimpulse_cnt_1.npy', np.asarray(one_cnt_list))
        np.save(f'{pigeon["cache_path"]}/diffimpulse_0_1.npy', np.asarray(impulse_0_1))
    logger.info(pigeon)
    progress_cb(100)
    return pigeon# }}}


##################################################################
## repnet
##################################################################


def engine_process(pigeon, progress_cb=None):# {{{
    if 'cache_path' not in pigeon:
        raise HandlerError(82001, 'not found cache_path')

    cache_path = pigeon['cache_path']
    pigeon['task'] = 'engine'

    if not os.path.isdir(pigeon['cache_path']):
        raise HandlerError(82002, f'cache_path[{cache_path}] cannot open!')

    def _send_progress(x, fix=False):
        if progress_cb:
            pigeon['progress'] = round(40 + 0.2 * x, 2)
            progress_cb(pigeon)
            logger.info(f"{round(x, 2)} {pigeon['progress']}")

    global model
    if model is None:
        model = get_repnet_model('/ckpts')

    if 'kstest_ecdfs_path' in pigeon:
        return _engine_kstest(pigeon, _send_progress)

    if 'direction_window_size' in pigeon:
        return _engine_direction(pigeon, _send_progress)

    if 'featpeak_window_size' in pigeon:
        return _engine_featpeak(pigeon, _send_progress)

    if 'stdwave_window_size' in pigeon:
        return _engine_stdwave(pigeon, _send_progress)

    if 'diffimpulse_window_size' in pigeon:
        return _engine_diffimpulse(pigeon, _send_progress)

    with open(f'{cache_path}/config.json', 'r') as fr:
        args = DotDict(json.load(fr))
        # logger.info(args)

    frames = np.load(f'{cache_path}/keepframe.npz')['x']

    devmode = pigeon['devmode']
    within_period_threshold = pigeon.get('within_period_threshold', 0.5)
    avg_pred_score = pigeon.get('avg_pred_score', 0.2)
    tsm_last_thresh = pigeon.get('tsm_last_threshold', 0.5)
    tsm_last_smooth = pigeon.get('tsm_last_smooth', False)

    pcaks = None
    if args.ef_is_send:
        try:
            epath = easy_wget(args.ef_url, cache_path)
            with open(epath, 'rb') as fr:
                pcaks = pickle.load(fr)
                pcaks['ef_alpha'] = args.ef_alpha
                pcaks['ef_beta'] = args.ef_beta
                pcaks['ef_gamma'] = args.ef_gamma
        except Exception:
            raise HandlerError(81001, f'wget embs filter[{args.ef_url}] fail!')

    t0 = time.time()
    result = get_counts(model, frames,
            within_period_threshold=within_period_threshold,
            avg_pred_score=avg_pred_score,
            strides=args.strides, batch_size=args.batch_size,
            tsm_last_thresh=tsm_last_thresh,
            tsm_last_smooth=tsm_last_smooth,
            pcaks=pcaks, progress_cb=_send_progress, devmode=devmode, logger=logger)
    logger.info('model inference time: %.3f' % (time.time() - t0))

    with open(f'{cache_path}/engine.pkl', 'wb') as fw:
        pickle.dump(result, fw)

    if args.osd_sims:
        embs_feat = result['final_embs']
        embs_sims = get_sims(embs_feat, temperature=args.temperature)
        embs_sims = np.squeeze(embs_sims, -1)
        np.save(f'{cache_path}/embs_feat.npy', embs_feat)
        np.save(f'{cache_path}/embs_sims.npy', embs_sims)
        logger.info(f'embs_feat.shape: {embs_feat.shape}  embs_sims.shape: {embs_sims.shape}')

    del result

    _send_progress(100)

    return pigeon# }}}
