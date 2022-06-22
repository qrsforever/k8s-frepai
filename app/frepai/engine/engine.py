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


def _engine_kstest(pigeon, progress_cb):
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
    return pigeon


def _engine_stdwave(pigeon, progress_cb):
    stdwave_sigma_count = pigeon['stdwave_sigma_count']
    stdwave_window_size = pigeon['stdwave_window_size']
    stdwave_distance_size = pigeon['stdwave_distance_size']

    progress_cb(10)
    stdwave_data = np.load(f'{pigeon["cache_path"]}/stdwave_data.npy')

    pad_r_size = (stdwave_window_size - 1) // 2
    pad_l_size = stdwave_window_size - 1 - pad_r_size

    # average = np.convolve(stdwave_data, np.ones(stdwave_window_size), 'valid') / stdwave_window_size
    # average = [average[0]] * pad_l_size + average.tolist() + [average[-1]] * pad_r_size

    pdata = np.pad(stdwave_data, (pad_r_size, pad_l_size), mode='reflect')
    wdata = np.lib.stride_tricks.sliding_window_view(pdata, stdwave_window_size)
    average = np.mean(wdata, axis=-1)

    stdwave_data = stdwave_data - average

    progress_cb(50)
    mean, std = stdwave_data.mean(), stdwave_data.std()
    threshold = mean + stdwave_sigma_count * std
    if stdwave_sigma_count < 0:
        outliers = np.where(stdwave_data < threshold)[0]
    else:
        outliers = np.where(stdwave_data > threshold)[0]
    progress_cb(80)
    if len(outliers) == 0:
        logger.error(threshold)
        indexes = []
    else:
        # indexes = np.where(np.subtract(outliers, np.roll(outliers, 1)) > stdwave_distance_size)[0]
        dd = np.diff(outliers)
        indexes = np.where(dd > stdwave_distance_size)[0]
        indexes = [outliers[0]] + [outliers[i] for i in indexes]
    if pigeon['devmode']:
        pigeon['stdwave_threshold'] = threshold
        pigeon['stdwave_mean'] = mean
        pigeon['stdwave_std'] = std
        pigeon['stdwave_dd'] = sorted(dd.tolist(), reverse=True)[:10]
        np.save(f'{pigeon["cache_path"]}/stdwave_post.npy', np.asarray(stdwave_data))
    progress_cb(90)
    np.save(f'{pigeon["cache_path"]}/stdwave_indexes.npy', np.asarray(indexes))
    logger.info(pigeon)
    progress_cb(100)
    return pigeon


def engine_process(pigeon, progress_cb=None):
    if 'cache_path' not in pigeon:
        raise HandlerError(82001, 'not found cache_path')

    cache_path = pigeon['cache_path']
    pigeon['task'] = 'engine'

    if not os.path.isdir(pigeon['cache_path']):
        raise HandlerError(82002, f'cache_path[{cache_path}] cannot open!')

    def _send_progress(x):
        if progress_cb:
            pigeon['progress'] = round(40 + 0.2 * x, 2)
            progress_cb(pigeon)
            logger.info(f"{round(x, 2)} {pigeon['progress']}")

    global model
    if model is None:
        model = get_repnet_model('/ckpts')

    if 'kstest_ecdfs_path' in pigeon:
        return _engine_kstest(pigeon, _send_progress)

    if 'stdwave_sigma_count' in pigeon:
        return _engine_stdwave(pigeon, _send_progress)

    with open(f'{cache_path}/config.json', 'r') as fr:
        args = DotDict(json.load(fr))
        # logger.info(args)

    frames = np.load(f'{cache_path}/keepframe.npz')['x']

    tsm_last_thresh, tsm_last_smooth = 0.5, False
    if args.tsm_last_enable:
        tsm_last_thresh = args.tsm_last_threshold
        tsm_last_smooth = args.tsm_last_smooth

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
            strides=args.strides, batch_size=args.batch_size,
            tsm_last_thresh=tsm_last_thresh,
            tsm_last_smooth=tsm_last_smooth,
            pcaks=pcaks, progress_cb=_send_progress)
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

    return pigeon
