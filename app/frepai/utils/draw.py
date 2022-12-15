#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file draw.py
# @brief
# @author QRS
# @version 1.0
# @date 2022-03-16 21:14


import cv2
import io
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LogNorm


def get_rect_points(w, h, box):
    if box[0] == box[2] and box[1] == box[3]:
        return None
    # TODO
    if box[0] == 0 and box[1] == 0 and box[2] == 1 and box[3] == 1:
        box = [0.4, 0.4, 0.6, 0.6]
    if box[0] < 1.0 and box[1] < 1.0 and box[2] <= 1.0 and box[3] <= 1.0:
        x1, y1 = int(w * box[0]), int(h * box[1])
        x2, y2 = int(w * box[2]), int(h * box[3])
    else:
        x1, y1 = box[0], box[1]
        x2, y2 = box[2], box[3]
    return x1, y1, x2, y2


def get_ploy_points(w, h, pts):
    result = []
    for point in pts:
        if point[0] < 1.0:
            x, y = int(w * point[0]), int(h * point[1])
        else:
            x, y = point
        result.append((x, y))
    return np.asarray(result)


def draw_osd_sim(sim, size=128):
    fig, ax = plt.subplots()
    plt.axis('off')
    fig.set_size_inches(size / 100.0, size / 100.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.imshow(sim, cmap='hot', interpolation='nearest', norm=LogNorm())
    with io.BytesIO() as fw:
        plt.savefig(fw, dpi=100.0, bbox_inches=0)
        buffer_ = np.frombuffer(fw.getvalue(), dtype=np.uint8)
        plt.close()
        return cv2.imdecode(buffer_, cv2.IMREAD_COLOR)
    raise


def draw_hist_density(x, bins, width, height):
    fig, ax = plt.subplots()
    fig.set_size_inches(width / 100.0, height / 100.0)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0,0)
    # density = stats.gaussian_kde(x)
    # plt.hist(x, bins=bins, histtype=u'step', density=True)
    # plt.plot(x, density(x))
    plt.hist(x, bins=bins, histtype=u'step', color='blue')
    with io.BytesIO() as fw:
        plt.savefig(fw, dpi=100.0, bbox_inches=0)
        buffer_ = np.frombuffer(fw.getvalue(), dtype=np.uint8)
        plt.close()
        return cv2.imdecode(buffer_, cv2.IMREAD_COLOR)
    raise RuntimeError
