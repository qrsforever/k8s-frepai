#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file routers.py
# @brief
# @author QRS
# @version 1.0
# @date 2022-04-11 16:05


import os
import json
import re
from frepai.utils import run_shell
from flask import Blueprint, request
from frepai.utils.logger import EasyLogger as logger
from frepai.utils.oss import coss3_put

net_ip = os.environ.get('NET_IP', '0.0.0.0')

api_srs = Blueprint("srs", __name__)


@api_srs.route('/on_publish', methods=['POST'])
def _srs_on_publish():
    reqjson = json.loads(request.get_data().decode())
    reqjson['external_ip'] = net_ip
    reqjson['external_port'] = 31990
    reqjson['webrtc_play'] = f'https://{net_ip}:30888/players/rtc_player.html{reqjson["param"]}&api=31990&app={reqjson["app"]}&stream={reqjson["stream"]}'
    api_srs.queue.put(reqjson)
    return '0'


@api_srs.route('/on_unpublish', methods=['POST'])
def _srs_on_unpublish():
    reqjson = json.loads(request.get_data().decode())
    reqjson['external_ip'] = net_ip
    reqjson['external_port'] = 31990
    api_srs.queue.put(reqjson)
    return '0'


@api_srs.route('/on_dvr', methods=['POST'])
def _srs_on_dvr():
    reqjson = json.loads(request.get_data().decode())
    basefile = os.path.basename(reqjson['file'])
    localfile = os.path.join(reqjson['cwd'], reqjson['file'])
    localpath = os.path.dirname(localfile)
    remotepath = os.path.join(reqjson['app'], reqjson['stream'], basefile[:8], 'videos')
    prefix_map = [localpath, remotepath]

    output = run_shell(f'/usr/local/srs/objs/ffmpeg/bin/ffmpeg -i {localfile}')
    result = re.search(r'Duration: (?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+)\.\d+,', output)
    if result:
        timesecs = result.groupdict()
        duration = 3600 * int(timesecs['hours']) + 60 * int(timesecs['minutes']) + int(timesecs['seconds'])
    else:
        if 'seg' in reqjson['vhost']:
            duration = int(reqjson['vhost'].split('.')[1][:-1])
        else:
            duration = -1

    if duration <= 0:
        logger.error('Error: The dvr video have no frames!')
        return '0'

    result = coss3_put(localfile, prefix_map=prefix_map)
    if len(result) == 1 and 'object' in result[0]:
        api_srs.queue.put({
            **result[0],
            **reqjson,
            'external_ip': net_ip,
            'external_port': 31990,
            'webrtc_play': f'https://{net_ip}:30888/players/rtc_player.html{reqjson["param"]}&api=31990&app={reqjson["app"]}&stream={reqjson["stream"]}',
            'duration': duration,
            'filename': os.path.basename(reqjson['file'])})
    os.remove(localfile)
    return '0'
