#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file master-app.py
# @brief
# @author QRS
# @version 1.0
# @date 2022-05-09 16:26


import os
import time
import json # noqa
import shutil
import subprocess
from flask import Flask, request
from flask_cors import CORS

from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client

app = Flask(__name__)


CORS(app, supports_credentials=True)


region = 'ap-beijing'
bucket = 'frepai'
bucket_name = f'{bucket}-1301930378'
coss3_domain = f'https://{bucket_name}.cos.{region}.myqcloud.com'
coss3_client = CosS3Client(CosConfig(
    Region='ap-beijing',
    SecretId='AKIDV7XjgOr42nMhneGdmiPs66rNioeFafeT',
    SecretKey='d190cxQk0CHCtLXjhQt65tUr2yf7KI1V',
    Token=None, Scheme='https'))


def _coss3_put(local_path, prefix_map=None):
    result = []

    def _upload_file(local_file):
        if not os.path.isfile(local_file):
            return
        if prefix_map and isinstance(prefix_map, list):
            lprefix = prefix_map[0].rstrip(os.path.sep)
            rprefix = prefix_map[1].strip(os.path.sep)
            remote_file = local_file.replace(lprefix, rprefix, 1)
        else:
            remote_file = local_file.lstrip(os.path.sep)

        file_size = os.stat(local_file).st_size
        with open(local_file, 'rb') as file_data:
            btime = time.time()
            response = coss3_client.put_object(
                    Bucket=bucket_name,
                    Body=file_data,
                    Key=remote_file)
            etime = time.time()
            result.append({
                'etag': response['ETag'].strip('"'),
                'bucket': bucket,
                'object': remote_file,
                'size': file_size,
                'time': [btime, etime]})

    if os.path.isdir(local_path):
        for root, directories, files in os.walk(local_path):
            for filename in files:
                _upload_file(os.path.join(root, filename))
    else:
        _upload_file(local_path)

    return result


@app.route('/master/logcat', methods=["GET"])
def _logcat():
    st = request.args.get("st")
    et = request.args.get("et")
    utc = int(time.time())
    try:
        # subprocess.call(['/master/logcat.sh', st, et])
        proc = subprocess.Popen(['/master/logcat.sh', st, et], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc.wait()
        if not os.path.exists('/tmp/logs.tar.gz'):
            raise RuntimeError('not create logs')
        ret = _coss3_put('/tmp/logs.tar.gz', prefix_map=['/tmp/logs.tar.gz', f'/logs/{utc}.tar.gz'])
        shutil.rmtree('/tmp/logs.tar.gz')
        if len(ret) == 1:
            return f"{coss3_domain}/{ret[0]['object']}", 200
    except Exception as err:
        return f'{err}', 201
    return '', 202


@app.route('/master/task/eyeai', methods=["POST"])
def _task_eyeai():
    try:
        params = request.get_data().decode()
        print(params)
        proc = subprocess.Popen(['/master/eyeai.sh', params], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc.wait()
        _, stderr = proc.communicate()
        if stderr:
            return stderr.decode(), 201
        return '', 200
    except subprocess.CalledProcessError as err0:
        return f'{err0}', 202
    except Exception as err:
        return f'{err}', 203
    return '', 204


@app.route('/echo', methods=["POST"])
def _echo():
    print(request.get_data().decode())
    return '', 200


@app.route('/probe/liveness')
def _liveness_probe():
    return '', 200


@app.route('/probe/readiness')
def _readiness_probe():
    return '', 200


if __name__ == '__main__':
    app.run(port="4848", host="0.0.0.0", debug=True)
