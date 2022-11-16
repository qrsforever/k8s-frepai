#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file oss.py
# @brief
# @author QRS
# @version 1.0
# @date 2022-04-11 16:18


import os
import time
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client

region = 'ap-beijing'
bucket = 'frepai'
bucket_name = f'{bucket}-1301930378'
access_key = os.environ.get('MINIO_ACCESS_KEY', 'AKIDV7XjgOr42nMhneGdmiPs66rNioeFafeT')
secret_key = os.environ.get('MINIO_SECRET_KEY', 'd190cxQk0CHCtLXjhQt65tUr2yf7KI1V')
coss3_domain = f'https://{bucket_name}.cos.{region}.myqcloud.com'
coss3_client = CosS3Client(CosConfig(Region=region, SecretId=access_key, SecretKey=secret_key, Token=None, Scheme='https'))


def coss3_put(local_path, prefix_map=None):
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


def coss3_list(prefix, recurive=False):
    delimiter = '' if recurive else '/'
    marker = ""
    result = []
    while True:
        response = coss3_client.list_objects(
            Bucket=bucket_name, Prefix=prefix, Marker=marker, MaxKeys=50, Delimiter=delimiter)

        if 'Contents' in response:
            for content in response['Contents']:
                result.append(content['Key'])

        if 'CommonPrefixes' in response:
            for folder in response['CommonPrefixes']:
                result.append(folder['Prefix'])

        if response['IsTruncated'] == 'false':
            break

        marker = response["NextMarker"]
    return result


def coss3_delete(path, logger=None):
    if logger:
        logger.info(path)
    files = coss3_list(path, recurive=True)
    for key in files:
        if logger:
            logger.info(key)
        coss3_client.delete_object(Bucket=bucket_name, Key=key)
