#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file kafka_consumer.py
# @brief
# @author QRS
# @version 1.0
# @date 2022-03-18 18:25


import signal
import os
import json
import sys
import time
from kafka import KafkaConsumer
import socket
hostname = socket.gethostname()

if hostname == 'storage':
    host = '172.21.0.4'
else:
    host = '82.157.36.183'

print(host)

consumer = KafkaConsumer(
        'frepai_output',
        bootstrap_servers=[f'{host}:19092'],
        group_id='frepai_output',
        value_deserializer=lambda v: json.loads(v.decode('utf-8')),
        auto_offset_reset='latest')

orders = {
}

finished = {}


def main():
    t0 = None # int(time.time())
    try:
        for msg in consumer:
            # sys.stderr.write(f'{msg.value}\n')
            if t0 is None:
                t0 = int(time.time())
                print('t0: ', t0)
            if 'pigeon' in msg.value and 'order' in msg.value['pigeon'] and 'task' in msg.value:
                tok = msg.value['pigeon']['token']
                oid = msg.value["pigeon"]["order"]
                tol = msg.value["pigeon"]["total"]
                if oid not in orders.keys():
                    orders[oid] = {'pre': 0.0, 'engine': 0.0, 'post': 0.0}
                orders[oid][msg.value['task']] = msg.value['progress']
                orders[oid]['from_host'] = f"{msg.value['task']}_{msg.value['from_host']}"
                orders[oid]['token'] = tok
                print(oid, json.dumps(orders[oid]))
                if int(msg.value['progress']) == 100:
                    print(json.dumps(orders, indent=2))
                    print(json.dumps(finished, indent=2))
                    if tok not in finished.keys():
                        finished[tok] = []
                    finished[tok].append(oid)
                    finished[tok] = sorted(finished[tok])
                    if len(finished[tok]) == tol:
                        used = int(time.time()) - t0
                        print('time_elapsed: ', int(time.time()) - t0, '%.3f' % float(used / tol))
                        orders[oid].clear()
                        finished[tok].clear()
                        t0 = None
    except Exception as err:
        sys.stderr.write(f'{err}')
        consumer.close()


def signal_handler(sig, frame):
    print(json.dumps(finished))
    print(json.dumps(orders))
    consumer.close()
    os._exit(os.EX_OK)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    main()
