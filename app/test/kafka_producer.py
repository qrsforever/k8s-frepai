#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file kafka_producer.py
# @brief
# @author QRS
# @version 1.0
# @date 2022-03-18 18:05

# kafka-topics.sh --create --zookeeper zookeeper:2181 --replication-factor 1 --partitions 1 --topic messages
# kafka-console-producer.sh --broker-list kafka:19092 --topic messages
# kafka-console-consumer.sh --bootstrap-server kafka:19092 --topic messages --from-beginning

import os
import json
import time
from kafka import KafkaProducer

import socket
hostname = socket.gethostname()

if hostname == 'storage':
    host = '172.21.0.4'
else:
    host = '82.157.36.183'

producer = KafkaProducer(
        bootstrap_servers=[f'{host}:19092'],
        # key_serializer=str.encode,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'))

with open('./ef.json', 'r') as fr:
    value = json.load(fr)

N = int(os.environ.get('N', 1))

print('Count:', N)

try:
    t0 = int(time.time())
    for i in range(N):
        token = int(time.time()) - t0
        value[0]['cfg']['pigeon']['token'] = token
        value[0]['cfg']['pigeon']['order'] = i
        value[0]['cfg']['pigeon']['total'] = N
        value[0]['cfg']['pigeon']['msgkey'] = 'frepai_output'
        future = producer.send(
                'frepai_input',
                # key='frepai',
                value=value[0])
        time.sleep(2)
        result = future.get(timeout=10)
        print(f'1: {i} {token} {result}')
        # value[1]['cfg']['pigeon']['token'] = token
        # value[1]['cfg']['pigeon']['order'] = i
        # value[1]['cfg']['pigeon']['total'] = N
        # value[1]['cfg']['pigeon']['msgkey'] = 'frepai_output'
        # future = producer.send(
        #         'frepai_input',
        #         # key='frepai',
        #         value=value[1])
        # time.sleep(2)
        # result = future.get(timeout=10)
        # print(f'2: {i} {token} {result}')

    producer.flush()

except Exception:
    producer.close()
