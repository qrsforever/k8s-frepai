#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file app_service.py
# @brief
# @author QRS
# @version 1.0
# @date 2022-03-15 18:23

import argparse
import time
import json
import socket
import signal
import os
import traceback

from queue import Empty
from flask import Flask
from flask_cors import CORS
from multiprocessing import Event, Queue

from frepai.utils.errcodes import HandlerError
from frepai.utils.logger import logger_subprocess
from frepai.utils.logger import EasyLogger as logger
from frepai.utils import rmdir_p

SHORT_HOSTNAME = socket.gethostname()[-5:]

logger.setup_logger(f'frepai-{SHORT_HOSTNAME}', mp=True)


class KafkaBreakError(Exception):
    pass


def _ossrs_main_process(host, port, topic_out, exit_event, msg_queue):
    from kafka import KafkaProducer

    server = f'{host}:{port}'
    producer = KafkaProducer(
            bootstrap_servers=[server],
            client_id=SHORT_HOSTNAME,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'))
    logger.info(f'{server}, {topic_out}')

    with open('/tmp/healthy', 'w') as f:
        f.write('')
    while not exit_event.is_set():
        try:
            msg = msg_queue.get(timeout=3)
        except Empty:
            continue
        logger.info(msg)
        producer.send(topic, value=msg)
        producer.flush()
    os.remove('/tmp/healthy')


def _repnet_main_process(host, port, topic_in, topic_out, message_handler):
    from kafka import KafkaConsumer
    from kafka import KafkaProducer
    from kafka.errors import KafkaTimeoutError

    server = f'{host}:{port}'

    producer = KafkaProducer(
            bootstrap_servers=[server],
            client_id=SHORT_HOSTNAME,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'))

    logger.info(f'{server}, {topic_in}, {topic_out}')
    consumer = KafkaConsumer(topic_in,
            bootstrap_servers=[server],
            client_id=SHORT_HOSTNAME,
            group_id=topic_in,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            enable_auto_commit=False,
            max_poll_records=1,
            auto_offset_reset='latest')

    def signal_handler(sig, frame):
        logger.warning('handle signal: [%d]' % sig)
        consumer.close()
        producer.close(timeout=2)
        os._exit(os.EX_OK)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    def _safe_send(topic, message):
        try:
            # logger.info(f'kafka topic[{topic}] {message}!')
            future = producer.send(topic, value=message)
            future.get(timeout=2)
        except KafkaTimeoutError:
            logger.warning(f'kafka topic[{topic}] timeout!')
            raise KafkaBreakError
        except Exception as err:
            logger.error(f'kafka topic[{topic}] {err}')
            raise KafkaBreakError

    def _send_errmsg(message, code, err):
        message['progress'] = 100
        message['errno'] = code
        message['errtxt'] = {
            'content': str(err),
            'traceback': traceback.format_exc(limit=10)
        }
        logger.error(message)
        _safe_send(message['pigeon']['msgkey'], message)
        if 'cache_path' in message:
            rmdir_p(message['cache_path'])

    try:
        with open('/tmp/healthy', 'w') as f:
            f.write('')

        for msg in consumer:
            logger.info(f'{msg.key} {msg.partition} {msg.offset}: {msg.value}')
            value = msg.value
            consumer.commit()
            if 'cfg' in value:
                value = value['cfg']
            if 'pigeon' not in value:
                logger.error('message no has pigeon!!!')
                continue
            try:
                value['pigeon']['from_host'] = SHORT_HOSTNAME
                t0 = int(time.time())
                result = message_handler(
                        value,
                        lambda x, key=value['pigeon']['msgkey']: _safe_send(key, x))
                logger.info('time_elapsed: %d' % (int(time.time()) - t0))
                if result is not None:
                    if topic_out:
                        _safe_send(topic_out, result)
            except HandlerError as err0:
                _send_errmsg(value, err0.code, err0.message)
            except json.decoder.JSONDecodeError as err1:
                _send_errmsg(value, 90001, err1)
            except ImportError as err2:
                _send_errmsg(value, 90002, err2)
            except KeyError as err3:
                _send_errmsg(value, 90003, err3)
            except ValueError as err4:
                _send_errmsg(value, 90004, err4)
            except AssertionError as err5:
                _send_errmsg(value, 90005, err5)
            except AttributeError as err6:
                _send_errmsg(value, 90006, err6)
            except KafkaBreakError:
                break
            except Exception as err99:
                _send_errmsg(value, 90099, err99)
    except json.decoder.JSONDecodeError:
        pass
    except Exception as err:
        logger.error(f'unkdown error: {err}')
    finally:
        os.remove('/tmp/healthy')
        logger.error('Kafka subprocess quit!')
        consumer.close()
        producer.close(timeout=2)


def _liveness_probe():
    return '', 200


def _readiness_probe():
    return '', 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, dest='task')
    parser.add_argument('--host', default='0.0.0.0', type=str, dest='host')
    parser.add_argument('--port', default=1818, type=int, dest='port')
    args = parser.parse_args()
    try:
        ahost = os.environ.get('APP_HOST', args.host)
        aport = os.environ.get('APP_PORT', args.port)

        khost = os.environ.get('KAFKA_HOST', None)
        kport = os.environ.get('KAFKA_PORT', None)
        topic = os.environ.get('KAFKA_TOPIC', None)

        assert khost is not None

        app = Flask(__name__)
        CORS(app, supports_credentials=True)

        kafka_subproc, exit_event, msg_queue = None, None, None

        if args.task == 'srs.dvr':
            from frepai.srs.routers import api_srs
            exit_event, msg_queue = Event(), Queue()
            app.register_blueprint(api_srs, url_prefix='/apis/srs/v1')
            api_srs.queue = msg_queue
            with logger_subprocess(_ossrs_main_process,
                    khost, int(kport),
                    topic, exit_event, msg_queue) as proc:
                proc.start()
                kafka_subproc = proc
        else:
            if args.task == 'videoprocess.pre':
                from frepai.videoprocess.pre import video_preprocess as message_handler
                topic_in, topic_out = topic, 'engine_input'
            elif args.task == 'engine':
                from frepai.engine.engine import engine_process as message_handler
                topic_in, topic_out = 'engine_input', 'engine_output'
            elif args.task == 'videoprocess.post':
                from frepai.videoprocess.post import video_postprocess as message_handler
                topic_in, topic_out = 'engine_output', None
            else:
                raise RuntimeError(f'{args.task} is not support')

            with logger_subprocess(_repnet_main_process,
                    khost, int(kport),
                    topic_in, topic_out,
                    message_handler) as proc:
                proc.start()
                kafka_subproc = proc

        def signal_handler(sig, frame):
            logger.warning('handle signal: [%d]' % sig)
            if exit_event is not None:
                exit_event.set()
            if kafka_subproc.is_alive():
                kafka_subproc.terminate()
                logger.warning('terminate kafka subprocess!')
                kafka_subproc.join()
            os._exit(os.EX_OK)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        app.add_url_rule('/k8s/probe/liveness', 'liveness', _liveness_probe)
        app.add_url_rule('/k8s/probe/readiness', 'readiness', _readiness_probe)
        app.run(host=ahost, port=int(aport))
        # from gevent import pywsgi
        # server = pywsgi.WSGIServer((ahost, int(aport)), app)
        # server.serve_forever()
    finally:
        pass
