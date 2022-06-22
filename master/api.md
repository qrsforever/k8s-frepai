# logcat: `/master/logcat`


# eyeai: `/master/task/eyeai`


### yolor

- train start:

```
curl -X POST -H 'Content-Type: application/json' http://factory.admin.hzcsdata.com/master/task/eyeai -d '{"pigeon": {"uuid": "admin"}, "uuid": "admin", "do": "train.start", "network": "yolor", "mserver_url": "http://master-service.system.svc.cluster.local:4848/echo", "dataset_url": "https://frepai-1301930378.cos.ap-beijing.myqcloud.com/datasets/maskfaces.zip", "dataset_fmt": "yolo_txt", "epochs_num": 3, "cos_key": "/weights/yolor/v1/", "weight_url": "https://frepai-1301930378.cos.ap-beijing.myqcloud.com/weights/yolor/v1/best.pt"}'
curl -X POST -H 'Content-Type: application/json' http://82.157.10.200:30048/master/task/eyeai -d '{"pigeon": {"uuid": "admin"}, "uuid": "admin", "do": "train.start", "network": "yolor", "mserver_url": "http://master-service.system.svc.cluster.local:4848/echo", "dataset_url": "https://frepai-1301930378.cos.ap-beijing.myqcloud.com/datasets/maskfaces.zip", "dataset_fmt": "yolo_txt", "epochs_num": 300, "cos_key": "/weights/yolor/v1/", "weight_url": "https://frepai-1301930378.cos.ap-beijing.myqcloud.com/weights/yolor/v1/best.pt"}'
```

```
curl -X POST -H 'Content-Type: application/json' http://82.157.10.200:30048/master/task/eyeai -d '{"pigeon":{"uuid":84},"uuid":84,"network":"yolor","cos_key":"/weights/task/0/1/","do":"train.start","mserver_url":"http://master-service.system.svc.cluster.local:4848/echo","dataset_url":"https://frepai-1301930378.cos.ap-beijing.myqcloud.com/datasets/maskfaces.zip","dataset_fmt":"yolo-txt","epochs_num":200}'
```

- train stop:

```
curl -X POST -H 'Content-Type: application/json' http://82.157.10.200:30048/master/task/eyeai -d '{"pigeon": {"uuid": "admin"}, "uuid": "admin", "do": "train.stop", "network": "yolor", "mserver_url": "http://master-service.system.svc.cluster.local:4848/echo"}'
```

- deploy install:

```
curl -X POST -H 'Content-Type: application/json' http://82.157.10.200:30048/master/task/eyeai -d '{"uuid": "admin", "do": "deploy.install", "network": "yolor", "weight_url": "https://frepai-1301930378.cos.ap-beijing.myqcloud.com/weights/yolor/v1/best.pt", "replicas":1}'
```

- deploy uninstall:

```
curl -X POST -H 'Content-Type: application/json' http://82.157.10.200:30048/master/task/eyeai -d '{"uuid": "admin", "do": "deploy.uninstall", "network": "yolor"}'
```

- inference

```
curl -X POST -H 'Content-Type: application/json' http://factory.admin.hzcsdata.com/eyeai/yolor/inference -d '{"sources": [{"id": "1", "url": "https://frepai-1301930378.cos.ap-beijing.myqcloud.com/datasets/test1.jpg"}, {"id": "2", "url": "https://frepai-1301930378.cos.ap-beijing.myqcloud.com/datasets/test2.jpg"}]}'
```
