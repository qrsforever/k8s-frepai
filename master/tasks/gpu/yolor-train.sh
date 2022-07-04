#!/bin/bash
#=================================================================
# date: 2022-06-10 13:54:19
# title: yolor-train
# author: QRS
#=================================================================


ACCELERATOR=${ACCELERATOR:-nvidia-tesla-t4}
BATCH_SIZE=${BATCH_SIZE:-8}
WORKER_NUM=${WORKER_NUM:-4}
EPOCHS_NUM=${EPOCHS_NUM:-2}

CURL_POST="curl --connect-timeout 3 --max-time 5 -X POST --header 'Content-Type: application/json'"
DATASET_URL=${DATASET_URL:-https://frepai-1301930378.cos.ap-beijing.myqcloud.com/datasets/dataset.zip}
MSERVER_URL=${MSERVER_URL:-master-service.system.svc.cluster.local:4848/echo}
NSERVER_URL=master-service.system.svc.cluster.local:4848/echo
CLOUD_COSS3=${CLOUD_COSS3:-weights/unkown}

PIGEON=${PIGEON:-"{}"}
JOBNAME=${JOBNAME:-unkown}
NAMESPACE=${NAMESPACE:-eyeai}
WEIGHT_URL=${WEIGHT_URL}

if [[ x$WEIGHT_URL == xnull ]]
then
    WEIGHT_URL=""
fi

job=$(kubectl create -o name -f - << EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOBNAME}
  namespace: ${NAMESPACE}
spec:
  parallelism: 1
  completions: 1
  backoffLimit: 0
  ttlSecondsAfterFinished: 1
  template:
    spec:
      terminationGracePeriodSeconds: 1
      containers:
      - name: eyeai-yolor-train
        image: hzcsk8s.io/models/yolor
        imagePullPolicy: IfNotPresent
        command: ['sh', '-c']
        args:
          - |
            mkdir -p /data
            wget -q $DATASET_URL -O /data/dataset.zip
            unzip -q /data/dataset.zip -d /data

            names=\$(cat /data/dataset/info.json | jq -c '.label_names')
            length=\$(cat /data/dataset/info.json | jq -c '.label_names | length')

            cat > /data/dataset/data.yaml << EOF
            train: /data/dataset/train/images
            val: /data/dataset/valid/images
            nc: \$length
            names: \$names
            EOF

            cd /app/yolor
            python train.py \
                --pigeon '${PIGEON}' \
                --mserver_url ${MSERVER_URL} \
                --cloud_coss3 ${CLOUD_COSS3} \
                --batch-size ${BATCH_SIZE} \
                --workers ${WORKER_NUM} \
                --epochs ${EPOCHS_NUM} \
                --img 640 640 \
                --data /data/dataset/data.yaml \
                --hyp data/hyp.scratch.640.yaml \
                --cfg cfg/yolor_p6.cfg \
                --weights "${WEIGHT_URL}" \
                --device "0" \
                --project /data/yolor \
                --name ${JOBNAME}

            ${CURL_POST} ${MSERVER_URL} -d '{"pigeon": $PIGEON, "errno": 0, "lifecycle": "quit"}'

        lifecycle:
          postStart:
            exec:
              command:
                - /bin/sh
                - -c
                - |
                  ${CURL_POST} ${NSERVER_URL} -d '{"pigeon": $PIGEON, "errno": 0, "lifecycle": "start"}'

          preStop:
            exec:
              command:
                - /bin/sh
                - -c
                - |
                  ${CURL_POST} ${NSERVER_URL} -d '{"pigeon": $PIGEON, "errno": 0, "lifecycle": "stop"}'

        resources:
          limits:
            nvidia.com/gpu: 1

        startupProbe:
          exec:
            command: ["test", "-e", "/data"]
          initialDelaySeconds: 10
          timeoutSeconds: 30
          periodSeconds: 60
          failureThreshold: 2

        volumeMounts:
        - mountPath: /dev/shm
          name: cache-volume

      volumes:
      - name: cache-volume
        emptyDir:
          medium: Memory
          sizeLimit: "8Gi"

      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule

      nodeSelector:
        accelerator: "${ACCELERATOR}"
      restartPolicy: Never
EOF
)

err=1

for i in `seq 180`
do
    status=$(kubectl get pods -n ${NAMESPACE} -l job-name=${JOBNAME} --no-headers | grep ${JOBNAME} | awk '{print $3}')
    if [[ $status == Running ]] || [[ x$status == x ]]
    then
        exit 0
    elif [[ $status == Pending ]]
    then
        if [[ $i == 5 ]]
        then
            break
        fi
        sleep 1
    elif [[ $status =~ "Err" ]] || [[ $status =~ "BackOff" ]] || [[ $status =~ "No" ]]
    then
        break
    fi
    sleep 1
done

if [[ $err == 1 ]]
then
    echo "job [${JOBNAME}] is [${status}]!" >&2
    kubectl delete jobs -n ${NAMESPACE} ${JOBNAME} 2>/dev/null
    exit 1
fi
