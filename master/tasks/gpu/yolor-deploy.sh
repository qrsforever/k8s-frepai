#!/bin/bash
#=================================================================
# date: 2022-06-13 17:54:38
# title: yolor-deploy
# author: QRS
#=================================================================


ACCELERATOR=${ACCELERATOR:-nvidia-tesla-t4}
DEPLOYNAME=${DEPLOYNAME:-unkown}
NAMESPACE=${NAMESPACE:-eyeai}
REPLICAS=${REPLICAS:-1}
WEIGHT_URL=${WEIGHT_URL}

deployments=$(kubectl create -o name -f - << EOF
kind: Service
apiVersion: v1
metadata:
  name: ${DEPLOYNAME}-service
  namespace: ${NAMESPACE}
spec:
  selector:
    app: ${DEPLOYNAME}-app
  type: NodePort
  ports:
    - name: infer-port
      port: 4871
      targetPort: infer-port
      nodePort: 31871

---

apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${DEPLOYNAME}-deployment
  namespace: ${NAMESPACE}
  labels:
    app: ${DEPLOYNAME}-app
spec:
  selector:
    matchLabels:
      app: ${DEPLOYNAME}-app
  replicas: ${REPLICAS}
  template:
    metadata:
      labels:
        app: ${DEPLOYNAME}-app
    spec:
      containers:
        - name: eyeai-yolor-infer
          image: hzcsk8s.io/models/yolor
          imagePullPolicy: IfNotPresent
          ports:
            - name: infer-port
              containerPort: 4871

          command:
            - /bin/sh
            - -c
            - |
              cd /app/yolor
              python3 inference.py --weights ${WEIGHT_URL} --img-size 640 --cfg cfg/yolor_p6.cfg --device "0"

          resources:
            limits:
              nvidia.com/gpu: 1

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
        accelerator: ${ACCELERATOR}

      restartPolicy: Always
      terminationGracePeriodSeconds: 3

---

apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ${DEPLOYNAME}-ingress
  namespace: ${NAMESPACE}
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/use-regex: 'true'
spec:
  rules:
    - host: factory.admin.hzcsdata.com
      http:
        paths:
          - path: /${NAMESPACE}/yolor/inference
            pathType: Prefix
            backend:
              service:
                name: ${DEPLOYNAME}-service
                port:
                  number: 4871
EOF
)

for i in `seq 160`
do
    status=$(kubectl get pods -n ${NAMESPACE} -l app=${DEPLOYNAME}-app --no-headers -o custom-columns=":status.phase")
    has_run=$(echo $status | grep "Running")
    if [[ x$has_run != x ]]
    then
        exit 0
    fi
    sleep 1
done

echo "deploy fail status:${status[*]}" >&2

exit 1
