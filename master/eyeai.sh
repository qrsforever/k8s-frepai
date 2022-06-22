#!/bin/bash
#=================================================================
# date: 2022-06-10 10:01:16
# title: eyeai
# author: QRS
#=================================================================

CUR_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)

export KUBECONFIG=/etc/kubernetes/admin.conf


inputs=$1
pigeon=$(echo $inputs | jq -rc '.pigeon')
dotask=$(echo $inputs | jq -rc '.do')
uuid=$(echo $inputs | jq -rc '.uuid')

task=${dotask%.*}
oper=${dotask#*.}

export NAMESPACE=eyeai

__do_train() {
    network=$(echo $inputs | jq -rc '.network')
    case $network in
        "yolor")
            jobname=yolor-train-${uuid}

            exist=$(kubectl get jobs -n ${NAMESPACE} --no-headers -o custom-columns=:metadata.name | grep -e "^${jobname}$")

            if [[ x$1 == xstop ]]
            then
                if [[ x$exist != x ]]
                then
                    kubectl -n ${NAMESPACE} delete jobs ${jobname} 2>/dev/null
                fi
                exit 0
            fi
            if [[ x$exist != x ]]
            then
                echo "job ${jobname} already exist!" >&2
                exit -1
            fi
            export PIGEON=$pigeon
            export JOBNAME=${jobname}
            export ACCELERATOR=nvidia-tesla-t4
            export BATCH_SIZE=8
            export WORKER_NUM=4
            export EPOCHS_NUM=$(echo $inputs | jq -rc '.epochs_num')
            export DATASET_URL=$(echo $inputs | jq -rc '.dataset_url')
            export MSERVER_URL=$(echo $inputs | jq -rc '.mserver_url')
            export CLOUD_COSS3=$(echo $inputs | jq -rc '.cos_key')
            export WEIGHT_URL=$(echo $inputs | jq -rc '.weight_url')
            ${CUR_DIR}/tasks/gpu/yolor-train.sh $1
            ;;
        *)
            ;;
    esac
}

__do_deploy() {
    network=$(echo $inputs | jq -rc '.network')
    case $network in
        "yolor")
            deployname=yolor-infer-${uuid}
            exist=$(kubectl get deployments -n ${NAMESPACE} --no-headers -o custom-columns=:metadata.name | grep -e "^${deployname}-deployment$")

            if [[ x$1 == xuninstall ]]
            then
                if [[ x$exist != x ]]
                then
                    kubectl -n ${NAMESPACE} delete ingress ${deployname}-ingress
                    kubectl -n ${NAMESPACE} delete deployments ${deployname}-deployment
                    kubectl -n ${NAMESPACE} delete services ${deployname}-service
                fi
                exit 0
            fi
            if [[ x$exist != x ]]
            then
                echo "deployment ${deployname} already exist!" >&2
                exit -1
            fi
            export DEPLOYNAME=${deployname}
            export ACCELERATOR=nvidia-tesla-t4
            export WEIGHT_URL=$(echo $inputs | jq -rc '.weight_url')
            export REPLICAS=$(echo $inputs | jq -rc '.replicas')
            ${CUR_DIR}/tasks/gpu/yolor-deploy.sh $1
            ;;
        *)
            echo "network:[$network] error!" >&2
            ;;
    esac
}


case $task in
    "train")
        __do_train $oper
        ;;
    "deploy")
        __do_deploy $oper
        ;;
    *)
        echo "task:[$task] error!" >&2
        ;;
esac
