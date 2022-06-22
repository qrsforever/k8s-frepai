#!/bin/bash
#=================================================================
# date: 2022-05-09 16:33:29
# title: logcat
# author: QRS
#=================================================================

export KUBECONFIG=/etc/kubernetes/admin.conf

TARGETS=(
    "frepai@videoprocess-pre-deployment"
    "frepai@engine-deployment"
    "frepai@videoprocess-post-deployment"

    "talentai@talentai-factory"
    "talentai@task-server"
    "talentai@mqs-server"
    "talentai@rpc-server"
)

__logcat() {
    utc=$(date -u +"%s")
    st=$1
    et=$2
    [[ x${st} == x ]] && st=$(date --date=@$(expr $utc - 300) -Iseconds)
    [[ x${et} == x ]] && et=$(date --date=@${utc} -Iseconds)

    if [[ ${#st} == 19 ]]
    then
        st="${st}+08:00"
    fi

    if [[ ${#et} == 19 ]]
    then
        et=$(date --date="${et}+08:00" -u -Iseconds | cut -c1-19)
    fi

    logdir=/tmp/${utc}

    mkdir -p ${logdir}

    cd ${logdir}

    cmdlog=cmd.log

    for item in ${TARGETS[@]}
    do
        nspace=${item%@*}
        deploy=${item#*@}
        echo "${nspace} / ${deploy}"
        kubectl --namespace ${nspace} logs deployments/${deploy} \
            --prefix=true --all-containers=true \
            --ignore-errors --timestamps --tail -1 \
            --since-time $st | tee ${item}.log | \
            awk -v til=${et} '{if ($2 < til) {print} else {print; exit}}'

        cat >> ${cmdlog} << EOF
kubectl --namespace ${nspace} logs deployments/${deploy} \
--prefix=true --all-containers=true \
--ignore-errors --timestamps --tail -1 \
--since-time $st | tee ${item}.log | \
awk -v til=${et} '{if ($2 < til) {print} else {print; exit}}'

EOF
    done

    tar zcf /tmp/logs.tar.gz *

    cd - >/dev/null
    rm -rf ${logdir}
    echo "${st} @ ${et}"

}

# TODO for flush k8s log
sleep 5

__logcat $*
