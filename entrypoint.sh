#!/bin/bash
#=================================================================
# date: 2022-03-15 19:55:52
# title: entrypoint
# author: QRS
#=================================================================

APP_MODULE=$1
APP_PORT=1818

shift

while getopts "p:" OPT;
do
    case $OPT in
        p)
            APP_PORT=$OPTARG
            ;;
        *)
            echo $OPTARG
            ;;
    esac
done

while true;
do
    case ${APP_MODULE} in
        "videoprocess")
            python3 app_service.py --host 0.0.0.0 --port ${APP_PORT}
            ;;
        "engine")
            python3 app_service.py --host 0.0.0.0 --port ${APP_PORT}
            ;;
        *)
            echo "APP_MODULE[$APP_MODULE] error"
            break;;
    esac
    sleep 3
done
