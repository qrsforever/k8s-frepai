#!/bin/bash
#=================================================================
# date: 2022-03-16 17:48:24
# title: set_node
# author: QRS
#=================================================================

node=$1
acce=${2:-"nvidia-tesla-t4"}

kubectl taint --overwrite nodes $node nvidia.com/gpu:NoSchedule
kubectl label --overwrite nodes $node nvidia.com/gpu="true"
kubectl label --overwrite nodes $node accelerator=$acce

# kubectl taint nodes $node nvidia.com/gpu:NoSchedule-
# kubectl create -f nvidia-device-plugin.yml
