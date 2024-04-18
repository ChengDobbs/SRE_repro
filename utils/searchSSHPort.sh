#!/bin/bash
HOST="10.1.3.100"
# read from applPorts.txt
ports=$(cat utils/applPorts.txt)

while [ true ]; do
    for port in $(echo $ports | tr " " "\n")
    do
        # echo $port
        ssh root@$HOST -p $port
    done
    sleep 300
done