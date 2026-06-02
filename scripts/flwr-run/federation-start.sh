#!/bin/bash

# Variables
rp="/home/admine/matej/federated-learning-poc" # repository path
lp="${rp}/src/federated/logs" # logs path
started_processes=""

# Virtual environment
cd ${rp} # !important
source .venv/bin/activate
# Updates
git pull

# Federated learning

## Preparation

### Superlink
flower-superlink --insecure &> ${lp}/super_link_log.txt &
started_processes="${started_processes},${!}"

### Supernodes
n=1
while [ ${n} -lt 4 ] # 3 hospitals
do
    echo "Hospital ${n}"
    port=9093+${n}
    dataset-filename="h_${n}202605311423.csv"

    flower-supernode \
      --insecure \
      --superlink 127.0.0.1:9092 \
      --clientappio-api-address 127.0.0.1:${port} \
      --node-config "dataset-filename=${dataset-filename}" &> ${lp}/h${n}_log.txt &

    started_processes="${started_processes},${!}"

    n=`expr $n + 1`
done

## Start (run) the process.
flwr run . local-deployment --stream &> ${lp}/run_log.txt &
started_processes="${started_processes},${!}"

echo $started_processes

# Manually running the federation.
## Hospital 1
#flower-supernode --insecure --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9094 --node-config 'dataset-filename="h1_202605311423.csv"' &> $lp/h1_log.txt &
## Hospital 2
#flower-supernode --insecure --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9095 --node-config 'dataset-filename="h2_202605311423.csv"' &> $lp/h2_log.txt &
## Hospital 3
#flower-supernode --insecure --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9096 --node-config 'dataset-filename="h3_202605311423.csv"' &> $lp/h3_log.txt &