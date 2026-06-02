#!/bin/bash

echo "FEDERATED LEARNING (Obstructive Apnea Predictions)"

# Variables
rp="/home/admine/matej/federated-learning-poc" # repository path
lp="${rp}/src/federated/logs" # logs path
started_processes=""
echo "Logs can be found at ${lp}"


# Virtual environment
echo "Setting up virtual environment"
cd ${rp} # !important
source .venv/bin/activate

# Updates
echo "Setting up git (fetching updates)"
{git stash && git pull} &> ${lp}/git.log

# Federated learning

## Preparation

### Superlink
echo "Setting up flower superlink"
flower-superlink --insecure &> ${lp}/super_link.log &
started_processes="${!}"

### Supernodes
echo "Setting up flower supernodes"
n=1
while [ ${n} -lt 4 ] # 3 hospitals
do
    port=9093+${n}
    dataset_filename="h${n}_202605311423.csv"

    echo "    Setting up hospital ${n}"
    flower-supernode \
      --insecure \
      --superlink 127.0.0.1:9092 \
      --clientappio-api-address 127.0.0.1:${port} \
      --node-config "dataset-filename=${dataset-filename}" &> ${lp}/h${n}.log &

    started_processes="${started_processes} ${!}"

    n=`expr $n + 1`
done

## Start (run) the process.
echo "Setting up flower run (starting the learning process)"
flwr run . local-deployment --stream &> ${lp}/run.log &
started_processes="${started_processes} ${!}"

# Save process ids.
echo "Started processes: ${started_processes}"
echo $started_processes &> ${lp}/processes.log
echo "Kill processes with command: kill ${started_processes}"


# Manually running the federation.
## Hospital 1
#flower-supernode --insecure --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9094 --node-config 'dataset-filename="h1_202605311423.csv"' &> $lp/h1.log &
## Hospital 2
#flower-supernode --insecure --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9095 --node-config 'dataset-filename="h2_202605311423.csv"' &> $lp/h2.log &
## Hospital 3
#flower-supernode --insecure --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9096 --node-config 'dataset-filename="h3_202605311423.csv"' &> $lp/h3.log &