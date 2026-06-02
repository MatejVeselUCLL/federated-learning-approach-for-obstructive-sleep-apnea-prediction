#!/bin/bash

echo "FEDERATED LEARNING (Obstructive Apnea Predictions)"

# Variables
repo_path="/home/admine/matej/federated-learning-poc" # repository path
log_path="${repo_path}/logs" # logs path
log_counter=1
started_processes=""
echo "Logs can be found at ${log_path}"


# Virtual environment
echo "Setting up virtual environment"
cd ${repo_path} # !important
source .venv/bin/activate

# Updates
echo "Setting up git (fetching updates)"
{
git stash
git pull
} &> ${log_path}/${log_counter}-git.log
log_counter=$((log_counter+1))

# Federated learning

## Preparation

### Superlink
echo "Setting up flower superlink"
flower-superlink --insecure &> ${log_path}/${log_counter}-super_link.log &
log_counter=$((log_counter+1))
another_superlink_pid=$(ps aux | grep '[f]lower-superlink.*--control-api-address 127.0.0.1:39093' | awk '{ print $2; }')
started_processes="${!} ${another_superlink_pid}"

### Supernodes
echo "Setting up flower supernodes"
n=1
while [ ${n} -lt 4 ] # 3 hospitals
do
    port=$((9093+${n}))
    echo "POORT ${port}"
    dataset_filename="h${n}_202605311423.csv"
    log_filename="${log_counter}-h${n}.log"

    echo "    Setting up hospital ${n}"
    flower-supernode \
      --insecure \
      --superlink 127.0.0.1:9092 \
      --clientappio-api-address 127.0.0.1:${port} \
      --node-config "dataset-filename=\"${dataset-filename}\"" &> ${log_path}/${log_filename} &

    started_processes="${started_processes} ${!}"

    log_counter=$((log_counter+1))
    n=`expr $n + 1`
done

## Start (run) the process.
echo "Setting up flower run (starting the learning process)"
flwr run . local-deployment --stream &> ${log_path}/${log_counter}-run.log &
log_counter=$((log_counter+1))
started_processes="${started_processes} ${!}"

# Save process ids.
echo $started_processes &> ${log_path}/${log_counter}-processes.log
log_counter=$((log_counter+1))
echo "Kill processes with command: kill ${started_processes}"


# Manually running the federation.
## Hospital 1
#flower-supernode --insecure --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9094 --node-config 'dataset-filename="h1_202605311423.csv"' &> $log_path/h1.log &
## Hospital 2
#flower-supernode --insecure --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9095 --node-config 'dataset-filename="h2_202605311423.csv"' &> $log_path/h2.log &
## Hospital 3
#flower-supernode --insecure --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9096 --node-config 'dataset-filename="h3_202605311423.csv"' &> $log_path/h3.log &