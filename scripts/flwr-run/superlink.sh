#!/bin/bash
cd ~/matej/federated-learning-poc
source .venv/bin/activate
git pull
flower-superlink --insecure 2>&1 | tee ~/matej/federated-learning-poc/src/federated/super_link_log.txt