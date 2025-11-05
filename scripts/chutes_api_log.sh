#! /bin/bash

while true; do
  curl -s 'https://api.chutes.ai/chutes/?template=vllm&include_public=true&limit=1000' | jq > api_logs/chutes_$(date +%m%dT%H).json
  sleep 3600
done
