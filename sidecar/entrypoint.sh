#!/bin/bash
set -e

exec vllm serve "$MODEL_ID" --trust-remote-code "$@"
