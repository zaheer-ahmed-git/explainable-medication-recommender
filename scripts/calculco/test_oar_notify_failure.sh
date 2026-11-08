#!/usr/bin/env bash
#OAR -n rm_test_notify_failure
#OAR -l /nodes=1/core=1,walltime=00:05:00
#OAR -p gpudevice='-1'

set -euo pipefail
echo "OAR notify failure probe started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
exit 42
