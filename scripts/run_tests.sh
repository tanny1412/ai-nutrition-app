#!/bin/bash
set -euo pipefail

python -m pytest backend/tests "$@"
