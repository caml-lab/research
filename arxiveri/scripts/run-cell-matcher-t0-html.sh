#!/bin/bash

DIR_ROOT="/Users/noel/projects/caml/research/arxiveri"
TEMPERATURE=0.0

python3 "${DIR_ROOT}/cell_matcher.py" --dir_root "${DIR_ROOT}" --temperature "${TEMPERATURE}"