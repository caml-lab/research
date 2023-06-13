#!/bin/bash

DIR_ROOT="/Users/noel/projects/caml/research/arxiveri"
EMBEDDING="ada-002"
# available embedding models: "ada-002", "embed-english-light-v2.0", "embed-english-v2.0", "embed-multilingual-v2.0"

python3 "${DIR_ROOT}/table_matcher.py" -e "${EMBEDDING}" -w