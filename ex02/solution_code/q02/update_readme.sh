#!/usr/bin/env bash
# File       : update_readme.sh
# Description: Update the Readme.html in the skeleton codes
# Copyright 2020 ETH Zurich. All Rights Reserved.
pandoc --standalone --mathml -t html \
    -c "../.github-pandoc.css" \
    -o "../../skeleton_code/ispc_gemm/Readme.html" "Readme.md"
