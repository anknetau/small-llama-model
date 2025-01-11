#!/bin/sh

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd ${script_dir}

export PYTHONPATH=${script_dir}/src:${script_dir}/src/small_llama_model
pytest --color=no
