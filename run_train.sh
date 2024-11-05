#!/bin/bash

export PYTHONPATH=$(pwd)
pip install matplotlib

python diffusion_model_v2/train_ddpm.py