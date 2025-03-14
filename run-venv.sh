#!/bin/bash

VENV_HOME=/srv/dev-disk-by-uuid-4315ace7-b3ba-428c-bdf1-bd9b23f4035c/venv

source $VENV_HOME/bin/activate
pip install -r requirements.txt
python main.py --config config-default.yaml

uvicorn main:app

