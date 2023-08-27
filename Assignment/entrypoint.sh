#!/bin/bash

# "$@" -> This indicates that pass whatever arguments supplied during docker run command to main.py
# In this case model-name & image-path both will get passed automatically

export SOMETHING='something'  # Not relevant just sample code

python3 main.py "$@"