#!/bin/bash

curl -vvvv http://localhost:11434/v1/images/generations \
  -H "Content-Type: application/json" \
  -d @../prompts/image_generation.json