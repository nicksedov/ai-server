#!/bin/sh

curl -X POST \
 -H 'Content-Type: application/json' \
 -H 'Content-Language: ru' \
 -d @../prompts/image_generation.json \
 http://localhost:8000/v1/images/generations