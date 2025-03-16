## Create docker image 
docker build -t ai-server .

## Run shell in docker container
docker exec -it ai-server-ai-server-1 /bin/bash

## Check container health
curl http://localhost:8000/v1/health
