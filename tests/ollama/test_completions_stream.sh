#!/bin/bash

curl -s http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwq:32b-q8_0",
    "messages": [{"role": "user", "content": "Напиши короткий рассказ о космосе"}],
    "stream": true
  }' | while IFS= read -r line; do
    if [[ $line == data:* ]]; then
      content=$(echo "$line" | sed 's/^data: //' | jq -r '.choices[0].delta.content // empty')
      if [ -n "$content" ]; then
        echo -n "$content"
      fi
    fi
  done
echo