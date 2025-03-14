#!/bin/bash

curl http://localhost:11434/v1/chat/completions \
-H "Content-Type: application/json" \
-d @../prompts/simple.json | while IFS= read -r line; do
    case "$line" in
      data:*)
        # Извлекаем данные после 'data: ' и проверяем на [DONE]
        data_content=$(echo "$line" | sed 's/^data: //')
        if [ "$data_content" != "[DONE]" ]; then
          # Обрабатываем только валидный JSON
          content=$(echo "$data_content" | jq -r '.choices[0].delta.content // empty' 2>/dev/null)
          if [ -n "$content" ]; then
            echo -n "$content"
          fi
        fi
        ;;
    esac
  done
echo