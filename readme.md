# AI Server

## Create docker image 
docker build -t ai-server .

## Run shell in docker container
docker exec -it ai-server-ai-server-1 /bin/bash

## Check container health
curl http://localhost:8000/v1/health

## Infer models with AI Server with Swagger UI
### Swagger UI:
http://omv-nas:8000/docs
### Endpoint: 
/v1/images/generations
### Header:
Content-Language: ru
### Body:
'''json
{
    "model": "black-forest-labs/FLUX.1-dev", 
    "prompt": "Изобрази мужчину-политика в энергичной позе: он поднимает сжатую в кулак правую руку вверх, словно обращаясь к толпе. Его внешность: светлые, почти белые волосы, зачесанные назад и вбок с характерным объемом, густые брови, слегка нахмуренные, выразительные морщины вокруг глаз и рта. Лицо округлое, с подчеркнутыми чертами, включая выразительные скулы, выступающий подбородок. На нем надета ярко-красная бейсболка с белой надписью: «Make DISK great again» (шрифт жирный, заглавный). Одет политик в классический синий костюм с широкими лацканами, белая рубашка и длинный красный галстук с узким узлом. Фигура слегка полноватая, но подчеркнуто уверенная. Фон — размытая толпа с американскими флагами и плакатами, создающая атмосферу митинга. Освещение яркое, акцент на лице и жесте. Стиль: реализм с легкой карикатурной exaggeration, портретное сходство с Дональдом Трампом.",
    "n": 1,
    "size": "1280x720",
    "steps": 50
}
'''

### Endpoint: 
/v1/images/generations
### Header:
Content-Language: ru
### Body:
'''json
{ 
     "model": "qwq:32b-q8_0", 
     "messages": [
          {"role": "system", 
          "content": "Тебя зовут ДискоБот, ты бот команды Диск."}, 
          {"role": "system", 
          "content": "Команда Диск - это владелец продукта Юля, тимлид Коля, бэкенд-разработчики: Кирилл, Валера, Дима и Леша, фронтенд-разработчики: Женя, Илья и Лера, DevOps-инженеры Егор, Женя и Дима. Аналитики: Марина, Даша, Катя и Лиза. Дизайнер Антон, Тест-лид Илья, тестировщики: Женя и Игорь"},
          {"role": "system", 
          "content": "Команде помогают инженеры сопровождения - Леша Волков и Леша Зайцев, Валя и Витя."},
          {"role": "system", 
          "content": "Учаснники команды живут в разных городах России - Новосибирске, Москве, Петербурге, Екатеринбурге, Наро-фоминске."},
          {"role": "system", 
          "content": "Команда развивает продукт Диск - корпоративное облачное хранилище, предназначенное для использования сотрудниками Сбера, дочерних обществ и компаний экосистемы."},
          {"role": "system", 
          "content": "Каждому пользователю Диска предоставляется персональное пространство для хранения файлов размером 10 гигабайт."}, 
          {"role": "system", 
          "content": "Твоя задача рассказывать команде Диск о важных событиях. Примеры событий - ИФТ (интеграционное тестирование), ПСИ (приемо-сдаточные испытания), внедрение продукта в промышленную эксплуатацию."}, 
          {"role": "system", 
          "content": "Отвечай в дружеском стиле с элементами иронии."}, 
          {"role": "system", 
          "content": "Следи за правильным употреблением падежей и родов существительных."},
          {"role": "user", 
          "content": "Расскажи что ты вернулся после долгого перерыва, во время которого немного подучился и поумнел. Напиши как ты рад всех снова увидеть и поприветствуй каждого члена команды лично."}
     ],
     "max_tokens": 8192,
     "temperature": 0.4
}
'''
### Available models
qwq:32b-q8_0
