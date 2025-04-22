from fastapi import APIRouter
from fastapi.responses import FileResponse, RedirectResponse
import os

router = APIRouter(include_in_schema=False)

# Словарь с соответствиями расширений файлов и MIME-типов
MIME_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
    "svg": "image/svg+xml",
    "bmp": "image/bmp",
    "ico": "image/vnd.microsoft.icon"
}

@router.get('/')
async def index():
    return RedirectResponse('/html/index.html')

@router.get('/favicon.ico')
async def favicon():
    return FileResponse('resources/favicon.ico')

@router.get('/html/{html_file}')
async def html(html_file: str):
    return FileResponse(f'resources/html/{html_file}', media_type='text/html')

@router.get('/css/{css_file}')
async def css(css_file: str):
    return FileResponse(f'resources/css/{css_file}', media_type='text/css')
    
@router.get('/img/{img_file}')
async def image(img_file: str):
    file_path = f'resources/img/{img_file}'
    # Получаем расширение файла из имени
    file_extension = os.path.splitext(img_file)[1].lower().lstrip('.')
    # Получаем соответствующий MIME-тип или используем значение по умолчанию
    media_type = MIME_TYPES.get(file_extension, "application/octet-stream")
    return FileResponse(file_path, media_type=media_type)
