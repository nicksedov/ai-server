from fastapi import APIRouter
from fastapi.responses import FileResponse, RedirectResponse

router = APIRouter(include_in_schema=False)

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
    
@router.get('/jpeg/{jpeg_file}')
async def avatar(jpeg_file: str):
    return FileResponse(f'resources/jpeg/{jpeg_file}', media_type='image/jpeg')
