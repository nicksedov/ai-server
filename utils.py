from googletrans import Translator
import asyncio

async def translate_to_english(prompt: str, language: str):
    if language != 'en':
        translator = Translator()
        prompt_task = asyncio.get_running_loop().create_task(
            translator.translate(prompt, src=language, dest='en'))
        translated = await prompt_task
        print(f"Translated from {language} to en: {translated.text}")
        return translated.text
    return prompt