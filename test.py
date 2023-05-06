from google.transliteration import transliterate_text
from googletrans import Translator

translator = Translator()



user_query = input()
print(translator.translate(user_query).text)