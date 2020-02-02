## Попытка выполнить черновое транскрибирование (speech-to-text) подкаста Radio-T различными SaaS решениями

Для того, чтобы транскрибирование Radio-T требовало как можно меньше ручной работы, имеет смысл черновую версию генерить автоматически.

Возможные варианты для этого (все сервисы ниже поддерживают русский):
 - Yandex Speech Kit
 - Google Speech-To-Text
 - MS Azure Speech-To-Text
 - Wit.ai

Т.к. YSK и Wit.ai не умеют сами распознавать говорящего, перед попыткой заливки к ним надо сначала распилить запись на записи отдельных ведущих 
(да и в остальные сервисы надо попробовать так залить, Гугл тот же весьма посредственно распознает говорящего).


#### Варианты проверки
    1. короткий (чуть меньше 5 минут) сэмпл эпизода 686 для изначальной проверки
    2. короткий сэмпл + убранная при помощи noise reduction унца
    3. короткий сэмпл + убранная при помощи invert (минусовки) унца - TBD
    4. короткий сэмпл, распиленный на отдельные файлы по ведущим - TBD
    5. полный эпизод 686 с примененной обработкой по лучшему из варинтов выше 


Используемый сэмпл лежит на GCS gs://radio-t-transcribing-examples/rt_podcast686-sample.mp3

Результаты проверок лежат в [./transcripts](https://github.com/q210/radio-t-transcriptions/tree/master/transcripts)

### Выполненные проверки

#### Google Speech-To-Text

    - будет стоить $2.8-$3.6 за эпизод в зависимости от длинны и готовности делиться данными с Гуглом
    - умеет в распознавание говорящего на записи (diarization) из коробки, хотя на практике с подкастом работает так себе - часто путает людей. Возможно дело в неверно указанном количестве говорящих на записи - оно отличается для сэмпла и полного эпизода подкаста
    - требует чтобы файл больше 1 минуты длинной был залит в Google Cloud Storage 
    - вреия распознания сэмпла: ~3 минуты
    - время распознания полного эпизода: ~1 час


## Для разработчика

При выполнении всех скриптов рабочей директорией следует указывать корень репы.

После изменений python скриптов, желательно прогнать следующие линтеры и форматтеры:
 - `isort --lines 119 -y`
 - `black --line-length 120 --target-version py38 ./*/*.py`
 - `flake8 . --max-line-length=120`
