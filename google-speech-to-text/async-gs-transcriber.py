#!/usr/bin/env python
"""
Transcribe file from Google Cloud Storage using Google Speech-To-Text
"""
import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime as dt
from datetime import timedelta
from time import time
from typing import Any, Dict, Generator, List, Union
from urllib.parse import quote_plus

from google.cloud import speech_v1p1beta1
from google.cloud.speech_v1p1beta1 import enums, types
from google.protobuf.json_format import MessageToDict

from speakers import get_speaker_name

TIMEOUT = 9000
RECOGNITION_CONFIG = {
    "encoding": enums.RecognitionConfig.AudioEncoding.MP3,
    "sample_rate_hertz": 44100,
    "enable_speaker_diarization": True,
    "diarization_speaker_count": 4,
    "language_code": "ru-RU",
}


RecognizedWordMeta = Dict[str, Union[str, int]]
to_time = lambda time_str: timedelta(milliseconds=int(float(time_str.rstrip("s")) * 1000))


@dataclass
class Phrase:
    start: timedelta
    finish: timedelta
    speaker: str
    text: str


def transcribe_gcs(gcs_uri: str) -> Dict[str, List[Any]]:
    """Asynchronously transcribes the audio file specified by the gcs_uri."""
    client = speech_v1p1beta1.SpeechClient()

    # The language of the supplied audio
    audio = types.RecognitionAudio(uri=gcs_uri)
    operation = client.long_running_recognize(RECOGNITION_CONFIG, audio)

    print(dt.now(), "Waiting for operation to complete...")
    print(dt.now(), "Operation", operation.operation)
    start = time()
    response = operation.result(timeout=90000)

    print(dt.now(), "Got response in ", time() - start)

    return MessageToDict(response)


def save_transcript(response_data, transcripts_dir: str) -> None:
    """
    Save converted transcipt into `./transcripts`
    """
    speakers = defaultdict(lambda: get_speaker_name().title())

    count = 0
    for result in response_data["results"]:
        if not (result["alternatives"][0] and "speakerTag" in result["alternatives"][0]["words"][0]):
            continue

        with open(os.path.join(transcripts_dir, f"transcript.{count}.txt"), "w", encoding="utf-8") as f:
            for phrase in get_phrases(result["alternatives"][0]["words"], speakers):
                f.write(f"[{phrase.start} - {phrase.finish}] {phrase.speaker} : {phrase.text}\n")

            count += 1


def get_phrases(words: List[RecognizedWordMeta], speakers: Dict[int, str]) -> Generator[Phrase, None, None]:
    """
    Convert separate words and recognition meta-data to individual speaker phrases
    """
    phrase_start = timedelta(milliseconds=0)
    phrase_end = timedelta(milliseconds=0)
    phrase_words = []
    phrase_speaker = None
    word_meta = None

    for word_meta in words:
        word_speaker = speakers[word_meta["speakerTag"]]
        if phrase_speaker is None:
            phrase_speaker = word_speaker

        if phrase_speaker != word_speaker:
            yield Phrase(
                start=phrase_start, finish=phrase_end, speaker=phrase_speaker, text=" ".join(phrase_words),
            )

            phrase_start = to_time(word_meta["startTime"])
            phrase_speaker = word_speaker
            phrase_words = []

        phrase_end = to_time(word_meta["endTime"])
        phrase_words.append(word_meta["word"])

    if not word_meta:
        return

    yield Phrase(
        start=phrase_start, finish=phrase_end, speaker=phrase_speaker, text=" ".join(phrase_words),
    )


def run(gcs_uri: str) -> None:
    """
    Try to transcribe file at {gcs_url} using Google Speech-To-Text.
    Writes result to `./transcripts/gs-<filename>-<timeastamp>/`
    """
    response_data = transcribe_gcs(gcs_uri)

    fname = quote_plus(os.path.basename(gcs_uri))
    transcripts_dir = os.path.join("transcripts", f"gs-{fname}-{int(dt.now().timestamp())}")
    os.mkdir(transcripts_dir)

    # save full Google Speech-To-Text service response to `./transcripts/gs-<filename>-<timeastamp>/response-full.json`
    with open(os.path.join(transcripts_dir, "response-full.json"), "w", encoding="utf-8") as f:
        json.dump(response_data, f, indent=2, ensure_ascii=False)

    # save transcript to `./transcripts/gs-<filename>-<timeastamp>/transcript.{index}.txt`
    save_transcript(response_data, transcripts_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="File or GCS path for audio file to be recognized")
    args = parser.parse_args()
    if not args.path.startswith("gs://"):
        RuntimeError("only GCS links are allowed")

    run(args.path)
