import json
import nemo.collections.asr as nemo_asr
import torch
import pandas as pd
from jiwer import wer, process_words, visualize_alignment


def convert_manifest_to_df(manifest_path):
    audio_filepath = list()
    duration = list()
    text = list()
    with open(manifest_path, encoding="utf8") as f:
        for line in f:
            metadata = json.loads(line)
            audio_filepath.append(metadata['audio_filepath'])
            duration.append(metadata['duration'])
            text.append(metadata['text'])

    return pd.DataFrame({'audio_filepath': audio_filepath,
                          'duration': duration,
                          'text': text})

if __name__ == "__main__":
    quartznet = nemo_asr.models.EncDecCTCModel.restore_from('quartznet_1.nemo')
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    quartznet.to(device)

    df = convert_manifest_to_df('data/vivos/vivos/test_manifest.json')
    audio_path = df['audio_filepath'].tolist()
    reference = df['text'].tolist()

    hypothesis = quartznet.transcribe(paths2audio_files=audio_path)
    print(wer(reference, hypothesis))

    out = process_words(reference, hypothesis)