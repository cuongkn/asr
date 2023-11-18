import glob
import json
import os
from pathlib import Path
import subprocess
import tarfile
import librosa
import wget
from torch.utils.data import Dataset
from collections import defaultdict


class VivosDataset(Dataset):
    def __init__(self, type: str, data_dir: str = 'data/vivos/') -> None:
        super().__init__()

        self.data_dir = data_dir

        self.prepare_data()    

        if type == "train":
            self.transcripts_path = os.path.join(self.data_dir, 'vivos/train/prompts.txt')
            self.manifest_path = os.path.join(self.data_dir, 'vivos/train_manifest.json')
            self.path = os.path.join(self.data_dir, 'vivos/train/waves')
        elif type == "test":
            self.transcripts_path = os.path.join(self.data_dir, 'vivos/test/prompts.txt')
            self.manifest_path = os.path.join(self.data_dir, 'vivos/test_manifest.json')
            self.path = os.path.join(self.data_dir, 'vivos/test/waves')
        else:
            assert False, "only type 'train' or 'test'"
        
        if not os.path.isfile(self.manifest_path):
            self.build_manifest(self.transcripts_path, self.manifest_path, self.path)
            print(f"{type} manifest created.")

        print(self.manifest_path)
        
        self.data = self.read_manifest(self.manifest_path)
        
        self.files = glob.glob(self.path + "/**/*.wav", recursive=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        item = self.data[index]
        return item

    def prepare_data(self):
        if not os.path.exists(self.data_dir + '/vivos.tar.gz'):
            Path(self.data_dir).mkdir(parents=True, exist_ok=True)
            vivos_url = "http://ailab.hcmus.edu.vn/assets/vivos.tar.gz"
            vivos_path = wget.download(vivos_url, self.data_dir)
            print(f"Dataset downloaded at: {vivos_path}")
        else:
            print("Tarfile already exists.")
            vivos_path = self.data_dir + '/vivos.tar.gz'

        tar = tarfile.open(vivos_path)
        tar.extractall(path=self.data_dir)

        if not os.path.exists(self.data_dir + '/vivos/'):
            print("Converting .sph to .wav...")   
            sph_list = glob.glob(self.data_dir + '/**/*.sph', recursive=True)
            for sph_path in sph_list:
                wav_path = sph_path[:-4] + '.wav'
                cmd = ["sox", sph_path, wav_path]
                subprocess.run(cmd)
        print("Finished conversion.\n******")

    def build_manifest(self, transcripts_path, manifest_path, wav_path):
        with open(transcripts_path, 'r', encoding="utf8") as fin:
            with open(manifest_path, 'w', encoding="utf8") as fout:
                for line in fin:

                    transcript = line[line.find(' ') : -1].lower()
                    transcript = transcript.strip()

                    file_id = line[: line.find(' ')]

                    audio_path = os.path.join(
                        wav_path,
                        file_id[file_id.find('V') : file_id.rfind('_')],
                        file_id + '.wav')

                    duration = librosa.core.get_duration(path=audio_path)

                    # Write the metadata to the manifest
                    metadata = {
                        "audio_filepath": audio_path,
                        "duration": duration,
                        "text": transcript
                        }
                    json.dump(metadata, fout, ensure_ascii=False)
                    fout.write('\n')

    def read_manifest(self, path):
        manifest = []
        with open(path, 'r', encoding='utf8', errors='ignore') as f:
            for line in f:
                line = line.replace("\n", "")
                data = json.loads(line)
                manifest.append(data)
        return manifest


def get_charset(manifest_data):
    charset = defaultdict(int)
    for row in manifest_data:
        text = row['text']
        for character in text:
            charset[character] += 1
    return charset

if __name__ == "__main__":
    train_dataset = VivosDataset(type="train")
    test_dataset = VivosDataset(type="test")

    from collections import defaultdict

    train_charset = get_charset(train_dataset.data)
    test_charset = get_charset(test_dataset.data)

    train_set = set(train_charset.keys())
    test_set = set(test_charset.keys())

    print(f"Number of tokens in train set : {len(train_set)}")
    print(f"Number of tokens in test set : {len(test_set)}")

    concatenated = train_set.union(test_set)
    concatenated_list = list(concatenated)

    json_file_path = os.path.join(train_dataset.data_dir, 'charset.json')

    with open(json_file_path, 'w', encoding="utf8") as json_file:
        json.dump(concatenated_list, json_file, ensure_ascii=False)

    print(f'The concatenated set has been written to {json_file_path}.')
