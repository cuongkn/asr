import glob
import json
import os
import subprocess
import tarfile
import librosa
import wget
from tqdm import tqdm

class LibriDataset():
    def __init__(self, option: str, data_dir: str = 'data/libri/') -> None:
        super().__init__()

        self.data_dir = data_dir
        self.option = option
        self.path = 'LibriSpeech/' + self.option

        self.prepare_data()

        # Get transcript path list
        transcript_path_lst = list()
        for r1 in os.listdir(os.path.join(self.data_dir, self.path)):
            for r2 in os.listdir(os.path.join(self.data_dir, self.path, r1)):
                for r3 in os.listdir(os.path.join(self.data_dir, self.path, r1, r2)):
                    if r3[-4:] == '.txt':
                        transcript_path_lst.append(os.path.join(self.path, r1, r2, r3))

        # Building Manifests
        print('Bulding Manifests for dataset...')

        self.manifest_path = self.data_dir + '/LibriSpeech/' + self.option + '-manifest.json'

        if not os.path.isfile(self.manifest_path):
            for transcript_dir in tqdm(transcript_path_lst):
                transcripts_path = os.path.join(self.data_dir, transcript_dir)
                self.build_manifest(transcripts_path, self.manifest_path, self.path)
        print("***Done***")

    def prepare_data(self):
        mirror = self.option + ".tar.gz"
        if not os.path.exists(self.data_dir + mirror):
            print(f"Downloading {self.option} dataset...")
            libri_url = "https://www.openslr.org/resources/12/" + mirror
            libri_path = wget.download(libri_url, self.data_dir)
            print(f"Dataset downloaded at: {libri_path}")
        else:
            print("Tarfile already exists.")
            libri_path = self.data_dir + mirror.replace("-", "_")
        
        if not os.path.exists(self.data_dir + self.path):
            tar = tarfile.open(libri_path)
            tar.extractall(path=self.data_dir)

            print("Converting .flac to .wav...")
            flac_list = glob.glob(self.data_dir + 'LibriSpeech/**/*.flac', recursive=True)
            for flac_path in flac_list:
                wav_path = flac_path[:-5] + '.wav'
                cmd = ["sox", flac_path, wav_path]
                subprocess.run(cmd)
        print("Finished conversion.\n******")

    def build_manifest(self, transcripts_path, manifest_path, wav_path):
        with open(transcripts_path, 'r') as fin:
            with open(manifest_path, 'a') as fout:
                for line in fin:

                    transcript = ' '.join(line.split(' ')[1:]).lower()
                    file_id = line.split(' ')[0]

                    audio_path = os.path.join(
                        self.data_dir,
                        wav_path,
                        file_id[:file_id.find('-')],
                        file_id[file_id.find('-')+1 : file_id.rfind('-')],
                        file_id + '.wav')

                    duration = librosa.core.get_duration(filename=audio_path)

                    metadata = {
                        "audio_filepath": audio_path,
                        "duration": duration,
                        "text": transcript
                    }
                    json.dump(metadata, fout)
                    fout.write('\n')

def concatenate_files(file1_path, file2_path, file3_path):
    try:
        with open(file1_path, 'r', encoding='utf8') as file1:
            content1 = file1.read()

        with open(file2_path, 'r', encoding='utf8') as file2:
            content2 = file2.read()

        concatenated_content = content1 + content2

        with open(file3_path, 'w', encoding='utf8') as file3:
            file3.write(concatenated_content)

        print(f"Contents of {file1_path} and {file2_path} have been concatenated and saved to {file3_path}")

    except FileNotFoundError:
        print("One or more files not found.")

if __name__ == "__main__":
    dev_clean = LibriDataset(option="dev-clean")
    dev_other = LibriDataset(option="dev-other")
    test_clean = LibriDataset(option="test-clean")
    test_other = LibriDataset(option="test-other")
    # train_clean = LibriDataset(option="train-clean-100")
    concatenate_files(dev_clean.manifest_path, dev_other.manifest_path, 'data/libri/LibriSpeech/libri_train.json')
    concatenate_files(test_clean.manifest_path, test_other.manifest_path, 'data/libri/LibriSpeech/libri_test.json')
