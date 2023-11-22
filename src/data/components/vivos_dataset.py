import glob
import subprocess
import tarfile
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
import os
import wget


class VivosDataset(Dataset):
    def __init__(self, subset: str, root: str, n_fft: int = 159):
        super().__init__()
        self.root = root
        self.subset = subset
        assert self.subset in ["train", "test"], "subset not found"

        path = os.path.join(self.root, self.subset)
        waves_path = os.path.join(path, "waves")
        transcript_path = os.path.join(path, "prompts.txt")

        # walker oof
        self.walker = list(Path(waves_path).glob("*/*"))

        with open(transcript_path, "r", encoding="utf-8") as f:
            transcripts = f.read().strip().split("\n")
            transcripts = [line.split(" ", 1) for line in transcripts]
            filenames = [i[0] for i in transcripts]
            trans = [i[1] for i in transcripts]
            self.transcripts = dict(zip(filenames, trans))

        self.feature_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft)

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        filepath = str(self.walker[idx])
        filename = filepath.rsplit(os.sep, 1)[-1].split(".")[0]

        wave, sr = torchaudio.load(filepath)
        specs = self.feature_transform(wave)  # channel, feature, time
        specs = specs.permute(0, 2, 1)  # channel, time, feature
        specs = specs.squeeze()  # time, feature

        trans = self.transcripts[filename].lower()

        return specs, trans
    
if __name__ == "__main__":
    dataset = VivosDataset(subset='train', root="data/vivos/vivos")
    an_item = dataset[0]
    print(f'type of each item: {type(an_item)}')
    print(an_item)