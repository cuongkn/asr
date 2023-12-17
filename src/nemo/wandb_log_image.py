# import os
# import librosa
# import numpy as np


# def save_spectogram_as_img(audio_path, datadir, plt_type='spec'):
#     filename = os.path.basename(audio_path)
#     out_path = os.path.join(datadir, filename.replace('.wav', '.png'))
#     audio, sample_rate = librosa.load(audio_path)
#     if plt_type=='spec':
#         spec = np.abs(librosa.stft(audio))
#         spec_db = librosa.amplitude_to_db(spec, ref=np.max)
#     else:
#         mel_spec = librosa.feature.melspectrogram(audio, sr=sample_rate)
#         mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
#     fig = plt.Figure()
#     ax = fig.add_subplot()
#     ax.set_axis_off()
    
#     librosa.display.specshow(
#         spec_db if plt_type=='spec' else mel_spec_db, 
#         y_axis='log' if plt_type=='spec' else 'mel', 
#         x_axis='time', ax=ax)


#     fig.savefig(out_path)


# # convert audio file to spectogram and mel spectogram images
# if not os.path.exists('./an4/melspectogram_images/'):
#     for path in tqdm(train_audio_paths): 
#         save_mel_spectogram_as_img(path, datadir='./an4/images/')
#         save_spectogram_as_img(path, datadir='./an4/melspectogram_images/', plt_type='mel')


# # log filename, playable audio, duration of audio, transcript, spectogram and mel spectogram to W&B for ease of reference
# if LOG_WANDB:
#     # create W&B Table
#     wandb.init(project="ASR")


#     audio_table = wandb.Table(
#         columns=['Filename', 'Audio File', 'Duration', 'Transcript', 'Spectogram', 'Mel-Spectogram'])


#     for path, duration, text in zip(train_audio_paths, train_durations, train_texts):
#         filename = os.path.basename(path)
#         img_fn   = filename.replace('.wav', '.png')
#         spec_pth = os.path.join('./an4/images', img_fn)
#         melspec_pth = os.path.join('./an4/melspectogram_images', img_fn)
#         audio_table.add_data(
#             filename, wandb.Audio(path), duration, text, wandb.Image(spec_pth), wandb.Image(melspec_pth))


#     wandb.log({"Train Data": audio_table})
#     wandb.finish();
