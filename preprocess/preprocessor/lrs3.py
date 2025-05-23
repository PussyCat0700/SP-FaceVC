import os
from glob import glob
from .base import BasePreproceccor

class Preprocessor(BasePreproceccor):
    def __init__(self, config):
        super().__init__(config)
    
    def gen_file_dict(self, input_path):
        speaker_dir_list = glob(os.path.join(input_path, '*'))
        file_list = []
        basename_list = []
        for speaker_dir in speaker_dir_list:
            if os.path.basename(speaker_dir) == 'RIRS_NOISES' or \
            os.path.basename(speaker_dir) == 'noise.csv' or \
            os.path.basename(speaker_dir) == 'reverb.csv' or \
            os.path.basename(speaker_dir) == 'rirs_noises.zip' or \
            os.path.basename(speaker_dir) == 'alignment_dir':
                print(speaker_dir)
                continue

            lst = glob(os.path.join(speaker_dir, '*.wav'))
            basename_list += ['_'.join(f.split('/')[-2:]) for f in lst]
            file_list += lst
        file_dict = dict(zip(file_list, basename_list))
        return file_dict
