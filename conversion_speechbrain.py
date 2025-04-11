import os
import pickle
import torch
import numpy as np
from math import ceil

from tqdm import tqdm
from model_vc_gan import Generator
from pathlib import Path 
from scipy.fft import idctn, dctn

########### parameters here ###############
basedir = '/data0/yfliu/lrs3/spfacevc'
result_dir = os.path.join(basedir, 'result_noised')
no_repar = False
no_attn = False
face_dir = '/data0/yfliu/lrs3/spfacevc/test/faceemb_lrs3_mtcnn_margin50/test/'
mel_dir = '/data0/yfliu/lrs3/spfacevc/test/noise/mel/'

os.makedirs(result_dir, exist_ok=True)
#############################################################

def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

def chunking_mel(melspectrogram, base = 128):
    data = []
    num_spectro = (melspectrogram.shape[0]//base)+1
    for index in range(num_spectro):
        if index < num_spectro - 1:
            mel = melspectrogram[index*base:index*base+base,:]
        else:
            mel = melspectrogram[index*base:, :]
            len_pad = base - melspectrogram.shape[0]%base        
            mel = np.pad(mel, ((0, len_pad), (0,0)), 'constant', constant_values=(0,0))
        data.append(mel)

    return torch.tensor(np.array(data)), len_pad

device = 'cuda'
G = Generator(32,512,512,32,no_attn, no_repar).eval().to(device)

g_checkpoint = torch.load(basedir+'/G.ckpt')
G.load_state_dict(g_checkpoint)

spect_vc = []
#############################  LRS3  #################################
# src_speaker_lst = ["0wpCZxiAQzw", "FD5ZKiSmoMU", "1gdKrtwBGqY", "pcNxS2i7SvQ", "11Mq9ZuxZMc", "F2hc2FLOdhI", "oXSyMUeAEec", "Yo5cKRmJaf0", "weyd0UMdP7g", "LKAhTELkOV8", "2L4BSVpvx1A", "xbagFzcyNiM", "6Af6bSwyiwI", "MgUTzbGakRw", "EZwxKPv1CwA", "5duz42kHqPs"]
# tgt_speaker_lst = ["0wpCZxiAQzw", "FD5ZKiSmoMU", "1gdKrtwBGqY", "pcNxS2i7SvQ", "11Mq9ZuxZMc", "F2hc2FLOdhI", "oXSyMUeAEec", "Yo5cKRmJaf0", "weyd0UMdP7g", "LKAhTELkOV8", "2L4BSVpvx1A", "xbagFzcyNiM", "6Af6bSwyiwI", "MgUTzbGakRw", "EZwxKPv1CwA", "5duz42kHqPs"]
src_file_lst_0 = []
tgt_file_lst_0 = []
with open('/home/yfliu/hifi-gan/test_speakers/N.txt', 'r') as fr:
    for line in fr.readlines():
        _, src_file, tgt_file = line.strip().split(' ')
        src_file_lst_0.append(src_file)
        tgt_file_lst_0.append(tgt_file)

src_file_lst_1 = []
tgt_file_lst_1 = []
with open('/home/yfliu/hifi-gan/test_speakers/P.txt', 'r') as fr:
    for line in fr.readlines():
        _, src_file, tgt_file = line.strip().split(' ')
        src_file_lst_1.append(src_file)
        tgt_file_lst_1.append(tgt_file)

src_file_lst = src_file_lst_0 + src_file_lst_1
tgt_file_lst = tgt_file_lst_0 + tgt_file_lst_1


pbar = tqdm(zip(src_file_lst, tgt_file_lst), total=len(src_file_lst))
for src_file, tgt_file in pbar:
        src_speaker, src_serial = src_file.split('/')[-2:]
        src_serial = src_serial.split('.')[0]
        tgt_speaker, tgt_serial = tgt_file.split('/')[-2:]
        tgt_serial = tgt_serial.split('.')[0]
        pbar.set_description(src_speaker + '_' + src_serial + '>' + tgt_speaker + '_' + tgt_serial)
        src_speaker_mel = np.load(os.path.join(mel_dir,src_speaker+f'_{src_serial}.wav.npy'))
        src_speaker_mel = src_speaker_mel.transpose()
        m = np.ones(src_speaker_mel.shape)
        m[:, 10:]=0
        src_speaker_mel = dctn(idctn(src_speaker_mel)*m)
            
        src_speaker_emb = None

        try:
            tgt_speaker_emb = np.load(os.path.join(face_dir, tgt_speaker, f'{tgt_serial}.npy'))
        
            src_speaker_mel, len_pad = chunking_mel(src_speaker_mel, 64)
            src_speaker_mel = src_speaker_mel.float().to(device)   
            if src_speaker_emb is not None:
                src_speaker_emb = torch.from_numpy(src_speaker_emb[np.newaxis, :]).repeat(src_speaker_mel.shape[0], 1).to(device)
            tgt_speaker_emb = torch.from_numpy(tgt_speaker_emb[np.newaxis, :]).repeat(src_speaker_mel.shape[0], 1).to(device)    

            with torch.no_grad():
                _, x_identic_psnt, _ = G(src_speaker_mel, src_speaker_emb, tgt_speaker_emb)
            x_identic_psnt = torch.cat([x_identic_psnt[i] for i in range(x_identic_psnt.shape[0])], 1).unsqueeze(0)        
                      
            if len_pad == 0:
                uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
            else:
                uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
            spect_vc.append((f'{src_speaker}_{src_serial}_{tgt_speaker}_{tgt_serial}', uttr_trg))
        except:
            pass  

result_path = os.path.join(result_dir, 'result.pkl')
with open(result_path, 'wb') as handle:
    pickle.dump(spect_vc, handle)
