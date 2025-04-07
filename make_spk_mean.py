import os
import numpy as np
rootDir = '/data0/yfliu/lrs3/spfacevc/test/faceemb_lrs3_mtcnn_margin50/'
out_dir = '/data0/yfliu/lrs3/spfacevc/test/faceemb_lrs3_mtcnn_margin50_mean/'

if not os.path.exists(out_dir):
        os.makedirs(out_dir)

dirName, subdirList, _ = next(os.walk(rootDir))

speakers = []
for speaker in sorted(subdirList):
    print('Processing speaker: %s' % speaker)
    _, subsubdirList, _ = next(os.walk(os.path.join(dirName,speaker)))
    for subdir in subsubdirList:
        _, _, fileList = next(os.walk(os.path.join(dirName,speaker, subdir)))
        embs = []
        for s in sorted(fileList):
            n = np.load(os.path.join(dirName, speaker, subdir, s))
            #print(n.shape)
            embs.append(n)
        if not os.path.exists(os.path.join(out_dir, speaker)):
            os.makedirs(os.path.join(out_dir, speaker))
        np.save(os.path.join(out_dir,speaker, speaker+'-'+subdir+'.npy'), np.mean(embs, axis=0))
    #print(np.mean(embs, axis=0).shape)
    #assert 0