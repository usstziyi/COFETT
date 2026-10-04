import os
from pathlib import Path
import mne
import numpy as np
import pickle
from pyprep.find_noisy_channels import NoisyChannels
from pyprep.prep_pipeline import PrepPipeline
from mne_icalabel import label_components
import pandas as pd 




try:
    root_folder = os.path.dirname(os.path.abspath(__file__)) # the folder containing this script
    IC_NUM = 30
    subject_id = 'A'
    method_str = 'default'
    TEST = False
    PREP = True
except Exception as e:
    print(e)
    
# other settings
sample_rate = 500
output_folder = os.path.join('preprocess_output', method_str + '_' + subject_id)

# select data folder by ID
if subject_id == 'A':
    data_folder = os.path.join(root_folder,'Chisco-2.0/sub-01')
elif subject_id == 'B':
    data_folder = os.path.join(root_folder,'Chisco-2.0/sub-02')
else:
    print("Invalid subject id")
    exit()



if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(os.path.join(output_folder,'pkl')):
    os.makedirs(os.path.join(output_folder,'pkl'))
if not os.path.exists(os.path.join(output_folder,'fif-epo')):
    os.makedirs(os.path.join(output_folder,'fif-epo'))
if not os.path.exists(os.path.join(output_folder,'fif')):
    os.makedirs(os.path.join(output_folder,'fif'))
if not os.path.exists(os.path.join(output_folder,'log')):
    os.makedirs(os.path.join(output_folder,'log'))



def process_edf_file(edf_file, events_file, output_folder):

    raw = mne.io.read_raw_edf(
        edf_file,
        preload=True,
        verbose=False,
    )

    raw.resample(sample_rate,verbose=False)

    if TEST:
        raw.crop(tmin=300, tmax=600)

    # 1. 先删除没有有效位置的 EEG
    raw.drop_channels([ch for ch in ["10", "111"] if ch in raw.ch_names])

    # 2. 明确通道类型
    raw.set_channel_types({
        "VEO": "eog",
        "HEO": "eog",
        "Trigger": "stim",
    })

    # 3. 再设置 montage
    montage = mne.channels.read_custom_montage("montage.csv")
    raw.set_montage(montage,on_missing="raise")

  
    if PREP:
        # run pyprep
        print("Running pyprep")
        prep_params = {
            "ref_chs": "eeg",
            "reref_chs": "eeg",
            "line_freqs": np.arange(50, sample_rate / 2, 50), # 50,100,150,200Hz
        }

        prep = PrepPipeline(raw, prep_params, montage, ransac=RANSAC)
        prep.fit()

        raw_new = prep.raw 

    else:
        print("Not running pyprep")
        raw_new = raw.copy()

    # 如果打印结果为空，说明所有坏道都被插值修复了
    print("Still bad channels: ", raw_new.info['bads'])


    # High-pass filter
    raw_new.filter(l_freq=1, h_freq=None)


    # save unsegmented processed data
    # 已插值坏道 + 已做鲁棒平均参考 + 已陷波 + 已高通，但还没切 epoch 的整段信号。
    raw_new.save(os.path.join(output_folder,'fif' ,f"{os.path.basename(edf_file).replace('.edf', '')}_{subject_id}_{method_str}-raw.fif"), overwrite=True)

        


def check_edf_events(edf_files, events_files):
    for edf_file,events_file in zip(edf_files,events_files):
        # 比较 EDF 和 events 文件名中 _eeg 与 _events 之前的前缀是否一致
        # 提取EDF文件名的前缀（_eeg 之前的部分）
        edf_prefix = edf_file.name.split('_eeg')[0]
        # 提取events文件名的前缀（_events 之前的部分）
        events_prefix = events_file.name.split('_events')[0]

        if edf_prefix == events_prefix:
            print("✓ 文件名前缀匹配成功")
        else:
            print("✗ 警告：文件名前缀不匹配！请检查文件对应关系")


for ses_dir in sorted(Path(data_folder).glob("ses-*")):
    eeg_dir = ses_dir / "eeg"

    edf_files = sorted(list(eeg_dir.glob("*.edf")))
    events_files = sorted(list(eeg_dir.glob("*_events.tsv")))
    
    check_edf_events(edf_files, events_files)
    for edf_file,events_file in zip(edf_files, events_files):
        process_edf_file(edf_file, events_file, output_folder)

