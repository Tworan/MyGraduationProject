import json
import math
import os
import numpy as np
import torch
import torch.utils.data as data
import librosa
import sys
import soundfile as sf
sys.path.append('/home/photon/MyGraduationProject/data')
from preprocess import preprocess_one_dir
from transform import get_preprocessing_pipeline

TRANSFORM_FLAG = False

class AudioDataset(data.Dataset):

    def __init__(self, json_dir, batch_size, mode='audio-only', sample_rate=8000, segment=2.0, cv_max_len=8.0):

        """
            Args:
                json_dir: directory including mix.json, s1.json and s2.json
                segment: duration of audio segment, when set to -1, use full audio

            xxx_list is a list and each item is a tuple (wav_file, #samples)
        """

        super(AudioDataset, self).__init__()
        self.mode = mode
        if self.mode not in ['audio-only', 'audio-visual']:
            raise KeyError
        # 拼接 json 文件地址
        mix_json = os.path.join(json_dir, 'mix.json')
        s1_json = os.path.join(json_dir, 's1.json')
        s2_json = os.path.join(json_dir, 's2.json')

        # 读取 json 文件（json 文件 => 数据地址 + 数据长度）
        with open(mix_json, 'r') as f:
            mix_list = json.load(f)

        with open(s1_json, 'r') as f:
            s1_list = json.load(f)

        with open(s2_json, 'r') as f:
            s2_list = json.load(f)

        # 按照数据长度排序
        def sort(wav_list):
            return sorted(wav_list, key=lambda info: int(info[-1]), reverse=True)

        # 按照长度降序排列
        sorted_mix_list = sort(mix_list)
        sorted_s1_list = sort(s1_list)
        sorted_s2_list = sort(s2_list)

        # 只读取长度为 4 秒的语音
        if segment >= 0.0:
            segment_len = int(segment * sample_rate)  # 4s * 8000/s = 32000 samples

            drop_utt = 0  # 语音数量
            drop_len = 0  # 语音点数

            # 统计小于 4 秒的语音
            for _, sample in sorted_mix_list:
                if sample < segment_len:
                    drop_utt += 1
                    drop_len += sample

            print("Drop {} utterance({:.2f} h) which is short than {} samples".format(drop_utt,
                                                                                      drop_len/sample_rate/36000,
                                                                                      segment_len))

            mini_batch = []
            start = 0

            while True:
                num_segments = 0
                end = start
                part_mix, part_s1, part_s2 = [], [], []

                while num_segments < batch_size and end < len(sorted_mix_list):

                    utterance_len = int(sorted_mix_list[end][-1])  # 当前语音数据长度

                    if utterance_len >= segment_len:  # 判断语音是否大于 4 秒

                        num_segments += math.ceil(utterance_len/segment_len)  # 向上取整

                        # 大于 4 秒丢弃
                        if num_segments > batch_size:
                            if start == end:
                                end += 1
                            break

                        part_mix.append(sorted_mix_list[end])
                        part_s1.append(sorted_s1_list[end])
                        part_s2.append(sorted_s2_list[end])

                    end += 1

                if len(part_mix) > 0:
                    mini_batch.append([part_mix, part_s1, part_s2, sample_rate, segment_len])

                if end == len(sorted_mix_list):
                    break

                start = end

            self.mini_batch = mini_batch
        # 读取所有数据
        else:
            mini_batch = []
            start = 0

            while True:
                # 所有语句长度和 start+batch_size 比较大小
                end = min(len(sorted_mix_list), start+batch_size)

                # 跳过较长音频避免内存不足问题
                if int(sorted_mix_list[start][1]) > cv_max_len * sample_rate:
                    start = end
                    continue

                mini_batch.append([sorted_mix_list[start:end],
                                  sorted_s1_list[start:end],
                                  sorted_s2_list[start:end],
                                  sample_rate,
                                  segment])

                if end == len(sorted_mix_list):
                    break

                start = end

            self.mini_batch = mini_batch

    def __getitem__(self, index):
        return self.mini_batch[index]

    def __len__(self):
        return len(self.mini_batch)


class AudioDataLoader(data.DataLoader):
    """
        NOTE: 这里只使用 batch_size = 1，所以 drop_last = True 在这里没有意义
    """

    def __init__(self, _mode, _type, *args, **kwargs):

        super(AudioDataLoader, self).__init__(*args, **kwargs)
        # print(kwargs)
        self.mode = _mode
        if _type == 'tr':
            if self.mode == 'audio-only':
                self.collate_fn = _collate_fn_audio
            elif self.mode == 'audio-visual':
                self.collate_fn = _collate_fn_audio_visual_train
        else:
            if self.mode == 'audio-only':
                self.collate_fn = _collate_fn_audio
            elif self.mode == 'audio-visual':
                self.collate_fn = _collate_fn_audio_visual_val

def _collate_fn_audio(batch):
    """
        Args:
            batch: list, len(batch) = 1. See AudioDataset.__getitem__()
        Returns:
            mixtures_pad: B x T, torch.Tensor
            ilens: B, torch.Tentor
            sources_pad: B x C x T, torch.Tensor
    """
    assert len(batch) == 1
    
    mixtures, sources = load_data_audio_only(batch[0])

    # 获取输入序列长度
    lens = np.array([mix.shape[0] for mix in mixtures])

    # 执行填充和转换为张量
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float() for mix in mixtures], pad_value)  # 补零，保证长度一样
    lens = torch.from_numpy(lens)  # 转换为张量
    sources_pad = pad_list([torch.from_numpy(s).float() for s in sources], pad_value)  # 补零，保证长度一样
    sources_pad = sources_pad.permute((0, 2, 1)).contiguous()  # N x T x C -> N x C x T
    # print(mixtures_pad.shape, sources_pad.shape)
    return mixtures_pad.unsqueeze(1), lens, sources_pad

def _collate_fn_audio_visual_train(batch):
    """
        Args:
            batch: list, len(batch) = 1. See AudioDataset.__getitem__()
        Returns:
            mixtures_pad: B x T, torch.Tensor
            ilens: B, torch.Tentor
            sources_pad: B x C x T, torch.Tensor
    """
    assert len(batch) == 1
    
    mixtures, sources, faces = load_data_audio_visual(batch[0], train=True)

    # 获取输入序列长度
    lens = np.array([mix.shape[0] for mix in mixtures])

    # 执行填充和转换为张量
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float() for mix in mixtures], pad_value)  # 补零，保证长度一样
    lens = torch.from_numpy(lens)  # 转换为张量
    sources_pad = pad_list([torch.from_numpy(s).float() for s in sources], pad_value)  # 补零，保证长度一样
    sources_pad = sources_pad.permute((0, 2, 1)).contiguous()  # N x T x C -> N x C x T
    # print(mixtures_pad.shape, sources_pad.shape)
    return mixtures_pad.unsqueeze(1), lens, sources_pad, torch.Tensor(faces)

def _collate_fn_audio_visual_val(batch):
    """
        Args:
            batch: list, len(batch) = 1. See AudioDataset.__getitem__()
        Returns:
            mixtures_pad: B x T, torch.Tensor
            ilens: B, torch.Tentor
            sources_pad: B x C x T, torch.Tensor
    """
    assert len(batch) == 1
    
    mixtures, sources, faces = load_data_audio_visual(batch[0], train=False)

    # 获取输入序列长度
    lens = np.array([mix.shape[0] for mix in mixtures])

    # 执行填充和转换为张量
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float() for mix in mixtures], pad_value)  # 补零，保证长度一样
    lens = torch.from_numpy(lens)  # 转换为张量
    sources_pad = pad_list([torch.from_numpy(s).float() for s in sources], pad_value)  # 补零，保证长度一样
    sources_pad = sources_pad.permute((0, 2, 1)).contiguous()  # N x T x C -> N x C x T
    # print(mixtures_pad.shape, sources_pad.shape)
    return mixtures_pad.unsqueeze(1), lens, sources_pad, torch.Tensor(faces)

def load_data_audio_visual(batch, train=True):
    """
    batch: 
        [0]: [mix, length]
        [1]: [s1, v1, length]
        [2]: [s2, v2, length]
        [3]: sample_rate
        [4]: length
    """
    mixtures, sources, faces = [], [], []
    mix_infos, s1_v1_infos, s2_v2_infos, sample_rate, segemnt_len = batch 
    transform = get_preprocessing_pipeline()['train' if train else 'val']
    for mix_info, s1_v1_info, s2_v2_info in zip(mix_infos, s1_v1_infos, s2_v2_infos):
        mix_path = mix_info[0]
        s1_path = s1_v1_info[0]
        v1_path = s1_v1_info[1]
        s2_path = s2_v2_info[0]
        v2_path = s2_v2_info[1]

        # print(mix_info, s1_v1_info, s2_v2_info)
        assert mix_info[1] == s1_v1_info[2] and s1_v1_info[2] == s2_v2_info[2]
        # 读取语音数据
        mix, _ = sf.read(mix_path)
        mix = mix[:32000]
        # print(mix_path)
        s1, _ = sf.read(s1_path)
        s1 = s1[:32000]
        s2, _ = sf.read(s2_path)
        s2 = s2[:32000]
        # 读取视频数据
        v1 = np.load(v1_path)['data']
        v2 = np.load(v2_path)['data']

        v1 = transform(v1)
        v2 = transform(v2)
        # v1 and v2 must be numpy array
        flag = np.random.choice(2, size=(1,), replace=True)
        # if flag:
        s = np.dstack((s1, s2))[0]  # 32000 x 2
        v = np.stack((v1, v2)) # 2 x 50 x 96 x 96
        # else:
        #     s = np.dstack((s2, s1))[0]  # 32000 x 2
        #     v = np.stack((v2, v1)) # 2 x 50 x 96 x 96
        mixtures.append(mix)
        sources.append(s)
        faces.append(v)
    
    return mixtures, sources, faces
        

def load_data_audio_only(batch):
    """
    batch: 
        [0]: [mix, length]
        [1]: [s1, length]
        [2]: [s2, length]
        [3]: sample_rate
        [4]: length
    """
    mixtures, sources = [], []
    mix_infos, s1_infos, s2_infos, sample_rate, segemnt_len = batch 

    for mix_info, s1_info, s2_info in zip(mix_infos, s1_infos, s2_infos):
        mix_path = mix_info[0]
        s1_path = s1_info[0]
        s2_path = s2_info[0]
        
        assert mix_info[-1] == s1_info[-1] and s1_info[-1] == s2_info[-1]
        # 读取语音数据
        mix, _ = sf.read(mix_path)
        mix = mix[:32000]
        # print(mix_path)
        s1, _ = sf.read(s1_path)
        s1 = s1[:32000]
        s2, _ = sf.read(s2_path)
        s2 = s2[:32000]
        # 读取视频数据
        # v1 and v2 must be numpy array
        s = np.dstack((s1, s2))[0]  # 32000 x 2

        mixtures.append(mix)
        sources.append(s)
    
    return mixtures, sources

def load_mixtures_and_sources(batch):
    """
        Each info include wav path and wav duration.
        Returns:
            mixtures: a list containing B items, each item is T np.ndarray
            sources: a list containing B items, each item is T x C np.ndarray
            T varies from item to item.
    """
    mixtures, sources = [], []
    mix_infos, s1_infos, s2_infos, sample_rate, segment_len = batch

    for mix_info, s1_info, s2_info in zip(mix_infos, s1_infos, s2_infos):
        mix_path = mix_info[0]
        s1_path = s1_info[0]
        s2_path = s2_info[0]

        assert mix_info[1] == s1_info[1] and s1_info[1] == s2_info[1]

        # 读取语音数据
        mix, _ = sf.read(mix_path)
        # print(mix_path)
        s1, _ = sf.read(s1_path)
        s2, _ = sf.read(s2_path)

        # 将 s1 与 s2 合并
        s = np.dstack((s1, s2))[0]  # 32000 x 2
        utt_len = mix.shape[-1]  # 32000

        if segment_len >= 0:
            for i in range(0, utt_len-segment_len+1, segment_len):
                mixtures.append(mix[i:i+segment_len])
                sources.append(s[i:i+segment_len])

            if utt_len % segment_len != 0:
                mixtures.append(mix[-segment_len:])
                sources.append(s[-segment_len:])
        else:
            mixtures.append(mix)
            sources.append(s)

    return mixtures, sources


def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


class EvalDataset(data.Dataset):

    def __init__(self, mix_dir, mix_json, batch_size, sample_rate=8000):
        """
            Args:
                mix_dir: directory including mixture wav files
                mix_json: json file including mixture wav files
        """
        super(EvalDataset, self).__init__()

        assert mix_dir!=None or mix_json!=None

        if mix_dir is not None:
            preprocess_one_dir(mix_dir, mix_dir, 'mix', sample_rate=sample_rate)
            mix_json = os.path.join(mix_dir, 'mix.json')

        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)

        def sort(infos):
            return sorted(infos, key=lambda info: int(info[1]), reverse=True)

        sorted_mix_infos = sort(mix_infos)

        mini_batch = []
        start = 0
        while True:
            end = min(len(sorted_mix_infos), start+batch_size)
            mini_batch.append([sorted_mix_infos[start:end], sample_rate])

            if end == len(sorted_mix_infos):
                break

            start = end

        self.minibatch = mini_batch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


class EvalDataLoader(data.DataLoader):
    """
        NOTE: just use batch_size = 1 here, so drop_last = True makes no sense here.
    """
    def __init__(self, *args, **kwargs):

        super(EvalDataLoader, self).__init__(*args, **kwargs)

        self.collate_fn = _collate_fn_eval


def _collate_fn_eval(num_batch):
    """
        Args:
            num_batch: list, len(batch) = 1. See AudioDataset.__getitem__()
        Returns:
            mixtures_pad: B x T, torch.Tensor
            ilens: B, torch.Tentor
            filenames: a list contain B strings
    """
    assert len(num_batch) == 1

    mixtures, filenames = load_mixtures(num_batch[0])

    ilens = np.array([mix.shape[0] for mix in mixtures])  # 获取输入序列长度的批处理

    pad_value = 0

    mixtures_pad = pad_list([torch.from_numpy(mix).float() for mix in mixtures], pad_value)  # 填充 0

    ilens = torch.from_numpy(ilens)

    return mixtures_pad, ilens, filenames


def load_mixtures(batch):
    """
        Returns:
            mixtures: a list containing B items, each item is T np.ndarray
            filenames: a list containing B strings
            T varies from item to item.
    """
    mixtures, filenames = [], []

    mix_infos, sample_rate = batch

    for mix_info in mix_infos:
        mix_path = mix_info[0]

        mix, _ = sf.read(mix_path)
        if len(mix) >= 32000:
            mix = mix[:32000]

        mixtures.append(mix)
        filenames.append(mix_path)

    return mixtures, filenames

