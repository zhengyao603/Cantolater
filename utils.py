import logging
import os
import time

import torch
from torch.utils.data import DataLoader, Dataset


class TSVDataset(Dataset):
    def __init__(self, data_path, sep=":", **kwargs):
        super(TSVDataset, self).__init__(**kwargs)
        self._sep = sep
        self.lines = open(data_path, "r", encoding="utf-8").readlines()

    def __getitem__(self, index):
        example = self.lines[index].strip().split(self._sep)
        example[0].strip()
        example[1].strip()
        return example

    def __len__(self):
        return len(self.lines)


def collect_fn(batch, src_tokenizer, tgt_tokenizer, max_src_len, max_tgt_len):
    # 将batch拆分为源语言和目标语言的两个列表
    batch_src, batch_tgt = zip(*batch)

    # 使用源语言的tokenizer对batch_src进行处理
    src_inputs = src_tokenizer(
        batch_src,
        padding="max_length",
        truncation=True,
        max_length=max_src_len,
        return_tensors="pt",
    )

    # 使用目标语言的tokenizer对batch_tgt进行处理
    tgt_inputs = tgt_tokenizer(
        batch_tgt,
        padding="max_length",
        truncation=True,
        max_length=max_tgt_len,
        return_tensors="pt",
    )

    # 根据目标语言的input_ids生成标签（labels）
    labels = tgt_inputs.input_ids.numpy().tolist()
    labels = [
        [
            # 将padding token的token id变为-100
            -100 if token_id == tgt_tokenizer.pad_token_id else token_id
            for token_id in label
        ]
        for label in labels
    ]

    # 返回处理后的数据作为元组
    labels = torch.LongTensor(labels)
    return (
        src_inputs.input_ids,
        src_inputs.attention_mask,
        tgt_inputs.input_ids,
        tgt_inputs.attention_mask,
        labels,
    )


def get_logger(log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_save_path = "log_{}.txt".format(
        time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time()))
    )
    log_save_path = os.path.join(log_path, log_save_path)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    filehandler = logging.FileHandler(log_save_path)
    filehandler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    return logger


def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    if token_ids_1 is None:
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    # 开始标识符
    bos = [self.bos_token_id]
    # 结束标识符
    eos = [self.eos_token_id]
    return bos + token_ids_0 + eos + token_ids_1 + eos


def to_list(tensor):
    return tensor.detach().cuda().tolist()
