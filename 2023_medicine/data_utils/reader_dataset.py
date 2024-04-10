import glob
import logging
import os
import pickle
import numpy as np
import copy, random

import torch
from torch.utils.data import Dataset

from utils import dist_utils
from .common_utils import read_json


logger = logging.getLogger()

def get_subsets(s):
    if len(s) == 0:
        return [[]]  # 返回空集合的子集

    subsets = []
    first = s[0]  # 取集合中的第一个元素
    remaining = s[1:]  # 剩余的元素

    # 递归获取剩余元素的所有子集
    subsets_without_first = get_subsets(remaining)

    # 将剩余元素的所有子集添加到结果中
    subsets.extend(subsets_without_first)

    # 将第一个元素加入剩余元素的所有子集中，并添加到结果中
    for subset in subsets_without_first:
        subsets.append([first] + subset)

    return subsets

def get_subsets_and_removed_subsets(s):
    subsets = get_subsets(s)
    removed_subsets = []
    for subset in subsets:
        removed_subset = list(set(s)-set(subset))
        removed_subsets.append(removed_subset)
    return subsets, removed_subsets

class DiaDataset(torch.utils.data.Dataset):
    def __init__(self, args, data_dir, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        if dist_utils.is_local_master():
            logger.info(f"Data dir: {data_dir}")
        data = read_json(path=data_dir)

        self.data = self.read_data(data)
        if dist_utils.is_local_master():
            logger.info(f"Total data size: {len(self.data)}")
    
    def read_data(self, data):
        processed_data = []
        for item_index, item in enumerate(data):
            exp_sx_ids = []
            exp_attr_ids = []
            imp_sx_ids = []
            imp_attr_ids = []
            label = None
            for sx, attr in item['exp_sxs'].items():
                if sx == "上呼吸道感染":
                    print('a')
                attr_id = self.tokenizer.convert_attr_to_id(attr)
                if attr_id == self.tokenizer.unk_token_id:
                    continue
                exp_attr_ids.append(attr_id)
                sx_id = self.tokenizer.convert_token_to_id(sx)
                exp_sx_ids.append(sx_id)
            
            for sx, attr in item['imp_sxs'].items():
                attr_id = self.tokenizer.convert_attr_to_id(attr)
                if attr_id == self.tokenizer.unk_token_id:
                    continue
                imp_attr_ids.append(attr_id)
                sx_id = self.tokenizer.convert_token_to_id(sx)
                imp_sx_ids.append(sx_id)

            if item["label"] == 'UNK':
                continue
            label = self.tokenizer.convert_token_to_id(item["label"])

            if len(exp_sx_ids) != 0 or len(imp_sx_ids) != 0:
                processed_data.append({'exp_sx_ids': exp_sx_ids, 'exp_attr_ids': exp_attr_ids, 'imp_sx_ids': imp_sx_ids, 'imp_attr_ids': imp_attr_ids, 'label': label})
        return processed_data


    def convert_example_to_feature(
        self,
        example,
        max_input_len=512,
        max_output_len=512,
    ):
        """
        每一阶段均预测疾病，从copy中复制得到
        """
        features = []
        # disease
        all_disease_ids = self.tokenizer.get_disease_ids()
        # dx_attr_ids = [self.tokenizer.mask_token_id] * len(disease_ids)
        input_dxs = all_disease_ids
        # 0: disease 1: symptom
        token_type_dxs = [0] * len(all_disease_ids) 
        # 0: true 1: false 2: mask
        attr_dx_labels = [0 if dx_id == example['label'] else 1 for dx_id in all_disease_ids]
        attr_dxs = [2] * len(all_disease_ids)
        input_sx = example['exp_sx_ids'] + example['imp_sx_ids']
        token_type_sx = [1] * len(input_sx)
        attr_sx_label = example['exp_attr_ids'] + example['imp_attr_ids']

        subsets, removed_subsets = get_subsets_and_removed_subsets(range(len(example['imp_sx_ids'])))
        # 子集过多，随机采样k个
        if len(subsets) > self.args.k:
            indices = random.sample(range(len(subsets)), self.args.k)
            subsets = [subsets[i] for i in indices]
            removed_subsets = [removed_subsets[i] for i in indices]

        if self.args.add_task_1:
            # task 1: 输入exp_sx和imp_sx及其attr, 输入某个具体的dx, 预测其attr
            # 最终阶段包含false的dx
            for (input_dx, token_type_dx, attr_dx, attr_dx_label) in zip(input_dxs, token_type_dxs, attr_dxs, attr_dx_labels):
                input_id = [input_dx] + input_sx
                type_id = [token_type_dx] + token_type_sx
                attr_id = [attr_dx] + attr_sx_label
                input_mask = [1] * len(input_id)
                attr_label = attr_dx_label
                token_label = -100
                features.append((input_id, type_id, attr_id, input_mask, attr_label, token_label))

        if self.args.add_task_2:
            # task 2:输入exp_sx和imp_sx及其attr, 给定dx_attr为true，预测dx (诊断任务预测方式)
            # 最终阶段
            input_id = [self.tokenizer.sx_mask_token_id] + input_sx
            type_id = [0] + token_type_sx
            attr_id = [0] + attr_sx_label
            input_mask = [1] * len(input_id)
            attr_label = -100
            token_label = example['label']
            features.append((input_id, type_id, attr_id, input_mask, attr_label, token_label))
        
        for subset in subsets:
            if len(subset) == len(example['imp_sx_ids']):
                continue
            if self.args.add_middle_task_1:
                # task 1: 中间阶段也预测true的dx
                input_id = [example['label']] + example['exp_sx_ids'] + [example['imp_sx_ids'][i] for i in subset]
                type_id = [0] + [1] * (len(input_id)-1)
                attr_id = [2] + example['exp_attr_ids'] + [example['imp_attr_ids'][i] for i in subset]
                input_mask = [1] * len(input_id)
                attr_label = 0
                token_label = -100
                features.append((input_id, type_id, attr_id, input_mask, attr_label, token_label))

            if self.args.add_middle_task_2:
                # task 2:中间阶段也预测
                input_id = [self.tokenizer.sx_mask_token_id]  + example['exp_sx_ids'] + [example['imp_sx_ids'][i] for i in subset]
                type_id = [0] + [1] * (len(input_id)-1)
                attr_id = [0] + example['exp_attr_ids'] + [example['imp_attr_ids'][i] for i in subset]
                input_mask = [1] * len(input_id)
                attr_label = -100
                token_label = example['label']
                features.append((input_id, type_id, attr_id, input_mask, attr_label, token_label))

        
        # 对imp_sx中的所有子集均预测
        for i, subset in enumerate(subsets):
            for sx_index in removed_subsets[i]:
                if self.args.add_task_3:
                    # task 3:给定attr为true的dx及其attr，随机mask一个(或多个)sx的attr，预测其attr
                    input_id = [example['imp_sx_ids'][sx_index]] + example['exp_sx_ids'] + [example['imp_sx_ids'][i] for i in subset] + [example['label']]
                    type_id = [1] * (len(input_id)-1) + [0]
                    # attr mask
                    attr_id = [2] + example['exp_attr_ids'] + [example['imp_attr_ids'][i] for i in subset] + [0]
                    input_mask = [1] * len(input_id)
                    attr_label = example['imp_attr_ids'][sx_index]
                    token_label = -100
                    features.append((input_id, type_id, attr_id, input_mask, attr_label, token_label))
                if self.args.add_task_4:
                    # task 4:给定除特定sx外所有已知的sx及其attr，对于该特定sx，输入其attr为true or false，预测其对应的sx (症状问询预测方式)，也可以与task 3结合作为症状问询预测方式
                    # 对imp_sx中的所有子集均预测，这里不加入dx
                    # sx mask
                    input_id = [self.tokenizer.sx_mask_token_id] + example['exp_sx_ids'] + [example['imp_sx_ids'][i] for i in subset]
                    type_id = [1] * (len(input_id))
                    attr_id = [example['imp_attr_ids'][sx_index]] + example['exp_attr_ids'] + [example['imp_attr_ids'][i] for i in subset]
                    input_mask = [1] * len(input_id)
                    attr_label = -100
                    token_label = example['imp_sx_ids'][sx_index]
                    features.append((input_id, type_id, attr_id, input_mask, attr_label, token_label))

        return features


    def convert_example_to_feature_v1(
        self,
        example,
        max_input_len=512,
        max_output_len=512,
    ):
        """
        每一阶段均预测疾病
        """
        features = []
        # disease
        all_disease_ids = self.tokenizer.get_disease_ids()
        # dx_attr_ids = [self.tokenizer.mask_token_id] * len(disease_ids)
        input_dxs = all_disease_ids
        # 0: disease 1: symptom
        token_type_dxs = [0] * len(all_disease_ids) 
        # 0: true 1: false 2: mask
        attr_dx_labels = [0 if dx_id == example['label'] else 1 for dx_id in all_disease_ids]
        attr_dxs = [2] * len(all_disease_ids)
        input_sx = example['exp_sx_ids'] + example['imp_sx_ids']
        token_type_sx = [1] * len(input_sx)
        attr_sx_label = example['exp_attr_ids'] + example['imp_attr_ids']

        subsets, removed_subsets = get_subsets_and_removed_subsets(range(len(example['imp_sx_ids'])))
        # 子集过多，随机采样k个
        k = 32
        if len(subsets) > k:
            indices = random.sample(range(len(subsets)), 8)
            subsets = [subsets[i] for i in indices]
            removed_subsets = [removed_subsets[i] for i in indices]

        # task 1: 输入exp_sx和imp_sx及其attr, 输入某个具体的dx, 预测其attr
        # 最终阶段包含false的dx
        for (input_dx, token_type_dx, attr_dx, attr_dx_label) in zip(input_dxs, token_type_dxs, attr_dxs, attr_dx_labels):
            input_id = [input_dx] + input_sx
            type_id = [token_type_dx] + token_type_sx
            attr_id = [attr_dx] + attr_sx_label
            input_mask = [1] * len(input_id)
            attr_label = attr_dx_label
            token_label = -100
            features.append((input_id, type_id, attr_id, input_mask, attr_label, token_label))

        # task 2:输入exp_sx和imp_sx及其attr, 给定dx_attr为true，预测dx (诊断任务预测方式)
        # 最终阶段
        input_id = [self.tokenizer.sx_mask_token_id] + input_sx
        type_id = [0] + token_type_sx
        attr_id = [0] + attr_sx_label
        input_mask = [1] * len(input_id)
        attr_label = -100
        token_label = example['label']
        features.append((input_id, type_id, attr_id, input_mask, attr_label, token_label))
        
        for subset in subsets:
            if len(subset) == len(example['imp_sx_ids']):
                continue
            # task 1: 中间阶段也预测true的dx
            input_id = [example['label']] + example['exp_sx_ids'] + [example['imp_sx_ids'][i] for i in subset]
            type_id = [0] + [1] * (len(input_id)-1)
            attr_id = [2] + example['exp_attr_ids'] + [example['imp_attr_ids'][i] for i in subset]
            input_mask = [1] * len(input_id)
            attr_label = 0
            token_label = -100
            features.append((input_id, type_id, attr_id, input_mask, attr_label, token_label))

            # task 2:中间阶段也预测
            input_id = [self.tokenizer.sx_mask_token_id]  + example['exp_sx_ids'] + [example['imp_sx_ids'][i] for i in subset]
            type_id = [0] + [1] * (len(input_id)-1)
            attr_id = [0] + example['exp_attr_ids'] + [example['imp_attr_ids'][i] for i in subset]
            input_mask = [1] * len(input_id)
            attr_label = -100
            token_label = example['label']
            features.append((input_id, type_id, attr_id, input_mask, attr_label, token_label))
        
        # 对imp_sx中的所有子集均预测
        for i, subset in enumerate(subsets):
            for sx_index in removed_subsets[i]:
                # task 3:给定attr为true的dx及其attr，随机mask一个(或多个)sx的attr，预测其attr
                input_id = [example['imp_sx_ids'][sx_index]] + example['exp_sx_ids'] + [example['imp_sx_ids'][i] for i in subset] + [example['label']]
                type_id = [1] * (len(input_id)-1) + [0]
                # attr mask
                attr_id = [2] + example['exp_attr_ids'] + [example['imp_attr_ids'][i] for i in subset] + [0]
                input_mask = [1] * len(input_id)
                attr_label = example['imp_attr_ids'][sx_index]
                token_label = -100
                features.append((input_id, type_id, attr_id, input_mask, attr_label, token_label))

                # task 4:给定除特定sx外所有已知的sx及其attr，对于该特定sx，输入其attr为true or false，预测其对应的sx (症状问询预测方式)，也可以与task 3结合作为症状问询预测方式
                # 对imp_sx中的所有子集均预测，这里不加入dx
                # sx mask
                input_id = [self.tokenizer.sx_mask_token_id] + example['exp_sx_ids'] + [example['imp_sx_ids'][i] for i in subset]
                type_id = [1] * (len(input_id))
                attr_id = [example['imp_attr_ids'][sx_index]] + example['exp_attr_ids'] + [example['imp_attr_ids'][i] for i in subset]
                input_mask = [1] * len(input_id)
                attr_label = -100
                token_label = example['imp_sx_ids'][sx_index]
                features.append((input_id, type_id, attr_id, input_mask, attr_label, token_label))

        return features

    def convert_example_to_feature_v0(
        self,
        example,
        max_input_len=512,
        max_output_len=512,
    ):
        """
        仅在最后阶段预测疾病
        """
        features = []
        # disease
        all_disease_ids = self.tokenizer.get_disease_ids()
        # dx_attr_ids = [self.tokenizer.mask_token_id] * len(disease_ids)
        input_dxs = all_disease_ids
        # 0: disease 1: symptom
        token_type_dxs = [0] * len(all_disease_ids) 
        # 0: true 1: false 2: mask
        attr_dx_labels = [0 if dx_id == example['label'] else 1 for dx_id in all_disease_ids]
        attr_dxs = [2] * len(all_disease_ids)
        input_sx = example['exp_sx_ids'] + example['imp_sx_ids']
        token_type_sx = [1] * len(input_sx)
        attr_sx_label = example['exp_attr_ids'] + example['imp_attr_ids']

        # task 1: 输入exp_sx和imp_sx及其attr, 输入某个具体的dx, 预测其attr
        for (input_dx, token_type_dx, attr_dx, attr_dx_label) in zip(input_dxs, token_type_dxs, attr_dxs, attr_dx_labels):
            input_id = [input_dx] + input_sx
            type_id = [token_type_dx] + token_type_sx
            attr_id = [attr_dx] + attr_sx_label
            input_mask = [1] * len(input_id)
            attr_label = attr_dx_label
            token_label = -100
            features.append((input_id, type_id, attr_id, input_mask, attr_label, token_label))
        
        # task 2:输入exp_sx和imp_sx及其attr, 给定dx_attr为true，预测dx (诊断任务预测方式)
        input_id = [self.tokenizer.sx_mask_token_id] + input_sx
        type_id = [0] + token_type_sx
        attr_id = [0] + attr_sx_label
        input_mask = [1] * len(input_id)
        attr_label = -100
        token_label = example['label']
        features.append((input_id, type_id, attr_id, input_mask, attr_label, token_label))

        # task 3:给定attr为true的dx及其attr，随机mask一个(或多个)sx的attr，预测其attr
        # 考虑到样本较少，对sx进行遍历
        # for index, sx in enumerate(input_sx):
        #     input_id = [sx] + input_sx[:index] + input_sx[index+1:] + [example['label']]
        #     type_id = [1] * len(input_sx) + [0]
        #     # attr mask
        #     attr_id = [2] + token_type_sx[:index] + token_type_sx[index+1:] + [0]
        #     input_mask = [1] * len(input_id)
        #     attr_label = token_type_sx[index]
        #     token_label = -100
        #     features.append((input_id, type_id, attr_id, input_mask, attr_label, token_label))

        # 对imp_sx中的所有子集均预测
        subsets, removed_subsets = get_subsets_and_removed_subsets(range(len(example['imp_sx_ids'])))
        for i, subset in enumerate(subsets):
            for sx_index in removed_subsets[i]:
                input_id = [example['imp_sx_ids'][sx_index]] + example['exp_sx_ids'] + [example['imp_sx_ids'][i] for i in subset] + [example['label']]
                type_id = [1] * (len(input_id)-1) + [0]
                # attr mask
                attr_id = [2] + example['exp_attr_ids'] + [example['imp_attr_ids'][i] for i in subset] + [0]
                input_mask = [1] * len(input_id)
                attr_label = example['imp_attr_ids'][sx_index]
                token_label = -100
                features.append((input_id, type_id, attr_id, input_mask, attr_label, token_label))

        # task 4:给定除特定sx外所有已知的sx及其attr，对于该特定sx，输入其attr为true or false，预测其对应的sx (症状问询预测方式)，也可以与task 3结合作为症状问询预测方式
        # 对imp_sx中的所有子集均预测，这里不加入dx
        for i, subset in enumerate(subsets):
            for sx_index in removed_subsets[i]:
                # sx mask
                input_id = [self.tokenizer.sx_mask_token_id] + example['exp_sx_ids'] + [example['imp_sx_ids'][i] for i in subset]
                type_id = [1] * (len(input_id))
                attr_id = [example['imp_attr_ids'][sx_index]] + example['exp_attr_ids'] + [example['imp_attr_ids'][i] for i in subset]
                input_mask = [1] * len(input_id)
                attr_label = -100
                token_label = example['imp_sx_ids'][sx_index]
                features.append((input_id, type_id, attr_id, input_mask, attr_label, token_label))

        return features

    def __getitem__(self, idx):
        return self.convert_example_to_feature(self.data[idx])

    def __len__(self):
        return len(self.data)

class ReaderMedDataset_gen(torch.utils.data.Dataset):
    def __init__(self, data_dir, tokenizer, stage='stage1', stage1_index_file=None):
        # stage='stage1','stage2'
        self.tokenizer = tokenizer
        self.stage = stage
        if dist_utils.is_local_master():
            logger.info(f"Data dir: {data_dir}")
        data = read_json(path=data_dir)
        self.data = []
        if self.stage == 'stage1':
            for example in data:
                if example["term_id"] == -1:
                    self.data.append(example)
        elif self.stage == 'stage2':
            if stage1_index_file is None:
                for example in data:
                    if example["term_id"] != -1:
                        self.data.append(example)
            else:
                stage1_index = read_json(stage1_index_file)
                for example in data:
                    if [example['dial_id'], example['window_id'], example['term_id']] in stage1_index:
                        self.data.append(example)

        if dist_utils.is_local_master():
            logger.info(f"Total data size: {len(self.data)}")

        # self.data = data

    def convert_example_to_feature(
        self,
        example,
        max_input_len=512,
        max_output_len=512,
    ):
        input = self.tokenizer.encode(example['context'].lower(), add_special_tokens=True)[:max_input_len]
        input_mask = [1] * len(input)
        label = self.tokenizer.encode(example['output'].lower())[1:max_output_len]
        dial_id = example['dial_id']
        window_id = example['window_id']
        term_id = example['term_id']
        

        return (input, input_mask, label, dial_id,  window_id, term_id)

    def __getitem__(self, idx):
        return self.convert_example_to_feature(self.data[idx])

    def __len__(self):
        return len(self.data)

class ReaderMedDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, tokenizer):
        self.tokenizer = tokenizer
        if dist_utils.is_local_master():
            logger.info(f"Data dir: {data_dir}")
        data = read_json(path=data_dir)

        if dist_utils.is_local_master():
            logger.info(f"Total data size: {len(data)}")

        self.data = data

    def convert_example_to_feature(
        self,
        example,
        max_input_len=512,
        max_output_len=512,
    ):
        input = self.tokenizer.encode(example['context'].lower(), add_special_tokens=True)[:max_input_len]
        input_mask = [1] * len(input)
        label = self.tokenizer.encode(example['output'].lower())[1:max_output_len]
        dial_id = example['dial_id']
        window_id = example['window_id']
        term_id = example['term_id']
        

        return (input, input_mask, label, dial_id,  window_id, term_id)

    def __getitem__(self, idx):
        return self.convert_example_to_feature(self.data[idx])

    def __len__(self):
        return len(self.data)


class ReaderDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        paths = glob.glob(os.path.join(data_dir, '*'))

        if dist_utils.is_local_master():
            logger.info(f"Data dir: {data_dir}")
            logger.info(f"Data paths: {paths}")

        assert paths, "No Data files found."
        data = []
        for path in paths:
            with open(path, "rb") as f:
                data.extend(pickle.load(f))

        if dist_utils.is_local_master():
            logger.info(f"Total data size: {len(data)}")

        for d in data:
            d.to_tensor()
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

