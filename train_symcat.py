import collections
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
from sys import path
path.append(os.getcwd())
# path.append('/home/Medical_Understanding/MSL')
from typing import List
import time
import heapq
from tqdm import tqdm

import argparse
import glob
import logging
import math
import numpy as np
import torch
import torch.nn.functional as F
import transformers as tfs
from transformers import BertTokenizer, T5ForConditionalGeneration, Text2TextGenerationPipeline
# from model import DiaT5
from modeling import BertForDiagnosis
from tokenizer import diaTokenizer
from data_utils import data_collator, reader_dataset
from utils import dist_utils
from utils import model_utils
from utils import options
from utils import sampler
from utils import utils
from tensorboardX import SummaryWriter

from data_utils.common_utils import read_json

try:
    from apex import amp
except:
    pass

"""
使用BERT作为backbone, 便于设计position embedding, token_type_embedding
修改其中的生成过程，先预测top 10，然后将top 10进行匹配过滤其中false的疾病，最后选择生成预测概率最大的。
"""

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


class ModelTrainer(object):
    
    def __init__(self, args):

        utils.print_section_bar('Initializing components for training')

        tokenizer = diaTokenizer(args.vocab_file)
        cfg = tfs.BertConfig.from_pretrained(args.model_cfg)
        cfg.vocab_size = len(tokenizer)
        cfg.attention_probs_dropout_prob = args.dropout
        cfg.hidden_dropout_prob = args.dropout
        model = BertForDiagnosis(cfg)


        if args.inference_only:
            optimizer = None
        else:
            optimizer = model_utils.get_optimizer(
                model,
                learning_rate=args.learning_rate,
                adam_eps=args.adam_eps,
                weight_decay=args.weight_decay)

        self.model, self.optimizer = model_utils.setup_for_distributed_mode(
            model,
            optimizer,
            args.device,
            args.n_gpu,
            args.local_rank,
            args.fp16,
            args.fp16_opt_level)

        self.start_epoch = 0
        self.start_offset = 0
        self.global_step = 0
        self.args = args
        
        self.tokenizer = tokenizer
        self.tb_writer = SummaryWriter(logdir=args.log_dir)


    def get_train_dataloader(self, train_dataset, shuffle=True, offset=0):
        if torch.distributed.is_initialized():
            train_sampler = sampler.DistributedSampler(
                train_dataset,
                num_replicas=self.args.distributed_world_size,
                rank=self.args.local_rank,
                shuffle=shuffle)
            train_sampler.set_offset(offset)
        else:
            assert self.args.local_rank == -1
            train_sampler = torch.utils.data.RandomSampler(train_dataset)

        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            pin_memory=True,
            sampler=train_sampler,
            num_workers=0,
            collate_fn=data_collator.collate_fn,
            drop_last=False)

        return dataloader

    def run_train(self):
        args = self.args

        train_dataset = reader_dataset.DiaDataset(args, args.train_file, self.tokenizer)
        train_dataloader = self.get_train_dataloader(
            train_dataset,
            shuffle=True,
            offset=self.start_offset)

        updates_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps)
        total_updates = updates_per_epoch * args.num_train_epochs

        dataloader_steps = self.start_offset // (
            args.distributed_world_size * args.batch_size)
        updated_steps = (dataloader_steps // 
                         args.gradient_accumulation_steps) + (
                             self.start_epoch * updates_per_epoch)
        remaining_updates = total_updates - updated_steps

        # global_step is added per dataloader step.
        calc_global_step = (self.start_epoch * len(train_dataloader) + 
                            dataloader_steps)

        assert self.global_step == calc_global_step, \
            (f'global step = {self.global_step}, '
             f'calc global step = {calc_global_step}')

        self.scheduler = model_utils.get_schedule_linear(
            self.optimizer,
            warmup_steps=args.warmup_steps,
            training_steps=total_updates,
            last_epoch=self.global_step-1)

        utils.print_section_bar('Training')
        if dist_utils.is_local_master():
            logger.info(f'Total updates = {total_updates}')
            logger.info(
                f'Updates per epoch (/gradient accumulation) = '
                f'{updates_per_epoch}')
            logger.info(
                f'Steps per epoch (dataloader) = {len(train_dataloader)}')
            logger.info(
                f'Gradient accumulation steps = '
                f'{args.gradient_accumulation_steps}')
            logger.info(
                f'Start offset of the epoch {self.start_epoch} (dataset) = '
                f'step {self.start_offset}')
            logger.info(
                f'Updated step of the epoch {self.start_epoch} (dataloader) = '
                f'step {updated_steps}')
            logger.info(
                f'Total remaining updates = {remaining_updates}')

        # Starts training here.
        max_score = 0
        epoch_max = 0
        for epoch in range(self.start_epoch, int(args.num_train_epochs)):
            utils.print_section_bar(f'Epoch {epoch}')

            if isinstance(train_dataloader.sampler, sampler.DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            self._train_epoch(epoch, train_dataloader)
            if epoch > 0:
                max_score, epoch_max = self.generate(max_score, epoch, epoch_max)

            if isinstance(train_dataloader.sampler, sampler.DistributedSampler):
                train_dataloader.sampler.set_offset(0)

        utils.print_section_bar('Training finished. epoch_max: {}'.format(epoch_max))

        return


    def _train_epoch(self, epoch, train_dataloader):
        args = self.args
        epoch_loss = 0
        rolling_train_losses = collections.defaultdict(int)
        rolling_train_others = collections.defaultdict(int)

        step_offset = 0

        train_batch_times = []
        # epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(train_dataloader, start=step_offset):
            self.model.train()
            step += 1

            batch_start_time = time.time()
            if step % args.gradient_accumulation_steps != 0 \
                    and args.local_rank != -1:
                with self.model.no_sync(): # 不进行梯度同步
                    losses, others = self._training_step(batch)
            else:
                losses, others = self._training_step(batch)
            batch_end_time = time.time()
            train_batch_times.append(batch_end_time - batch_start_time)

            self.global_step += 1

            '''
                record loss
            '''
            epoch_loss += losses['total']
            for k, loss in losses.items():
                # add
                if dist_utils.is_local_master():
                    self.tb_writer.add_scalar(k + '_loss', loss, self.global_step)

                rolling_train_losses[k] += loss
            for k, other in others.items():
                # other could be -1 if adv_loss not applicable
                rolling_train_others[k] += max(other, 0)

            '''
                parameters update
            '''
            if (step - step_offset) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(self.optimizer), args.max_grad_norm
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), args.max_grad_norm
                        )
    
                self.scheduler.step()
                self.optimizer.step()
                self.model.zero_grad()

        epoch_loss = epoch_loss / len(train_dataloader)

        if dist_utils.is_local_master():
            logger.info(f'Train: global step = {self.global_step}; '
                        f'step = {step}')
            logger.info(f'Avg. total Loss of epoch {epoch} ={epoch_loss:.3f}')
            # logger.info(
            #     "** ** * Saving fine-tuned model ** ** * ")
            output_model_file = os.path.join(
                args.output_dir, "model.{0}.bin".format(epoch))
            if hasattr(self.model, 'module'):
                torch.save(self.model.module.state_dict(), output_model_file)
            else:
                torch.save(self.model.state_dict(), output_model_file)

    def _training_step(self, batch) -> torch.Tensor:
        args = self.args
        batch = tuple(t.to(args.device) for t in batch)
        # input_ids, input_masks, lm_labels, attr_labels = batch
        input_ids, input_masks, type_id, attr_id, attr_label, token_label = batch
        # input_ids, input_masks, position_id, lm_labels, decoder_position_id, attr_labels = batch
        output = self.model(input_ids=input_ids, attention_mask=input_masks, token_type_ids=type_id, attr_ids=attr_id, \
                            next_sentence_label=attr_label, labels=token_label)
        losses = {'total': output['loss']}

        losses = {k: loss.mean() for k, loss in losses.items()}

        if args.fp16:
            with amp.scale_loss(losses['total'], self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            losses['total'].backward()

        return {k: v.item() for k, v in losses.items()}, {}

    # Use generation to simulate the diagnostic process
    def generate(self, max_score=0, epoch=0, epoch_max=0):
        # 综合dx匹配和生成选择最终的诊断结果
        testdata = read_json(self.args.dev_file)

        # the result list
        reslist = []

        # record of symptom inquiry
        mc_acc = 0
        pred_No = 0
        imp_acc = 0
        imp_all = 0
        imp_recall = 0
        vocab_size = len(self.tokenizer)
        # start simulation for each testing data
        self.model.eval()
        with torch.no_grad():
            for item_index, item in enumerate(testdata):
                exp_sx_ids = []
                exp_attr_ids = []
                expslots = {}
                imp_sx_ids = []
                imp_attr_ids = []
                impslots = {}
                label = None
                # Expset records explicit symptoms
                for sx, attr in item['exp_sxs'].items():
                    if attr not in ['0', '1']:
                        continue
                    sx_id = self.tokenizer.convert_token_to_id(sx)
                    exp_sx_ids.append(sx_id)
                    attr_id = self.tokenizer.convert_attr_to_id(attr)
                    exp_attr_ids.append(attr_id)
                    expslots[sx_id] = attr_id
                # reserve the implicit symptoms
                for sx, attr in item['imp_sxs'].items():
                    if attr not in ['0', '1']:
                        continue
                    sx_id = self.tokenizer.convert_token_to_id(sx)
                    imp_sx_ids.append(sx_id)
                    attr_id = self.tokenizer.convert_attr_to_id(attr)
                    imp_attr_ids.append(attr_id)
                    impslots[sx_id] = attr_id
                imp_all += len(imp_sx_ids)
                
                label = self.tokenizer.convert_token_to_id(item["label"])

                disease_ids = self.tokenizer.get_disease_ids()
                input_sx = exp_sx_ids + []
                token_type_sx = [1] * len(input_sx)
                attr_sx = exp_attr_ids + []
                # features = []
                # task 4:给定除特定sx外所有已知的sx及其attr，对于该特定sx，输入其attr为true or false，预测其对应的sx

                # save all the requiry symptom
                generated = []

                for _ in range(self.args.max_turn + 1):
                    input_id = [self.tokenizer.sx_mask_token_id] + input_sx
                    type_id = [1] * (len(input_id))
                    # attr为true
                    attr_id_0 = [0] + attr_sx
                    # attr为false
                    attr_id_1 = [1] + attr_sx
                    input_mask = [1] * len(input_id)

                    curr_input_tensor = torch.tensor([input_id, input_id]).long().to(self.args.device)
                    input_masks_tensor = torch.tensor([input_mask, input_mask]).long().to(self.args.device)
                    type_id_tensor = torch.tensor([type_id, type_id]).long().to(self.args.device)
                    attr_id_tensor = torch.tensor([attr_id_0, attr_id_1]).long().to(self.args.device)

                    outputs = self.model(input_ids=curr_input_tensor, attention_mask=input_masks_tensor, token_type_ids=type_id_tensor, attr_ids=attr_id_tensor, \
                                next_sentence_label=None, labels=None)
                    token_logits = F.softmax(outputs['prediction_logits'], dim=-1)
                    token_logits = token_logits.cpu().tolist()
                    token_logits = token_logits[0] + token_logits[1]
                    sorted_list = sorted(enumerate(token_logits), key=lambda x: x[1], reverse=True)
                    sorted_indices, sorted_logits = zip(*sorted_list)
                    sorted_logits, sorted_indices = list(sorted_logits), list(sorted_indices)

                    # whether stop inquring symptoms
                    isDiease = False

                    for index, id in enumerate(sorted_indices):
                        token_id = id % vocab_size
                        if len(generated) >= self.args.max_turn:
                            isDiease = True
                            break
                        elif sorted_logits[index] < self.args.min_probability:
                            isDiease = True
                            break
                        elif token_id in exp_sx_ids:
                        # check if the symptom inquired is a explicit symptoms
                            continue
                        elif token_id in generated:
                        # check if the symptom has been inquired 
                            continue
                        elif token_id in self.tokenizer.special_tokens_id or token_id in self.tokenizer.disease_tokens_id:
                            continue
                        else:
                            # inquire symptom
                            if token_id in impslots:
                                # in implicit symptom set
                                imp_acc += 1
                                generated.append(token_id)
                                input_sx.append(token_id)
                                attr_sx.append(impslots[token_id])
                                break
                            else:
                                # not in implicit symptom set
                                generated.append(token_id)
                    
                    if isDiease:
                        # input_sx = exp_sx_ids + imp_sx_ids
                        token_type_sx = [1] * len(input_sx)
                        attr_sx_label = attr_sx
                        # features = []
                        input_ids, type_ids, attr_ids, attention_masks = [], [], [], []
                        # task 1: 输入exp_sx和imp_sx及其attr, 输入某个具体的dx, 预测其attr
                        for dx_id in disease_ids:
                            input_id = [dx_id] + input_sx
                            type_id = [0] + token_type_sx
                            attr_id = [2] + attr_sx_label
                            input_mask = [1] * len(input_id)
                            input_ids.append(input_id)
                            type_ids.append(type_id)
                            attr_ids.append(attr_id)
                            attention_masks.append(input_mask)
                            # features.append((input_id, type_id, attr_id, input_mask))

                        # task 2:输入exp_sx和imp_sx及其attr, 给定dx_attr为true，预测dx
                        input_id = [self.tokenizer.sx_mask_token_id] + input_sx
                        type_id = [0] + token_type_sx
                        attr_id = [0] + attr_sx_label
                        input_mask = [1] * len(input_id)
                        input_ids.append(input_id)
                        type_ids.append(type_id)
                        attr_ids.append(attr_id)
                        attention_masks.append(input_mask)
                        # input_ids, type_ids, attr_ids, attention_masks = list(zip(*features))
                        curr_input_tensor = torch.tensor(input_ids).long().to(self.args.device)
                        # curr_input_tensor = torch.tensor([torch.tensor(instance) for i, instance in enumerate(input_ids)]).long().to(self.args.device)
                        attention_masks_tensor = torch.tensor(attention_masks).long().to(self.args.device)
                        type_id_tensor = torch.tensor(type_ids).long().to(self.args.device)
                        attr_id_tensor = torch.tensor(attr_ids).long().to(self.args.device)

                        output = self.model(input_ids=curr_input_tensor, attention_mask=attention_masks_tensor, token_type_ids=type_id_tensor, attr_ids=attr_id_tensor, \
                                    next_sentence_label=None, labels=None)
                        seq_relationship_logits = output['seq_relationship_logits'][:-1]
                        token_logits = output['prediction_logits'][-1, len(self.tokenizer) - len(disease_ids): len(self.tokenizer)]
                        # 只有生成
                        if self.args.only_gen:
                            label_index = torch.argmax(token_logits, -1).item()
                            flag = '生成v0'
                        else:
                            _, gen_indices = torch.sort(token_logits, descending=True)
                            gen_indices = gen_indices.cpu().tolist()
                            # 取 top 20
                            gen_indices = gen_indices[:20]
                            attr_pred = torch.argmax(seq_relationship_logits, -1)
                            dx_index = torch.where(attr_pred==0)[0]
                            dx_index = dx_index.cpu().tolist()
                            if len(dx_index) == 0:
                                label_index = torch.argmax(token_logits, -1).item()
                                flag = '生成v0'
                            else:
                                attr_tensor = torch.where(attr_pred == 0, torch.tensor(1).to(self.args.device), torch.tensor(0).to(self.args.device))
                                token_logits = F.softmax(token_logits, dim=-1)
                                token_logits = torch.mul(token_logits, attr_tensor)
                                label_index = torch.argmax(token_logits, -1).item()
                                flag = '生成v2'
                                if label_index not in gen_indices:
                                    label_index = gen_indices[0]
                                    flag = '生成v3'

                        if disease_ids[label_index] == label:
                            mc_acc += 1
                        break
                imp_recall += len(generated)
                res = {'explicit_symptoms':item['exp_sxs'],'implicit_symptoms':item['imp_sxs'],'target_disease':item['label'], \
                        'inquiry_symptom': [self.tokenizer.convert_id_to_token(x) for x in generated] , \
                        'pred_disease':self.tokenizer.convert_id_to_token(disease_ids[label_index]), 'num_turn': len(generated), 'flag': flag}
                reslist.append(res)
            
        # total metric
        tscore =  0.8*mc_acc/len(testdata)+0.2*imp_acc/(imp_all)
        
        if tscore > max_score: 
            max_score = tscore
            epoch_max = epoch
            logger.info(
                "** ** * Saving fine-tuned model ** ** * ")
            output_model_file = os.path.join(
                self.args.output_dir, "model.bin")
            if hasattr(self.model, 'module'):
                torch.save(self.model.module.state_dict(), output_model_file)
            else:
                torch.save(self.model.state_dict(), output_model_file)
            if self.args.result_output_path is not None:
                logger.info('** ** * results saved ** ** * ')
                with open(self.args.result_output_path,'w') as f:
                    json.dump(reslist,f,ensure_ascii=False,indent=4)
        logger.info('epoch {}:generative results\n sym_recall:{}, disease:{}, avg_turn:{}, pred_no:{}'.format(epoch, imp_acc/imp_all,mc_acc/len(testdata),imp_recall/len(testdata), pred_No))
        return max_score, epoch_max

def main():
    parser = argparse.ArgumentParser()

    options.add_model_params(parser)
    options.add_cuda_params(parser)
    options.add_training_params(parser)
    options.add_data_params(parser)
    args = parser.parse_args()

    args.output_dir += '_k_{}'.format(args.k)
    if args.add_task_1:
        args.output_dir += '_1'
    if args.add_task_2:
        args.output_dir += '_2'
    if args.add_task_3:
        args.output_dir += '_3'
    if args.add_task_4:
        args.output_dir += '_4'
    if args.add_middle_task_1:
        args.output_dir += '_add_middle_1'
    if args.add_middle_task_2:
        args.output_dir += '_add_middle_2'
    if args.dropout != 0.1:
        args.output_dir += '_drop_{}'.format(args.dropout)

    args.log_dir = os.path.join(args.output_dir, args.log_dir) 
    args.result_output_path = os.path.join(args.output_dir, args.result_output_path)
    args.train_file = os.path.join(args.data_dir, args.train_file)
    args.dev_file = os.path.join(args.data_dir, args.dev_file)
    options.setup_args_gpu(args)
    # Makes sure random seed is fixed.
    # set_seed must be called after setup_args_gpu.
    options.set_seed(args)

    if dist_utils.is_local_master():
        utils.print_args(args)

    trainer = ModelTrainer(args)
    trainer.run_train()
    # for i in range(199):
    #     trainer.model.load_state_dict(torch.load(os.path.join(args.output_dir, 'model.{}.bin'.format(i)), map_location=args.device), strict=True)
    #     # trainer.model.eval()
    #     max_score = trainer.generate(max_score, i)
    # trainer.model.load_state_dict(torch.load(os.path.join(args.output_dir, 'model.bin'), map_location=args.device))
    # trainer.generate()

    # max_score = 0
    # trainer.model.load_state_dict(torch.load(os.path.join(args.output_dir, 'model.{}.bin'.format(199)), map_location=args.device))
    # max_score = trainer.generate(max_score)
    # max_score = trainer.predict(max_score)


if __name__ == '__main__':
    main()