#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
 Pipeline to train DPR Biencoder
"""

import argparse
import glob
import logging
import math
import os
import random
import time

import torch

from typing import Tuple
from torch import nn
from torch import Tensor as T

from dpr.models import init_encoder_components
from dpr.models.biencoder import BiEncoder, BiEncoderNllLoss, BiEncoderBatch
from dpr.options import add_encoder_params, add_training_params, setup_args_gpu, set_seed, print_args, \
    get_encoder_params_state, add_tokenizer_params, set_encoder_params_from_state, add_tuning_params
from dpr.utils.data_utils import ShardedDataIterator, read_data_from_json_files, Tensorizer
from dpr.utils.dist_utils import all_gather_list
from dpr.utils.model_utils import setup_for_distributed_mode, move_to_device, get_schedule_linear, CheckpointState, \
    get_model_file, get_model_obj, load_states_from_checkpoint

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


class BiEncoderTrainer(object):
    """
    BiEncoder training pipeline component. Can be used to initiate or resume training and validate the trained model
    using either binary classification's NLL loss or average rank of the question's gold passages across dataset
    provided pools of negative passages. For full IR accuracy evaluation, please see generate_dense_embeddings.py
    and dense_retriever.py CLI tools.
    """

    def __init__(self, args):
        self.args = args

        self.prefix = args.prefix
        self.prompt = args.prompt
        self.adapter = args.adapter

        self.shard_id = args.local_rank if args.local_rank != -1 else 0
        self.distributed_factor = args.distributed_world_size or 1

        logger.info("***** Initializing components for training *****")

        # if model file is specified, encoder parameters from saved state should be used for initialization
        model_file = get_model_file(self.args, self.args.checkpoint_file_name)
        saved_state = None
        if model_file and not self.adapter:
            saved_state = load_states_from_checkpoint(model_file)
            set_encoder_params_from_state(saved_state.encoder_params, args)
            

        tensorizer, model, optimizer = init_encoder_components("dpr", args)

        model, optimizer = setup_for_distributed_mode(model, optimizer, args.device, args.n_gpu,
                                                      args.local_rank,
                                                      args.fp16,
                                                      args.fp16_opt_level)
        self.biencoder = model
        self.optimizer = optimizer
        self.tensorizer = tensorizer
        self.start_epoch = 0
        self.start_batch = 0
        self.scheduler_state = None
        self.best_validation_result = None
        self.best_cp_name = None

        if saved_state:
            self._load_saved_state(saved_state)

    def get_data_iterator(self, path: str, batch_size: int, shuffle=True,
                          shuffle_seed: int = 0,
                          offset: int = 0, upsample_rates: list = None) -> ShardedDataIterator:
        data_files = glob.glob(path)

        if len(data_files) > 1:
            upsample_rates = []
            for path in data_files:
                if "trec" in path or "webq" in path:
                    upsample_rates.append(4)
                else:
                    upsample_rates.append(1)
            print("upsample ratse =", upsample_rates)
            
        data = read_data_from_json_files(data_files, upsample_rates)
    
        # filter those without positive ctx
        data = [r for r in data if len(r['positive_ctxs']) > 0]
        logger.info('Total cleaned data size: {}'.format(len(data)))

        return ShardedDataIterator(data, shard_id=self.shard_id,
                                   num_shards=self.distributed_factor,
                                   batch_size=batch_size, shuffle=shuffle, shuffle_seed=shuffle_seed, offset=offset,
                                   strict_batch_size=True,  # this is not really necessary, one can probably disable it
                                   )

    def run_train(self, ):
        args = self.args
        upsample_rates = None
        if args.train_files_upsample_rates is not None:
            upsample_rates = eval(args.train_files_upsample_rates)

        train_iterator = self.get_data_iterator(args.train_file, args.batch_size,
                                                shuffle=True,
                                                shuffle_seed=args.seed, offset=self.start_batch,
                                                upsample_rates=upsample_rates)

        logger.info("  Total iterations per epoch=%d", train_iterator.max_iterations)
        updates_per_epoch = train_iterator.max_iterations // args.gradient_accumulation_steps
        total_updates = max(updates_per_epoch * (args.num_train_epochs - self.start_epoch - 1), 0) + \
                        (train_iterator.max_iterations - self.start_batch) // args.gradient_accumulation_steps
        logger.info(" Total updates=%d", total_updates)
        if args.warmup_percentage is not None:
            warmup_steps = round(args.warmup_percentage * total_updates)
        else:
            warmup_steps = args.warmup_steps
        logger.info(" Total warmup=%d", warmup_steps)
        scheduler = get_schedule_linear(self.optimizer, warmup_steps, total_updates)

        if self.scheduler_state:
            logger.info("Loading scheduler state %s", self.scheduler_state)
            scheduler.load_state_dict(self.scheduler_state)

        eval_step = math.ceil(updates_per_epoch / args.eval_per_epoch)
        logger.info("  Eval step = %d", eval_step)
        logger.info("***** Training *****")

        for epoch in range(self.start_epoch, int(args.num_train_epochs)):
            logger.info("***** Epoch %d *****", epoch)
            self._train_epoch(scheduler, epoch, eval_step, train_iterator)

        if args.local_rank in [-1, 0]:
            logger.info('Training finished. Best validation checkpoint %s', self.best_cp_name)

    def validate_and_save(self, epoch: int, iteration: int, scheduler):
        args = self.args
        # for distributed mode, save checkpoint for only one process
        save_cp = args.local_rank in [-1, 0]

        if epoch == args.val_av_rank_start_epoch:
            self.best_validation_result = None

        if epoch >= args.val_av_rank_start_epoch:
            validation_loss = self.validate_average_rank()
        else:
            validation_loss = self.validate_nll()

        if save_cp:
            cp_name = self._save_checkpoint(scheduler, epoch, iteration)
            logger.info('Saved checkpoint to %s', cp_name)
            
            if validation_loss < (self.best_validation_result or validation_loss + 1):
                self.best_validation_result = validation_loss
                self.best_cp_name = cp_name
                logger.info('New Best validation checkpoint %s', cp_name)

    def validate_nll(self) -> float:
        logger.info('NLL validation ...')
        args = self.args
        self.biencoder.eval()
        data_iterator = self.get_data_iterator(args.dev_file, args.dev_batch_size, shuffle=False)

        total_loss = 0.0
        start_time = time.time()
        total_correct_predictions = 0
        num_hard_negatives = args.hard_negatives
        num_other_negatives = args.other_negatives
        log_result_step = args.log_batch_step
        batches = 0
        for i, samples_batch in enumerate(data_iterator.iterate_data()):
            biencoder_input = BiEncoder.create_biencoder_input(samples_batch, self.tensorizer, self.args.ctx_field,
                                                               True,
                                                               num_hard_negatives, num_other_negatives, shuffle=False)

            loss, correct_cnt = _do_biencoder_fwd_pass(self.biencoder, biencoder_input, self.tensorizer, args)
            total_loss += loss.item()
            total_correct_predictions += correct_cnt
            batches += 1
            if (i + 1) % log_result_step == 0:
                logger.info('Eval step: %d , used_time=%f sec., loss=%f ', i, time.time() - start_time, loss.item())

        total_loss = total_loss / batches
        total_samples = batches * args.dev_batch_size * self.distributed_factor
        correct_ratio = float(total_correct_predictions / total_samples)
        logger.info('NLL Validation: loss = %f. correct prediction ratio  %d/%d ~  %f', total_loss,
                    total_correct_predictions,
                    total_samples,
                    correct_ratio
                    )
        return total_loss

    def validate_average_rank(self) -> float:
        """
        Validates biencoder model using each question's gold passage's rank across the set of passages from the dataset.
        It generates vectors for specified amount of negative passages from each question (see --val_av_rank_xxx params)
        and stores them in RAM as well as question vectors.
        Then the similarity scores are calculted for the entire
        num_questions x (num_questions x num_passages_per_question) matrix and sorted per quesrtion.
        Each question's gold passage rank in that  sorted list of scores is averaged across all the questions.
        :return: averaged rank number
        """
        logger.info('Average rank validation ...')

        args = self.args
        self.biencoder.eval()
        distributed_factor = self.distributed_factor

        data_iterator = self.get_data_iterator(args.dev_file, args.dev_batch_size, shuffle=False)

        sub_batch_size = args.val_av_rank_bsz
        sim_score_f = BiEncoderNllLoss.get_similarity_function()
        q_represenations = []
        ctx_represenations = []
        positive_idx_per_question = []

        num_hard_negatives = args.val_av_rank_hard_neg
        num_other_negatives = args.val_av_rank_other_neg

        log_result_step = args.log_batch_step

        for i, samples_batch in enumerate(data_iterator.iterate_data()):
            # samples += 1
            if len(q_represenations) > args.val_av_rank_max_qs / distributed_factor:
                break

            biencoder_input = BiEncoder.create_biencoder_input(samples_batch, self.tensorizer, self.args.ctx_field,
                                                               True,
                                                               num_hard_negatives, num_other_negatives, shuffle=False)
            total_ctxs = len(ctx_represenations)
            ctxs_ids = biencoder_input.context_ids
            ctxs_segments = biencoder_input.ctx_segments
            bsz = ctxs_ids.size(0)

            # split contexts batch into sub batches since it is supposed to be too large to be processed in one batch
            for j, batch_start in enumerate(range(0, bsz, sub_batch_size)):

                q_ids, q_segments = (biencoder_input.question_ids, biencoder_input.question_segments) if j == 0 \
                    else (None, None)

                if j == 0 and args.n_gpu > 1 and q_ids.size(0) == 1:
                    # if we are in DP (but not in DDP) mode, all model input tensors should have batch size >1 or 0,
                    # otherwise the other input tensors will be split but only the first split will be called
                    continue

                ctx_ids_batch = ctxs_ids[batch_start:batch_start + sub_batch_size]
                ctx_seg_batch = ctxs_segments[batch_start:batch_start + sub_batch_size]

                q_attn_mask = self.tensorizer.get_attn_mask(q_ids)
                ctx_attn_mask = self.tensorizer.get_attn_mask(ctx_ids_batch)
                with torch.no_grad():
                    q_dense, ctx_dense = self.biencoder(q_ids, q_segments, q_attn_mask, ctx_ids_batch, ctx_seg_batch,
                                                        ctx_attn_mask)

                if q_dense is not None:
                    q_represenations.extend(q_dense.cpu().split(1, dim=0))

                ctx_represenations.extend(ctx_dense.cpu().split(1, dim=0))

            batch_positive_idxs = biencoder_input.is_positive
            positive_idx_per_question.extend([total_ctxs + v for v in batch_positive_idxs])

            if (i + 1) % log_result_step == 0:
                logger.info('Av.rank validation: step %d, computed ctx_vectors %d, q_vectors %d', i,
                            len(ctx_represenations), len(q_represenations))

        ctx_represenations = torch.cat(ctx_represenations, dim=0)
        q_represenations = torch.cat(q_represenations, dim=0)

        logger.info('Av.rank validation: total q_vectors size=%s', q_represenations.size())
        logger.info('Av.rank validation: total ctx_vectors size=%s', ctx_represenations.size())

        q_num = q_represenations.size(0)
        assert q_num == len(positive_idx_per_question)

        scores = sim_score_f(q_represenations, ctx_represenations)
        values, indices = torch.sort(scores, dim=1, descending=True)

        rank = 0
        for i, idx in enumerate(positive_idx_per_question):
            # aggregate the rank of the known gold passage in the sorted results for each question
            gold_idx = (indices[i] == idx).nonzero()
            rank += gold_idx.item()

        if distributed_factor > 1:
            # each node calcuated its own rank, exchange the information between node and calculate the "global" average rank
            # NOTE: the set of passages is still unique for every node
            eval_stats = all_gather_list([rank, q_num], max_size=100)
            for i, item in enumerate(eval_stats):
                remote_rank, remote_q_num = item
                if i != args.local_rank:
                    rank += remote_rank
                    q_num += remote_q_num

        av_rank = float(rank / q_num)
        logger.info('Av.rank validation: average rank %s, total questions=%d', av_rank, q_num)
        return av_rank

    def _train_epoch(self, scheduler, epoch: int, eval_step: int,
                     train_data_iterator: ShardedDataIterator, ):
        logger.info('Best validation checkpoint %s: %f', self.best_cp_name, self.best_validation_result if self.best_validation_result else 0)

        args = self.args
        rolling_train_loss = 0.0
        epoch_loss = 0
        epoch_correct_predictions = 0

        log_result_step = args.log_batch_step
        rolling_loss_step = args.train_rolling_loss_step
        num_hard_negatives = args.hard_negatives
        num_other_negatives = args.other_negatives
        seed = args.seed
        self.biencoder.train()
        epoch_batches = train_data_iterator.max_iterations
        data_iteration = 0
        start_time = time.time()
        for i, samples_batch in enumerate(train_data_iterator.iterate_data(epoch=epoch)):
            # to be able to resume shuffled ctx- pools
            data_iteration = train_data_iterator.get_iteration()
            random.seed(seed + epoch + data_iteration)
            biencoder_batch = BiEncoder.create_biencoder_input(samples_batch, self.tensorizer, self.args.ctx_field,
                                                               True,
                                                               num_hard_negatives, num_other_negatives, shuffle=True,
                                                               shuffle_positives=args.shuffle_positive_ctx
                                                               )

            loss, correct_cnt = _do_biencoder_fwd_pass(self.biencoder, biencoder_batch, self.tensorizer, args)

            epoch_correct_predictions += correct_cnt
            epoch_loss += loss.item()
            rolling_train_loss += loss.item()

            if args.fp16:
                from apex import amp
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), args.max_grad_norm)
            else:
                loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.biencoder.parameters(), args.max_grad_norm)

            if (i + 1) % args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                scheduler.step()
                self.biencoder.zero_grad()

            if i % log_result_step == 0:
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    'Epoch: %d: Step: %d/%d, loss=%f, lr=%f, used_time=%f sec.', epoch, data_iteration, epoch_batches, loss.item(), lr, time.time() - start_time)

            if (i + 1) % rolling_loss_step == 0:
                logger.info('Train batch %d', data_iteration)
                latest_rolling_train_av_loss = rolling_train_loss / rolling_loss_step
                logger.info('Avg. loss per last %d batches: %f', rolling_loss_step, latest_rolling_train_av_loss)
                rolling_train_loss = 0.0

            # if data_iteration % eval_step == 0:
            #     logger.info('Validation: Epoch: %d Step: %d/%d', epoch, data_iteration, epoch_batches)
            #     self.validate_and_save(epoch, train_data_iterator.get_iteration(), scheduler)
            #     self.biencoder.train()


        self.validate_and_save(epoch, data_iteration, scheduler)

        epoch_loss = (epoch_loss / epoch_batches) if epoch_batches > 0 else 0
        logger.info('Av Loss per epoch=%f', epoch_loss)
        logger.info('epoch total correct predictions=%d', epoch_correct_predictions)

    def _save_checkpoint(self, scheduler, epoch: int, offset: int) -> str:
        args = self.args
        model_to_save = get_model_obj(self.biencoder)
        cp = os.path.join(args.output_dir,
                          args.checkpoint_file_name + '.' + str(epoch) + ('.' + str(offset) if offset > 0 else ''))

        meta_params = get_encoder_params_state(args)

        if self.prefix or self.prompt:
            model_dict = {key: value for (key, value) in model_to_save.state_dict().items() if "prefix_encoder" in key}
        elif self.adapter:
            model_to_save.question_model.save_adapter(cp+".q", "adapter")
            model_to_save.ctx_model.save_adapter(cp+".ctx", "adapter")
            return cp 
        else:
            model_dict = model_to_save.state_dict()

        state = CheckpointState(model_dict,
                                self.optimizer.state_dict(),
                                scheduler.state_dict(),
                                offset,
                                epoch, meta_params
                                )

        torch.save(state._asdict(), cp)
        logger.info('Saved checkpoint at %s', cp)
        return cp

    def _load_saved_state(self, saved_state: CheckpointState):
        epoch = saved_state.epoch
        offset = saved_state.offset
        if offset == 0:  # epoch has been completed
            epoch += 1
        logger.info('Loading checkpoint @ batch=%s and epoch=%s', offset, epoch)

        self.start_epoch = epoch
        self.start_batch = offset

        model_to_load = get_model_obj(self.biencoder)
        logger.info('Loading saved model state ...')
        if self.adapter:
            adapter_name = model_to_load.question_model.load_adapter(self.args.model_file+".q")
            model_to_load.question_model.set_active_adapters(adapter_name)
            return
        else:
            model_to_load.load_state_dict(saved_state.model_dict, strict=not self.prefix and not self.prompt) 
        
        if saved_state.optimizer_dict:
            logger.info('Loading saved optimizer state ...')
            self.optimizer.load_state_dict(saved_state.optimizer_dict)
            
        if saved_state.scheduler_dict:
            self.scheduler_state = saved_state.scheduler_dict

    def _load_pretrained_state(self, saved_state: CheckpointState):

        model_to_load = get_model_obj(self.biencoder)
        logger.info('Loading saved model state ...')
        if self.prefix:
            model_to_load.load_state_dict(saved_state.model_dict, strict=False)
        else:
            model_to_load.load_state_dict(saved_state.model_dict)  # set strict=False if you use extra projection

    

def _calc_loss(args, loss_function, local_q_vector, local_ctx_vectors, local_positive_idxs,
               local_hard_negatives_idxs: list = None,
               ) -> Tuple[T, bool]:
    """
    Calculates In-batch negatives schema loss and supports to run it in DDP mode by exchanging the representations
    across all the nodes.
    """
    distributed_world_size = args.distributed_world_size or 1
    if distributed_world_size > 1:
        q_vector_to_send = torch.empty_like(local_q_vector).cpu().copy_(local_q_vector).detach_()
        ctx_vector_to_send = torch.empty_like(local_ctx_vectors).cpu().copy_(local_ctx_vectors).detach_()

        global_question_ctx_vectors = all_gather_list(
            [q_vector_to_send, ctx_vector_to_send, local_positive_idxs, local_hard_negatives_idxs],
            max_size=args.global_loss_buf_sz)

        global_q_vector = []
        global_ctxs_vector = []

        # ctxs_per_question = local_ctx_vectors.size(0)
        positive_idx_per_question = []
        hard_negatives_per_question = []

        total_ctxs = 0

        for i, item in enumerate(global_question_ctx_vectors):
            q_vector, ctx_vectors, positive_idx, hard_negatives_idxs = item

            if i != args.local_rank:
                global_q_vector.append(q_vector.to(local_q_vector.device))
                global_ctxs_vector.append(ctx_vectors.to(local_q_vector.device))
                positive_idx_per_question.extend([v + total_ctxs for v in positive_idx])
                hard_negatives_per_question.extend([[v + total_ctxs for v in l] for l in hard_negatives_idxs])
            else:
                global_q_vector.append(local_q_vector)
                global_ctxs_vector.append(local_ctx_vectors)
                positive_idx_per_question.extend([v + total_ctxs for v in local_positive_idxs])
                hard_negatives_per_question.extend([[v + total_ctxs for v in l] for l in local_hard_negatives_idxs])
            total_ctxs += ctx_vectors.size(0)

        global_q_vector = torch.cat(global_q_vector, dim=0)
        global_ctxs_vector = torch.cat(global_ctxs_vector, dim=0)

    else:
        global_q_vector = local_q_vector
        global_ctxs_vector = local_ctx_vectors
        positive_idx_per_question = local_positive_idxs
        hard_negatives_per_question = local_hard_negatives_idxs

    loss, is_correct = loss_function.calc(global_q_vector, global_ctxs_vector, positive_idx_per_question,
                                          hard_negatives_per_question)

    return loss, is_correct


def _do_biencoder_fwd_pass(model: nn.Module, input: BiEncoderBatch, tensorizer: Tensorizer, args) -> Tuple[
        torch.Tensor, int]:
    input = BiEncoderBatch(**move_to_device(input._asdict(), args.device))

    q_attn_mask = tensorizer.get_attn_mask(input.question_ids)
    ctx_attn_mask = tensorizer.get_attn_mask(input.context_ids)

    if model.training:
        model_out = model(input.question_ids, input.question_segments, q_attn_mask, input.context_ids,
                          input.ctx_segments, ctx_attn_mask)
    else:
        with torch.no_grad():
            model_out = model(input.question_ids, input.question_segments, q_attn_mask, input.context_ids,
                              input.ctx_segments, ctx_attn_mask)

    local_q_vector, local_ctx_vectors = model_out

    loss_function = BiEncoderNllLoss()

    loss, is_correct = _calc_loss(args, loss_function, local_q_vector, local_ctx_vectors, input.is_positive,
                                  input.hard_negatives)

    is_correct = is_correct.sum().item()

    if args.n_gpu > 1:
        loss = loss.mean()
    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps

    return loss, is_correct


def main():
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_training_params(parser)
    add_tokenizer_params(parser)
    add_tuning_params(parser)

    # biencoder specific training features
    parser.add_argument("--eval_per_epoch", default=1, type=int,
                        help="How many times it evaluates on dev set per epoch and saves a checkpoint")

    parser.add_argument("--global_loss_buf_sz", type=int, default=150000,
                        help='Buffer size for distributed mode representations al gather operation. \
                                Increase this if you see errors like "encoded data exceeds max_size ..."')

    parser.add_argument("--fix_ctx_encoder", action='store_true')
    parser.add_argument("--shuffle_positive_ctx", action='store_true')

    # input/output src params
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model checkpoints will be written or resumed from")

    # data handling parameters
    parser.add_argument("--hard_negatives", default=1, type=int,
                        help="amount of hard negative ctx per question")
    parser.add_argument("--other_negatives", default=0, type=int,
                        help="amount of 'other' negative ctx per question")
    parser.add_argument("--train_files_upsample_rates", type=str,
                        help="list of up-sample rates per each train file. Example: [1,2,1]")

    # parameters for Av.rank validation method
    parser.add_argument("--val_av_rank_start_epoch", type=int, default=10000,
                        help="Av.rank validation: the epoch from which to enable this validation")
    parser.add_argument("--val_av_rank_hard_neg", type=int, default=30,
                        help="Av.rank validation: how many hard negatives to take from each question pool")
    parser.add_argument("--val_av_rank_other_neg", type=int, default=30,
                        help="Av.rank validation: how many 'other' negatives to take from each question pool")
    parser.add_argument("--val_av_rank_bsz", type=int, default=128,
                        help="Av.rank validation: batch size to process passages")
    parser.add_argument("--val_av_rank_max_qs", type=int, default=10000,
                        help="Av.rank validation: max num of questions")
    parser.add_argument('--checkpoint_file_name', type=str, default='dpr_biencoder', help="Checkpoints file prefix")

    parser.add_argument("--warmup_percentage", default=None, type=float)
    args = parser.parse_args()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    setup_args_gpu(args)
    set_seed(args)
    print_args(args)

    trainer = BiEncoderTrainer(args)

    if args.train_file is not None:
        trainer.run_train()
    elif args.model_file and args.dev_file:
        logger.info("No train files are specified. Run 2 types of validation for specified model file")
        trainer.validate_nll()
        trainer.validate_average_rank()
    else:
        logger.warning("Neither train_file or (model_file & dev_file) parameters are specified. Nothing to do.")


if __name__ == "__main__":
    main()
