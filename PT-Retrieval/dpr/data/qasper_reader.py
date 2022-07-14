import argparse
import collections
import glob
import json
import logging
import os
from enum import Enum
from collections import defaultdict
from typing import List, Dict, Tuple, Any

import numpy as np
import torch
from torch import Tensor as T

from transformers import AutoTokenizer

logger = logging.getLogger()

ReaderBatch = collections.namedtuple('ReaderBatch', ['input_ids', 'attention_mask'])

class AnswerType(Enum):
    EXTRACTIVE = 1
    ABSTRACTIVE = 2
    BOOLEAN = 3
    NONE = 4

class QasperReader():
    def __init__(self,
                 tokenizer: AutoTokenizer,
                 context: str = "full_text",
                 max_query_length: int = 128,
                 max_document_length: int = 16384,
                 paragraph_separator: bool = False,
                 is_train: bool = True):

        self._context = context
        self._tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_document_length = max_document_length
        self._paragraph_separator = paragraph_separator
        self.is_train = is_train
            
    def parser_data(self, path: str):
        with open(path, 'r', encoding="utf-8") as f:
            logger.info('Reading file %s' % path)
            data = json.load(f)
            
            ret = []
            for article_id, article in data.items():
                article['article_id'] = article_id
                d = self._article_to_dict(article)
                ret.extend(d)
            logger.info('Aggregated data size: {}'.format(len(ret)))
        return ret

    def _article_to_dict(self, article: Dict[str, Any]):
        paragraphs = self._get_paragraphs_from_article(article)
        tokenized_context = None
        paragraph_start_indices = None

        if not self._context == 'question_and_evidence':
            tokenized_context, paragraph_start_indices = self._tokenize_paragraphs(paragraphs)

        data_dicts = []
        for question_answer in article['qas']:

            all_answers = []
            all_evidence = []
            all_evidence_masks = []
            for answer_annotation in question_answer['answers']:
                answer, evidence, answer_type = self._extract_answer_and_evidence(answer_annotation['answer'])

                all_answers.append({'text': answer, 'type': answer_type}) 
                all_evidence.append(evidence)
                all_evidence_masks.append(self._get_evidence_mask(evidence, paragraphs))

                tokenized_question = self._tokenizer.encode(question_answer['question'])

            metadata = {
                'question': question_answer['question'],
                'all_answers': all_answers
            }
            if self.is_train:
                for answer, evidence_mask in zip(all_answers, all_evidence_masks):
                    data_dicts.append({
                        'tokenized_question': tokenized_question,
                        'tokenized_context': tokenized_context,
                        'paragraph_start_indices': paragraph_start_indices,
                        'tokenized_answer': self._tokenizer.encode(answer['text']),
                        'evidence_mask': None, #torch.tensor(evidence_mask),
                        'metadata': metadata
                    })
            else:
                data_dicts.append({
                    'tokenized_question': tokenized_question,
                    'tokenized_context': tokenized_context,
                    'paragraph_start_indices': paragraph_start_indices,
                    'tokenized_answer': self._tokenizer.encode(all_answers[0]['text']),
                    'evidence_mask': None, #torch.tensor(evidence_mask),
                    'metadata': metadata
                })

        return data_dicts
            
    @staticmethod
    def _get_evidence_mask(evidence: List[str], paragraphs: List[str]) -> List[int]:
        """
        Takes a list of evidence snippets, and the list of all the paragraphs from the
        paper, and returns a list of indices of the paragraphs that contain the evidence.
        """
        evidence_mask = []
        for paragraph in paragraphs:
            for evidence_str in evidence:
                if evidence_str in paragraph:
                    evidence_mask.append(1)
                    break
            else:
                evidence_mask.append(0)
        return evidence_mask

    def _extract_answer_and_evidence(self, answer: List[Dict]) -> Tuple[str, List[str]]:
        evidence_spans = [x.replace("\n", " ").strip() for x in answer["evidence"]]
        evidence_spans = [x for x in evidence_spans if x != ""]

        answer_string = None
        answer_type = None

        if answer.get("unanswerable", False):
            answer_string = "Unanswerable"
            answer_type = AnswerType.NONE

        elif answer.get("yes_no") is not None:
            answer_string = "Yes" if answer["yes_no"] else "No"
            answer_type = AnswerType.BOOLEAN

        elif answer.get("extractive_spans", []):

            answer_string = ", ".join(answer["extractive_spans"])
            answer_type = AnswerType.EXTRACTIVE

        else:
            answer_string = answer.get("free_form_answer", "")
            answer_type = AnswerType.ABSTRACTIVE

        return answer_string, evidence_spans, answer_type

    def _tokenize_paragraphs(self, paragraphs: List[str]) -> Tuple[List[int], List[int]]:
        tokenized_context = []
        paragraph_start_indices = []
        for paragraph in paragraphs:
            tokenized_paragraph = self._tokenizer.encode(paragraph)
            paragraph_start_indices.append(len(tokenized_context))
            # if self._paragraph_separator:
            #     tokenized_context.append(self._tokenizer.sep_token_id)
            tokenized_context.extend(tokenized_paragraph)
            if self._paragraph_separator:
                tokenized_context.append(self._tokenizer.sep_token_id)
        if self._paragraph_separator:
            # We added the separator after every paragraph, so we remove it after the last one.
            tokenized_context = tokenized_context[:-1]
        return tokenized_context, paragraph_start_indices

    def _get_paragraphs_from_article(self, article: Dict) -> List[str]:
        # if self._context == "question_only":
        #     return []
        # if self._context == "question_and_abstract":
        #     return [article["abstract"]]
        full_text = article["full_text"]
        paragraphs = []
        for section_info in full_text:
            if section_info["section_name"] is not None:
                paragraphs.append(section_info["section_name"])
            for paragraph in section_info["paragraphs"]:
                paragraph_text = paragraph.replace("\n", " ").strip()
                if paragraph_text:
                    paragraphs.append(paragraph_text)
            # if self._context == "question_and_introduction":
            #     # Assuming the first section is the introduction and stopping here.
            #     break
        return paragraphs

    # def _get_sections_paragraphs_from_article(self, article: Dict) -> List[List[str]]:
    #     full_text = article["full_text"]
    #     sections = []
    #     for section_info in full_text:




    # def _create_question_answer_tensors(self,
    #                                     question: str,
    #                                     paragraphs: List[str],
    #                                     tokenized_context: List[int],
    #                                     paragraph_start_indices: List[int] = None,
    #                                     evidence_mask: List[int] = None,
    #                                     answer: str = None,
    #                                     evidence: List[str] = None):
        
    #     tokenized_question = self._tokenizer.encode(question)
    #     if len(tokenized_question) > self.max_query_length:
    #         tokenized_question = tokenized_question[:self.max_document_length]
        
    #     allow_context_length = self.max_document_length - len(tokenized_question) - 1 - 1

    #     if len(tokenized_context) > allow_context_length:
    #         tokenized_context = tokenized_context[:allow_context_length]
    #         paragraph_start_indices = [index for index in paragraph_start_indices
    #                                    if index <= allow_context_length]
    #         if evidence_mask is not None:
    #             num_paragraphs = len(paragraph_start_indices)
    #             evidence_mask = evidence_mask[:num_paragraphs]

    #     return {
    #         'question': tokenized_question,
    #         'context': tokenized_context,
    #         'paragraph_start_indices': paragraph_start_indices,
    #         'evidence_mask': evidence_mask,
    #         'answer': answer,
    #         'evidence': evidence
    #     }

        # question_and_context = ([self._tokenizer.cls_token_id]
        #                         + tokenized_question
        #                         + [self._tokenizer.sep_token_id]
        #                         + tokenized_context)

        # start_of_context = 1 + len(tokenized_question)
        # paragraph_indices_list = [x + start_of_context for x in paragraph_start_indices]
        # mask_indices = set(list(range(start_of_context)) + paragraph_indices_list)
        # global_attention_mask = [
        #     True if i in mask_indices else False for i in range(len(question_and_context))
        # ]
        # print('question:', tokenized_question)
        # print('context', tokenized_context)
        # print('cls', self._tokenizer.cls_token_id, 'sep', self._tokenizer.sep_token_id)
        # print('input_id', question_and_context)
        # print(len(tokenized_question), len(tokenized_context), len(question_and_context), len(global_attention_mask))
        # return torch.tensor(question_and_context), torch.tensor(global_attention_mask)
