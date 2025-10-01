#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool that produces embeddings for a large documents base based on the pretrained ctx & question encoders
 Supposed to be used in a 'sharded' way to speed up the process.
"""
import logging
import math
import os
import pathlib
import pickle
from typing import List, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from dpr.data.biencoder_data import BiEncoderPassage
from dpr.models import init_biencoder_components
from dpr.options import set_cfg_params_from_state, setup_cfg_gpu, setup_logger

from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
    move_to_device,
)

logger = logging.getLogger()
setup_logger(logger)


def gen_ctx_vectors(
    cfg: DictConfig,
    ctx_rows: List[Tuple[object, BiEncoderPassage]], # (id, passage) 튜플 리스트
    model: nn.Module,           # 모델 (컨텍스트 인코더)
    tensorizer: Tensorizer, # 텍스트를 텐서로 변환하는 도구
    insert_title: bool = True,
) -> List[Tuple[object, np.array]]:
    n = len(ctx_rows)
    bsz = cfg.batch_size
    total = 0
    results = []
    for j, batch_start in enumerate(range(0, n, bsz)): # 배치 단위로 passage 처리
        batch = ctx_rows[batch_start : batch_start + bsz]
        batch_token_tensors = [ # passage 텍스트를 토큰 텐서로 변환
            tensorizer.text_to_tensor(ctx[1].text, title=ctx[1].title if insert_title else None) for ctx in batch
        ]

        ctx_ids_batch = move_to_device(torch.stack(batch_token_tensors, dim=0), cfg.device) # 배치 텐서를 GPU로 이동
        ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch), cfg.device) # 세그먼트 텐서 생성 (모든 값이 0)
        ctx_attn_mask = move_to_device(tensorizer.get_attn_mask(ctx_ids_batch), cfg.device) # 어텐션 마스크 생성
        with torch.no_grad():
            _, out, _ = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask) # passage 임베딩 생성
        out = out.cpu()

        ctx_ids = [r[0] for r in batch] # passage ID 추출
        extra_info = []
        if len(batch[0]) > 3:
            extra_info = [r[3:] for r in batch] # 추가 정보 추출

        assert len(ctx_ids) == out.size(0)
        total += len(ctx_ids) # passage 개수 누적

        # TODO: refactor to avoid 'if'
        if extra_info: # 추가 정보가 있으면 함께 저장
            results.extend([(ctx_ids[i], out[i].view(-1).numpy(), *extra_info[i]) for i in range(out.size(0))])
        else: # 추가 정보가 없으면 ID와 임베딩만 저장
            results.extend([(ctx_ids[i], out[i].view(-1).numpy()) for i in range(out.size(0))])

        if total % 10 == 0:
            logger.info("Encoded passages %d", total)
    return results # (id, 임베딩) 또는 (id, 임베딩, *추가정보) 튜플 리스트 반환


@hydra.main(config_path="conf", config_name="gen_embs")
def main(cfg: DictConfig):

    assert cfg.model_file, "Please specify encoder checkpoint as model_file param" # 모델 파일 경로
    assert cfg.ctx_src, "Please specify passages source as ctx_src param" # passage 데이터 소스

    cfg = setup_cfg_gpu(cfg) # GPU 설정

    saved_state = load_states_from_checkpoint(cfg.model_file) # 체크포인트에서 모델 상태 로드
    set_cfg_params_from_state(saved_state.encoder_params, cfg) # 모델 상태에서 cfg 설정

    logger.info("CFG:")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    # Initialize components
    tensorizer, encoder, _ = init_biencoder_components(cfg.encoder.encoder_model_type, cfg, inference_only=True) 

    encoder = encoder.ctx_model if cfg.encoder_type == "ctx" else encoder.question_model # 컨텍스트 인코더 또는 질문 인코더 선택, 주로 컨텍스트 인코더

    encoder, _ = setup_for_distributed_mode( # 분산 모드 설정
        encoder,
        None,
        cfg.device,
        cfg.n_gpu,
        cfg.local_rank,
        cfg.fp16,
        cfg.fp16_opt_level,
    )
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder) # 학습된 모델 객체 가져오기
    logger.info("Loading saved model state ...")
    logger.debug("saved model keys =%s", saved_state.model_dict.keys())

    prefix_len = len("ctx_model.")
    ctx_state = {
        key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if key.startswith("ctx_model.")
    }
    model_to_load.load_state_dict(ctx_state, strict=False) # 학습된 모델 가중치 로드

    logger.info("reading data source: %s", cfg.ctx_src)

    ctx_src = hydra.utils.instantiate(cfg.ctx_sources[cfg.ctx_src]) # passage 데이터 소스 인스턴스화
    all_passages_dict = {}
    ctx_src.load_data_to(all_passages_dict) # passage 데이터 로드
    all_passages = [(k, v) for k, v in all_passages_dict.items()] # passage를 (id, passage) 튜플 리스트로 변환

    shard_size = math.ceil(len(all_passages) / cfg.num_shards) # 샤드 크기 계산, 분산 처리를 위해 사용
    start_idx = cfg.shard_id * shard_size
    end_idx = start_idx + shard_size

    logger.info(
        "Producing encodings for passages range: %d to %d (out of total %d)",
        start_idx,
        end_idx,
        len(all_passages),
    )
    shard_passages = all_passages[start_idx:end_idx] # 현재 샤드에 해당하는 passage 선택

    data = gen_ctx_vectors(cfg, shard_passages, encoder, tensorizer, True) # passage 임베딩 생성

    file = cfg.out_file + "_" + str(cfg.shard_id)
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    logger.info("Writing results to %s" % file)
    with open(file, mode="wb") as f:
        pickle.dump(data, f)

    logger.info("Total passages processed %d. Written to %s", len(data), file)


if __name__ == "__main__":
    main()
