# from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
# from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
from typing import Union, List, Dict, Tuple
from tqdm.autonotebook import trange
import torch

import sys
from pathlib import Path
BASE_DIR = Path.resolve(Path(__file__)).parent.parent
sys.path.append(str(BASE_DIR))
print(BASE_DIR)


from dpr.models import init_encoder_components
from dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint,move_to_device


class DPRForBeir():
    def __init__(self, args, **kwargs):
        if not args.adapter:
            saved_state = load_states_from_checkpoint(args.model_file)
            set_encoder_params_from_state(saved_state.encoder_params, args)

        args.sequence_length = 512
        tensorizer, encoder, _ = init_encoder_components("dpr", args, inference_only=True)
        encoder, _ = setup_for_distributed_mode(encoder, None, args.device, args.n_gpu,
                                                args.local_rank,
                                                args.fp16,
                                                args.fp16_opt_level)

        self.device = args.device
        
        self.tensorizer = tensorizer

        model_to_load = get_model_obj(encoder).question_model
        model_to_load.eval()
        if args.adapter:
            adapter_name = model_to_load.load_adapter(args.model_file+".q")
            model_to_load.set_active_adapters(adapter_name)
        else:
            encoder_name = "question_model."
            prefix_len = len(encoder_name)
            if args.prefix or args.prompt:
                encoder_name += "prefix_encoder."
            q_state = {key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if
                        key.startswith(encoder_name)}
            model_to_load.load_state_dict(q_state, strict=not args.prefix and not args.prompt)
        self.q_model = model_to_load
        

        model_to_load = get_model_obj(encoder).ctx_model
        model_to_load.eval()
        if args.adapter:
            adapter_name = model_to_load.load_adapter(args.model_file+".ctx")
            model_to_load.set_active_adapters(adapter_name)
        else:
            encoder_name = "ctx_model."
            prefix_len = len(encoder_name)
            if args.prefix or args.prompt:
                encoder_name += "prefix_encoder."
            ctx_state = {key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if
                        key.startswith(encoder_name)}
            model_to_load.load_state_dict(ctx_state, strict=not args.prefix and not args.prompt)
        self.ctx_model = model_to_load
    
    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> torch.Tensor:
        query_embeddings = []
        with torch.no_grad():
            for start_idx in trange(0, len(queries), batch_size):
                batch_token_tensors = [self.tensorizer.text_to_tensor(q) for q in
                                       queries[start_idx:start_idx + batch_size]]
                q_ids_batch = move_to_device(torch.stack(batch_token_tensors, dim=0), self.device)
                q_seg_batch = move_to_device(torch.zeros_like(q_ids_batch), self.device)
                q_attn_mask = move_to_device(self.tensorizer.get_attn_mask(q_ids_batch), self.device)
                _, out, _ = self.q_model(q_ids_batch, q_seg_batch, q_attn_mask)

                query_embeddings.extend(out.cpu())

        result = torch.stack(query_embeddings)
        return result
        
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs) -> torch.Tensor:
        
        corpus_embeddings = []
        with torch.no_grad():
            for start_idx in trange(0, len(corpus), batch_size):
                batch_token_tensors = [self.tensorizer.text_to_tensor(ctx["text"], title=ctx["title"]) for ctx in
                                       corpus[start_idx:start_idx + batch_size]]

                ctx_ids_batch = move_to_device(torch.stack(batch_token_tensors, dim=0), self.device)
                ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch), self.device)
                ctx_attn_mask = move_to_device(self.tensorizer.get_attn_mask(ctx_ids_batch), self.device)
                _, out, _ = self.ctx_model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)                
                out = out.cpu()
                corpus_embeddings.extend([out[i].view(-1) for i in range(out.size(0))])  

        
        result = torch.stack(corpus_embeddings)
        return result