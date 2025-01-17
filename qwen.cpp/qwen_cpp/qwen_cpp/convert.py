"""
Convert Hugging Face Qwen models to GGML format
"""
import argparse
import platform
import struct
import sys
from enum import Enum
from pathlib import Path
from typing import BinaryIO

import torch
from tabulate import tabulate
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


GGML_QK8_0 = 32
GGML_QK4_0 = 32
GGML_QK4_1 = 32
GGML_QK5_0 = 32
GGML_QK5_1 = 32

GGML_MEM_ALIGN = 16

if platform.system() == "Darwin":
    # cpm_kernels doesn't support macOS but transformers will check missing packages, so mock it
    sys.modules["cpm_kernels"] = object()


class GGMLType(Enum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8


def quantize_q8_0(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ggml_quantize_q8_0 in ggml.c
    assert tensor.shape[1] % GGML_QK8_0 == 0
    tensor = tensor.view(-1, GGML_QK8_0)
    scale = tensor.abs().max(dim=-1, keepdim=True).values / ((1 << 7) - 1)
    tensor = (tensor / scale).round().clamp(min=-128, max=127).char()
    # add scale into each block
    tensor = torch.cat((scale.half().view(torch.int8), tensor), dim=-1)
    return tensor


def quantize_q4_0(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ggml_quantize_q4_0 in ggml.c
    assert tensor.shape[1] % GGML_QK4_0 == 0
    tensor = tensor.view(-1, GGML_QK4_0)
    abs_max_indices = tensor.abs().max(dim=-1, keepdim=True).indices
    max_values = torch.take_along_dim(tensor, abs_max_indices, dim=-1)
    scale = max_values / -8
    tensor = (tensor / scale + 8).round().clamp(min=0, max=15).char()
    # compress two int4 weights into an int8
    tensor = tensor[:, :16] | (tensor[:, 16:] << 4)
    # add scale into each block
    tensor = torch.cat((scale.half().view(torch.int8), tensor), dim=-1)
    return tensor


def quantize_q4_1(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ggml_quantize_q4_1 in ggml.c
    assert tensor.shape[1] % GGML_QK4_1 == 0
    tensor = tensor.view(-1, GGML_QK4_1)
    min_vals = tensor.min(dim=-1, keepdim=True).values
    max_vals = tensor.max(dim=-1, keepdim=True).values
    scale = (max_vals - min_vals) / ((1 << 4) - 1)
    tensor = ((tensor - min_vals) / scale).round().clamp(min=0, max=15).char()
    # compress two int4 weights into an int8
    tensor = tensor[:, :16] | (tensor[:, 16:] << 4)
    # add scale & min into each block
    tensor = torch.cat((scale.half().view(torch.int8), min_vals.half().view(torch.int8), tensor), dim=-1)
    return tensor


def quantize_q5_0(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ggml_quantize_q5_0 in ggml.c
    assert tensor.shape[1] % GGML_QK5_0 == 0
    tensor = tensor.view(-1, GGML_QK5_0)
    abs_max_indices = tensor.abs().max(dim=-1, keepdim=True).indices
    max_values = torch.take_along_dim(tensor, abs_max_indices, dim=-1)
    scale = max_values / -16
    tensor = (tensor / scale + 16).round().clamp(min=0, max=31).char()
    qs = (tensor[:, :16] & 0x0F) | (tensor[: 16:] << 4)
    qh = torch.zeros(tensor.shape[:-1], dtype=torch.int32)
    for i in range(32):
        qh |= ((tensor[:, i] & 0x10) >> 4).int() << i

    # add scale into each block
    tensor = torch.cat((scale.half().view(torch.int8), qh[..., None].view(torch.int8), qs), dim=-1)
    return tensor


def quantize_q5_1(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ggml_quantize_q5_1 in ggml.c
    assert tensor.shape[1] % GGML_QK5_1 == 0
    tensor = tensor.view(-1, GGML_QK5_1)
    min_vals = tensor.min(dim=-1, keepdim=True).values
    max_vals = tensor.max(dim=-1, keepdim=True).values
    scale = (max_vals - min_vals) / ((1 << 5) - 1)
    tensor = ((tensor - min_vals) / scale).round().clamp(min=0, max=31).char()
    qs = (tensor[:, :16] & 0x0F) | (tensor[:, 16:] << 4)
    qh = torch.zeros(tensor.shape[:-1], dtype=torch.int32)
    for i in range(32):
        qh |= ((tensor[:, i] & 0x10) >> 4).int() << i

    # add scale & min into each block
    tensor = torch.cat(
        (scale.half().view(torch.int8), min_vals.half().view(torch.int8), qh[..., None].view(torch.int8), qs), dim=-1
    )
    return tensor


def dump_tensor(f, name: str, tensor: torch.Tensor, ggml_type: GGMLType):
    assert tensor.dtype == torch.float32

    # tensor name
    f.write(struct.pack("i", len(name.encode())))
    f.write(name.encode())

    # tensor shape & dtype
    f.write(struct.pack("i" * (2 + tensor.ndim), tensor.ndim, *tensor.shape, ggml_type.value))

    # tensor data
    if ggml_type == GGMLType.F32:
        tensor = tensor.float()
    elif ggml_type == GGMLType.F16:
        tensor = tensor.half()
    elif ggml_type == GGMLType.Q8_0:
        tensor = quantize_q8_0(tensor)
    elif ggml_type == GGMLType.Q4_0:
        tensor = quantize_q4_0(tensor)
    elif ggml_type == GGMLType.Q4_1:
        tensor = quantize_q4_1(tensor)
    elif ggml_type == GGMLType.Q5_0:
        tensor = quantize_q5_0(tensor)
    elif ggml_type == GGMLType.Q5_1:
        tensor = quantize_q5_1(tensor)
    else:
        raise NotImplementedError(f"Cannot dump tensor of dtype {tensor.dtype}")

    # align address
    aligned_pos = (f.tell() + (GGML_MEM_ALIGN - 1)) // GGML_MEM_ALIGN * GGML_MEM_ALIGN
    f.seek(aligned_pos)
    tensor.numpy().tofile(f)


def dump_state_dict(f, weight_names, state_dict, ggml_type):
    tensor_info = []
    for name in tqdm(weight_names, desc="Processing model states"):
        tensor = state_dict[name]
        if tensor.ndim == 2:
            # 2d weight: should quantize it if needed

            # step 1: de-quantize it back to float32
            tensor = tensor.float()

            # step 2: quantize it into ggml format
            tensor_ggml_type = ggml_type
        else:
            # 1d weight: convert it to float32
            assert tensor.ndim == 1
            tensor = tensor.float()
            tensor_ggml_type = GGMLType.F32

        dump_tensor(f, name, tensor, tensor_ggml_type)
        tensor_info.append((name, tensor.shape, tensor_ggml_type.name))

    print(tabulate(tensor_info, headers=["name", "shape", "dtype"], tablefmt="psql"))


class Qwen2Converter:
    @classmethod
    def convert(cls, f, model, tokenizer, ggml_type):
        f.write(b"ggml")  # magic
        cls.dump_config(f, model.config, model.generation_config, tokenizer, ggml_type)
        cls.dump_model(f, model, ggml_type)

# Qwen2 has different config
# got from manually checking the config and tokenizer 
# 
# compare  
#     https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/tokenizer_config.json
#     https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/config.json
#     https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/generation_config.json
# with
#     https://huggingface.co/Qwen/Qwen2-0.5B/blob/main/tokenizer_config.json
#     https://huggingface.co/Qwen/Qwen2-0.5B/blob/main/config.json
#     https://huggingface.co/Qwen/Qwen2-0.5B/blob/main/generation_config.json
#
# and 
#
# >>> from transformers import AutoTokenizer, AutoModelForCausalLM
# >>> model_name_or_path = "Qwen/Qwen2-0.5B"
# >>> tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# >>> tokenizer
# Qwen2TokenizerFast(name_or_path='Qwen/Qwen2-0.5B', vocab_size=151643, model_max_length=32768, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'eos_token': '<|endoftext|>', 'pad_token': '<|endoftext|>', 'additional_special_tokens': ['<|im_start|>', '<|im_end|>']}, clean_up_tokenization_spaces=False),  added_tokens_decoder={
# 	151643: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
# 	151644: AddedToken("<|im_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
# 	151645: AddedToken("<|im_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
# }
# }
    @staticmethod
    # https://huggingface.co/Qwen/Qwen2-0.5B/blob/main/config.json
    # https://huggingface.co/Qwen/Qwen2-0.5B/blob/main/tokenizer_config.json
    def dump_config(f, config, generation_config, tokenizer, ggml_type):
        config_values = [
            ggml_type.value,
            config.vocab_size,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,# config.num_attention_heads,
            config.num_hidden_layers,
            config.intermediate_size,
            tokenizer.model_max_length, #config.seq_length,
            config.eos_token_id, #generation_config.eos_token_id,
            list(tokenizer.added_tokens_decoder.keys())[0], # generation_config.pad_token_id,
            list(tokenizer.added_tokens_decoder.keys())[1], # tokenizer.im_start_id,
            list(tokenizer.added_tokens_decoder.keys())[2], # tokenizer.im_end_id,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

# Qwen2 has different weight names, we can read them by
#
# >>> from transformers import AutoTokenizer, AutoModelForCausalLM
# >>> model_name_or_path = "Qwen/Qwen2-0.5B"
# >>> model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
# >>> print(model.state_dict().keys())
#
# or from the model card of Qwen2-7B-Instruct:
# https://huggingface.co/Qwen/Qwen2-0.5B?show_file_info=model.safetensors

    @staticmethod
    def dump_model(f, model, ggml_type):
        weight_names = ["model.embed_tokens.weight"] # ["transformer.wte.weight"]
        for i in range(model.config.num_hidden_layers):
            weight_names += [
                # f"transformer.h.{i}.ln_1.weight",
                f"model.layers.{i}.input_layernorm.weight", # first layer norm
                # f"transformer.h.{i}.attn.c_attn.weight",
                # f"transformer.h.{i}.attn.c_attn.bias",
                # f"transformer.h.{i}.attn.c_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.bias",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.k_proj.bias",
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.bias",
                f"model.layers.{i}.self_attn.o_proj.weight", # q,k,v are separate in attn
                # f"transformer.h.{i}.ln_2.weight",
                f"model.layers.{i}.post_attention_layernorm.weight", # second layer norm
                # f"transformer.h.{i}.mlp.w1.weight",
                f"model.layers.{i}.mlp.gate_proj.weight", # first mlp
                # f"transformer.h.{i}.mlp.w2.weight",
                f"model.layers.{i}.mlp.up_proj.weight", # second mlp
                # f"transformer.h.{i}.mlp.c_proj.weight",
                f"model.layers.{i}.mlp.down_proj.weight", # final projection
            ]
        weight_names += [
            "model.norm.weight",
            "lm_head.weight",
        ]
        # weight_names += [
        #     "transformer.ln_f.weight",
        #     "lm_head.weight",
        # ]
        dump_state_dict(f, weight_names, model.state_dict(), ggml_type)


def convert(f: BinaryIO, model_name_or_path: str, dtype: str = "q4_0"):
    ggml_type = GGMLType[dtype.upper()]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)

    Qwen2Converter.convert(f, model, tokenizer, ggml_type)


def main():
    parser = argparse.ArgumentParser("qwen-convert")
    parser.add_argument(
        "-i",
        "--model_name_or_path",
        default="Qwen/Qwen2-0.5B",
        type=str,
        help="Model name or path used in AutoModel.from_pretrained",
    )
    parser.add_argument(
        "-o", "--save_path", default="Qwen2-0.5B.bin", type=Path, help="Path to save the generated GGML model"
    )
    parser.add_argument(
        "-t",
        "--type",
        default="q4_0",
        type=str,
        choices=["f32", "f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1"],
        help="GGML model quantization type",
    )
    args = parser.parse_args()

    with open(args.save_path, "wb") as f:
        convert(f, args.model_name_or_path, dtype=args.type)

    print(f"GGML model saved to {args.save_path}")


if __name__ == "__main__":
    main()
