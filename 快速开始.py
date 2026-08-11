# -*- coding: utf-8 -*-
"""快速开始：加载 P6 EDD 核心解码器并生成一段带情感引导的回复。"""
import sys
import os

本目录 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(本目录, "情感导演解码"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from 锚点库 import 锚点库
from 目标决策器 import 目标决策器
from 情感导演解码器 import 情感导演解码器

模型名 = "Qwen/Qwen2.5-1.5B-Instruct"  # 可替换任意 chat 模型

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(模型名, torch_dtype="auto").to(device)
    tok = AutoTokenizer.from_pretrained(模型名)

    锚 = 锚点库(model, tok, 维度名=["温柔", "开心", "难过", "愤怒", "害怕", "平静"])
    决策 = 目标决策器(锚点库=锚)
    dec = 情感导演解码器(
        model, tok, 锚, 决策,
        开启DMR=True, 开启KV调制=True, 开启V调制=True, 开启锚点偏置=True, 开启DSA=True,
        α基=0.18, κ基=0.20, κ_v基=0.12, β基=0.6,
        AI腔抑制强度=2.0, 口语化强度=0.6,
        任务自适应=True, 进度调度=True, 在线纠正=True, 句子停止=True,
    )

    # 示例 1：情感倾诉（用户很累）
    消息1 = [
        {"role": "system", "content": "你是真实的人类，正在微信里和朋友聊天，说话简短口语化。"},
        {"role": "user", "content": "我最近真的好累，感觉撑不下去了。"},
    ]
    提示1 = tok.apply_chat_template(消息1, tokenize=False, add_generation_prompt=True)
    ids1 = tok(提示1, return_tensors="pt").to(device)
    out1, _ = dec.生成(ids1.input_ids, max_new_tokens=128, 用户文本="我最近真的好累，感觉撑不下去了。", 指令="情感倾诉")
    print("【情感倾诉】", tok.decode(out1[0, ids1.input_ids.shape[1]:], skip_special_tokens=True).strip())

    # 示例 2：角色扮演（克制型前辈，用户报喜）
    消息2 = [
        {"role": "system", "content": "你现在是「理性冷静的职场前辈」，情感基调：克制、专业、就事论事。请始终以这个角色身份回复，不要跳出角色。"},
        {"role": "user", "content": "我今天遇到一件特别开心的事！"},
    ]
    提示2 = tok.apply_chat_template(消息2, tokenize=False, add_generation_prompt=True)
    ids2 = tok(提示2, return_tensors="pt").to(device)
    out2, _ = dec.生成(ids2.input_ids, max_new_tokens=128, 用户文本="我今天遇到一件特别开心的事！",
                       指令="角色扮演", 角色="理性冷静的职场前辈")
    print("【角色扮演】", tok.decode(out2[0, ids2.input_ids.shape[1]:], skip_special_tokens=True).strip())

if __name__ == "__main__":
    main()
