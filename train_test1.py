#!/usr/bin/env python3
import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import List, Dict
import torch
import torchaudio
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from transformers import LlamaModel
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.tokenizers.tokenizer import MTLTokenizer
from chatterbox.models.s3tokenizer import S3Tokenizer
from chatterbox.models.voice_encoder.voice_encoder import VoiceEncoder
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.mtl_tts import Conditionals
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors

try:
    import pyarabic.araby as araby
    PYARABIC_AVAILABLE = True
except ImportError:
    PYARABIC_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_test1")

def sanitize_text(text: str) -> str:
    if PYARABIC_AVAILABLE:
        text = araby.strip_tashkeel(text)
        text = araby.strip_tatweel(text)
    return text

def load_t3_model(ckpt_dir, device):
    class PatchedT3(T3):
        def __init__(self, hp=None):
            super().__init__(hp)
            cfg = self.cfg
            cfg._attn_implementation = "eager"
            self.tfmr = LlamaModel(cfg)

    t3_config = T3Config.multilingual()
    t3 = PatchedT3(t3_config)
    
    if (ckpt_dir / "t3_23lang.safetensors").exists():
        t3_state = load_safetensors(ckpt_dir / "t3_23lang.safetensors")
    if "model" in t3_state.keys():
        t3_state = t3_state["model"][0]
    t3.load_state_dict(t3_state, strict=False)
    t3.to(device)
    return t3

def main():
    # CONFIG
    input_wav = "loRa/ناروتو أبن كوراما.wav"
    input_json = "loRa/ناروتو أبن كوراما_training_data.json"
    output_name = "hussein_test1"
    epochs = 50
    lr = 1e-4 # Aggressive LR for overfitting
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Starting OVERFITTING TEST on {input_wav}")
    
    # 1. Load Data
    with open(input_json, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    
    # Sanitize
    for seg in segments:
        seg["text"] = sanitize_text(seg["text"])
        
    # Load Audio
    full_audio, sr = torchaudio.load(input_wav)
    if full_audio.shape[0] > 1:
        full_audio = torch.mean(full_audio, dim=0, keepdim=True)
    full_audio = full_audio.to(device)
    
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000).to(device)
        full_audio_16k = resampler(full_audio)
    else:
        full_audio_16k = full_audio

    # 2. Setup Model
    model_dir = Path("model_cache/ResembleAI_chatterbox")
    mtl_tokenizer = MTLTokenizer(str(model_dir / "mtl_tokenizer.json"))
    s3_tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
    ve = VoiceEncoder().to(device).eval()
    ve.load_state_dict(torch.load(model_dir / "ve.pt", map_location=device))
    
    t3_model = load_t3_model(model_dir, device)
    
    peft_config = LoraConfig(
        r=32, # High rank for memorization
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Target ALL attention
        lora_dropout=0.0, # No dropout for overfitting
        task_type=None
    )
    t3_model.tfmr = get_peft_model(t3_model.tfmr, peft_config)
    t3_model.train()
    
    optimizer = torch.optim.AdamW(t3_model.tfmr.parameters(), lr=lr)
    
    # 3. Training Loop
    logger.info("Training...")
    
    for epoch in range(epochs):
        total_loss = 0
        steps = 0
        
        # Iterate over ALL segments in the file
        for seg in segments:
            start_sec = seg.get("start", 0)
            end_sec = seg.get("end", start_sec + 5)
            text = seg["text"].strip()
            
            if not text: continue
            
            # Extract audio
            start_sample = int(start_sec * 16000)
            end_sample = int(end_sec * 16000)
            audio_chunk = full_audio_16k[:, start_sample:end_sample]
            
            if audio_chunk.size(1) < 1600: continue # Skip tiny bits
            
            try:
                # Prepare Inputs
                wav_np = audio_chunk.squeeze().cpu().numpy()
                spk_emb = torch.from_numpy(ve.embeds_from_wavs([wav_np], 16000, False)).to(device)
                
                text_ids = mtl_tokenizer.encode(text, language_id="ar")
                text_tokens = torch.tensor([text_ids], dtype=torch.long).to(device)
                sot = t3_model.hp.start_text_token
                eot = t3_model.hp.stop_text_token
                text_tokens = F.pad(text_tokens, (1, 0), value=sot)
                text_tokens = F.pad(text_tokens, (0, 1), value=eot)
                
                speech_tokens, _ = s3_tokenizer(audio_chunk)
                speech_tokens = speech_tokens.to(device)
                
                t3_cond_obj = T3Cond(spk_emb, None, torch.ones(1, 1, 1).to(device))
                
                text_lens = torch.tensor([text_tokens.size(1)], device=device)
                speech_lens = torch.tensor([speech_tokens.size(1)], device=device)

                # Forward
                optimizer.zero_grad()
                out = t3_model.forward(
                    t3_cond=t3_cond_obj,
                    text_tokens=text_tokens,
                    text_token_lens=text_lens,
                    speech_tokens=speech_tokens,
                    speech_token_lens=speech_lens,
                    training=True
                )
                
                # Loss
                mask = torch.arange(speech_tokens.size(1), device=device)[None] >= speech_lens[:, None]
                masked_st = speech_tokens.masked_fill(mask, -100)
                loss = F.cross_entropy(out.speech_logits.permute(0,2,1), masked_st, ignore_index=-100)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1
                
            except Exception as e:
                pass
        
        avg_loss = total_loss / max(steps, 1)
        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            
    # Save
    save_path = Path(f"voices/loras/{output_name}")
    save_path.mkdir(parents=True, exist_ok=True)
    t3_model.tfmr.save_pretrained(save_path)
    logger.info(f"Saved test model to {save_path}")

if __name__ == "__main__":
    main()
