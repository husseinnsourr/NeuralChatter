#!/usr/bin/env python3
"""
Professional LoRA Training Script with Cross-Validation
Features:
- Multi-file support (all audio+transcript pairs in folder)
- Train/Validation split
- K-Fold Cross-Validation (optional)
- Early stopping
- Best model checkpointing
- Learning rate scheduler
"""

import argparse
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torchaudio
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============ Data Loading ============

try:
    import pyarabic.araby as araby
    PYARABIC_AVAILABLE = True
except ImportError:
    PYARABIC_AVAILABLE = False
    logger.warning("PyArabic not found. Text sanitization will be stuck.")

def sanitize_text(text: str) -> str:
    """Strip tashkeel and tatweel for consistent training."""
    if PYARABIC_AVAILABLE:
        text = araby.strip_tashkeel(text)
        text = araby.strip_tatweel(text)
    return text

def parse_txt_transcript(txt_path: Path) -> List[Dict]:
    """Parse TXT format transcript (timestamp + text pairs)."""
    segments = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    
    current_time = None
    for line in lines:
        line = line.strip()
        if not line or line == "SENTIMENTS":
            continue
        
        # Check if line is a timestamp
        time_match = re.match(r'^(\d{2}):(\d{2}):(\d{2})(?:\s*/.*)?$', line)
        if time_match:
            h, m, s = map(int, time_match.groups()[:3])
            current_time = h * 3600 + m * 60 + s
        elif current_time is not None:
            # This is text following a timestamp
            segments.append({
                "start": current_time,
                "text": sanitize_text(line)
            })
            current_time = None
    
    # Set end times based on next segment start
    for i in range(len(segments) - 1):
        segments[i]["end"] = segments[i + 1]["start"]
    
    # Last segment: estimate 10 seconds or based on text length
    if segments:
        segments[-1]["end"] = segments[-1]["start"] + max(5, len(segments[-1]["text"]) / 20)
    
    return segments


def parse_json_transcript(json_path: Path) -> List[Dict]:
    """Parse JSON format transcript."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Sanitize
    for seg in data:
        if "text" in seg:
            seg["text"] = sanitize_text(seg["text"])
    return data


def load_all_data(data_dir: Path) -> List[Dict]:
    """
    Load all audio+transcript pairs from directory.
    Returns list of: {"audio_path": Path, "segments": List[Dict]}
    """
    data_pairs = []
    
    for audio_file in data_dir.iterdir():
        if not audio_file.suffix.lower() in ['.wav', '.mp3']:
            continue
        
        # Look for matching transcript
        json_transcript = data_dir / f"{audio_file.stem}_training_data.json"
        txt_transcript = audio_file.with_suffix('.txt')
        
        segments = None
        if json_transcript.exists():
            try:
                segments = parse_json_transcript(json_transcript)
                logger.info(f"Loaded JSON transcript: {json_transcript.name} ({len(segments)} segments)")
            except Exception as e:
                logger.warning(f"Failed to parse {json_transcript.name}: {e}")
        elif txt_transcript.exists():
            try:
                segments = parse_txt_transcript(txt_transcript)
                logger.info(f"Loaded TXT transcript: {txt_transcript.name} ({len(segments)} segments)")
            except Exception as e:
                logger.warning(f"Failed to parse {txt_transcript.name}: {e}")
        
        if segments and len(segments) > 0:
            data_pairs.append({
                "audio_path": audio_file,
                "segments": segments
            })
    
    logger.info(f"Loaded {len(data_pairs)} audio+transcript pairs")
    return data_pairs


def split_data(data_pairs: List[Dict], val_split: float = 0.2, seed: int = 42) -> Tuple[List, List]:
    """Split data into train and validation sets."""
    random.seed(seed)
    shuffled = data_pairs.copy()
    random.shuffle(shuffled)
    
    val_size = int(len(shuffled) * val_split)
    val_data = shuffled[:val_size]
    train_data = shuffled[val_size:]
    
    logger.info(f"Data split: {len(train_data)} train, {len(val_data)} validation")
    return train_data, val_data


def k_fold_split(data_pairs: List[Dict], k: int = 5, seed: int = 42) -> List[Tuple[List, List]]:
    """Create K-fold splits."""
    random.seed(seed)
    shuffled = data_pairs.copy()
    random.shuffle(shuffled)
    
    fold_size = len(shuffled) // k
    folds = []
    
    for i in range(k):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < k - 1 else len(shuffled)
        
        val_data = shuffled[val_start:val_end]
        train_data = shuffled[:val_start] + shuffled[val_end:]
        folds.append((train_data, val_data))
        logger.info(f"Fold {i+1}: {len(train_data)} train, {len(val_data)} validation")
    
    return folds


# ============ Model Setup ============

def load_t3_model(ckpt_dir, device):
    """Loads the T3 model components."""
    logger.info("Loading T3 Model...")
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    from transformers import LlamaModel
    from safetensors.torch import load_file as load_safetensors

    # Recreate T3 with patched config instructions
    class PatchedT3(T3):
        def __init__(self, hp=None):
            super().__init__(hp)
            cfg = self.cfg
            cfg._attn_implementation = "eager"
            self.tfmr = LlamaModel(cfg)

    t3_config = T3Config.multilingual()
    t3 = PatchedT3(t3_config)
    
    # Try safetensors first, then pt
    if (ckpt_dir / "t3_23lang.safetensors").exists():
        t3_state = load_safetensors(ckpt_dir / "t3_23lang.safetensors")
    else:
        # Fallback or different path 
        # (original script used t3_23lang.safetensors)
        # Check standard checkpoints
        pass

    if "model" in t3_state.keys():
        t3_state = t3_state["model"][0]
    t3.load_state_dict(t3_state, strict=False)
    t3.to(device)
    t3.train()
    return t3

def setup_model(device: str):
    """Load the base model and setup LoRA."""
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from chatterbox.models.tokenizers.tokenizer import MTLTokenizer
    from chatterbox.models.s3tokenizer import S3Tokenizer
    from chatterbox.models.voice_encoder.voice_encoder import VoiceEncoder
    
    # Download model if needed
    model_dir_path = Path("model_cache/ResembleAI_chatterbox")
    if not model_dir_path.exists():
        logger.info("Downloading model from HuggingFace...")
        model_dir = snapshot_download(repo_id="ResembleAI/chatterbox")
        model_dir_path = Path(model_dir)
    
    # Load Tokenizers
    mtl_tokenizer = MTLTokenizer(str(model_dir_path / "mtl_tokenizer.json"))
    s3_tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
    
    # Load Voice Encoder
    ve = VoiceEncoder()
    ve.load_state_dict(torch.load(model_dir_path / "ve.pt", map_location=device))
    ve.to(device).eval()
    
    # Load T3 model using custom loader
    t3_model = load_t3_model(model_dir_path, device)
    
    # Apply LoRA
    logger.info("Setting up LoRA configuration...")
    peft_config = LoraConfig(
        r=16,  # Higher rank for better quality
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],  # Target Attention layers
    )
    t3_model.tfmr = get_peft_model(t3_model.tfmr, peft_config)
    t3_model.tfmr.print_trainable_parameters()
    
    return t3_model, mtl_tokenizer, s3_tokenizer, ve, model_dir_path


# ============ Training Loop ============

def train_epoch(
    t3_model,
    mtl_tokenizer,
    s3_tokenizer,
    ve,
    train_data: List[Dict],
    optimizer,
    device: str,
    steps_per_epoch: int
) -> float:
    """Train for one epoch, return average loss."""
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    from chatterbox.mtl_tts import Conditionals
    import torch.nn.functional as F
    
    t3_model.train()
    total_loss = 0
    step_count = 0
    
    # Flatten all segments
    all_segments = []
    for pair in train_data:
        audio_path = pair["audio_path"]
        for seg in pair["segments"]:
            all_segments.append({
                "audio_path": audio_path,
                **seg
            })
    
    random.shuffle(all_segments)
    
    # Pre-load audio files
    audio_cache = {}
    for pair in train_data:
        path = pair["audio_path"]
        if path not in audio_cache:
            try:
                audio, sr = torchaudio.load(path)
                if audio.shape[0] > 1:
                    audio = torch.mean(audio, dim=0, keepdim=True)
                audio_cache[path] = (audio.to(device), sr)
            except Exception as e:
                logger.warning(f"Error loading {path}: {e}")
                continue
    
    resampler_16k = torchaudio.transforms.Resample(24000, 16000).to(device)
    
    pbar = tqdm(total=min(steps_per_epoch, len(all_segments)), desc="Training")
    
    for seg in all_segments:
        if step_count >= steps_per_epoch:
            break
        
        audio_path = seg["audio_path"]
        if audio_path not in audio_cache:
            continue
            
        start_sec = seg.get("start", 0)
        end_sec = seg.get("end", start_sec + 5)
        text = seg["text"].strip()
        
        if not text or end_sec - start_sec < 1.0 or end_sec - start_sec > 15.0:
            continue
        
        full_audio, sr = audio_cache[audio_path]
        
        # Extract chunk
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        
        if end_sample > full_audio.shape[1]:
            continue
        
        audio_chunk = full_audio[:, start_sample:end_sample]
        
        # Resample for 16k usage
        if sr != 16000:
            temp_resampler = torchaudio.transforms.Resample(sr, 16000).to(device)
            audio_16k = temp_resampler(audio_chunk)
        else:
            audio_16k = audio_chunk
        
        try:
            # 1. Compute Speaker Embedding
            wav_np = audio_16k.squeeze().cpu().numpy()
            spk_emb_np = ve.embeds_from_wavs([wav_np], sample_rate=16000, as_spk=False)
            spk_emb = torch.from_numpy(spk_emb_np).to(device)
            
            # 2. Tokenize Text (assume Arabic for simplicity, or add lang detect later)
            text_ids = mtl_tokenizer.encode(text, language_id="ar")
            text_tokens = torch.tensor([text_ids], dtype=torch.long).to(device)
            
            # Pad with SOT and EOT
            sot = t3_model.hp.start_text_token
            eot = t3_model.hp.stop_text_token
            text_tokens = F.pad(text_tokens, (1, 0), value=sot)
            text_tokens = F.pad(text_tokens, (0, 1), value=eot)
            
            # 3. Tokenize Speech (Targets)
            with torch.no_grad():
                speech_tokens, speech_lens = s3_tokenizer(audio_16k)
                speech_tokens = speech_tokens.to(device)
            
            # 4. Conditioning
            t3_cond_obj = T3Cond(
                speaker_emb=spk_emb,
                cond_prompt_speech_tokens=None,
                emotion_adv=torch.ones(1, 1, 1).to(device)
            )
            t3_cond = Conditionals(t3=t3_cond_obj, gen={})
            
            # Forward pass using T3's forward method
            optimizer.zero_grad()
            
            # Prepare lengths
            text_lens = torch.tensor([text_tokens.size(1)], device=device)
            speech_lens = torch.tensor([speech_tokens.size(1)], device=device)

            out = t3_model.forward(
                t3_cond=t3_cond_obj,
                text_tokens=text_tokens,
                text_token_lens=text_lens,
                speech_tokens=speech_tokens,
                speech_token_lens=speech_lens,
                training=True
            )
            
            # Compute loss
            # cross_entropy expects (B, C, L), so we permute logits from (B, L, C)
            IGNORE_ID = -100
            
            # Create mask for valid speech positions
            mask_speech = torch.arange(speech_tokens.size(1), device=device)[None] >= speech_lens[:, None]
            masked_speech = speech_tokens.masked_fill(mask_speech, IGNORE_ID)
            
            speech_logits_permuted = out.speech_logits.permute(0, 2, 1)
            loss = F.cross_entropy(speech_logits_permuted, masked_speech, ignore_index=IGNORE_ID)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(t3_model.tfmr.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            step_count += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        except Exception as e:
            # logger.warning(f"Error processing segment: {e}")
            continue
    
    pbar.close()
    return total_loss / max(step_count, 1)


def validate(
    t3_model,
    mtl_tokenizer,
    s3_tokenizer,
    ve,
    val_data: List[Dict],
    device: str
) -> float:
    """Compute validation loss."""
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    from chatterbox.mtl_tts import Conditionals
    import torch.nn.functional as F
    
    t3_model.eval()
    total_loss = 0
    step_count = 0
    
    with torch.no_grad():
        for pair in val_data:
            path = pair["audio_path"]
            try:
                full_audio, sr = torchaudio.load(path)
                if full_audio.shape[0] > 1:
                    full_audio = torch.mean(full_audio, dim=0, keepdim=True)
                full_audio = full_audio.to(device)
            except:
                continue
            
            for seg in pair["segments"][:5]:  # Limit validation samples per file
                start_sec = seg.get("start", 0)
                end_sec = seg.get("end", start_sec + 5)
                text = seg["text"].strip()
                
                if not text or end_sec - start_sec < 1.0:
                    continue
                
                start_sample = int(start_sec * sr)
                end_sample = int(end_sec * sr)
                
                if end_sample > full_audio.shape[1]:
                    continue
                
                audio_chunk = full_audio[:, start_sample:end_sample]
                
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000).to(device)
                    audio_16k = resampler(audio_chunk)
                else:
                    audio_16k = audio_chunk
                
                try:
                    wav_np = audio_16k.squeeze().cpu().numpy()
                    spk_emb_np = ve.embeds_from_wavs([wav_np], sample_rate=16000, as_spk=False)
                    spk_emb = torch.from_numpy(spk_emb_np).to(device)
                    
                    text_ids = mtl_tokenizer.encode(text, language_id="ar")
                    text_tokens = torch.tensor([text_ids], dtype=torch.long).to(device)
                    
                    sot = t3_model.hp.start_text_token
                    eot = t3_model.hp.stop_text_token
                    text_tokens = F.pad(text_tokens, (1, 0), value=sot)
                    text_tokens = F.pad(text_tokens, (0, 1), value=eot)
                    
                    speech_tokens, _ = s3_tokenizer(audio_16k)
                    speech_tokens = speech_tokens.to(device)
                    
                    t3_cond_obj = T3Cond(
                        speaker_emb=spk_emb,
                        cond_prompt_speech_tokens=None,
                        emotion_adv=torch.ones(1, 1, 1).to(device)
                    )
                    
                    text_lens = torch.tensor([text_tokens.size(1)], device=device)
                    speech_lens = torch.tensor([speech_tokens.size(1)], device=device)

                    out = t3_model.forward(
                        t3_cond=t3_cond_obj,
                        text_tokens=text_tokens,
                        text_token_lens=text_lens,
                        speech_tokens=speech_tokens,
                        speech_token_lens=speech_lens,
                        training=True
                    )
                    
                    IGNORE_ID = -100
                    mask_speech = torch.arange(speech_tokens.size(1), device=device)[None] >= speech_lens[:, None]
                    masked_speech = speech_tokens.masked_fill(mask_speech, IGNORE_ID)
                    
                    speech_logits_permuted = out.speech_logits.permute(0, 2, 1)
                    loss = F.cross_entropy(speech_logits_permuted, masked_speech, ignore_index=IGNORE_ID)
                    
                    total_loss += loss.item()
                    step_count += 1
                    
                except Exception as e:
                    continue
    
    return total_loss / max(step_count, 1)

def save_lora(t3_model, output_dir: Path):
    """Save LoRA weights."""
    output_dir.mkdir(parents=True, exist_ok=True)
    t3_model.tfmr.save_pretrained(output_dir)
    logger.info(f"Saved LoRA to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Professional LoRA Training")
    parser.add_argument("--data_dir", type=str, default="loRa", help="Directory with audio+transcript files")
    parser.add_argument("--output_name", type=str, default="hussein_pro", help="Output LoRA name")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--k_folds", type=int, default=0, help="K-fold CV (0=disabled)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="Steps per epoch")
    parser.add_argument("--early_stopping", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load all data
    data_dir = Path(args.data_dir)
    all_data = load_all_data(data_dir)
    
    if len(all_data) < 2:
        logger.error("Need at least 2 audio+transcript pairs!")
        return
    
    output_dir = Path(f"voices/loras/{args.output_name}")
    
    # Split data
    if args.k_folds > 1:
        folds = k_fold_split(all_data, args.k_folds, args.seed)
        logger.info(f"Running {args.k_folds}-fold cross-validation")
    else:
        train_data, val_data = split_data(all_data, args.val_split, args.seed)
        folds = [(train_data, val_data)]
    
    best_val_loss = float('inf')
    
    for fold_idx, (train_data, val_data) in enumerate(folds):
        logger.info(f"\n{'='*50}")
        logger.info(f"Fold {fold_idx + 1}/{len(folds)}")
        logger.info(f"{'='*50}")
        
        # Setup model (fresh for each fold)
        t3_model, mtl_tokenizer, s3_tokenizer, ve, model_dir = setup_model(device)
        optimizer = torch.optim.AdamW(t3_model.tfmr.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        
        patience_counter = 0
        fold_best_loss = float('inf')
        
        for epoch in range(args.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
            
            # Train
            train_loss = train_epoch(
                t3_model, mtl_tokenizer, s3_tokenizer, ve, train_data,
                optimizer, device, args.steps_per_epoch
            )
            logger.info(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = validate(t3_model, mtl_tokenizer, s3_tokenizer, ve, val_data, device)
            logger.info(f"Val Loss: {val_loss:.4f}")
            
            # Learning rate scheduler
            scheduler.step(val_loss)
            
            # Check for improvement
            if val_loss < fold_best_loss:
                fold_best_loss = val_loss
                patience_counter = 0
                
                # Save if best across all folds
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_lora(t3_model, output_dir)
                    logger.info(f"✓ New best model saved! Val Loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{args.early_stopping}")
            
            # Early stopping
            if patience_counter >= args.early_stopping:
                logger.info("Early stopping triggered!")
                break
        
        # Clean up
        del t3_model, optimizer
        torch.cuda.empty_cache()
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Training complete! Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"LoRA saved to: {output_dir}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()
