import os
import argparse
import logging
import torch
from pathlib import Path
from tqdm import tqdm
import whisper
import torchaudio
import numpy as np

# Chatterbox imports
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.tokenizers import MTLTokenizer
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.mtl_tts import Conditionals
from chatterbox.models.s3tokenizer import S3Tokenizer
import torch.nn.functional as F # Import F globally

from transformers import LlamaConfig, LlamaModel
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_lora")

def load_t3_model(ckpt_dir, device):
    """Loads the T3 model components."""
    logger.info("Loading T3 Model...")
    from safetensors.torch import load_file as load_safetensors

    # Recreate T3 with patched config instructions (same as engine.py)
    # We need to ensure we use the same structure as the inference engine
    class PatchedT3(T3):
        def __init__(self, hp=None):
            super().__init__(hp)
            cfg = self.cfg
            cfg._attn_implementation = "eager"
            self.tfmr = LlamaModel(cfg)

    t3_config = T3Config.multilingual()
    t3 = PatchedT3(t3_config)
    
    t3_state = load_safetensors(ckpt_dir / "t3_23lang.safetensors")
    if "model" in t3_state.keys():
        t3_state = t3_state["model"][0]
    t3.load_state_dict(t3_state, strict=False)
    t3.to(device)
    # We put it in training mode, but usually we eval() the frozen parts.
    # Typically we want the transformer in train mode for dropout etc? 
    # For LoRA, we set model to train() but only adapters are trainable.
    t3.train() 
    
    return t3

def main():
    parser = argparse.ArgumentParser(description="Train LoRA for Chatterbox TTS")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the long audio file (e.g. 15 min wav/mp3)")
    parser.add_argument("--output_name", type=str, default="my_voice", help="Name of the training output (will save in voices/loras/)")
    parser.add_argument("--steps", type=int, default=300, help="Number of training steps")
    parser.add_argument("--model_path", type=str, default="model_cache/ResembleAI_chatterbox", help="Path to base model directory")
    parser.add_argument("--transcript_path", type=str, default=None, help="Path to custom transcript text file with timestamps")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("Training on CPU is very slow. CUDA is recommended.")

    # 1. Prepare Paths
    lora_save_dir = Path("voices/loras") / args.output_name
    lora_save_dir.mkdir(parents=True, exist_ok=True)
    
    segments = []
    
    # 2. Transcribe or Load Transcript
    if args.transcript_path and os.path.exists(args.transcript_path):
        logger.info(f"Loading custom transcript from {args.transcript_path}...")
        
        def parse_time(t_str):
            h, m, s = map(float, t_str.split(':'))
            return h * 3600 + m * 60 + s

        with open(args.transcript_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        current_time = None
        current_text = []
        
        # Simple parser for:
        # 00:00:00
        # Text...
        
        parsed_entries = [] # (start_time, text)
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # Check if line is timestamp
            if ':' in line and len(line.split(':')) == 3:
                try:
                    t = parse_time(line)
                    # If we have a previous entry pending
                    if current_time is not None and current_text:
                         parsed_entries.append((current_time, " ".join(current_text)))
                         current_text = []
                    current_time = t
                    continue
                except ValueError:
                    pass # Not a timestamp, treat as text
            
            if current_time is not None:
                current_text.append(line)
                
        # Append last entry
        if current_time is not None and current_text:
             parsed_entries.append((current_time, " ".join(current_text)))
             
        # Convert to segments with start/end
        # We infer end time from next start time. Last segment needs an arbitrary duration or max duration.
        # Let's assume max 15s or 10s if not specified.
        
        for i, (start, text) in enumerate(parsed_entries):
            if i < len(parsed_entries) - 1:
                end = parsed_entries[i+1][0]
            else:
                end = start + 10.0 # Default 10s for last segment
            
            # Sanity check duration
            if end - start > 30.0:
                 end = start + 30.0
            
            segments.append({
                "start": start,
                "end": end,
                "text": text
            })
            
        logger.info(f"Parsed {len(segments)} segments from transcript file.")
        
    else:
        # Use Whisper
        logger.info("Transcribing and segmenting audio with Whisper...")
        model = whisper.load_model("base", device=device) # 'base' is strictly sufficient for alignment usually
        result = model.transcribe(args.audio_path, language="ar") # Assume Arabic based on user context, or auto
        
        segments = result["segments"]
        logger.info(f"Found {len(segments)} segments.")
        
        # FREE VRAM: Unload Whisper
        del model
        torch.cuda.empty_cache()
        logger.info("Unloaded Whisper to free VRAM.")

    # 3. Load Chatterbox Components
    # We assume 'from_pretrained' logic downloaded models to cache, or we use explicit path
    # Usually ~/.cache/huggingface/hub/... or local folder.
    # For simplified script, let's assume standard cache or user provided path.
    # If standard cache is tricky to find programmatically without chatterbox logic, we might need simple scan or re-download check.
    # To correspond with server, we can reuse logic or just assume standard path if server ran once.
    
    # Try to resolve model path from standard HF cache if default doesn't exist
    model_dir_path = Path(args.model_path)
    if not model_dir_path.exists():
        # Try finding in HF cache
        try:
            from huggingface_hub import snapshot_download
            logger.info(f"Model path {model_dir_path} not found. Downloading/Checking from HF...")
            model_dir = snapshot_download(repo_id="ResembleAI/chatterbox")
            model_dir_path = Path(model_dir)
        except Exception as e:
            logger.error(f"Could not find model: {e}")
            return

    # Load Tokenizers
    mtl_tokenizer = MTLTokenizer(str(model_dir_path / "mtl_tokenizer.json"))
    s3_tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
    
    # Load Models
    t3_model = load_t3_model(model_dir_path, device)
    ve = VoiceEncoder().to(device).eval()
    ve.load_state_dict(torch.load(model_dir_path / "ve.pt", map_location=device))

    # 4. Setup LoRA
    logger.info("Setting up LoRA configuration...")
    
    adapter_path = lora_save_dir / "adapter_model.safetensors"
    if adapter_path.exists():
        logger.info(f"Existing LoRA found at {lora_save_dir}. Resuming training...")
        # Load existing adapter and make it trainable
        # Note: is_trainable=True is crucial for fine-tuning an existing LoRA
        t3_model.tfmr = PeftModel.from_pretrained(t3_model.tfmr, lora_save_dir, is_trainable=True)
    else:
        logger.info("No existing LoRA found. Starting fresh training...")
        peft_config = LoraConfig(
            task_type=None, # Generic task
            inference_mode=False,
            r=16,           # Rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"], # Target Attention layers
        )
        # Wrap the transformer backbone
        t3_model.tfmr = get_peft_model(t3_model.tfmr, peft_config)
    
    t3_model.tfmr.print_trainable_parameters()
    
    # Optimizer - lower LR to prevent overfitting
    optimizer = torch.optim.AdamW(t3_model.tfmr.parameters(), lr=5e-5)
    
    # 5. Training Loop
    full_audio, sr = torchaudio.load(args.audio_path)
    # Convert to mono if needed
    if full_audio.shape[0] > 1:
        full_audio = torch.mean(full_audio, dim=0, keepdim=True)
    
    # Resamplers
    resampler_16k = torchaudio.transforms.Resample(sr, 16000).to(device)
    # T3 needs 24k? No, S3Tokenizer uses 16k usually? 
    # S3Tokenizer in s3gen expects 16k? Let's check. s3gen/s3gen.py uses tokenizer(ref_wav_16).
    # Correct.
    
    full_audio = full_audio.to(device)
    
    logger.info(f"Starting training for {args.steps} steps...")
    
    step = 0
    pbar = tqdm(total=args.steps)
    
    import random
    
    while step < args.steps:
        # Shuffle segments
        random.shuffle(segments)
        
        for seg in segments:
            if step >= args.steps:
                break
                
            start_sec = seg["start"]
            end_sec = seg["end"]
            text = seg["text"].strip()
            
            if end_sec - start_sec < 1.0 or end_sec - start_sec > 15.0:
                 continue # Skip too short or too long
            
            # Extract Audio Chunk
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            wav_chunk = full_audio[:, start_sample:end_sample] # (1, T)
            
            # Resample for Tokenizer & VE
            wav_16k = resampler_16k(wav_chunk)
            
            # 1. Compute Speaker Embedding (from this chunk)
            wav_np = wav_16k.squeeze().cpu().numpy()
            # embeds_from_wavs expects List[np.ndarray]
            # output is np.ndarray (B, E)
            # We use as_spk=False to get (1, E) for this single chunk
            spk_emb_np = ve.embeds_from_wavs([wav_np], sample_rate=16000, as_spk=False)
            spk_emb = torch.from_numpy(spk_emb_np).to(device)
            
            # 2. Tokenize Text
            # We need to use valid language ID?
            # MTLTokenizer usage: encode(text, lang_id=...)
            # We can default to 'ar' or detect.
            text_ids = mtl_tokenizer.encode(text, language_id="ar")
            text_tokens = torch.tensor([text_ids], dtype=torch.long).to(device) # (1, L_text)
            
            # Pad with SOT and EOT tokens manually, as T3 expects them
            sot = t3_model.hp.start_text_token
            eot = t3_model.hp.stop_text_token
            # text_tokens shape is (1, L)
            # Pad left with SOT, right with EOT
            text_tokens = F.pad(text_tokens, (1, 0), value=sot)
            text_tokens = F.pad(text_tokens, (0, 1), value=eot)
            
            text_lens = torch.tensor([text_tokens.size(1)], device=device)
            
            # 3. Tokenize Speech (Targets)
            # S3Tokenizer expects 16k
            with torch.no_grad():
                speech_tokens, speech_lens = s3_tokenizer(wav_16k)
                speech_tokens = speech_tokens.to(device)
                speech_lens = speech_lens.to(device)

            # 4. Prepare Conditioning (T3Cond)
            # For "Voice Cloning", usually we prompt with a slice of the audio.
            # In training, we can use the *same* audio as prompt (forcing alignment) or empty prompt?
            # Zero-shot usually takes a prompt. If we fine-tune, we might want to Condition on Speaker Embedding ONLY?
            # T3 usually typically requires some cond.
            # Let's verify T3Cond structure.
            # From engine.py/models, it seems T3 taking `t3_cond`.
            # We construct T3Cond.
            
            # 4. Prepare Conditioning (T3Cond)
            # T3Cond needs (speaker_emb, cond_prompt_speech_tokens, emotion_adv)
            from chatterbox.models.t3.modules.cond_enc import T3Cond
            
            # We need T3Cond to contain speaker_emb
            # emotion_adv defaults to 0 or 1? Usually 1.0 for neutral? 
            # engine.py uses: emotion_adv=exaggeration * torch.ones(1, 1, 1)
            # Let's use 1.0
            
            t3_cond_obj = T3Cond(
                speaker_emb=spk_emb,
                cond_prompt_speech_tokens=None, # Training without prompt tokens for now (unconditioned on text prompt history)
                emotion_adv=torch.ones(1, 1, 1).to(device)
            )
            
            # Conditionals wrapper
            # s3gen needs 'gen' dict. For T3 training, s3gen conds might not be used?
            # T3 forward only uses t3_cond.
            t3_cond = Conditionals(
                t3=t3_cond_obj,
                gen={} # Dummy dict for s3gen since we are only training T3
            )

            # 5. Forward & Loss
            optimizer.zero_grad()
            
            # Manual forward pass since t3_model.loss() has cross_entropy dimension issues
            out = t3_model.forward(
                t3_cond=t3_cond_obj,
                text_tokens=text_tokens,
                text_token_lens=text_lens,
                speech_tokens=speech_tokens,
                speech_token_lens=speech_lens,
                training=True
            )
            
            # Calculate loss manually with correct tensor dimensions
            # cross_entropy expects (B, C, L) but we have (B, L, C), so we permute
            IGNORE_ID = -100
            len_speech_val = speech_tokens.size(1)
            
            # Create mask for valid speech positions
            mask_speech = torch.arange(len_speech_val, device=device)[None] >= speech_lens[:, None]
            masked_speech = speech_tokens.masked_fill(mask_speech, IGNORE_ID)
            
            # Permute logits from (B, L, C) to (B, C, L) for cross_entropy
            speech_logits_permuted = out.speech_logits.permute(0, 2, 1)
            loss = F.cross_entropy(speech_logits_permuted, masked_speech, ignore_index=IGNORE_ID) 
            
            loss.backward()
            optimizer.step()
            
            step += 1
            pbar.update(1)
            pbar.set_description(f"Loss: {loss.item():.4f}")
            
            # Checkpoint occasionally
            if step % 100 == 0:
                 t3_model.tfmr.save_pretrained(lora_save_dir)

    pbar.close()
    
    # Final Save
    logger.info(f"Saving final LoRA to {lora_save_dir}")
    t3_model.tfmr.save_pretrained(lora_save_dir)
    logger.info("Done!")

if __name__ == "__main__":
    main()
