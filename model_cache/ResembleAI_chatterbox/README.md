---
license: mit
language:
- ar
- da
- de
- el
- en
- es
- fi
- fr
- he
- hi
- it
- ja
- ko
- ms
- nl
- no
- pl
- pt
- ru
- sv
- sw
- tr
- zh
pipeline_tag: text-to-speech
tags:
- text-to-speech
- speech
- speech-generation
- voice-cloning
- multilingual-tts
library_name: chatterbox
---

<img width="800" alt="cb-big2" src="https://github.com/user-attachments/assets/bd8c5f03-e91d-4ee5-b680-57355da204d1" />

<h1 style="font-size: 32px">Chatterbox TTS</h1>

<div style="display: flex; align-items: center; gap: 12px">
  <a href="https://resemble-ai.github.io/chatterbox_demopage/">
    <img src="https://img.shields.io/badge/listen-demo_samples-blue" alt="Listen to Demo Samples" />
  </a>
  <a href="https://huggingface.co/spaces/ResembleAI/Chatterbox">
    <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg" alt="Open in HF Spaces" />
  </a>
  <a href="https://podonos.com/resembleai/chatterbox">
    <img src="https://static-public.podonos.com/badges/insight-on-pdns-sm-dark.svg" alt="Insight on Podos" />
  </a>
</div>

<div style="display: flex; align-items: center; gap: 8px;">
  <span style="font-style: italic;white-space: pre-wrap">Made with  ❤️  by</span>
  <img width="100" alt="resemble-logo-horizontal" src="https://github.com/user-attachments/assets/35cf756b-3506-4943-9c72-c05ddfa4e525" />
</div>

**09/04 🔥 Introducing Chatterbox Multilingual in 23 Languages!**

We're excited to introduce **Chatterbox** and **Chatterbox Multilingual**, [Resemble AI's](https://resemble.ai) production-grade open source TTS models. Chatterbox Multilingual supports **Arabic, Danish, German, Greek, English, Spanish, Finnish, French, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Dutch, Norwegian, Polish, Portuguese, Russian, Swedish, Swahili, Turkish, Chinese** out of the box. Licensed under MIT, Chatterbox has been benchmarked against leading closed-source systems like ElevenLabs, and is consistently preferred in side-by-side evaluations.

Whether you're working on memes, videos, games, or AI agents, Chatterbox brings your content to life. It's also the first open source TTS model to support **emotion exaggeration control**, a powerful feature that makes your voices stand out. Try it now on our [Hugging Face Gradio app.](https://huggingface.co/spaces/ResembleAI/Chatterbox)

If you like the model but need to scale or tune it for higher accuracy, check out our competitively priced TTS service (<a href="https://resemble.ai">link</a>). It delivers reliable performance with ultra-low latency of sub 200ms—ideal for production use in agents, applications, or interactive media.

# Key Details
- Multilingual, zero-shot TTS supporting 23 languages
- SoTA zeroshot English TTS
- 0.5B Llama backbone
- Unique exaggeration/intensity control
- Ultra-stable with alignment-informed inference
- Trained on 0.5M hours of cleaned data
- Watermarked outputs
- Easy voice conversion script
- [Outperforms ElevenLabs](https://podonos.com/resembleai/chatterbox)

# Tips
- **General Use (TTS and Voice Agents):**
  - The default settings (`exaggeration=0.5`, `cfg=0.5`) work well for most prompts.
  - If the reference speaker has a fast speaking style, lowering `cfg` to around `0.3` can improve pacing.

- **Expressive or Dramatic Speech:**
  - Try lower `cfg` values (e.g. `~0.3`) and increase `exaggeration` to around `0.7` or higher.
  - Higher `exaggeration` tends to speed up speech; reducing `cfg` helps compensate with slower, more deliberate pacing.

  
***Note:*** Ensure that the reference clip matches the specified language tag. Otherwise, language transfer outputs may inherit the accent of the reference clip’s language.  
***To mitigate this, set the CFG weight to 0.***


# Installation
```
pip install chatterbox-tts
```


# Usage
```python
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
wav = model.generate(text)
ta.save("test-1.wav", wav, model.sr)

# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH="YOUR_FILE.wav"
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
ta.save("test-2.wav", wav, model.sr)
```

# Multilingual Quickstart
```python
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

french_text = "Bonjour, comment ça va? Ceci est le modèle de synthèse vocale multilingue Chatterbox, il prend en charge 23 langues."
wav_french = multilingual_model.generate(french_text, language_id="fr")
ta.save("test-french.wav", wav_french, model.sr)

chinese_text = "你好，今天天气真不错，希望你有一个愉快的周末。"
wav_chinese = multilingual_model.generate(chinese_text, language_id="zh")
ta.save("test-chinese.wav", wav_chinese, model.sr)
```
See `example_tts.py` for more examples.


# Acknowledgements
- [Cosyvoice](https://github.com/FunAudioLLM/CosyVoice)
- [HiFT-GAN](https://github.com/yl4579/HiFTNet)
- [Llama 3](https://github.com/meta-llama/llama3)

# Built-in PerTh Watermarking for Responsible AI

Every audio file generated by Chatterbox includes [Resemble AI's Perth (Perceptual Threshold) Watermarker](https://github.com/resemble-ai/perth) - imperceptible neural watermarks that survive MP3 compression, audio editing, and common manipulations while maintaining nearly 100% detection accuracy.

# Disclaimer
Don't use this model to do bad things. Prompts are sourced from freely available data on the internet.