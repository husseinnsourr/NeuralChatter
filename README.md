# 🎙️ NeuralChatter TTS 🌍

### Multilingual TTS Server with LoRA Fine-Tuning

> **🔀 Enhanced Fork by [Hussein](https://github.com/husseinnsourr)**

## 📺 Demo Video

<video src="voice.mp4" controls width="100%"></video>

---

**NeuralChatter** is an enhanced [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) server with **multilingual support**, **LoRA fine-tuning**, and a **modern React UI**. Generate high-quality speech in 24+ languages with voice cloning and custom voice training capabilities.

---

## ⭐ What's New in NeuralChatter

| Feature                       | Description                                                                 |
| ----------------------------- | --------------------------------------------------------------------------- |
| 🧠 **LoRA Fine-Tuning**       | Train custom voice models using your own audio data                         |
| 🌍 **Cross-Language Cloning** | Train with Arabic audio → Generate speech in English, French, any language! |
| ⚛️ **Modern React UI**        | Beautiful, responsive web interface with Tailwind CSS                       |
| 📊 **Training Scripts**       | `train_lora.py` and `train_lora_pro.py` for voice training                  |
| 🎙️ **Arabic Voice Training**  | Pre-configured training data for Arabic voice cloning                       |
| 🔧 **Enhanced Configuration** | Extended `config.py` with more customization options                        |

> **🎯 Clone your voice with just ~10 audio files!** Record yourself speaking in Arabic, train the LoRA model, and your cloned voice can speak in 24+ languages!

---

## ✨ Key Features

### From Original Project

- 🌐 **24+ Languages:** Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swedish, Swahili, Turkish
- 🎤 **Voice Cloning:** Clone any voice using reference audio files
- 📚 **Audiobook Generation:** Process entire books with automatic text chunking
- 🚀 **Easy Windows Installation:** One-click setup with `install.bat`
- ⚡ **GPU Acceleration:** NVIDIA (CUDA) and AMD (ROCm) support
- 📡 **FastAPI Server:** RESTful API with interactive documentation

### New in NeuralChatter

- 🧠 **LoRA Fine-Tuning:** Train personalized voice models
- ⚛️ **React Web UI:** Modern interface built with React + Vite + Tailwind CSS
- 📁 **Training Data Management:** Organized `loRa/` folder for training datasets
- 🔊 **Audio Processing:** Tools for audio trimming and preprocessing

---

## 🧠 LoRA Fine-Tuning (New!)

Train custom voice models using your own audio recordings:

### Quick Start

```bash
# Activate environment
source venv/bin/activate  # Linux
# or
.\venv\Scripts\Activate.ps1  # Windows

# Run training
python train_lora.py --audio_file "path/to/audio.wav" --transcript "path/to/text.txt"
```

### Training Data Structure

Place your training data in the `loRa/` folder:

```
loRa/
├── your_audio.wav          # Audio file (WAV/MP3)
├── your_audio.txt          # Transcript text file
└── your_audio_training_data.json  # Generated training data
```

---

## 🚀 Quick Installation (Windows)

1. **Download**

   ```bash
   git clone https://github.com/husseinnsourr/NeuralChatter.git
   cd NeuralChatter
   ```

2. **Install**
   - Double-click `install.bat`
   - Choose your hardware:
     - Option 1: NVIDIA GPU (CUDA)
     - Option 2: CPU Only

3. **Run**
   - Double-click `win-run.bat`

---

## 🔧 Configuration

Edit `config.yaml` for settings:

```yaml
server:
  host: "0.0.0.0"
  port: 8004

tts_engine:
  device: "auto" # auto, cuda, cpu
```

---

## 📜 License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. Contributions must be shared back with the community.

---

## 🤝 Contributing

Contributions welcome! Report bugs, suggest features, or submit pull requests. [Open issue here](https://github.com/husseinnsourr/NeuralChatter/issues).

---

_🎙️ Generate speech in 24+ languages with LoRA fine-tuning and one-click Windows installation!_
