# 🌍 Multilingual Chatterbox TTS Server with LoRA Fine-Tuning

> **🔀 Enhanced Fork by [Hussein](https://github.com/husseinnsourr)**

Enhanced [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) server with **multilingual support**, **LoRA fine-tuning**, and a **modern React UI**. Generate high-quality speech in 24+ languages with voice cloning and custom voice training capabilities.

---

## ⭐ What's New in This Fork

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

### New in This Fork

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

### Advanced Training

For more control over training parameters:

```bash
python train_lora_pro.py --help
```

---

## 🖥️ Platform Support

- **Windows:** Full support with easy installation script
- **Linux:** Manual installation supported
- **macOS:** Use the [main repository](https://github.com/devnen/Chatterbox-TTS-Server)

## 🔩 System Requirements

- **Windows 10/11** (64-bit) or **Linux**
- **Python 3.10+**
- **4GB+ RAM** (8GB+ recommended)
- **5GB+ storage** for models
- **GPU (Optional):** NVIDIA with 4GB+ VRAM or AMD RX 6000/7000 series

---

## 🚀 Quick Installation (Windows)

1. **Download**

   ```bash
   git clone https://github.com/husseinnsourr/chatterbox-server-tts-lora.git
   cd chatterbox-server-tts-lora
   ```

2. **Install**
   - Double-click `install.bat`
   - Choose your hardware:
     - Option 1: NVIDIA GPU (CUDA)
     - Option 2: CPU Only
     - Option 3: AMD GPU (ROCm)

3. **Run**
   - Double-click `win-run.bat`

4. **Access**
   - Open `http://localhost:8004` in your browser

---

## 💻 Manual Installation (Linux/Other)

1. **Clone Repository**

   ```bash
   git clone https://github.com/husseinnsourr/chatterbox-server-tts-lora.git
   cd chatterbox-server-tts-lora
   ```

2. **Setup Environment**

   > **Note:** Make sure you're using Python 3.10 for compatibility

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   ```

3. **Install Dependencies**

   Choose based on hardware:

   ```bash
   # NVIDIA GPU
   pip install -r requirements-nvidia.txt

   # AMD GPU
   pip install -r requirements-rocm.txt

   # CPU Only
   pip install -r requirements.txt
   ```

4. **Run Server**
   ```bash
   source venv/bin/activate  # if not already activated
   python server.py
   ```

---

## ⚛️ React UI Development

The modern React UI is located in `ui_src/`:

```bash
cd ui_src
npm install
npm run dev
```

Build for production:

```bash
npm run build
```

---

## ▶️ Running the Application

**Windows:** Double-click `win-run.bat`

**Linux/Manual:**

```bash
source venv/bin/activate
python server.py
```

**Access:** Open `http://localhost:8004`

**API Docs:** Visit `http://localhost:8004/docs`

---

## 🎯 Usage

### Web Interface

1. Enter text in any supported language
2. Select target language from dropdown
3. Choose voice mode (predefined or clone)
4. Adjust generation settings
5. Generate and listen to speech

### API Examples

**Main TTS Endpoint:**

```bash
POST /tts
{
  "text": "Hello world!",
  "language": "en",
  "voice_mode": "predefined",
  "temperature": 0.7,
  "speed_factor": 1.0
}
```

**OpenAI Compatible:**

```bash
POST /v1/audio/speech
{
  "input": "Your text here",
  "voice": "S1",
  "language": "en",
  "response_format": "wav"
}
```

---

## 🎤 Voice Management

**Predefined Voices:** Place `.wav`/`.mp3` files in `./voices` directory

**Voice Cloning:** Upload reference audio via Web UI (stored in `./reference_audio`)

**LoRA Models:** Trained models are saved in the model cache directory

---

## ⚙️ Configuration

Edit `config.yaml` for settings:

```yaml
server:
  host: "0.0.0.0"
  port: 8004

tts_engine:
  device: "auto" # auto, cuda, cpu

generation_defaults:
  temperature: 0.7
  language: "en"
```

---

## 📁 Project Structure

```
├── server.py              # Main FastAPI server
├── engine.py              # TTS engine implementation
├── config.py              # Configuration management
├── train_lora.py          # LoRA training script
├── train_lora_pro.py      # Advanced LoRA training
├── loRa/                  # Training data folder
├── ui_src/                # React UI source code
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   └── index.css
│   └── package.json
├── voices/                # Predefined voice files
├── reference_audio/       # Voice cloning references
└── outputs/               # Generated audio files
```

---

## 🔍 Troubleshooting

**Common Issues:**

- GPU not detected: Verify drivers and restart
- Import errors: Ensure virtual environment is activated
- Port conflicts: Change port in `config.yaml`
- LoRA training fails: Check CUDA/PyTorch compatibility

---

## 🤝 Contributing

Contributions welcome! Report bugs, suggest features, or submit pull requests.

---

## 📜 License

MIT License

---

## 🙏 Acknowledgements

- [Resemble AI](https://www.resemble.ai/) for [Chatterbox TTS](https://github.com/resemble-ai/chatterbox)
- [devnen](https://github.com/devnen) for the original [server implementation](https://github.com/devnen/Chatterbox-TTS-Server)
- [mirbehnam](https://github.com/mirbehnam) for the [Windows easy installation](https://github.com/mirbehnam/Chatterbox-TTS-Server-windows-easyInstallation)

---

## 📞 Support

- **This Fork Issues:** [Open issue here](https://github.com/husseinnsourr/chatterbox-server-tts-lora/issues)
- **Original Project:** [Main repository](https://github.com/devnen/Chatterbox-TTS-Server)

---

_🎙️ Generate speech in 24+ languages with LoRA fine-tuning and one-click Windows installation!_
