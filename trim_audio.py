
import torchaudio
import torch

def trim_audio(input_path, output_path, duration_sec=10):
    try:
        # Load audio
        print(f"Loading {input_path}...")
        audio, sr = torchaudio.load(input_path)
        
        # Calculate tokens
        num_frames = int(duration_sec * sr)
        
        # Trim
        if audio.shape[1] > num_frames:
            audio = audio[:, :num_frames]
        
        # Save
        print(f"Saving to {output_path}...")
        torchaudio.save(output_path, audio, sr)
        print("Done!")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    input_file = "loRa/Vice World War AI.MP3"
    output_file = "loRa/ref_hussein_short.wav"
    trim_audio(input_file, output_file)
