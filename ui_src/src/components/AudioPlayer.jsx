import React, { useEffect, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import { Play, Pause, Download, Volume2 } from 'lucide-react';

const AudioPlayer = ({ audioUrl }) => {
  const containerRef = useRef(null);
  const wavesurfer = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState('0:00');
  const [currentTime, setCurrentTime] = useState('0:00');

  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const secondsRemainder = Math.floor(seconds % 60);
    const paddedSeconds = secondsRemainder < 10 ? `0${secondsRemainder}` : `${secondsRemainder}`;
    return `${minutes}:${paddedSeconds}`;
  };

  useEffect(() => {
    if (!containerRef.current || !audioUrl) return;

    if (wavesurfer.current) {
        wavesurfer.current.destroy();
    }

    wavesurfer.current = WaveSurfer.create({
      container: containerRef.current,
      waveColor: '#4f46e5',
      progressColor: '#818cf8',
      cursorColor: '#fff',
      barWidth: 2,
      barRadius: 3,
      cursorWidth: 1,
      height: 80,
      barGap: 3,
      responsive: true,
      backend: 'MediaElement'
    });

    wavesurfer.current.load(audioUrl);

    wavesurfer.current.on('ready', () => {
      setDuration(formatTime(wavesurfer.current.getDuration()));
    });

    wavesurfer.current.on('audioprocess', () => {
      setCurrentTime(formatTime(wavesurfer.current.getCurrentTime()));
    });
    
    wavesurfer.current.on('finish', () => {
        setIsPlaying(false);
    });

    return () => {
      if (wavesurfer.current) {
        wavesurfer.current.destroy();
      }
    };
  }, [audioUrl]);

  const togglePlay = () => {
    if (wavesurfer.current) {
      if (isPlaying) {
        wavesurfer.current.pause();
      } else {
        wavesurfer.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };
  
  const handleDownload = () => {
      if (!audioUrl) return;
      const link = document.createElement('a');
      link.href = audioUrl;
      link.download = `speech-${Date.now()}.wav`; // Assuming wav, or extract ext
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
  }

  if (!audioUrl) return null;

  return (
    <div className="glass border border-white/10 rounded-2xl p-6 shadow-xl animate-fade-in relative z-20 w-full mt-8">
      <div className="flex items-center gap-4 mb-4">
        <button
          onClick={togglePlay}
          className="w-12 h-12 bg-indigo-600 rounded-full flex items-center justify-center hover:bg-indigo-500 transition-colors shadow-lg shadow-indigo-500/30"
        >
          {isPlaying ? <Pause className="w-5 h-5 text-white" fill="currentColor" /> : <Play className="w-5 h-5 text-white ml-1" fill="currentColor" />}
        </button>
        <div className="flex-col">
             <div className="text-white font-medium">Generated Audio</div>
             <div className="text-xs text-gray-400 font-mono">{currentTime} / {duration}</div>
        </div>
        <div className="flex-grow"></div>
        <button onClick={handleDownload} className="p-2 text-gray-400 hover:text-white transition-colors" title="Download">
            <Download className="w-5 h-5" />
        </button>
      </div>
      <div ref={containerRef} className="w-full" />
    </div>
  );
};

export default AudioPlayer;
