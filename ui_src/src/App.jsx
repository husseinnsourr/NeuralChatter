import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import TTSForm from './components/TTSForm';
import AudioPlayer from './components/AudioPlayer';
import { getInitialData } from './api';

function App() {
  const [darkMode, setDarkMode] = useState(true);
  const [initialData, setInitialData] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Determine theme from localStorage or system
    const isDark = true; // Defaulting to dark as per design
    setDarkMode(isDark);
    document.documentElement.classList.toggle('dark', isDark);
    
    // Fetch Initial Data
    getInitialData().then(data => {
        setInitialData(data);
    }).catch(err => {
        console.error("Failed to load initial data", err);
        setError("فشل الاتصال بالسيرفر. تأكد من تشغيل server.py");
    });
  }, []);

  const toggleTheme = () => {
    const newMode = !darkMode;
    setDarkMode(newMode);
    document.documentElement.classList.toggle('dark', newMode);
  };

  const handleAudioGenerated = (url) => {
      setAudioUrl(url);
      setError(null);
  };

  const handleError = (msg) => {
      setError(msg);
  };

  return (
    <div className={`min-h-screen flex flex-col relative overflow-x-hidden ${darkMode ? 'bg-[#0f172a]' : 'bg-gray-50'}`}>
      {/* Background Gradients */}
      <div className="fixed inset-0 z-0 pointer-events-none overflow-hidden">
        <div className="absolute top-[-10%] right-[-5%] w-[600px] h-[600px] bg-indigo-600/10 rounded-full blur-[100px] opacity-40 animate-pulse"></div>
        <div className="absolute bottom-[-10%] left-[-5%] w-[600px] h-[600px] bg-purple-600/10 rounded-full blur-[100px] opacity-30"></div>
      </div>

      <Header darkMode={darkMode} toggleTheme={toggleTheme} />

      <main className="relative z-10 flex-grow py-10 px-4 sm:px-6 lg:px-8">
        <div className="max-w-5xl mx-auto space-y-8">
          
          {error && (
             <div className="bg-red-500/10 border border-red-500/50 text-red-200 p-4 rounded-xl flex items-center gap-3 animate-fade-in text-right" dir="rtl">
                 <span className="font-bold">تنبيه:</span> {error}
             </div>
          )}

          <TTSForm 
            initialData={initialData} 
            onAudioGenerated={handleAudioGenerated}
            onError={handleError}
          />

          <AudioPlayer audioUrl={audioUrl} />
          
        </div>
      </main>

      <footer className="mt-auto py-8 border-t border-white/5 bg-black/40 text-center relative z-10 backdrop-blur-sm">
        <p className="text-xs text-gray-500">
          Powered by Chatterbox TTS &bull; &copy; 2026
        </p>
      </footer>
    </div>
  );
}

export default App;
