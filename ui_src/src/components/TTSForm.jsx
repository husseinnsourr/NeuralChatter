import React, { useState, useEffect, useRef } from 'react';
import { Mic, FileAudio, Settings, Upload, RefreshCw, Volume2, Save, Globe, BookOpen } from 'lucide-react';
import { generateTTS, uploadReference, uploadPredefined, loadMultilingualModel } from '../api';

const TTSForm = ({ initialData, onAudioGenerated, onError }) => {
  const [text, setText] = useState('');
  const [voiceMode, setVoiceMode] = useState('predefined');
  const [selectedPredefinedVoice, setSelectedPredefinedVoice] = useState('none');
  const [selectedReferenceFile, setSelectedReferenceFile] = useState('none');
  const [loading, setLoading] = useState(false);
  const [multilingualLoading, setMultilingualLoading] = useState(false);
  
  // Lists
  const [predefinedVoices, setPredefinedVoices] = useState([]);
  const [referenceFiles, setReferenceFiles] = useState([]);
  const [loras, setLoras] = useState([]);
  const [selectedLora, setSelectedLora] = useState('none');
  const [presets, setPresets] = useState([]);

  // Advanced Settings
  const [splitText, setSplitText] = useState(false);
  const [chunkSize, setChunkSize] = useState(120);
  const [temperature, setTemperature] = useState(0.8);
  const [exaggeration, setExaggeration] = useState(0.5);
  const [cfgWeight, setCfgWeight] = useState(0.5);
  const [speedFactor, setSpeedFactor] = useState(1.0);
  const [seed, setSeed] = useState(0);
  const [language, setLanguage] = useState('en');

  const predefinedInputRef = useRef(null);
  const referenceInputRef = useRef(null);

  useEffect(() => {
    if (initialData) {
      setPredefinedVoices(initialData.predefined_voices || []);
      setReferenceFiles(initialData.reference_files || []);
      setLoras(initialData.loras || []);
      setPresets(initialData.presets || []);
      
      // Set defaults from config if available
      if (initialData.config) {
          setTemperature(initialData.config.gen_defaults?.temperature || 0.8);
          // ... other defaults
      }
    }
  }, [initialData]);

  const handleMultilingualLoad = async () => {
      setMultilingualLoading(true);
      try {
          await loadMultilingualModel();
          alert('تم تحميل نموذج تعدد اللغات بنجاح!');
      } catch (e) {
          onError('فشل تحميل نموذج تعدد اللغات');
      } finally {
          setMultilingualLoading(false);
      }
  };

  const applyPreset = (presetText) => {
      setText(presetText);
  };
  
  // Custom fetch wrapper for Blob because standard axios setup might expect JSON
  const generateAudioBlob = async () => {
      if (!text.trim()) {
        onError('أدخل النص أولاً');
        return;
      }
      setLoading(true);
      try {
           const payload = {
            text,
            voice_mode: voiceMode,
            predefined_voice_id: selectedPredefinedVoice !== 'none' ? selectedPredefinedVoice : null,
            reference_audio_filename: selectedReferenceFile !== 'none' ? selectedReferenceFile : null,
            lora_id: voiceMode === 'lora' && selectedLora !== 'none' ? selectedLora : null,
            split_text: splitText,
            chunk_size: parseInt(chunkSize),
            temperature: parseFloat(temperature),
            exaggeration: parseFloat(exaggeration),
            cfg_weight: parseFloat(cfgWeight),
            speed_factor: parseFloat(speedFactor),
            seed: parseInt(seed),
            language: language,
            output_format: 'wav'
          };
          
          const response = await fetch('/tts', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload)
          });
          
          if (!response.ok) {
              const errJson = await response.json();
              throw new Error(errJson.detail || 'Generation failed');
          }
          
          const blob = await response.blob();
          const url = URL.createObjectURL(blob);
          onAudioGenerated(url);
          
      } catch (e) {
          onError(e.message);
      } finally {
          setLoading(false);
      }
  };

  const onUploadPredefined = async (e) => {
      if(!e.target.files.length) return;
      const formData = new FormData();
      for(let i=0; i<e.target.files.length; i++){
          formData.append('files', e.target.files[i]);
      }
      try {
          const res = await uploadPredefined(formData);
          setPredefinedVoices(res.data.all_predefined_voices);
      } catch(e) { onError('فشل رفع الملفات'); }
  };

    const onUploadRef = async (e) => {
      if(!e.target.files.length) return;
      const formData = new FormData();
      for(let i=0; i<e.target.files.length; i++){
          formData.append('files', e.target.files[i]);
      }
      try {
          const res = await uploadReference(formData);
          setReferenceFiles(res.data.all_reference_files);
          // Auto select first uploaded
          if(res.data.uploaded_files.length) setSelectedReferenceFile(res.data.uploaded_files[0]);
      } catch(e) { onError('فشل رفع الملفات المرجعية'); }
  };


  return (
    <div className="glass border border-white/10 rounded-3xl shadow-2xl overflow-hidden ring-1 ring-white/5 relative z-10 w-full mb-8">
      <div className="p-8 sm:p-10 space-y-8">
        
        {/* Header & Action */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 border-b border-white/5 pb-8">
          <div>
            <h1 className="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-white to-gray-300 mb-2">توليد الكلام</h1>
            <p className="text-gray-400 text-sm leading-relaxed max-w-lg">
              منصة الذكاء الاصطناعي لتحويل النصوص إلى صوت طبيعي ومتقن.
            </p>
          </div>
          <div className="flex gap-3">
             <button
                type="button"
                onClick={handleMultilingualLoad}
                disabled={multilingualLoading}
                className="btn-secondary-glow group relative inline-flex items-center justify-center px-4 py-3 bg-gray-800 rounded-xl overflow-hidden transition-all hover:bg-gray-700 border border-white/10"
                title="تحميل نموذج تعدد اللغات"
              >
                 {multilingualLoading ? <RefreshCw className="w-5 h-5 animate-spin text-white" /> : <Globe className="w-5 h-5 text-white" />}
              </button>
              
              <button
                type="button"
                onClick={generateAudioBlob}
                disabled={loading}
                className="btn-primary-glow group relative inline-flex items-center justify-center px-8 py-3.5 bg-indigo-600 rounded-xl overflow-hidden transition-all hover:scale-105 active:scale-95 shadow-lg shadow-indigo-600/30 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <span className="relative z-10 flex items-center gap-2 font-bold text-white">
                  {loading ? (
                    <>
                        <RefreshCw className="w-5 h-5 animate-spin" />
                        جاري التوليد...
                    </>
                  ) : (
                    <>
                        <Mic className="w-5 h-5 group-hover:animate-pulse" />
                         توليد المقطع الصوتي
                    </>
                  )}
                </span>
              </button>
          </div>
        </div>

        {/* Input */}
        <div className="space-y-3">
          <label htmlFor="text" className="flex justify-between text-sm font-semibold text-gray-300">
            <span>النص</span>
            <div className="flex items-center gap-2">
                 {/* Presets Button */}
                 <div className="relative group">
                    <button type="button" className="text-xs font-normal text-indigo-400 hover:text-indigo-300 flex items-center gap-1 transition-colors">
                        <BookOpen className="w-3 h-3" /> نماذج جاهزة
                    </button>
                    {/* Presets Dropdown */}
                    <div className="absolute left-0 top-full mt-2 w-64 bg-gray-900 border border-white/10 rounded-xl shadow-xl p-2 hidden group-hover:block z-50">
                        <div className="text-xs text-gray-500 mb-2 px-2">اختر نصاً للتجربة:</div>
                        {presets.map((preset, idx) => (
                            <button 
                                key={idx} 
                                onClick={() => applyPreset(preset.text)}
                                className="w-full text-right px-3 py-2 text-sm text-gray-300 hover:bg-white/10 rounded-lg truncate transition-colors"
                            >
                                {preset.name}
                            </button>
                        ))}
                         {presets.length === 0 && <div className="text-xs text-center text-gray-600 py-2">لا توجد نماذج</div>}
                    </div>
                 </div>
                <span className="text-xs font-normal text-gray-500 bg-gray-800/50 px-2 py-0.5 rounded border border-white/5">
                  <span className="text-indigo-400">{text.length}</span> حرف
                </span>
            </div>
          </label>
          <div className="relative group">
            <textarea
              id="text"
              rows={5}
              value={text}
              onChange={(e) => setText(e.target.value)}
              className="w-full bg-gray-900/40 border border-gray-700/50 text-gray-100 placeholder-gray-600 text-lg rounded-xl focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 focus:bg-gray-900/60 transition-all p-5 resize-y shadow-inner outline-none"
              placeholder="أدخل النص المراد تحويله هنا..."
            />
          </div>
        </div>

        {/* Voice & Settings Settings */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
             {/* Left Column: Voice Selection */}
             <div className="lg:col-span-7 space-y-6">
                <label className="text-sm font-semibold text-gray-300 border-b border-gray-700/50 pb-2 block">اختيار الصوت</label>
                
                <div className="bg-gray-800/30 p-1 rounded-xl flex gap-1 border border-white/5">
                    <button
                         onClick={() => setVoiceMode('clone')}
                         className={`flex-1 py-2.5 px-4 rounded-lg text-center text-sm font-medium transition-all ${voiceMode === 'clone' ? 'bg-gray-700 text-white shadow-sm' : 'text-gray-400 hover:text-gray-200'}`}
                    >
                        استنساخ (Clone)
                    </button>
                    <button
                        onClick={() => setVoiceMode('predefined')}
                        className={`flex-1 py-2.5 px-4 rounded-lg text-center text-sm font-medium transition-all ${voiceMode === 'predefined' ? 'bg-gray-700 text-white shadow-sm' : 'text-gray-400 hover:text-gray-200'}`}
                    >
                        أصوات جاهزة
                    </button>
                    <button
                        onClick={() => setVoiceMode('lora')}
                        className={`flex-1 py-2.5 px-4 rounded-lg text-center text-sm font-medium transition-all ${voiceMode === 'lora' ? 'bg-gradient-to-r from-purple-600 to-indigo-600 text-white shadow-sm' : 'text-gray-400 hover:text-gray-200'}`}
                    >
                        🎤 LoRA
                    </button>
                </div>

                {voiceMode === 'predefined' && (
                    <div className="space-y-3 animate-slide-down">
                        <div className="relative flex-grow group">
                            <select
                                value={selectedPredefinedVoice}
                                onChange={(e) => setSelectedPredefinedVoice(e.target.value)}
                                className="w-full bg-gray-900 border border-gray-700 text-white rounded-xl px-5 py-3 focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 outline-none transition-all cursor-pointer"
                                style={{ colorScheme: 'dark', color: 'white' }} 
                            >
                                <option value="none" style={{ backgroundColor: '#111827', color: 'white' }}>-- اختر صوتاً --</option>
                                {predefinedVoices && predefinedVoices.length > 0 ? (
                                    predefinedVoices.map((v) => (
                                        <option key={v.filename} value={v.filename} style={{ backgroundColor: '#111827', color: 'white' }}>{v.display_name}</option>
                                    ))
                                ) : (
                                    <option disabled style={{ backgroundColor: '#374151', color: '#9ca3af' }}>لا توجد أصوات متاحة</option>
                                )}
                            </select>
                            <div className="absolute inset-y-0 left-0 flex items-center px-4 pointer-events-none text-gray-400 group-hover:text-indigo-400 transition-colors">
                            </div>
                        </div>
                        <div className="flex gap-2">
                             <button
                                type="button"
                                onClick={handleMultilingualLoad}
                                disabled={multilingualLoading}
                                className="flex-1 py-2 text-xs text-indigo-300 hover:text-white bg-indigo-900/30 border border-indigo-500/30 hover:bg-indigo-900/50 rounded-lg transition-all flex items-center justify-center gap-2"
                              >
                                 {multilingualLoading ? <RefreshCw className="w-3 h-3 animate-spin" /> : <Globe className="w-3 h-3" />}
                                 تفعيل تعدد اللغات
                              </button>
                        </div>
                         <div className="flex gap-2">
                            <button onClick={()=> predefinedInputRef.current?.click()} type="button" className="flex-1 py-2 text-xs text-gray-400 hover:text-indigo-400 bg-gray-800/50 border border-gray-700/50 hover:border-indigo-500/30 rounded-lg transition-all flex items-center justify-center gap-1">
                                <Upload className="w-3 h-3" /> استيراد ملفات صوتية
                            </button>
                            <input ref={predefinedInputRef} type="file" className="hidden" multiple accept=".wav,.mp3" onChange={onUploadPredefined} />
                        </div>
                    </div>
                )}

                {voiceMode === 'clone' && (
                    <div className="space-y-3 animate-slide-down">
                        <div className="relative flex-grow group">
                             <select
                                value={selectedReferenceFile}
                                onChange={(e) => setSelectedReferenceFile(e.target.value)}
                                className="w-full bg-gray-900 border border-gray-700 text-white rounded-xl px-5 py-3 focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 outline-none transition-all cursor-pointer"
                                style={{ colorScheme: 'dark', color: 'white' }}
                            >
                                <option value="none" style={{ backgroundColor: '#111827', color: 'white' }}>-- اختر ملف مرجعي --</option>
                                {referenceFiles.map((f, idx) => (
                                    <option key={idx} value={f} style={{ backgroundColor: '#111827', color: 'white' }}>{f}</option>
                                ))}
                            </select>
                         </div>
                          <div className="flex gap-2">
                             <button
                                type="button"
                                onClick={handleMultilingualLoad}
                                disabled={multilingualLoading}
                                className="flex-1 py-2 text-xs text-indigo-300 hover:text-white bg-indigo-900/30 border border-indigo-500/30 hover:bg-indigo-900/50 rounded-lg transition-all flex items-center justify-center gap-2"
                              >
                                 {multilingualLoading ? <RefreshCw className="w-3 h-3 animate-spin" /> : <Globe className="w-3 h-3" />}
                                 تفعيل تعدد اللغات
                              </button>
                        </div>
                         <div className="flex gap-2">
                             <button onClick={()=> referenceInputRef.current.click()} type="button" className="flex-1 py-2 text-xs text-gray-400 hover:text-purple-400 bg-gray-800/50 border border-gray-700/50 hover:border-purple-500/30 rounded-lg transition-all flex items-center justify-center gap-1">
                                <Upload className="w-3 h-3" /> استيراد
                            </button>
                            <input ref={referenceInputRef} type="file" className="hidden" multiple accept=".wav,.mp3" onChange={onUploadRef} />
                        </div>
                    </div>
                )}

                {voiceMode === 'lora' && (
                    <div className="space-y-3 animate-slide-down">
                        <div className="relative flex-grow group">
                            <select
                                value={selectedLora}
                                onChange={(e) => setSelectedLora(e.target.value)}
                                className="w-full bg-gradient-to-r from-purple-900/40 to-indigo-900/40 border border-purple-500/30 text-white rounded-xl px-5 py-3 focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500 outline-none transition-all cursor-pointer"
                                style={{ colorScheme: 'dark', color: 'white' }}
                            >
                                <option value="none" style={{ backgroundColor: '#111827', color: 'white' }}>-- اختر صوت LoRA --</option>
                                {loras && loras.length > 0 ? (
                                    loras.map((l) => (
                                        <option key={l.lora_id} value={l.lora_id} style={{ backgroundColor: '#111827', color: 'white' }}>{l.display_name}</option>
                                    ))
                                ) : (
                                    <option disabled style={{ backgroundColor: '#374151', color: '#9ca3af' }}>لا توجد أصوات LoRA متاحة</option>
                                )}
                            </select>
                        </div>
                        <div className="bg-purple-500/10 border border-purple-500/20 rounded-lg p-3 text-xs text-purple-300">
                            <strong>ملاحظة:</strong> أصوات LoRA هي أصوات مدربة خصيصاً. تأكد من تفعيل "تعدد اللغات" قبل الاستخدام.
                        </div>
                        <div className="flex gap-2">
                             <button
                                type="button"
                                onClick={handleMultilingualLoad}
                                disabled={multilingualLoading}
                                className="flex-1 py-2 text-xs text-purple-300 hover:text-white bg-purple-900/30 border border-purple-500/30 hover:bg-purple-900/50 rounded-lg transition-all flex items-center justify-center gap-2"
                              >
                                 {multilingualLoading ? <RefreshCw className="w-3 h-3 animate-spin" /> : <Globe className="w-3 h-3" />}
                                 تفعيل تعدد اللغات
                              </button>
                        </div>
                    </div>
                )}
             </div>

             {/* Right Column: Advanced */}
             <div className="lg:col-span-5 space-y-6">
                 {/* Language Selection */}
                <div className="bg-gray-800/20 rounded-xl p-4 border border-white/5 space-y-2">
                    <label className="text-sm font-semibold text-gray-300 block mb-2">اللغة (Language)</label>
                    <select
                        value={language}
                        onChange={(e) => {
                            const newLang = e.target.value;
                            setLanguage(newLang);
                            console.log('Language changed to:', newLang);
                            if (newLang !== 'en') {
                                // Optional: Auto-suggest or remind user
                                // toast('Remember to enable Multilingual model for best results with ' + newLang); 
                            }
                        }}
                        className="w-full bg-gray-900 border border-gray-700 text-white rounded-lg px-3 py-2 text-sm focus:border-indigo-500 outline-none cursor-pointer"
                         style={{ colorScheme: 'dark', color: 'white' }}
                    >
                        <option value="en" style={{ backgroundColor: '#111827' }}>English (en)</option>
                        <option value="ar" style={{ backgroundColor: '#111827' }}>Arabic (ar)</option>
                        <option value="es" style={{ backgroundColor: '#111827' }}>Spanish (es)</option>
                        <option value="fr" style={{ backgroundColor: '#111827' }}>French (fr)</option>
                        <option value="de" style={{ backgroundColor: '#111827' }}>German (de)</option>
                        <option value="it" style={{ backgroundColor: '#111827' }}>Italian (it)</option>
                        <option value="pt" style={{ backgroundColor: '#111827' }}>Portuguese (pt)</option>
                        <option value="zh" style={{ backgroundColor: '#111827' }}>Chinese (zh)</option>
                        <option value="ja" style={{ backgroundColor: '#111827' }}>Japanese (ja)</option>
                    </select>
                    {language !== 'en' && (
                        <p className="text-xs text-yellow-400/80 mt-1">
                            * تأكد من تفعيل "تعدد اللغات" إذا لم تكن قد فعلته.
                        </p>
                    )}
                </div>

                <label className="text-sm font-semibold text-gray-300 border-b border-gray-700/50 pb-2 block">خيارات متقدمة</label>
                
                <div className="bg-gray-800/20 rounded-xl p-4 border border-white/5 space-y-4">
                     <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-300">تقسيم النص (Chunking)</span>
                        <label className="relative inline-flex items-center cursor-pointer">
                            <input type="checkbox" checked={splitText} onChange={(e) => setSplitText(e.target.checked)} className="sr-only peer" />
                            <div className="w-9 h-5 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-indigo-600"></div>
                        </label>
                    </div>
                    {splitText && (
                        <div className="pt-2 border-t border-white/5 animate-slide-down">
                             <div className="flex justify-between mb-2">
                                <label className="text-xs text-gray-400">حجم المقطع</label>
                                <span className="text-xs font-mono text-indigo-400">{chunkSize}</span>
                            </div>
                            <input type="range" min="50" max="1000" step="10" value={chunkSize} onChange={(e)=> setChunkSize(e.target.value)} className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-indigo-500" />
                        </div>
                    )}
                </div>
                
                 {/* Seed Field */}
                <div className="bg-gray-800/20 rounded-xl p-4 border border-white/5 space-y-2">
                    <label className="text-xs text-gray-400 block">Seed (ترك 0 للعشوائي)</label>
                    <input type="number" value={seed} onChange={(e) => setSeed(e.target.value)} className="w-full bg-gray-900 border border-gray-700 text-white rounded-lg px-3 py-2 text-sm focus:border-indigo-500 outline-none" placeholder="Seed" />
                </div>
             </div>
        </div>

        {/* Bottom Parameters Accordion */}
         <details className="group bg-gray-900/30 rounded-xl border border-gray-800 overflow-hidden transition-all duration-300" open>
             <summary className="flex justify-between items-center p-4 cursor-pointer hover:bg-gray-800/30 transition-colors list-none">
                <span className="font-medium text-sm text-gray-300 flex items-center gap-2">
                    <Settings className="w-4 h-4 text-indigo-400" />
                    ضبط الأداء (Parameters)
                </span>
             </summary>
             <div className="p-5 pt-0 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 animate-slide-down">
                <div className="space-y-2">
                    <label className="text-xs text-gray-400 flex justify-between">Temperature <span className="text-indigo-400">{temperature}</span></label>
                    <input type="range" step="0.01" min="0" max="1.5" value={temperature} onChange={(e)=>setTemperature(e.target.value)} className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-indigo-500" />
                </div>
                 <div className="space-y-2">
                    <label className="text-xs text-gray-400 flex justify-between">Exaggeration <span className="text-indigo-400">{exaggeration}</span></label>
                    <input type="range" step="0.01" min="0" max="2.0" value={exaggeration} onChange={(e)=>setExaggeration(e.target.value)} className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-indigo-500" />
                </div>
                 <div className="space-y-2">
                    <label className="text-xs text-gray-400 flex justify-between">CFG Weight <span className="text-indigo-400">{cfgWeight}</span></label>
                    <input type="range" step="0.01" min="0" max="2.0" value={cfgWeight} onChange={(e)=>setCfgWeight(e.target.value)} className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-indigo-500" />
                </div>
                 <div className="space-y-2">
                    <label className="text-xs text-gray-400 flex justify-between">Speed <span className="text-indigo-400">{speedFactor}</span></label>
                    <input type="range" step="0.05" min="0.25" max="4.0" value={speedFactor} onChange={(e)=>setSpeedFactor(e.target.value)} className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-indigo-500" />
                </div>
             </div>
         </details>

      </div>
    </div>
  );
};

export default TTSForm;
