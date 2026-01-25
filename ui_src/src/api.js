import axios from 'axios';

// Create axios instance with base URL if needed, or relative path
const api = axios.create({
  baseURL: '/', // Relative path since we serve from same origin
});

export const getInitialData = async () => {
  const response = await api.get('/api/ui/initial-data');
  return response.data;
};

export const generateTTS = async (data) => {
  return await api.post('/tts', data);
};

export const saveSettings = async (settings) => {
  return await api.post('/save_settings', settings);
};

export const resetSettings = async () => {
  return await api.post('/reset_settings');
};

export const getReferenceFiles = async () => {
  const response = await api.get('/get_reference_files');
  return response.data;
};

export const getPredefinedVoices = async () => {
  const response = await api.get('/get_predefined_voices');
  return response.data;
};

export const uploadReference = async (formData) => {
   return await api.post('/upload_reference', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
   });
};

export const uploadPredefined = async (formData) => {
   return await api.post('/upload_predefined_voice', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
   });
};

export const loadMultilingualModel = async () => {
    return await api.post('/load_multilingual_model');
};
