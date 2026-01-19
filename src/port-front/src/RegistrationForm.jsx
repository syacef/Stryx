import React, { useState } from 'react';
import { CloudRain, Link, Radio, Target, AlertCircle } from 'lucide-react';

const RegistrationForm = ({ sourceType, setSourceType, videoName, setVideoName, ...props }) => {
  const [urlError, setUrlError] = useState('');

  const tabs = [
    { id: 'mp4', label: 'File', icon: <CloudRain size={16} /> },
    { id: 'http', label: 'Link', icon: <Link size={16} /> },
    { id: 'rtsp', label: 'Stream', icon: <Radio size={16} /> },
  ];

  // Helper to validate URLs on the fly
  const handleUrlChange = (value, type) => {
    if (type === 'rtsp') {
      props.setRtspUrl(value);
      setUrlError(value && !value.startsWith('rtsp://') ? 'Must start with rtsp://' : '');
    } else {
      props.setHttpUrl(value);
      const isHttp = value.startsWith('http://') || value.startsWith('https://');
      setUrlError(value && !isHttp ? 'Must start with http:// or https://' : '');
    }
  };

  return (
    <div className="bg-stone-900/60 backdrop-blur-md rounded-3xl p-8 border border-stone-800 shadow-xl">
      {/* Navigation Tabs */}
      <div className="flex gap-2 mb-8 bg-stone-950 p-1.5 rounded-2xl w-fit">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => {
              setSourceType(tab.id);
              setUrlError('');
            }}
            className={`flex items-center gap-2 px-5 py-2 rounded-xl text-sm font-medium transition-all
              ${sourceType === tab.id ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-900/40' : 'text-stone-400 hover:text-stone-200'}`}
          >
            {tab.icon} {tab.label}
          </button>
        ))}
      </div>

      <div className="space-y-6">
        {/* Common Identity Field */}
        <div className="space-y-2">
          <label className="text-[10px] uppercase tracking-[0.2em] text-emerald-500 font-black ml-1">
            Feed Identity
          </label>
          <input
            type="text"
            value={videoName}
            onChange={(e) => setVideoName(e.target.value)}
            placeholder="e.g. Amazon_Canopy_West"
            className="w-full bg-stone-950 border border-stone-800 rounded-xl px-4 py-3 text-stone-200 focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 outline-none transition-all placeholder:text-stone-700"
          />
        </div>
        
        {/* Source Specific Inputs */}
        <div className="space-y-2">
          <label className="text-[10px] uppercase tracking-[0.2em] text-emerald-500 font-black ml-1">
            {sourceType === 'mp4' ? 'Video Source' : 'Connection String'}
          </label>

          {sourceType === 'mp4' && (
            <label className="flex flex-col items-center justify-center w-full h-40 border-2 border-dashed border-stone-800 rounded-2xl hover:bg-emerald-900/5 hover:border-emerald-500/50 cursor-pointer transition-all group">
              <Target className="text-stone-600 group-hover:text-emerald-500 mb-2 transition-colors" size={32} />
              <span className="text-sm text-stone-400 group-hover:text-stone-200">
                {props.file ? props.file.name : "Drop jungle footage here"}
              </span>
              <input type="file" accept="video/mp4" className="hidden" onChange={props.handleFileChange} />
            </label>
          )}

          {sourceType === 'http' && (
            <input
              type="text"
              value={props.httpUrl}
              onChange={(e) => handleUrlChange(e.target.value, 'http')}
              placeholder="https://storage.nature.com/clip.mp4"
              className={`w-full bg-stone-950 border rounded-xl px-4 py-3 text-stone-200 outline-none transition-all
                ${urlError ? 'border-red-900/50 focus:border-red-500' : 'border-stone-800 focus:border-emerald-500'}`}
            />
          )}

          {sourceType === 'rtsp' && (
            <input
              type="text"
              value={props.rtspUrl}
              onChange={(e) => handleUrlChange(e.target.value, 'rtsp')}
              placeholder="rtsp://192.168.1.50:554/live"
              className={`w-full bg-stone-950 border rounded-xl px-4 py-3 text-stone-200 outline-none transition-all
                ${urlError ? 'border-red-900/50 focus:border-red-500' : 'border-stone-800 focus:border-emerald-500'}`}
            />
          )}

          {/* Error Message */}
          {urlError && (
            <div className="flex items-center gap-2 text-red-400 text-xs mt-2 ml-1 animate-in fade-in slide-in-from-top-1">
              <AlertCircle size={14} />
              {urlError}
            </div>
          )}
        </div>
        
        {/* Action Button */}
        <button
          onClick={props.onSubmit}
          disabled={!!urlError || !videoName}
          className={`w-full font-bold py-4 rounded-xl shadow-lg transition-all flex items-center justify-center gap-2
            ${urlError || !videoName 
              ? 'bg-stone-800 text-stone-600 cursor-not-allowed opacity-50' 
              : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-900/20 active:scale-[0.98]'}`}
        >
          Activate Feed
        </button>
      </div>
    </div>
  );
};

export default RegistrationForm;
