import React, { useState, useEffect } from 'react';
import { Eye, Plus, Trees, Activity } from 'lucide-react';
import FeedCard from './FeedCard';
import RegistrationForm from './RegistrationForm'

export const API_CONFIG = {
  ingestionService: import.meta.env.VITE_INGESTION_SERVICE,
  isDev: import.meta.env.DEV,
};

function App() {
  const [activeTab, setActiveTab] = useState('feeds');
  const [sourceType, setSourceType] = useState('mp4');
  const [file, setFile] = useState(null);
  const [rtspUrl, setRtspUrl] = useState('');
  const [httpUrl, setHttpUrl] = useState('');
  const [videoName, setVideoName] = useState('');
  const [status, setStatus] = useState('idle');
  const [message, setMessage] = useState('');
  const [progress, setProgress] = useState(0);
  const [feeds, setFeeds] = useState([]);
  const [loadingFeeds, setLoadingFeeds] = useState(false);
  const [selectedFeed, setSelectedFeed] = useState(null);

  useEffect(() => {
    if (activeTab === 'feeds') {
      fetchFeeds();
    }
  }, [activeTab]);

  const fetchFeeds = async () => {
    setLoadingFeeds(true);
    try {
      const response = await fetch(`${API_CONFIG.ingestionService}/streams`);
      if (!response.ok) throw new Error('Failed to fetch feeds');
      const data = await response.json();
      setFeeds(data || []);
    } catch (error) {
      console.error('Error fetching feeds:', error);
      setFeeds([]);
    } finally {
      setLoadingFeeds(false);
    }
  };

  const deleteFeed = async (feedId) => {
    try {
      const response = await fetch(`${API_CONFIG.ingestionService}/streams/${feedId}`, {
        method: 'DELETE'
      });
      if (!response.ok) throw new Error('Failed to delete feed');
      fetchFeeds();
      if (selectedFeed?.id === feedId) {
        setSelectedFeed(null);
      }
    } catch (error) {
      console.error('Error deleting feed:', error);
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type === 'video/mp4') {
      setFile(selectedFile);
      setVideoName(selectedFile.name.replace('.mp4', ''));
      setStatus('idle');
      setMessage('');
    } else {
      setMessage('Please select a valid MP4 file');
      setStatus('error');
    }
  };

  const uploadMP4AndRegister = async () => {
    if (!file || !videoName.trim()) {
      setMessage('Please select a file and provide a name');
      setStatus('error');
      return;
    }

    setStatus('uploading');
    setMessage('Uploading video to ingestion service...');
    setProgress(0);

    try {
      const formData = new FormData();
      formData.append('video', file);
      formData.append('name', videoName);

      const response = await fetch(`${API_CONFIG.ingestionService}/streams/upload`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Service error: ${response.statusText}`);
      }

      const result = await response.json();

      setProgress(100);
      setStatus('success');
      setMessage(`Video processed successfully! Stream ID: ${result.streamId || result.id || 'N/A'}`);

      setTimeout(() => {
        setFile(null);
        setVideoName('');
        setProgress(0);
        setStatus('idle');
        setMessage('');
      }, 3000);

    } catch (error) {
      setStatus('error');
      setMessage(`Error: ${error.message}`);
      setProgress(0);
    }
  };

  const registerRTSP = async () => {
    if (!rtspUrl.trim() || !videoName.trim()) {
      setMessage('Please provide both RTSP URL and video name');
      setStatus('error');
      return;
    }

    if (!rtspUrl.startsWith('rtsp://')) {
      setMessage('Please provide a valid RTSP URL (must start with rtsp://)');
      setStatus('error');
      return;
    }

    setStatus('registering');
    setMessage('Registering RTSP stream...');

    try {
      const response = await fetch(`${API_CONFIG.ingestionService}/streams/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          name: videoName,
          rtspUrl: rtspUrl,
          source: 'direct'
        })
      });

      if (!response.ok) {
        throw new Error(`Registration failed: ${response.statusText}`);
      }

      const result = await response.json();

      setStatus('success');
      setMessage(`RTSP stream registered successfully! Stream ID: ${result.streamId || 'N/A'}`);

      setTimeout(() => {
        setRtspUrl('');
        setVideoName('');
        setStatus('idle');
        setMessage('');
      }, 3000);

    } catch (error) {
      setStatus('error');
      setMessage(`Error: ${error.message}`);
    }
  };

  const registerSource = async (url, sourceLabel) => {
    setStatus('registering');
    setMessage(`Connecting to ${sourceLabel}...`);

    try {
      const response = await fetch(`${API_CONFIG.ingestionService}/streams/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: videoName,
          rtsp_url: url, // Backend handles relaying if this is http/https
        })
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Registration failed');
      }

      const result = await response.json();
      setStatus('success');
      setMessage(`Stream active! ID: ${result.stream_id}`);

    } catch (error) {
      setStatus('error');
      setMessage(error.message);
    }
  };

  const handleSubmit = async () => {
    if (sourceType === 'mp4') {
      await uploadMP4AndRegister();
    } else if (sourceType === 'http') {
      await registerSource(httpUrl, 'HTTP Link');
    } else {
      await registerSource(rtspUrl, 'RTSP Stream');
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'uploading':
      case 'registering':
        return <Loader2 className="animate-spin text-emerald-400" size={20} />;
      case 'success':
        return <CheckCircle className="text-emerald-400" size={20} />;
      case 'error':
        return <XCircle className="text-red-400" size={20} />;
      default:
        return <AlertCircle className="text-gray-400" size={20} />;
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0f0d] text-stone-300 font-sans selection:bg-emerald-500/30">
      {/* Sidebar / Nav */}
      <nav className="fixed left-0 top-0 h-full w-20 flex flex-col items-center py-8 bg-stone-950 border-r border-stone-900 z-50">
        <div className="mb-12 text-emerald-500">
          <Trees size={32} weight="fill" />
        </div>
        <div className="flex flex-col gap-8">
          <button
            onClick={() => setActiveTab('feeds')}
            className={`p-3 rounded-2xl transition-all ${activeTab === 'feeds' ? 'bg-emerald-500/10 text-emerald-500' : 'text-stone-600 hover:text-stone-400'}`}
          >
            <Eye size={24} />
          </button>
          <button
            onClick={() => setActiveTab('register')}
            className={`p-3 rounded-2xl transition-all ${activeTab === 'register' ? 'bg-emerald-500/10 text-emerald-500' : 'text-stone-600 hover:text-stone-400'}`}
          >
            <Plus size={24} />
          </button>
        </div>
      </nav>

      <main className="pl-28 pr-8 py-12 max-w-7xl mx-auto">
        <header className="mb-12">
          <h1 className="text-4xl font-black text-white tracking-tight flex items-center gap-3">
            Stryx <span className="text-emerald-500 text-sm bg-emerald-500/10 px-3 py-1 rounded-full uppercase tracking-tighter">Wild-Life v2</span>
          </h1>
          <p className="text-stone-500 mt-2">Monitoring the digital ecosystem.</p>
        </header>

        {activeTab === 'feeds' ? (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-1 space-y-4 overflow-y-auto max-h-[70vh] pr-2 custom-scrollbar">
              {feeds.map(feed => (
                <FeedCard
                  key={feed.stream_id}
                  feed={feed}
                  isSelected={selectedFeed?.stream_id === feed.stream_id}
                  onSelect={setSelectedFeed}
                  onDelete={() => deleteFeed(feed.stream_id)}
                />
              ))}
            </div>
            <div className="lg:col-span-2">
              {/* Modern Video Player Placeholder */}
              <div className="aspect-video bg-stone-950 rounded-3xl border border-stone-800 overflow-hidden flex items-center justify-center group relative">
                {selectedFeed ? (
                  <video src={selectedFeed.public_url} controls className="w-full h-full object-cover" />
                ) : (
                  <div className="text-center">
                    <Activity className="mx-auto text-stone-800 mb-4 animate-pulse" size={48} />
                    <p className="text-stone-600 font-medium">Select a pulse to monitor</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="max-w-xl">
            <RegistrationForm
              sourceType={sourceType}
              setSourceType={setSourceType}
              videoName={videoName}
              setVideoName={setVideoName}
              httpUrl={httpUrl}
              setHttpUrl={setHttpUrl}
              rtspUrl={rtspUrl}
              setRtspUrl={setRtspUrl}
              file={file}
              handleFileChange={handleFileChange}
              onSubmit={handleSubmit}
            />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
