import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  // ── Camera state ──────────────────────────────────────
  const [alerts, setAlerts]             = useState([]);
  const [isStreaming, setIsStreaming]   = useState(false);
  const [currentDescription, setCurrentDescription] = useState('Waiting...');
  const [connectionStatus, setConnectionStatus]     = useState('disconnected');

  // ── Search state ──────────────────────────────────────
  const [activeTab, setActiveTab]       = useState('live');   // 'live' | 'search'
  const [searchQuery, setSearchQuery]   = useState('');
  const [searchWindow, setSearchWindow] = useState(0);
  const [searchResults, setSearchResults] = useState(null);  // null = not searched yet
  const [isSearching, setIsSearching]   = useState(false);
  const [logStats, setLogStats]         = useState({ total: 0, threats: 0, normal: 0 });

  // ── Refs ──────────────────────────────────────────────
  const videoRef    = useRef(null);
  const canvasRef   = useRef(null);
  const wsRef       = useRef(null);
  const streamRef   = useRef(null);
  const intervalRef = useRef(null);

  // Fetch stats on load and every 10s
  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 10000);
    return () => clearInterval(interval);
  }, []);

  const fetchStats = () => {
    fetch('http://localhost:8000/api/logs/stats')
      .then(r => r.json())
      .then(setLogStats)
      .catch(() => {});
  };

  // ── Camera functions ───────────────────────────────────
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });
      videoRef.current.srcObject = stream;
      streamRef.current = stream;

      wsRef.current = new WebSocket('ws://localhost:8000/ws/camera');

      wsRef.current.onopen = () => {
        setConnectionStatus('connected');
        setIsStreaming(true);
        startFrameCapture();
      };

      wsRef.current.onmessage = (event) => {
        const result = JSON.parse(event.data);
        setCurrentDescription(result.description);

        if (result.processed_frame && canvasRef.current) {
          const img  = new Image();
          img.onload = () => {
            const ctx = canvasRef.current.getContext('2d');
            ctx.drawImage(img, 0, 0, canvasRef.current.width, canvasRef.current.height);
          };
          img.src = `data:image/jpeg;base64,${result.processed_frame}`;
        }

        if (result.is_threat) {
          const frameTime    = new Date(result.frame_timestamp);
          const analysisTime = new Date(result.analysis_timestamp);
          setAlerts(prev => [{
            id:             Date.now(),
            description:    result.description,
            frameTime:      frameTime.toLocaleTimeString('en-US', { hour12: false }),
            frameDate:      frameTime.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
            analysisTime:   analysisTime.toLocaleTimeString('en-US', { hour12: false }),
            frameTimestamp: result.frame_timestamp
          }, ...prev].slice(0, 20));
        }

        // Refresh stats after each analysis
        fetchStats();
      };

      wsRef.current.onerror = () => setConnectionStatus('error');
      wsRef.current.onclose = () => setConnectionStatus('disconnected');

    } catch (err) {
      alert('Cannot access camera. Please allow camera permissions.');
    }
  };

  const startFrameCapture = () => {
    const canvas = document.createElement('canvas');
    const ctx    = canvas.getContext('2d');
    intervalRef.current = setInterval(() => {
      if (!videoRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
      canvas.width  = videoRef.current.videoWidth  || 640;
      canvas.height = videoRef.current.videoHeight || 480;
      ctx.drawImage(videoRef.current, 0, 0);
      canvas.toBlob((blob) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          wsRef.current.send(JSON.stringify({
            frame:     reader.result.split(',')[1],
            timestamp: new Date().toISOString()
          }));
        };
        reader.readAsDataURL(blob);
      }, 'image/jpeg', 0.7);
    }, 1000);
  };

  const stopCamera = () => {
    if (streamRef.current)   streamRef.current.getTracks().forEach(t => t.stop());
    if (wsRef.current)       wsRef.current.close();
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (canvasRef.current) {
      canvasRef.current.getContext('2d').clearRect(0, 0, 640, 480);
    }
    setIsStreaming(false);
    setConnectionStatus('disconnected');
    setCurrentDescription('Stopped');
  };

  const copyTimestamp = (ts) => {
    navigator.clipboard.writeText(ts)
      .then(() => alert('Timestamp copied!\nUse this to find the incident in camera footage.'))
      .catch(() => {});
  };

  // ── Search functions ───────────────────────────────────
  const handleSearch = async (e) => {
    e?.preventDefault();
    if (!searchQuery.trim()) return;

    setIsSearching(true);
    setSearchResults(null);

    try {
      const params = new URLSearchParams({
        q:      searchQuery.trim(),
        window: searchWindow
      });
      const res  = await fetch(`http://localhost:8000/api/logs/search?${params}`);
      const data = await res.json();
      setSearchResults(data);
    } catch (err) {
      alert('Search failed. Is backend running?');
    } finally {
      setIsSearching(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') handleSearch();
  };

  const formatTimestamp = (isoString) => {
    try {
      const d = new Date(isoString);
      return {
        date: d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
        time: d.toLocaleTimeString('en-US', { hour12: false })
      };
    } catch {
      return { date: '—', time: isoString };
    }
  };

  const highlightQuery = (text, query) => {
    if (!query.trim()) return text;
    const parts = text.split(new RegExp(`(${query})`, 'gi'));
    return parts.map((part, i) =>
      part.toLowerCase() === query.toLowerCase()
        ? <mark key={i} className="highlight">{part}</mark>
        : part
    );
  };

  const TIME_WINDOWS = [
    { label: 'All Time',    value: 0    },
    { label: 'Last 5 min',  value: 5    },
    { label: 'Last 15 min', value: 15   },
    { label: 'Last 30 min', value: 30   },
    { label: 'Last 1 hr',   value: 60   },
    { label: 'Last 3 hrs',  value: 180  },
    { label: 'Last 24 hrs', value: 1440 },
  ];

  // ── Render ─────────────────────────────────────────────
  return (
    <div className="App">
      <header className="header">
        <div className="header-left">
          <h1>🎥 CCTV Surveillance System</h1>
          <p>AI-Powered Real-Time Threat Detection — Qwen3-VL-8B</p>
        </div>

        <div className="header-right">
          <div className="stats-bar">
            <div className="stat-item">
              <span className="stat-value">{logStats.total}</span>
              <span className="stat-label">Total</span>
            </div>
            <div className="stat-item threat">
              <span className="stat-value">{logStats.threats}</span>
              <span className="stat-label">Threats</span>
            </div>
            <div className="stat-item safe">
              <span className="stat-value">{logStats.normal}</span>
              <span className="stat-label">Safe</span>
            </div>
          </div>

          <div className={`connection-badge ${connectionStatus}`}>
            <span className="badge-dot"></span>
            {connectionStatus === 'connected' ? 'Live'
             : connectionStatus === 'error'   ? 'Error'
             : 'Offline'}
          </div>
        </div>
      </header>

      {/* Tab Navigation */}
      <div className="tab-nav">
        <button
          className={`tab-btn ${activeTab === 'live' ? 'active' : ''}`}
          onClick={() => setActiveTab('live')}
        >
          📹 Live Monitor
        </button>
        <button
          className={`tab-btn ${activeTab === 'search' ? 'active' : ''}`}
          onClick={() => setActiveTab('search')}
        >
          🔍 Log Search
          {logStats.total > 0 && (
            <span className="tab-badge">{logStats.total}</span>
          )}
        </button>
      </div>

      {/* ── LIVE TAB ── */}
      {activeTab === 'live' && (
        <div className="container">
          <div className="video-section">
            <video ref={videoRef} autoPlay playsInline muted style={{ display: 'none' }} />

            <div className="video-container">
              <canvas ref={canvasRef} width={640} height={480} className="video-feed" />
              {!isStreaming && (
                <div className="video-overlay">
                  <p>📷 Camera feed will appear here</p>
                </div>
              )}
            </div>

            <div className="controls">
              {!isStreaming
                ? <button onClick={startCamera} className="btn btn-start">🎥 Start Camera</button>
                : <button onClick={stopCamera}  className="btn btn-stop">⏹ Stop Camera</button>
              }
            </div>

            <div className={`status-bar ${isStreaming ? 'active' : ''}`}>
              <div className="status-indicator"></div>
              <p className="status-text">{currentDescription}</p>
            </div>
          </div>

          <div className="alerts-section">
            <h2>🚨 Threat Alerts</h2>
            <div className="alerts-count">{alerts.length} {alerts.length === 1 ? 'Alert' : 'Alerts'} (this session)</div>

            <div className="alerts-list">
              {alerts.length === 0
                ? (
                  <div className="no-alerts">
                    <div className="no-alerts-icon">✅</div>
                    <p>No threats detected</p>
                    <small>System is monitoring in real-time</small>
                  </div>
                )
                : alerts.map(alert => (
                  <div key={alert.id} className="alert-item">
                    <div className="alert-header">
                      <div className="alert-date-badge">{alert.frameDate}</div>
                      <button className="copy-btn" onClick={() => copyTimestamp(alert.frameTimestamp)}>
                        📋 Copy
                      </button>
                    </div>
                    <div className="alert-time-row">
                      <div className="alert-time-item">
                        <span className="time-label">📹 Captured</span>
                        <span className="time-value">{alert.frameTime}</span>
                      </div>
                      <div className="alert-time-item">
                        <span className="time-label">🚨 Detected</span>
                        <span className="time-value">{alert.analysisTime}</span>
                      </div>
                    </div>
                    <div className="alert-description">{alert.description}</div>
                  </div>
                ))
              }
            </div>

            {alerts.length > 0 && (
              <button className="clear-alerts-btn" onClick={() => setAlerts([])}>
                Clear Session Alerts
              </button>
            )}
          </div>
        </div>
      )}

      {/* ── SEARCH TAB ── */}
      {activeTab === 'search' && (
        <div className="search-page">

          {/* Search Bar */}
          <div className="search-header">
            <div className="search-bar-row">
              <div className="search-input-wrap">
                <span className="search-icon">🔍</span>
                <input
                  type="text"
                  className="search-input"
                  placeholder="Search logs... e.g. screwdriver, gun, running"
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  onKeyDown={handleKeyDown}
                  autoFocus
                />
                {searchQuery && (
                  <button className="clear-input-btn" onClick={() => { setSearchQuery(''); setSearchResults(null); }}>
                    ✕
                  </button>
                )}
              </div>
              <button
                className="btn btn-search"
                onClick={handleSearch}
                disabled={isSearching || !searchQuery.trim()}
              >
                {isSearching ? '⏳ Searching...' : 'Search'}
              </button>
            </div>

            {/* Time Window Chips */}
            <div className="window-chips">
              <span className="chips-label">Time window:</span>
              {TIME_WINDOWS.map(w => (
                <button
                  key={w.value}
                  className={`chip ${searchWindow === w.value ? 'active' : ''}`}
                  onClick={() => setSearchWindow(w.value)}
                >
                  {w.label}
                </button>
              ))}
            </div>
          </div>

          {/* Quick Search Tags */}
          <div className="quick-tags">
            <span className="tags-label">Quick search:</span>
            {['gun', 'knife', 'screwdriver', 'running', 'suspicious', 'fire', 'fighting', 'theft'].map(tag => (
              <button
                key={tag}
                className="quick-tag"
                onClick={() => { setSearchQuery(tag); setTimeout(handleSearch, 50); }}
              >
                {tag}
              </button>
            ))}
          </div>

          {/* Results */}
          <div className="search-results">

            {/* Not searched yet */}
            {searchResults === null && !isSearching && (
              <div className="search-empty">
                <div className="search-empty-icon">📋</div>
                <p>Search the surveillance log</p>
                <small>
                  {logStats.total > 0
                    ? `${logStats.total} entries logged (${logStats.threats} threats)`
                    : 'No logs yet — start the camera to begin logging'}
                </small>
              </div>
            )}

            {/* Loading */}
            {isSearching && (
              <div className="search-empty">
                <div className="spinner"></div>
                <p>Searching {logStats.total} log entries...</p>
              </div>
            )}

            {/* Results found */}
            {searchResults !== null && !isSearching && (
              <>
                <div className="results-meta">
                  {searchResults.count > 0
                    ? <>Found <strong>{searchResults.count}</strong> result{searchResults.count !== 1 ? 's' : ''} for <strong>"{searchResults.query}"</strong>
                        {searchResults.window_minutes > 0 && ` in last ${searchResults.window_minutes} minutes`}</>
                    : <>No results for <strong>"{searchResults.query}"</strong>
                        {searchResults.window_minutes > 0 && ` in last ${searchResults.window_minutes} minutes`}</>
                  }
                </div>

                {searchResults.count === 0 && (
                  <div className="search-empty">
                    <div className="search-empty-icon">🔎</div>
                    <p>No matches found</p>
                    <small>Try a different keyword or expand the time window</small>
                  </div>
                )}

                {searchResults.results.map((entry, idx) => {
                  const { date, time } = formatTimestamp(entry.frame_timestamp);
                  return (
                    <div key={idx} className={`log-entry ${entry.is_threat ? 'threat' : 'safe'}`}>
                      <div className="log-entry-left">
                        <div className="log-icon">{entry.is_threat ? '🚨' : '✅'}</div>
                      </div>

                      <div className="log-entry-body">
                        <div className="log-description">
                          {highlightQuery(entry.description, searchResults.query)}
                        </div>
                        <div className="log-meta">
                          <span className="log-date">📅 {date}</span>
                          <span className="log-time">🕐 {time}</span>
                          {entry.frames_analyzed && (
                            <span className="log-frames">🎞 {entry.frames_analyzed} frames</span>
                          )}
                          <span className={`log-badge ${entry.is_threat ? 'threat' : 'safe'}`}>
                            {entry.is_threat ? 'THREAT' : 'SAFE'}
                          </span>
                        </div>
                      </div>

                      <div className="log-entry-right">
                        <button
                          className="copy-btn-sm"
                          onClick={() => copyTimestamp(entry.frame_timestamp)}
                          title="Copy timestamp for camera investigation"
                        >
                          📋
                        </button>
                      </div>
                    </div>
                  );
                })}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
