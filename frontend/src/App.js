import { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [alerts, setAlerts] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentDescription, setCurrentDescription] = useState('Waiting...');
  const [cameras, setCameras] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const videoRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);

  // Fetch available cameras on load
  useEffect(() => {
    fetch('http://localhost:8000/api/cameras')
      .then(res => res.json())
      .then(data => setCameras(data.cameras))
      .catch(err => console.error('Cannot fetch cameras:', err));
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      });
      
      videoRef.current.srcObject = stream;
      streamRef.current = stream;
      
      // Connect WebSocket
      wsRef.current = new WebSocket('ws://localhost:8000/ws/camera');
      
      wsRef.current.onopen = () => {
        console.log('✓ Connected to backend');
        setConnectionStatus('connected');
        setIsStreaming(true);
        startFrameCapture();
      };
      
      wsRef.current.onmessage = (event) => {
        const result = JSON.parse(event.data);
        setCurrentDescription(result.description);
        
        if (result.is_threat) {
          const frameTime = new Date(result.frame_timestamp);
          const analysisTime = new Date(result.analysis_timestamp);
          
          const alert = {
            id: Date.now(),
            description: result.description,
            frameTime: frameTime.toLocaleTimeString('en-US', { 
              hour12: false, 
              hour: '2-digit', 
              minute: '2-digit', 
              second: '2-digit' 
            }),
            frameDate: frameTime.toLocaleDateString('en-US', {
              month: 'short',
              day: 'numeric'
            }),
            analysisTime: analysisTime.toLocaleTimeString('en-US', { 
              hour12: false, 
              hour: '2-digit', 
              minute: '2-digit', 
              second: '2-digit' 
            }),
            frameTimestamp: result.frame_timestamp  // For investigation
          };
          
          setAlerts(prev => [alert, ...prev].slice(0, 20));
          
          // Play alert sound
          try {
            const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIF2i76+adTRALUKXh8LdjGwU2jdXty3cqBSJ1xe/ekjwJE12y5+qnUw0JQ5zd8r9pHQUsgs/z2Ik2CBdou+vmnU0QC0+j4O+2YxsFNo3V7Mt3KgUidb/u3pI8CRNcsuXqp1INB0Gb3PG+aR0FKn/O8tiKNQgXZ7rq551NEApPo+DvtmIbBTWM1OvKdikEIXS+7t6ROggSXLDk6aZRDAdCmtvxvWkcBSl+zfLXijQIF2a56+ecTQ8KUKL');
            audio.play().catch(e => console.log('Audio play failed'));
          } catch (e) {
            console.log('Audio error:', e);
          }
        }
      };
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('error');
        alert('Backend connection failed. Is it running on port 8000?');
      };
      
      wsRef.current.onclose = () => {
        console.log('WebSocket closed');
        setConnectionStatus('disconnected');
      };
      
    } catch (err) {
      console.error('Camera error:', err);
      alert('Cannot access camera. Please allow camera permissions.');
    }
  };

  const startFrameCapture = () => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    intervalRef.current = setInterval(() => {
      if (!videoRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        return;
      }
      
      canvas.width = videoRef.current.videoWidth || 640;
      canvas.height = videoRef.current.videoHeight || 480;
      ctx.drawImage(videoRef.current, 0, 0);
      
      canvas.toBlob((blob) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64 = reader.result.split(',')[1];
          
          // Send frame with accurate timestamp for camera investigation
          wsRef.current.send(JSON.stringify({ 
            frame: base64,
            timestamp: new Date().toISOString()
          }));
        };
        reader.readAsDataURL(blob);
      }, 'image/jpeg', 0.7);
      
    }, 1000); // Send 1 frame per second
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    if (wsRef.current) {
      wsRef.current.close();
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    setIsStreaming(false);
    setConnectionStatus('disconnected');
    setCurrentDescription('Stopped');
  };

  const copyTimestamp = (timestamp) => {
    navigator.clipboard.writeText(timestamp).then(() => {
      alert('Timestamp copied to clipboard!\nUse this to find the incident in camera footage.');
    }).catch(err => {
      console.error('Copy failed:', err);
    });
  };

  return (
    <div className="App">
      <header className="header">
        <h1>🎥 CCTV Surveillance System</h1>
        <p>AI-Powered Real-Time Threat Detection</p>
        <div className={`connection-badge ${connectionStatus}`}>
          <span className="badge-dot"></span>
          {connectionStatus === 'connected' ? 'Connected' : 
           connectionStatus === 'error' ? 'Connection Error' : 'Disconnected'}
        </div>
      </header>
      
      <div className="container">
        <div className="video-section">
          <div className="video-container">
            <video 
              ref={videoRef} 
              autoPlay 
              playsInline 
              muted
              className="video-feed"
            />
            {!isStreaming && (
              <div className="video-overlay">
                <p>Camera feed will appear here</p>
              </div>
            )}
          </div>
          
          <div className="controls">
            {!isStreaming ? (
              <button onClick={startCamera} className="btn btn-start">
                🎥 Start Camera
              </button>
            ) : (
              <button onClick={stopCamera} className="btn btn-stop">
                ⏹ Stop Camera
              </button>
            )}
          </div>
          
          <div className={`status-bar ${isStreaming ? 'active' : ''}`}>
            <div className="status-indicator"></div>
            <p className="status-text">{currentDescription}</p>
          </div>
        </div>
        
        <div className="alerts-section">
          <h2>🚨 Threat Alerts</h2>
          <div className="alerts-count">
            {alerts.length} {alerts.length === 1 ? 'Alert' : 'Alerts'}
          </div>
          
          <div className="alerts-list">
            {alerts.length === 0 ? (
              <div className="no-alerts">
                <div className="no-alerts-icon">✅</div>
                <p>No threats detected</p>
                <small>System is monitoring in real-time</small>
              </div>
            ) : (
              alerts.map(alert => (
                <div key={alert.id} className="alert-item">
                  <div className="alert-header">
                    <div className="alert-date-badge">{alert.frameDate}</div>
                    <button 
                      className="copy-btn" 
                      onClick={() => copyTimestamp(alert.frameTimestamp)}
                      title="Copy timestamp for camera investigation"
                    >
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
            )}
          </div>
          
          {alerts.length > 0 && (
            <button 
              className="clear-alerts-btn"
              onClick={() => setAlerts([])}
            >
              Clear All Alerts
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
