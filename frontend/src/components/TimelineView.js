import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import axios from 'axios';

const API_URL =
  process.env.REACT_APP_API_URL ||
  (process.env.NODE_ENV === 'development' ? 'http://localhost:8000' : '');

const TimelineView = () => {
  const { caseId } = useParams();
  const [timeline, setTimeline] = useState(null);

  useEffect(() => {
    fetchTimeline();
  }, [caseId]);

  const fetchTimeline = async () => {
    try {
      const res = await axios.get(`${API_URL}/api/timeline/${caseId}`);
      setTimeline(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  if (!timeline) return <div className="container"><p>Loading timeline...</p></div>;

  return (
    <div>
      <header className="header">
        <div className="logo">
          <Link to={`/case/${caseId}`} style={{ color: 'white', textDecoration: 'none' }}>← Back</Link>
          <h1>Investigation Timeline</h1>
        </div>
      </header>

      <div className="container">
        <div className="card">
          <h2 className="card-title">Case {caseId}</h2>
          <p>{timeline.total_events} events recorded</p>
        </div>

        {timeline.total_events === 0 ? (
          <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
            <p>No timeline events yet. Upload and analyze evidence first.</p>
          </div>
        ) : (
          <div className="timeline">
            {timeline.timeline.map((event, idx) => (
              <div key={idx} className="timeline-item">
                <div className="timeline-time">
                  {new Date(event.timestamp).toLocaleString()}
                </div>
                <div className="timeline-content">
                  <p style={{ fontWeight: 500 }}>{event.description}</p>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-light)' }}>
                    Source: {event.source?.substring(0, 12)}... | 
                    Camera: {event.camera_id || 'N/A'}
                    {event.confidence && ` | Confidence: ${(event.confidence * 100).toFixed(1)}%`}
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default TimelineView;