import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import axios from 'axios';

const API_URL =
  process.env.REACT_APP_API_URL ||
  (process.env.NODE_ENV === 'development' ? 'http://localhost:8000' : '');

const CaseDetail = () => {
  const { caseId } = useParams();
  const [caseData, setCaseData] = useState(null);
  const [report, setReport] = useState(null);

  useEffect(() => {
    fetchCase();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [caseId]);

  const fetchCase = async () => {
    try {
      const res = await axios.get(`${API_URL}/api/cases/${caseId}`);
      setCaseData(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  const generateReport = async () => {
    try {
      const res = await axios.post(`${API_URL}/api/report/generate`, {
        case_id: caseId,
        include_rag: true
      });
      setReport(res.data.report);
      alert('Report generated!');
    } catch (err) {
      alert('Error: ' + err.message);
    }
  };

  if (!caseData) return <div className="container"><p>Loading...</p></div>;

  return (
    <div>
      <header className="header">
        <div className="logo">
          <Link to="/" style={{ color: 'white', textDecoration: 'none' }}>← Back</Link>
          <h1>{caseData.case_name}</h1>
        </div>
        <div className="header-actions">
          <button className="btn btn-primary" onClick={generateReport}>📄 Generate Report</button>
        </div>
      </header>

      <div className="container">
        <div className="card">
          <h2 className="card-title">Case Information</h2>
          <p><strong>ID:</strong> {caseData.case_id}</p>
          <p><strong>Investigator:</strong> {caseData.investigator}</p>
          <p><strong>Status:</strong> <span className={`badge badge-${caseData.status.toLowerCase()}`}>{caseData.status}</span></p>
          <p><strong>Created:</strong> {new Date(caseData.created_at).toLocaleString()}</p>
        </div>

        <h2 style={{ margin: '1.5rem 0 1rem', color: 'var(--primary)' }}>Evidence Items</h2>
        
        {caseData.evidence?.length === 0 ? (
          <div className="card" style={{ textAlign: 'center', padding: '2rem' }}>
            <p>No evidence yet. <Link to={`/upload/${caseId}`}>Upload now</Link></p>
          </div>
        ) : (
          <div className="grid">
            {caseData.evidence.map(ev => (
              <div key={ev.evidence_id} className="card">
                <h3 style={{ fontSize: '1rem', marginBottom: '0.5rem' }}>{ev.original_name}</h3>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-light)' }}>{ev.evidence_id}</p>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-light)' }}>Type: {ev.type} | Camera: {ev.camera_id || 'N/A'}</p>
                
                {/* Show image with detection boxes */}
                <div style={{ marginTop: '1rem', textAlign: 'center', background: '#f0f0f0', padding: '1rem', borderRadius: '8px' }}>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-light)', marginBottom: '0.5rem' }}>
                    📷 Evidence Photo
                  </p>
                  <img 
                    src={`${API_URL}/api/evidence/image/${ev.evidence_id}?case_id=${caseId}&annotated=true`}
                    alt="Evidence with detection"
                    style={{ maxWidth: '100%', maxHeight: '300px', borderRadius: '8px', border: '2px solid #ddd' }}
                    onError={(e) => {
                      e.target.src = `${API_URL}/api/evidence/image/${ev.evidence_id}?case_id=${caseId}&annotated=false`;
                    }}
                  />
                </div>
                
                {ev.analysis && (
                  <div style={{ marginTop: '1rem', padding: '0.75rem', background: '#f8f9fa', borderRadius: '6px' }}>
                    <p style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
                      🔍 AI Detection Results:
                    </p>
                    {ev.analysis.detections && (
                      <div>
                        <p style={{ fontSize: '0.8rem', marginBottom: '0.5rem' }}>
                          <strong>{ev.analysis.detections.total_objects}</strong> objects detected
                        </p>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                          {ev.analysis.detections.detections.map((det, idx) => (
                            <span key={idx} style={{ 
                              display: 'inline-flex', 
                              alignItems: 'center',
                              gap: '0.25rem',
                              padding: '0.4rem 0.8rem', 
                              background: det.is_priority ? '#f8d7da' : '#d4edda',
                              color: det.is_priority ? '#721c24' : '#155724',
                              borderRadius: '20px',
                              fontSize: '0.8rem',
                              fontWeight: 500,
                              border: `2px solid ${det.is_priority ? '#dc3545' : '#28a745'}`
                            }}>
                              {det.is_priority ? '🔴' : '🟢'} {det.class} ({(det.confidence * 100).toFixed(0)}%)
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                    {ev.analysis.ocr?.full_text && (
                      <div style={{ marginTop: '0.75rem', padding: '0.5rem', background: 'white', borderRadius: '4px' }}>
                        <p style={{ fontSize: '0.75rem', color: 'var(--text-light)', marginBottom: '0.25rem' }}>📝 OCR Text:</p>
                        <p style={{ fontSize: '0.85rem', fontStyle: 'italic' }}>"{ev.analysis.ocr.full_text}"</p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {report && (
          <div className="card" style={{ marginTop: '2rem' }}>
            <h2 className="card-title">Generated Report</h2>
            <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit', fontSize: '0.9rem', lineHeight: 1.6 }}>
              {report.markdown_report}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};

export default CaseDetail;