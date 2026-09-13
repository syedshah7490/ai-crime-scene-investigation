import React, { useState } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import axios from 'axios';

const API_URL =
  process.env.REACT_APP_API_URL ||
  (process.env.NODE_ENV === 'development' ? 'http://localhost:8000' : '');

const UploadEvidence = () => {
  const { caseId } = useParams();
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [evidenceType, setEvidenceType] = useState('photo');
  const [description, setDescription] = useState('');
  const [cameraId, setCameraId] = useState('');
  const [uploading, setUploading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      alert('Please select a file');
      return;
    }

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      await axios.post(
        `${API_URL}/api/upload?case_id=${caseId}&evidence_type=${evidenceType}&description=${description}&camera_id=${cameraId}`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      alert('Upload successful!');
      navigate(`/case/${caseId}`);
    } catch (err) {
      alert('Upload failed: ' + err.message);
    }
    setUploading(false);
  };

  return (
    <div>
      <header className="header">
        <div className="logo">
          <Link to={`/case/${caseId}`} style={{ color: 'white', textDecoration: 'none' }}>← Back</Link>
          <h1>Upload Evidence</h1>
        </div>
      </header>

      <div className="container" style={{ maxWidth: '600px' }}>
        <div className="card">
          <h2 className="card-title">Upload to Case {caseId}</h2>
          
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label>Select File (Photo or Video)</label>
              <input 
                type="file" 
                onChange={e => setFile(e.target.files[0])}
                accept="image/*,video/*"
                required
              />
              {file && <p style={{ marginTop: '0.5rem', fontSize: '0.85rem', color: 'var(--text-light)' }}>Selected: {file.name}</p>}
            </div>

            <div className="form-group">
              <label>Evidence Type</label>
              <select value={evidenceType} onChange={e => setEvidenceType(e.target.value)}>
                <option value="photo">Crime Scene Photo</option>
                <option value="cctv">CCTV Footage</option>
                <option value="document">Document</option>
                <option value="video">Video</option>
              </select>
            </div>

            {evidenceType === 'cctv' && (
              <div className="form-group">
                <label>Camera ID</label>
                <input 
                  type="text" 
                  value={cameraId}
                  onChange={e => setCameraId(e.target.value)}
                  placeholder="e.g., CAM_01"
                />
              </div>
            )}

            <div className="form-group">
              <label>Description</label>
              <textarea 
                value={description}
                onChange={e => setDescription(e.target.value)}
                placeholder="Brief description of this evidence..."
                rows="3"
              />
            </div>

            <button type="submit" className="btn btn-primary" disabled={uploading}>
              {uploading ? 'Uploading...' : '📤 Upload Evidence'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default UploadEvidence;