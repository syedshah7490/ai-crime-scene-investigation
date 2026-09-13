import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';

const API_URL =
  process.env.REACT_APP_API_URL ||
  (process.env.NODE_ENV === 'development' ? 'http://localhost:8000' : '');

const Dashboard = () => {
  const [cases, setCases] = useState([]);
  const [loading, setLoading] = useState(true);
  const [newCase, setNewCase] = useState({ case_name: '', description: '', investigator: '' });

  useEffect(() => {
    fetchCases();
  }, []);

  const fetchCases = async () => {
    try {
      const res = await axios.get(`${API_URL}/api/cases`);
      setCases(res.data);
    } catch (err) {
      console.error('Error fetching cases:', err);
    }
    setLoading(false);
  };

  const createCase = async (e) => {
    e.preventDefault();
    try {
      await axios.post(`${API_URL}/api/cases`, newCase);
      setNewCase({ case_name: '', description: '', investigator: '' });
      fetchCases();
    } catch (err) {
      alert('Error creating case: ' + err.message);
    }
  };

  const runDemo = async () => {
    try {
      const res = await axios.post(`${API_URL}/api/demo/run-pipeline`);
      alert(`Demo completed! Case ID: ${res.data.case_id}`);
      fetchCases();
    } catch (err) {
      alert('Demo failed: ' + err.message);
    }
  };

  return (
    <div>
      <header className="header">
        <div className="logo">
          <h1>🔍 AI Crime Scene Investigation</h1>
        </div>
        <div className="header-actions">
          <button className="btn btn-secondary" onClick={runDemo}>
            ▶ Run Demo
          </button>
        </div>
      </header>

      <div className="container">
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Create New Case</h2>
          </div>
          <form onSubmit={createCase}>
            <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))' }}>
              <div className="form-group">
                <label>Case Name</label>
                <input 
                  type="text" 
                  value={newCase.case_name}
                  onChange={e => setNewCase({...newCase, case_name: e.target.value})}
                  placeholder="e.g., Bank Robbery"
                  required
                />
              </div>
              <div className="form-group">
                <label>Investigator</label>
                <input 
                  type="text" 
                  value={newCase.investigator}
                  onChange={e => setNewCase({...newCase, investigator: e.target.value})}
                  placeholder="e.g., Officer ZH"
                  required
                />
              </div>
            </div>
            <div className="form-group">
              <label>Description</label>
              <textarea 
                value={newCase.description}
                onChange={e => setNewCase({...newCase, description: e.target.value})}
                placeholder="Brief description of the case..."
                rows="2"
              />
            </div>
            <button type="submit" className="btn btn-primary">+ Create Case</button>
          </form>
        </div>

        <h2 style={{ marginBottom: '1rem', color: 'var(--primary)' }}>Investigation Cases</h2>
        
        {loading ? (
          <p>Loading...</p>
        ) : cases.length === 0 ? (
          <div className="card" style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-light)' }}>
            <p>No cases yet. Create one above or run the demo.</p>
          </div>
        ) : (
          <div className="grid">
            {cases.map(c => (
              <div key={c.case_id} className="card">
                <div className="card-header">
                  <h3 style={{ fontSize: '1.1rem', color: 'var(--primary)' }}>{c.case_name}</h3>
                  <span className={`badge badge-${c.status.toLowerCase()}`}>{c.status}</span>
                </div>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-light)', marginBottom: '0.5rem' }}>
                  {c.case_id}
                </p>
                <p style={{ marginBottom: '1rem', fontSize: '0.9rem' }}>{c.description}</p>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: 'var(--text-light)', marginBottom: '1rem' }}>
                  <span>👤 {c.investigator}</span>
                  <span>📁 {c.evidence?.length || 0} evidence</span>
                </div>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <Link to={`/case/${c.case_id}`} className="btn btn-primary btn-sm">View</Link>
                  <Link to={`/upload/${c.case_id}`} className="btn btn-secondary btn-sm">Upload</Link>
                  <Link to={`/timeline/${c.case_id}`} className="btn btn-secondary btn-sm">Timeline</Link>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;