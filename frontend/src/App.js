import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import CaseDetail from './components/CaseDetail';
import UploadEvidence from './components/UploadEvidence';
import TimelineView from './components/TimelineView';
import './App.css';

function App() {
  return (
    <Router>
      <div className="app">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/case/:caseId" element={<CaseDetail />} />
          <Route path="/upload/:caseId" element={<UploadEvidence />} />
          <Route path="/timeline/:caseId" element={<TimelineView />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;