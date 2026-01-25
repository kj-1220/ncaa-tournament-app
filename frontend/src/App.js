import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './styles/App.css';

// Import page components (we'll create these)
import Sidebar from './components/Sidebar';
import Summary from './pages/Summary';
import Matchups from './pages/Matchups';
import Bracket from './pages/Bracket';
import Women from './pages/Women';
import ModelPerformance from './pages/ModelPerformance';

function App() {
  return (
    <Router>
      <div className="App">
        <Sidebar />
        <div className="main-content">
          <Routes>
            <Route path="/" element={<Summary />} />
            <Route path="/summary" element={<Summary />} />
            <Route path="/matchups" element={<Matchups />} />
            <Route path="/bracket" element={<Bracket />} />
            <Route path="/women" element={<Women />} />
            <Route path="/model-performance" element={<ModelPerformance />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
