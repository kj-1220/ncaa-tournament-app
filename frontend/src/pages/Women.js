import React, { useState, useEffect } from 'react';
import axios from 'axios';
import '../styles/Women.css';

// Import components (we'll create these)
import Banner from '../components/Banner';
import KPICards from '../components/KPICards';
import Scatterplot from '../components/Scatterplot';
import Heatmap from '../components/Heatmap';
import TeamTable from '../components/TeamTable';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const Women = () => {
  const [teams, setTeams] = useState([]);
  const [kpis, setKpis] = useState({});
  const [scatterData, setScatterData] = useState([]);
  const [heatmapData, setHeatmapData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      
      // Fetch all data in parallel
      const [teamsRes, kpisRes, scatterRes, heatmapRes] = await Promise.all([
        axios.get(`${API_URL}/teams`),
        axios.get(`${API_URL}/kpis`),
        axios.get(`${API_URL}/scatterplot`),
        axios.get(`${API_URL}/probabilities/heatmap`)
      ]);

      setTeams(teamsRes.data.teams || []);
      setKpis(kpisRes.data.kpis || {});
      setScatterData(scatterRes.data.data || []);
      setHeatmapData(heatmapRes.data.data || []);
      
      setLoading(false);
    } catch (err) {
      console.error('Error fetching data:', err);
      setError('Failed to load data. Please try again.');
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="women-page">
        <div className="loading">Loading Women's Tournament data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="women-page">
        <div className="error">{error}</div>
      </div>
    );
  }

  return (
    <div className="women-page">
      {/* Banner Section */}
      <Banner title="Women's NCAA Tournament 2026" />

      {/* KPI Cards */}
      <KPICards kpis={kpis} />

      {/* Visualizations Row */}
      <div className="visualizations-row">
        <div className="viz-container">
          <h3>Team Tiers vs Composite Score</h3>
          <Scatterplot data={scatterData} />
        </div>
        
        <div className="viz-container">
          <h3>Win Probability Heatmap (Seeds 1-3)</h3>
          <Heatmap data={heatmapData} />
        </div>
      </div>

      {/* Team Rankings Table */}
      <div className="table-container">
        <h3>Complete Team Rankings & Predictions</h3>
        <TeamTable teams={teams} />
      </div>
    </div>
  );
};

export default Women;
