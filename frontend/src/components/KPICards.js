import React from 'react';
import '../styles/KPICards.css';

const KPICards = ({ kpis }) => {
  const kpiData = [
    {
      label: 'Total Teams',
      value: kpis.total_teams || 0,
      icon: '🏀'
    },
    {
      label: 'Average Score',
      value: kpis.average_score ? kpis.average_score.toFixed(2) : '0.00',
      icon: '📊'
    },
    {
      label: 'Top Seed Favorite',
      value: kpis.top_seed_favorite || 'TBD',
      icon: '🏆'
    },
    {
      label: 'Upset Potential',
      value: kpis.upset_potential ? `${(kpis.upset_potential * 100).toFixed(1)}%` : '0%',
      icon: '⚡'
    }
  ];

  return (
    <div className="kpi-cards">
      {kpiData.map((kpi, index) => (
        <div key={index} className="kpi-card">
          <div className="kpi-icon">{kpi.icon}</div>
          <div className="kpi-content">
            <div className="kpi-label">{kpi.label}</div>
            <div className="kpi-value">{kpi.value}</div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default KPICards;
