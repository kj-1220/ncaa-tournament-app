import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import '../styles/Sidebar.css';

const Sidebar = () => {
  const location = useLocation();
  
  const menuItems = [
    { path: '/summary', label: 'Summary', icon: '📊' },
    { path: '/matchups', label: 'Matchups', icon: '⚔️' },
    { path: '/bracket', label: 'Bracket', icon: '🏀' },
    { path: '/women', label: 'Women', icon: '👩' },
    { path: '/model-performance', label: 'Model Performance', icon: '📈' }
  ];

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h2>NCAA Tournament</h2>
        <p>Predictions 2026</p>
      </div>
      <nav className="sidebar-nav">
        {menuItems.map((item) => (
          <Link
            key={item.path}
            to={item.path}
            className={`nav-item ${location.pathname === item.path ? 'active' : ''}`}
          >
            <span className="nav-icon">{item.icon}</span>
            <span className="nav-label">{item.label}</span>
          </Link>
        ))}
      </nav>
    </div>
  );
};

export default Sidebar;
