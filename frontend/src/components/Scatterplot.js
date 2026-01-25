import React from 'react';
import Plot from 'react-plotly.js';

const Scatterplot = ({ data }) => {
  // Transform data for Plotly
  const plotData = [{
    x: data.map(d => d.tier),
    y: data.map(d => d.composite_score),
    mode: 'markers',
    type: 'scatter',
    text: data.map(d => d.team_name),
    marker: {
      size: 12,
      color: data.map(d => d.seed),
      colorscale: 'Viridis',
      showscale: true,
      colorbar: {
        title: 'Seed'
      }
    },
    hovertemplate: '<b>%{text}</b><br>' +
                   'Tier: %{x}<br>' +
                   'Composite Score: %{y:.2f}<br>' +
                   '<extra></extra>'
  }];

  const layout = {
    xaxis: {
      title: 'Tier',
      gridcolor: '#e0e0e0'
    },
    yaxis: {
      title: 'Composite Score',
      gridcolor: '#e0e0e0'
    },
    hovermode: 'closest',
    plot_bgcolor: '#f9f9f9',
    paper_bgcolor: '#ffffff',
    margin: { t: 40, r: 40, b: 60, l: 60 }
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false
  };

  return (
    <Plot
      data={plotData}
      layout={layout}
      config={config}
      style={{ width: '100%', height: '100%' }}
    />
  );
};

export default Scatterplot;
