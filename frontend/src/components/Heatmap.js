import React from 'react';
import Plot from 'react-plotly.js';

const Heatmap = ({ data }) => {
  // Expected data format:
  // {
  //   teams: ['Team A', 'Team B', ...],
  //   matrix: [[0.5, 0.65, ...], [0.35, 0.5, ...], ...]
  // }
  
  const plotData = [{
    z: data.matrix || [],
    x: data.teams || [],
    y: data.teams || [],
    type: 'heatmap',
    colorscale: [
      [0, '#d32f2f'],      // Red for low probability
      [0.5, '#fff3e0'],    // Light orange for 50%
      [1, '#2e7d32']       // Green for high probability
    ],
    hovertemplate: '<b>%{y}</b> vs <b>%{x}</b><br>' +
                   'Win Probability: %{z:.1%}<br>' +
                   '<extra></extra>',
    showscale: true,
    colorbar: {
      title: 'Win Prob',
      tickformat: '.0%'
    }
  }];

  const layout = {
    xaxis: {
      title: 'Opponent',
      tickangle: -45,
      side: 'bottom'
    },
    yaxis: {
      title: 'Team',
      autorange: 'reversed'
    },
    plot_bgcolor: '#ffffff',
    paper_bgcolor: '#ffffff',
    margin: { t: 40, r: 100, b: 120, l: 120 }
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

export default Heatmap;
