import React, { useState } from 'react';
import '../styles/TeamTable.css';

const TeamTable = ({ teams }) => {
  const [sortConfig, setSortConfig] = useState({ key: 'rank', direction: 'asc' });

  const sortedTeams = React.useMemo(() => {
    let sortableTeams = [...teams];
    if (sortConfig.key) {
      sortableTeams.sort((a, b) => {
        if (a[sortConfig.key] < b[sortConfig.key]) {
          return sortConfig.direction === 'asc' ? -1 : 1;
        }
        if (a[sortConfig.key] > b[sortConfig.key]) {
          return sortConfig.direction === 'asc' ? 1 : -1;
        }
        return 0;
      });
    }
    return sortableTeams;
  }, [teams, sortConfig]);

  const requestSort = (key) => {
    let direction = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  const formatPercentage = (value) => {
    return value ? `${(value * 100).toFixed(1)}%` : '0.0%';
  };

  const getSortIcon = (columnKey) => {
    if (sortConfig.key !== columnKey) return '↕';
    return sortConfig.direction === 'asc' ? '↑' : '↓';
  };

  return (
    <div className="team-table-wrapper">
      <table className="team-table">
        <thead>
          <tr>
            <th onClick={() => requestSort('rank')}>
              Rank {getSortIcon('rank')}
            </th>
            <th onClick={() => requestSort('team_name')}>
              Team {getSortIcon('team_name')}
            </th>
            <th onClick={() => requestSort('region')}>
              Region {getSortIcon('region')}
            </th>
            <th onClick={() => requestSort('seed')}>
              Seed {getSortIcon('seed')}
            </th>
            <th onClick={() => requestSort('bracket_value')}>
              Bracket Value {getSortIcon('bracket_value')}
            </th>
            <th onClick={() => requestSort('tier')}>
              Tier {getSortIcon('tier')}
            </th>
            <th onClick={() => requestSort('overall')}>
              Overall {getSortIcon('overall')}
            </th>
            <th onClick={() => requestSort('offense')}>
              Offense {getSortIcon('offense')}
            </th>
            <th onClick={() => requestSort('defense')}>
              Defense {getSortIcon('defense')}
            </th>
            <th onClick={() => requestSort('champion_pct')}>
              Champion % {getSortIcon('champion_pct')}
            </th>
            <th onClick={() => requestSort('championship_pct')}>
              Championship % {getSortIcon('championship_pct')}
            </th>
            <th onClick={() => requestSort('final_four_pct')}>
              Final Four % {getSortIcon('final_four_pct')}
            </th>
            <th onClick={() => requestSort('elite_eight_pct')}>
              Elite 8 % {getSortIcon('elite_eight_pct')}
            </th>
            <th onClick={() => requestSort('sweet_sixteen_pct')}>
              Sweet 16 % {getSortIcon('sweet_sixteen_pct')}
            </th>
            <th onClick={() => requestSort('round_2_pct')}>
              Round 2 % {getSortIcon('round_2_pct')}
            </th>
          </tr>
        </thead>
        <tbody>
          {sortedTeams.map((team, index) => (
            <tr key={index}>
              <td>{team.rank}</td>
              <td className="team-name">{team.team_name}</td>
              <td>{team.region}</td>
              <td>{team.seed}</td>
              <td>{team.bracket_value?.toFixed(2) || 'N/A'}</td>
              <td>{team.tier}</td>
              <td>{team.overall?.toFixed(2) || 'N/A'}</td>
              <td>{team.offense?.toFixed(2) || 'N/A'}</td>
              <td>{team.defense?.toFixed(2) || 'N/A'}</td>
              <td>{formatPercentage(team.champion_pct)}</td>
              <td>{formatPercentage(team.championship_pct)}</td>
              <td>{formatPercentage(team.final_four_pct)}</td>
              <td>{formatPercentage(team.elite_eight_pct)}</td>
              <td>{formatPercentage(team.sweet_sixteen_pct)}</td>
              <td>{formatPercentage(team.round_2_pct)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default TeamTable;
