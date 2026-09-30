import { Bell, Sun, Moon, Search, Menu } from 'lucide-react';
import { useTheme } from '../../context/ThemeContext';
import { useAuth } from '../../context/AuthContext';
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function Navbar({ onMenuToggle, pageTitle }) {
  const { toggle, isDark } = useTheme();
  const { user } = useAuth();
  const navigate = useNavigate();
  const [search, setSearch] = useState('');

  const handleSearch = (e) => {
    e.preventDefault();
    if (search.trim()) {
      const num = parseInt(search.trim());
      if (!isNaN(num)) navigate(`/transactions/${num}`);
    }
  };

  return (
    <header className="topbar">
      {/* Mobile menu */}
      <button
        onClick={onMenuToggle}
        style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', display: 'flex', padding: 6 }}
      >
        <Menu size={20}/>
      </button>

      {/* Page title */}
      <div style={{ fontWeight: 700, fontSize: 15, color: 'var(--text-primary)', marginRight: 'auto' }}>
        {pageTitle}
      </div>

      {/* Search bar */}
      <form onSubmit={handleSearch} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <div className="input-group" style={{ width: 220 }}>
          <Search className="input-icon" size={15}/>
          <input
            className="input"
            placeholder="Search node ID…"
            value={search}
            onChange={e => setSearch(e.target.value)}
            style={{ height: 36, paddingTop: 0, paddingBottom: 0 }}
          />
        </div>
      </form>

      {/* Model status */}
      <div className="live-indicator">
        <span className="live-dot"/>
        <span>Model Live</span>
      </div>

      {/* Notifications */}
      <button
        style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: 6, borderRadius: 8, position: 'relative' }}
      >
        <Bell size={18}/>
        <span style={{
          position: 'absolute', top: 2, right: 2,
          width: 8, height: 8, borderRadius: '50%',
          background: 'var(--accent-red)',
          border: '2px solid var(--bg-secondary)'
        }}/>
      </button>

      {/* Theme toggle */}
      <button
        onClick={toggle}
        style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: 6, borderRadius: 8 }}
        title={isDark ? 'Switch to light' : 'Switch to dark'}
      >
        {isDark ? <Sun size={18}/> : <Moon size={18}/>}
      </button>

      {/* Avatar */}
      <div style={{
        width: 34, height: 34, borderRadius: '50%',
        background: 'linear-gradient(135deg, var(--brand), var(--cat-4))',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 13, fontWeight: 700, color: 'white', cursor: 'pointer',
        flexShrink: 0
      }}>
        {user?.username?.[0]?.toUpperCase() || 'U'}
      </div>
    </header>
  );
}
