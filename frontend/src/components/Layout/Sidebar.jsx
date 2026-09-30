import { NavLink, useLocation } from 'react-router-dom';
import {
  LayoutDashboard, CreditCard, Brain, Network, BarChart3, Activity, FileText, Settings, Shield, ChevronLeft, ChevronRight, Zap, LogOut, Upload, ShieldAlert, SlidersHorizontal, HeartPulse
} from 'lucide-react';
import { useAuth } from '../../context/AuthContext';

const NAV = [
  { section: 'Overview' },
  { to: '/',             icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/transactions', icon: CreditCard,      label: 'Transactions' },
  { to: '/alerts',       icon: ShieldAlert,     label: 'Review Queue' },
  { section: 'AI Engine' },
  { to: '/predict',      icon: Zap,             label: 'Prediction' },
  { to: '/upload-predict', icon: Upload,        label: 'Dataset Scanner' },
  { to: '/explain',      icon: Brain,           label: 'Explainability' },
  { to: '/graph',        icon: Network,         label: 'Graph Viewer' },
  { section: 'Analytics' },
  { to: '/analytics',    icon: BarChart3,       label: 'Analytics' },
  { to: '/performance',  icon: Activity,        label: 'Model Performance' },
  { to: '/operating-point', icon: SlidersHorizontal, label: 'Operating Point' },
  { to: '/model-health', icon: HeartPulse,      label: 'Model Health' },
  { section: 'System' },
  { to: '/reports',      icon: FileText,        label: 'Reports' },
  { to: '/settings',     icon: Settings,        label: 'Settings' },
];

export default function Sidebar({ collapsed, onToggle }) {
  const { user, logout } = useAuth();
  const location = useLocation();

  return (
    <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
      {/* Logo */}
      <div className="sidebar-logo">
        <div className="logo-icon">
          <Shield size={18} color="var(--on-brand)" />
        </div>
        {!collapsed && (
          <div>
            <div className="logo-text">Fraud<span>Shield</span> AI</div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 1 }}>GraphSAGE Platform</div>
          </div>
        )}
        <button
          onClick={onToggle}
          style={{
            marginLeft: 'auto', background: 'none', border: 'none',
            color: 'var(--text-muted)', cursor: 'pointer', padding: 4,
            borderRadius: 6, display: 'flex', alignItems: 'center'
          }}
        >
          {collapsed ? <ChevronRight size={16}/> : <ChevronLeft size={16}/>}
        </button>
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav">
        {NAV.map((item, i) => {
          if (item.section) {
            return collapsed ? (
              <div key={i} style={{ height: 8 }} />
            ) : (
              <div key={i} className="nav-section-title">{item.section}</div>
            );
          }
          const isActive = item.to === '/'
            ? location.pathname === '/'
            : location.pathname.startsWith(item.to);
          return (
            <NavLink
              key={item.to}
              to={item.to}
              className={`nav-item ${isActive ? 'active' : ''}`}
              title={collapsed ? item.label : ''}
            >
              <item.icon className="nav-icon" size={18} />
              {!collapsed && <span>{item.label}</span>}
            </NavLink>
          );
        })}
      </nav>

      {/* User footer */}
      <div style={{
        padding: collapsed ? '12px 0' : '12px',
        borderTop: '1px solid var(--border)',
        flexShrink: 0
      }}>
        {!collapsed ? (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 10,
            padding: '10px 12px', borderRadius: 10,
            background: 'var(--bg-primary)'
          }}>
            <div style={{
              width: 34, height: 34, borderRadius: '50%',
              background: 'linear-gradient(135deg, var(--brand), var(--cat-4))',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 13, fontWeight: 700, color: 'white', flexShrink: 0
            }}>
              {user?.username?.[0]?.toUpperCase() || 'U'}
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }} className="truncate">
                {user?.username}
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }} className="truncate">
                {user?.role === 'admin' ? '⭐ Administrator' : '👤 Analyst'}
              </div>
            </div>
            <button
              onClick={logout}
              title="Logout"
              style={{
                background: 'none', border: 'none', cursor: 'pointer',
                color: 'var(--text-muted)', display: 'flex', padding: 4,
                borderRadius: 6, transition: 'var(--transition)'
              }}
              onMouseEnter={e => e.currentTarget.style.color = 'var(--status-critical)'}
              onMouseLeave={e => e.currentTarget.style.color = 'var(--text-muted)'}
            >
              <LogOut size={16}/>
            </button>
          </div>
        ) : (
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <button
              onClick={logout}
              title="Logout"
              style={{
                background: 'none', border: 'none', cursor: 'pointer',
                color: 'var(--text-muted)', padding: 8, borderRadius: 8
              }}
            >
              <LogOut size={18}/>
            </button>
          </div>
        )}
      </div>
    </aside>
  );
}
