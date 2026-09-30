import { CheckCircle, XCircle, AlertTriangle, Info, X } from 'lucide-react';

const ICONS = { success: CheckCircle, error: XCircle, warning: AlertTriangle, info: Info };
const COLORS = { success: 'var(--status-good)', error: 'var(--status-critical)', warning: 'var(--status-warning)', info: 'var(--brand)' };

export default function ToastContainer({ toasts, onRemove }) {
  return (
    <div className="toast-container">
      {toasts.map(t => {
        const Icon = ICONS[t.type] || Info;
        return (
          <div key={t.id} className={`toast ${t.type}`}>
            <Icon size={18} color={COLORS[t.type]} style={{ flexShrink: 0, marginTop: 1 }} />
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', textTransform: 'capitalize' }}>
                {t.type}
              </div>
              <div style={{ fontSize: 12.5, color: 'var(--text-secondary)', marginTop: 2 }}>
                {t.message}
              </div>
            </div>
            <button
              onClick={() => onRemove(t.id)}
              style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', padding: 2, display: 'flex' }}
            >
              <X size={14}/>
            </button>
          </div>
        );
      })}
    </div>
  );
}
