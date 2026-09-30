import { useState } from 'react';
import Sidebar from './Sidebar';
import Navbar from './Navbar';
import { useNotification } from '../../context/NotificationContext';
import ToastContainer from '../UI/ToastContainer';
import { useLocation } from 'react-router-dom';

const PAGE_TITLES = {
  '/':            'Dashboard',
  '/transactions':'Transactions',
  '/predict':     'Fraud Prediction',
  '/upload-predict':'Dataset Scanner',
  '/explain':     'Explainability',
  '/graph':       'Graph Viewer',
  '/analytics':   'Analytics',
  '/performance': 'Model Performance',
  '/operating-point': 'Operating Point',
  '/alerts':      'Review Queue',
  '/model-health':'Model Health',
  '/reports':     'Reports',
  '/settings':    'Settings',
};

export default function AppLayout({ children }) {
  const [collapsed, setCollapsed] = useState(false);
  const { toasts, remove } = useNotification();
  const location = useLocation();

  const title = PAGE_TITLES[location.pathname] ||
    Object.entries(PAGE_TITLES).find(([k]) => location.pathname.startsWith(k) && k !== '/')?.[1] ||
    'FraudShield AI';

  return (
    <div className="app-layout">
      <Sidebar collapsed={collapsed} onToggle={() => setCollapsed(c => !c)} />
      <div className={`main-content`} style={{ marginLeft: collapsed ? 72 : 260, transition: 'margin-left 400ms cubic-bezier(0.4,0,0.2,1)' }}>
        <Navbar onMenuToggle={() => setCollapsed(c => !c)} pageTitle={title} />
        <div className="page-content animate-fade-in">
          {children}
        </div>
      </div>
      <ToastContainer toasts={toasts} onRemove={remove} />
    </div>
  );
}
