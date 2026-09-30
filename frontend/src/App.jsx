import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import { ThemeProvider } from './context/ThemeContext';
import { NotificationProvider } from './context/NotificationContext';
import ProtectedRoute from './components/Layout/ProtectedRoute';
import AppLayout from './components/Layout/AppLayout';
import Login from './pages/Login';
import SkeletonLoader from './components/UI/SkeletonLoader';

// Every page past login is its own chunk, fetched on first visit. Recharts
// alone is several hundred kB, and most sessions touch only a few pages.
// Login stays eager: it is the first screen, so splitting it would only add
// a round trip before anything appears.
const Dashboard         = lazy(() => import('./pages/Dashboard'));
const Transactions      = lazy(() => import('./pages/Transactions'));
const TransactionDetail = lazy(() => import('./pages/TransactionDetail'));
const Predict           = lazy(() => import('./pages/Predict'));
const UploadPredict     = lazy(() => import('./pages/UploadPredict'));
const Explainability    = lazy(() => import('./pages/Explainability'));
const GraphViewer       = lazy(() => import('./pages/GraphViewer'));
const Analytics         = lazy(() => import('./pages/Analytics'));
const ModelPerformance  = lazy(() => import('./pages/ModelPerformance'));
const OperatingPoint    = lazy(() => import('./pages/OperatingPoint'));
const AlertQueue        = lazy(() => import('./pages/AlertQueue'));
const ModelHealth       = lazy(() => import('./pages/ModelHealth'));
const Reports           = lazy(() => import('./pages/Reports'));
const Settings          = lazy(() => import('./pages/Settings'));

function PageFallback() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <SkeletonLoader type="kpi"/>
      <SkeletonLoader type="chart"/>
    </div>
  );
}

function NotFound() {
  return (
    <div style={{ minHeight: '60vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 16 }}>
      <div style={{ fontSize: 72, fontWeight: 900, color: 'var(--border-light)' }}>404</div>
      <div style={{ fontSize: 20, fontWeight: 700 }}>Page not found</div>
      <div style={{ fontSize: 13.5, color: 'var(--text-muted)' }}>The page you're looking for doesn't exist.</div>
      <a href="/" className="btn btn-primary">Back to Dashboard</a>
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <NotificationProvider>
        <AuthProvider>
          <BrowserRouter>
            <Routes>
              <Route path="/login" element={<Login/>}/>
              <Route path="/*" element={
                <ProtectedRoute>
                  <AppLayout>
                    <Suspense fallback={<PageFallback/>}>
                    <Routes>
                      <Route path="/"                element={<Dashboard/>}/>
                      <Route path="/transactions"     element={<Transactions/>}/>
                      <Route path="/transactions/:id" element={<TransactionDetail/>}/>
                      <Route path="/predict"          element={<Predict/>}/>
                      <Route path="/upload-predict"   element={<UploadPredict/>}/>
                      <Route path="/explain"          element={<Explainability/>}/>
                      <Route path="/explain/:id"      element={<Explainability/>}/>
                      <Route path="/graph"            element={<GraphViewer/>}/>
                      <Route path="/graph/:id"        element={<GraphViewer/>}/>
                      <Route path="/analytics"        element={<Analytics/>}/>
                      <Route path="/performance"      element={<ModelPerformance/>}/>
                      <Route path="/operating-point"  element={<OperatingPoint/>}/>
                      <Route path="/alerts"           element={<AlertQueue/>}/>
                      <Route path="/model-health"     element={<ModelHealth/>}/>
                      <Route path="/reports"          element={<Reports/>}/>
                      <Route path="/settings"         element={<Settings/>}/>
                      <Route path="*"                 element={<NotFound/>}/>
                    </Routes>
                    </Suspense>
                  </AppLayout>
                </ProtectedRoute>
              }/>
            </Routes>
          </BrowserRouter>
        </AuthProvider>
      </NotificationProvider>
    </ThemeProvider>
  );
}
