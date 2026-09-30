export default function SkeletonLoader({ rows = 4, type = 'table' }) {
  if (type === 'kpi') {
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
        {Array(4).fill(0).map((_, i) => (
          <div key={i} className="kpi-card">
            <div className="skeleton" style={{ width: 48, height: 48, borderRadius: 10, marginBottom: 16 }}/>
            <div className="skeleton" style={{ width: '60%', height: 28, marginBottom: 8 }}/>
            <div className="skeleton" style={{ width: '40%', height: 12 }}/>
          </div>
        ))}
      </div>
    );
  }

  if (type === 'chart') {
    return (
      <div className="chart-container">
        <div className="skeleton" style={{ width: '30%', height: 16, marginBottom: 24 }}/>
        <div className="skeleton" style={{ width: '100%', height: 220 }}/>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {Array(rows).fill(0).map((_, i) => (
        <div key={i} style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
          <div className="skeleton" style={{ width: 32, height: 32, borderRadius: '50%', flexShrink: 0 }}/>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 6 }}>
            <div className="skeleton" style={{ width: `${60 + Math.random() * 30}%`, height: 12 }}/>
            <div className="skeleton" style={{ width: `${30 + Math.random() * 20}%`, height: 10 }}/>
          </div>
          <div className="skeleton" style={{ width: 80, height: 24, borderRadius: 999 }}/>
        </div>
      ))}
    </div>
  );
}
