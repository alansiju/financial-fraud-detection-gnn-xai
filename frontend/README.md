# 💻 FraudShield AI — Frontend Dashboard

This directory contains the React 19 single-page application (SPA) for **FraudShield AI**, powered by Vite, Lucide React icons, Recharts, and 2D Force-Directed Graph visualizer (`react-force-graph-2d`).

---

## 🛠️ Technology Stack

- **Framework**: [React 19](https://react.dev/)
- **Build Tool / Bundler**: [Vite](https://vitejs.dev/)
- **Routing**: [React Router DOM v7](https://reactrouter.com/)
- **Graph Visualization**: [react-force-graph-2d](https://github.com/vasturiano/react-force-graph)
- **Charts & Data Viz**: [Recharts](https://recharts.org/)
- **Icons**: [Lucide React](https://lucide.dev/)
- **HTTP Client**: [Axios](https://axios-http.com/)
- **Linter**: [Oxlint](https://oxc.rs/)

---

## 📂 Architecture & Directory Layout

```
frontend/
├── package.json               # Frontend dependencies & npm scripts
├── vite.config.js             # Vite configuration
├── index.html                 # HTML5 template
├── src/
│   ├── main.jsx               # Application entry point
│   ├── App.jsx                # Router & protected layouts
│   ├── index.css              # Global design tokens & CSS system
│   │
│   ├── api/                   # Centralized Axios API services
│   ├── context/               # AuthContext & state management
│   ├── components/            # Reusable UI components (Sidebar, Navbar, Cards, Charts)
│   └── pages/                 # Full feature views:
│       ├── Dashboard.jsx          # Executive overview & real-time KPI metrics
│       ├── AlertQueue.jsx         # Analyst alert management & triage workflow
│       ├── Transactions.jsx       # Searchable & filterable transaction data grid
│       ├── TransactionDetail.jsx  # Deep dive transaction inspection & factor breakdown
│       ├── Predict.jsx            # Single-transaction real-time inference simulator
│       ├── UploadPredict.jsx      # Batch CSV upload & graph-attachment scoring
│       ├── GraphViewer.jsx        # 2D force-directed fraud cluster & community explorer
│       ├── Explainability.jsx     # GNNExplainer feature attributions & neighbor impact
│       ├── Analytics.jsx          # Multi-dimensional fraud pattern distributions
│       ├── ModelPerformance.jsx  # Baseline comparisons, ROC/PR curves & confusion matrix
│       ├── ModelHealth.jsx        # Data drift monitoring & inference latency tracking
│       ├── OperatingPoint.jsx     # Dynamic threshold tuning & cost matrix optimizer
│       ├── Reports.jsx            # Executive PDF & Excel report exports
│       ├── Settings.jsx           # Global thresholds, model switcher & preferences
│       └── Login.jsx              # JWT authentication & session management
```

---

## 🚀 Available Scripts

### `npm run dev`
Runs the development server on `http://127.0.0.1:5173`.

### `npm run build`
Bundles production assets to `dist/`.

### `npm run preview`
Locally preview production build.

### `npm run lint`
Runs Oxlint for high-speed static code analysis.

### `npm run audit:contrast`
Audits WCAG color contrast across the design tokens.
