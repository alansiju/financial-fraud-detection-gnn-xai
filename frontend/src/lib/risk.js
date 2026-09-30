/*
 * Risk levels, in one place.
 *
 * These used to be redefined per page, and the copies had drifted: none knew
 * about MINIMAL (the backend's lowest band), so every low-score transaction
 * rendered as `badge undefined`; and TransactionDetail coloured MEDIUM in the
 * brand green while the badge stylesheet makes it amber. Every page now reads
 * from here, and the order matches server/ml/predictor._risk_level.
 */

export const RISK_LEVELS = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL'];

/** Badge class for a risk level — pairs with the .badge-* rules in index.css. */
export const RISK_CLASS = {
  CRITICAL: 'badge-critical',
  HIGH: 'badge-high',
  MEDIUM: 'badge-medium',
  LOW: 'badge-low',
  MINIMAL: 'badge-minimal',
};

/** Colour token for a risk level, matching the badge it would render as. */
export const RISK_COLOR = {
  CRITICAL: 'var(--status-critical)',
  HIGH: 'var(--status-serious)',
  MEDIUM: 'var(--status-warning)',
  LOW: 'var(--status-good)',
  MINIMAL: 'var(--text-muted)',
};

export const riskClass = (level) => RISK_CLASS[level] ?? 'badge-neutral';
export const riskColor = (level) => RISK_COLOR[level] ?? 'var(--text-muted)';
