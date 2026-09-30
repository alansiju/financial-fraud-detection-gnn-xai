/**
 * WCAG text-contrast audit against the running app.
 *
 * Walks every text-bearing leaf element on each page, composites its real
 * background (including translucent tints stacked over cards over the page),
 * and reports anything under AA — 4.5:1 for body text, 3:1 for large text.
 *
 * Colour parsing note: Chrome resolves `color-mix(in srgb, X 8%, transparent)`
 * to `color(srgb 0.286275 0.439216 0.196078 / 0.08)` — components in 0..1, not
 * 0..255, with alpha after a slash. Reading those as 0..255 makes every tinted
 * surface look nearly black and produces a page of phantom failures.
 *
 *   node audit-contrast.mjs [light|dark]
 */
import puppeteer from "puppeteer";

const APP = "http://127.0.0.1:5173";
const API = "http://127.0.0.1:8000";
const THEME = process.argv[2] || "light";

const PAGES = [
  ["/", "Dashboard"],
  ["/transactions", "Transactions"],
  ["/analytics", "Analytics"],
  ["/performance", "Model Performance"],
  ["/predict", "Prediction"],
  ["/settings", "Settings"],
];

const auth = await (await fetch(`${API}/auth/login`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ username: "admin", password: "fraud123" }),
})).json();

const browser = await puppeteer.launch({
  headless: "new",
  args: ["--no-sandbox", "--force-color-profile=srgb"],
});
const page = await browser.newPage();
await page.setViewport({ width: 1580, height: 1000 });

// Seed storage BEFORE any page script runs. Setting it after a goto races the
// app: ThemeContext mounts, finds nothing, and writes its default back — which
// is why a "dark" run was silently auditing the light palette.
await page.evaluateOnNewDocument((a, theme) => {
  localStorage.setItem("fs_token", a.access_token);
  localStorage.setItem("fs_user", JSON.stringify({ username: a.username, role: a.role, email: a.email }));
  localStorage.setItem("fs_theme", theme);
}, auth, THEME);

const AUDIT = () => {
  // ── colour parsing ────────────────────────────────────────────────────────
  const toRGBA = (str) => {
    if (!str) return null;
    const s = str.trim();
    if (s === "transparent") return [0, 0, 0, 0];
    // color(srgb r g b / a) — components 0..1
    let m = s.match(/^color\(\s*srgb\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*(?:\/\s*([\d.eE+-]+%?))?\s*\)$/i);
    if (m) {
      const a = m[4] == null ? 1 : (m[4].endsWith("%") ? parseFloat(m[4]) / 100 : parseFloat(m[4]));
      return [+m[1] * 255, +m[2] * 255, +m[3] * 255, a];
    }
    // rgb()/rgba() — components 0..255
    m = s.match(/^rgba?\(([^)]+)\)$/i);
    if (m) {
      const p = m[1].split(/[\s,/]+/).filter(Boolean).map((v) =>
        v.endsWith("%") ? parseFloat(v) / 100 : parseFloat(v));
      return [p[0], p[1], p[2], p.length > 3 ? p[3] : 1];
    }
    return null;
  };

  const lum = ([r, g, b]) => {
    const f = (v) => {
      v = Math.max(0, Math.min(255, v)) / 255;
      return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    };
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
  };

  // ── composite the real background behind an element ──────────────────────
  const bgOf = (el) => {
    const stack = [];
    let n = el;
    while (n && n !== document.documentElement) {
      const cs = getComputedStyle(n);
      // A gradient can't be reduced to one colour; skip rather than guess.
      if (cs.backgroundImage && /gradient/i.test(cs.backgroundImage)) return null;
      const c = toRGBA(cs.backgroundColor);
      if (c && c[3] > 0) {
        stack.push(c);
        if (c[3] >= 0.999) break;
      }
      n = n.parentElement;
    }
    let base = [255, 255, 255];
    for (let i = stack.length - 1; i >= 0; i--) {
      const [r, g, b, a] = stack[i];
      base = [r * a + base[0] * (1 - a), g * a + base[1] * (1 - a), b * a + base[2] * (1 - a)];
    }
    return base;
  };

  const out = [];
  document.querySelectorAll("body *").forEach((el) => {
    const txt = (el.textContent || "").trim();
    if (!txt || el.children.length > 0) return;
    const cs = getComputedStyle(el);
    if (cs.visibility === "hidden" || cs.display === "none" || +cs.opacity === 0) return;
    const rect = el.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) return;

    const fg = toRGBA(cs.color);
    const bg = bgOf(el);
    if (!fg || !bg) return;

    const size = parseFloat(cs.fontSize);
    const weight = parseInt(cs.fontWeight) || 400;
    const L1 = lum(fg), L2 = lum(bg);
    const ratio = (Math.max(L1, L2) + 0.05) / (Math.min(L1, L2) + 0.05);
    const large = size >= 24 || (size >= 18.66 && weight >= 700);
    const need = large ? 3 : 4.5;
    if (ratio < need) {
      out.push({
        text: txt.slice(0, 46),
        ratio: +ratio.toFixed(2),
        need,
        size: +size.toFixed(1),
        fg: cs.color,
        bg: `rgb(${bg.map((v) => Math.round(v)).join(",")})`,
      });
    }
  });

  const seen = new Set();
  return out.filter((o) => {
    const k = o.text + "|" + o.ratio;
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
};

let total = 0;
let themeMismatch = 0;
console.log(`\nWCAG AA text-contrast audit — ${THEME} theme`);
console.log("=".repeat(78));

for (const [path, name] of PAGES) {
  await page.goto(APP + path, { waitUntil: "networkidle2", timeout: 120000 }).catch(() => {});
  await new Promise((r) => setTimeout(r, 2200));

  // Verify the theme actually took effect. Without this a "dark" run can
  // silently audit the light palette and report a meaningless pass — which is
  // exactly what happened when both runs returned identical light-theme colours.
  const applied = await page.evaluate(() =>
    document.documentElement.getAttribute("data-theme"));
  if (applied !== THEME) {
    console.log(`
  [SKIP] ${name.padEnd(20)} page is in "${applied}" theme, expected "${THEME}"`);
    themeMismatch += 1;
    continue;
  }

  const report = await page.evaluate(AUDIT);
  total += report.length;
  const mark = report.length === 0 ? "PASS" : "FAIL";
  console.log(`\n  [${mark}] ${name.padEnd(20)} ${report.length} failure(s)`);
  for (const r of report.slice(0, 12)) {
    console.log(`        ${String(r.ratio).padStart(5)}:1 (need ${r.need})  ${String(r.size).padStart(5)}px  ` +
      `${r.fg} on ${r.bg}  "${r.text}"`);
  }
}

console.log("\n" + "=".repeat(78));
console.log(`  ${total === 0 ? "ALL PASS" : total + " total failures"}` + (themeMismatch ? `  (${themeMismatch} page(s) skipped: wrong theme)` : ""));
await browser.close();
process.exit(total === 0 && themeMismatch === 0 ? 0 : 1);
