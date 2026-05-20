import { useState, useRef, useCallback } from "react";

const API = "http://localhost:8000";

const LABEL_META = {
  contradiction: {
    icon: "⚡",
    label: "Contradicción",
    badgeClass: "badge-contradiction",
    color: "oklch(75% 0.22 25)",
    barColor: "oklch(65% 0.25 25)",
  },
  entailment: {
    icon: "✓",
    label: "Implicación",
    badgeClass: "badge-entailment",
    color: "oklch(75% 0.18 145)",
    barColor: "oklch(70% 0.2 145)",
  },
  neutral: {
    icon: "◎",
    label: "Neutral",
    badgeClass: "badge-neutral",
    color: "oklch(76% 0.16 270)",
    barColor: "oklch(66% 0.22 270)",
  },
};

function ScoreBar({ label, score, color }) {
  const pct = Math.round(score * 100);
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs" style={{ fontFamily: "var(--font-mono)" }}>
        <span className="capitalize" style={{ color }}>{label}</span>
        <span className="text-white/60">{pct}%</span>
      </div>
      <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "oklch(25% 0.02 270)" }}>
        <div
          className="h-full rounded-full score-bar-fill"
          style={{ "--target-w": `${pct}%`, background: color }}
        />
      </div>
    </div>
  );
}

function Spinner() {
  return (
    <svg className="spin" width="20" height="20" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeOpacity="0.25" />
      <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

function RLBadge({ rl }) {
  if (!rl) return null;
  const stateColor = { LOW: "#f59e0b", MEDIUM: "#8b5cf6", HIGH: "#ef4444" }[rl.state] ?? "#6b7280";
  return (
    <div className="flex items-center gap-2 flex-wrap text-xs">
      <span className="px-2 py-0.5 rounded-full font-mono font-medium" style={{ background: `${stateColor}22`, color: stateColor, border: `1px solid ${stateColor}44` }}>
        {rl.state}
      </span>
      <span className="text-white/40">{rl.decision_type === "exploration" ? "🎲 exploración" : "🧠 explotación"}</span>
      {rl.show_alert && <span className="text-yellow-400">⚠ alerta activada</span>}
    </div>
  );
}

export default function App() {
  const [premise, setPremise] = useState("");
  const [hypothesis, setHypothesis] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [feedbackLoading, setFeedbackLoading] = useState(false);
  const [correctLabel, setCorrectLabel] = useState("");
  const [totalCorrections, setTotalCorrections] = useState(null);
  const [history, setHistory] = useState([]);
  const resultsRef = useRef(null);

  const detect = useCallback(async () => {
    if (!premise.trim() || !hypothesis.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);
    setFeedbackSent(false);
    setCorrectLabel("");
    try {
      const res = await fetch(`${API}/detect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ premise: premise.trim(), hypothesis: hypothesis.trim() }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setResult(data);
      setHistory((h) => [{ premise: premise.trim(), hypothesis: hypothesis.trim(), ...data, ts: Date.now() }, ...h.slice(0, 9)]);
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [premise, hypothesis]);

  const sendFeedback = useCallback(async (accepted) => {
    if (!result) return;
    setFeedbackLoading(true);
    const confidence = result.scores?.[result.label] ?? 0.5;
    const action = result.rl_decision?.action ?? 1;
    try {
      const body = { confidence, action, accepted };
      if (!accepted && correctLabel) {
        body.correct_label = correctLabel;
        body.premise = premise.trim();
        body.hypothesis = hypothesis.trim();
      }
      const res = await fetch(`${API}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setTotalCorrections(data.total_corrections);
      setFeedbackSent(true);
      if (data.ready_to_retrain) alert(`🎉 ¡${data.total_corrections} correcciones! El modelo está listo para reentrenarse.`);
    } catch (e) {
      setError(e.message);
    } finally {
      setFeedbackLoading(false);
    }
  }, [result, correctLabel, premise, hypothesis]);

  const labelKey = result?.label?.toLowerCase();
  const meta = LABEL_META[labelKey] ?? LABEL_META.neutral;
  const scores = result?.scores ?? {};

  return (
    <div className="relative min-h-dvh overflow-x-hidden" style={{ fontFamily: "var(--font-sans)" }}>
      {/* Background orbs */}
      <div className="bg-orb w-96 h-96 opacity-20 top-[-100px] left-[-80px]" style={{ background: "oklch(58% 0.26 270)" }} />
      <div className="bg-orb w-80 h-80 opacity-15 bottom-32 right-[-60px]" style={{ background: "oklch(65% 0.25 25)" }} />

      <div className="relative z-10 max-w-3xl mx-auto px-4 py-12 space-y-10">

        {/* Header */}
        <header className="text-center space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium mb-2"
            style={{ background: "oklch(58% 0.26 270 / 0.15)", border: "1px solid oklch(58% 0.26 270 / 0.3)", color: "oklch(76% 0.16 270)" }}>
            <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: "oklch(76% 0.16 270)" }} />
            IA · Aprendizaje por Refuerzo
          </div>
          <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight gradient-text">
            ContradictAI
          </h1>
          <p className="text-white/50 text-base max-w-md mx-auto leading-relaxed">
            Detecta contradicciones entre enunciados usando NLI multilingüe con agente RL adaptativo.
          </p>
        </header>

        {/* Input Card */}
        <div className="glass rounded-2xl p-6 space-y-4">
          <div className="grid sm:grid-cols-2 gap-4">
            {[
              { id: "premise", label: "Premisa", value: premise, set: setPremise, placeholder: "El cielo es azul durante el día." },
              { id: "hypothesis", label: "Hipótesis", value: hypothesis, set: setHypothesis, placeholder: "El cielo es completamente negro a mediodía." },
            ].map(({ id, label, value, set, placeholder }) => (
              <div key={id} className="space-y-2">
                <label htmlFor={id} className="text-xs font-semibold uppercase tracking-widest text-white/40">{label}</label>
                <div className="glow-focus rounded-xl transition-all duration-300">
                  <textarea
                    id={id}
                    rows={4}
                    value={value}
                    onChange={(e) => set(e.target.value)}
                    placeholder={placeholder}
                    onKeyDown={(e) => { if (e.key === "Enter" && e.ctrlKey) detect(); }}
                    className="w-full rounded-xl px-4 py-3 text-sm resize-none leading-relaxed placeholder-white/20 transition-all duration-200"
                    style={{
                      background: "oklch(19% 0.02 270)",
                      border: "1px solid oklch(30% 0.025 270 / 0.6)",
                      color: "oklch(92% 0.01 270)",
                      fontFamily: "var(--font-sans)",
                    }}
                  />
                </div>
              </div>
            ))}
          </div>

          <button
            id="detect-btn"
            onClick={detect}
            disabled={loading || !premise.trim() || !hypothesis.trim()}
            className="w-full flex items-center justify-center gap-2 py-3 rounded-xl font-semibold text-sm transition-all duration-200 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed"
            style={{
              background: "linear-gradient(135deg, oklch(58% 0.26 270), oklch(50% 0.28 270))",
              color: "white",
              boxShadow: "0 0 20px oklch(58% 0.26 270 / 0.3)",
            }}
          >
            {loading ? <><Spinner /> Analizando…</> : <><span>🔍</span> Detectar contradicción</>}
          </button>
          <p className="text-center text-white/25 text-xs">Ctrl + Enter para analizar</p>
        </div>

        {/* Error */}
        {error && (
          <div className="animate-fade-in-up glass rounded-xl px-5 py-4 flex items-center gap-3 text-sm"
            style={{ borderColor: "oklch(65% 0.25 25 / 0.4)", color: "oklch(75% 0.22 25)" }}>
            <span>⚠</span> {error} — ¿El backend está corriendo en puerto 8000?
          </div>
        )}

        {/* Result */}
        {result && (
          <div ref={resultsRef} className="animate-fade-in-up glass rounded-2xl p-6 space-y-6">
            {/* Label header */}
            <div className="flex items-start justify-between flex-wrap gap-3">
              <div className="space-y-2">
                <p className="text-xs font-semibold uppercase tracking-widest text-white/40">Resultado</p>
                <div className="flex items-center gap-3">
                  <span className="text-3xl">{meta.icon}</span>
                  <div>
                    <span className={`px-3 py-1 rounded-full text-sm font-bold ${meta.badgeClass}`}>
                      {meta.label}
                    </span>
                    {result.source === "cache" && (
                      <span className="ml-2 text-xs text-white/40 font-mono">· caché humano</span>
                    )}
                  </div>
                </div>
              </div>
              {result.is_contradiction && result.rl_decision && (
                <div className="space-y-1">
                  <p className="text-xs font-semibold uppercase tracking-widest text-white/40">Agente RL</p>
                  <RLBadge rl={result.rl_decision} />
                </div>
              )}
            </div>

            {/* Scores */}
            {Object.keys(scores).length > 0 && (
              <div className="space-y-3">
                <p className="text-xs font-semibold uppercase tracking-widest text-white/40">Probabilidades</p>
                <div className="space-y-3">
                  {Object.entries(scores)
                    .sort(([, a], [, b]) => b - a)
                    .map(([lbl, score]) => {
                      const m = LABEL_META[lbl.toLowerCase()] ?? LABEL_META.neutral;
                      return <ScoreBar key={lbl} label={m.label} score={score} color={m.barColor} />;
                    })}
                </div>
              </div>
            )}

            {/* Feedback */}
            {!feedbackSent ? (
              <div className="space-y-3 pt-2 border-t" style={{ borderColor: "oklch(30% 0.025 270 / 0.4)" }}>
                <p className="text-xs font-semibold uppercase tracking-widest text-white/40">¿Es correcto este resultado?</p>
                <div className="flex gap-3 flex-wrap">
                  <button
                    id="feedback-accept"
                    onClick={() => sendFeedback(true)}
                    disabled={feedbackLoading}
                    className="flex-1 py-2 rounded-xl text-sm font-semibold transition-all duration-200 active:scale-95 disabled:opacity-40"
                    style={{ background: "oklch(70% 0.2 145 / 0.15)", border: "1px solid oklch(70% 0.2 145 / 0.4)", color: "oklch(75% 0.18 145)" }}
                  >
                    {feedbackLoading ? <Spinner /> : "👍 Correcto"}
                  </button>
                  <button
                    id="feedback-reject"
                    onClick={() => sendFeedback(false)}
                    disabled={feedbackLoading}
                    className="flex-1 py-2 rounded-xl text-sm font-semibold transition-all duration-200 active:scale-95 disabled:opacity-40"
                    style={{ background: "oklch(65% 0.25 25 / 0.15)", border: "1px solid oklch(65% 0.25 25 / 0.4)", color: "oklch(75% 0.22 25)" }}
                  >
                    👎 Incorrecto
                  </button>
                </div>
                <div className="space-y-1">
                  <label htmlFor="correct-label" className="text-xs text-white/40">Etiqueta correcta (opcional):</label>
                  <select
                    id="correct-label"
                    value={correctLabel}
                    onChange={(e) => setCorrectLabel(e.target.value)}
                    className="w-full rounded-lg px-3 py-2 text-sm"
                    style={{ background: "oklch(19% 0.02 270)", border: "1px solid oklch(30% 0.025 270 / 0.6)", color: "oklch(92% 0.01 270)" }}
                  >
                    <option value="">— Seleccionar —</option>
                    <option value="contradiction">Contradicción</option>
                    <option value="entailment">Implicación</option>
                    <option value="neutral">Neutral</option>
                  </select>
                </div>
              </div>
            ) : (
              <div className="animate-fade-in-up text-sm px-4 py-3 rounded-xl flex items-center gap-2"
                style={{ background: "oklch(58% 0.26 270 / 0.12)", border: "1px solid oklch(58% 0.26 270 / 0.3)", color: "oklch(76% 0.16 270)" }}>
                ✅ Feedback enviado — el agente RL ha actualizado su política.
                {totalCorrections !== null && <span className="text-white/40 text-xs ml-auto">{totalCorrections} correcciones totales</span>}
              </div>
            )}
          </div>
        )}

        {/* History */}
        {history.length > 0 && (
          <div className="space-y-3">
            <p className="text-xs font-semibold uppercase tracking-widest text-white/40">Historial reciente</p>
            <div className="space-y-2">
              {history.map((h) => {
                const hMeta = LABEL_META[h.label?.toLowerCase()] ?? LABEL_META.neutral;
                return (
                  <button
                    key={h.ts}
                    onClick={() => { setPremise(h.premise); setHypothesis(h.hypothesis); setResult(h); setFeedbackSent(false); }}
                    className="w-full text-left glass rounded-xl px-4 py-3 flex items-center gap-3 group transition-all duration-200 hover:border-white/20"
                  >
                    <span className="text-lg shrink-0">{hMeta.icon}</span>
                    <div className="min-w-0 flex-1">
                      <p className="text-xs text-white/50 truncate">{h.premise}</p>
                      <p className="text-xs text-white/30 truncate">{h.hypothesis}</p>
                    </div>
                    <span className={`text-xs px-2 py-0.5 rounded-full shrink-0 ${hMeta.badgeClass}`}>{hMeta.label}</span>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Footer */}
        <footer className="text-center text-white/20 text-xs space-y-1 pb-4">
          <p>ContradictAI · xlm-roberta-large-xnli + Q-Learning</p>
          <p>Backend en <code className="font-mono">localhost:8000</code></p>
        </footer>
      </div>
    </div>
  );
}
