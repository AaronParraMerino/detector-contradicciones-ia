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
  not_contradiction: {
    icon: "✓",
    label: "No Contradicción",
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
        <span className="capitalize" style={{ color }}>
          {label}
        </span>
        <span className="text-white/60">{pct}%</span>
      </div>
      <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "oklch(25% 0.02 270)" }}>
        <div className="h-full rounded-full score-bar-fill" style={{ "--target-w": `${pct}%`, background: color }} />
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

const splitText = (input) => {
  if (input.includes(".")) {
    const parts = input.split(".");
    return [parts[0].trim(), parts.slice(1).join(".").trim()];
  }
  if (input.includes(",")) {
    const parts = input.split(",");
    return [parts[0].trim(), parts.slice(1).join(",").trim()];
  }
  const words = input.trim().split(/\s+/);
  const mid = Math.floor(words.length / 2);
  return [words.slice(0, mid).join(" "), words.slice(mid).join(" ")];
};

export default function App() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [feedbackLoading, setFeedbackLoading] = useState(false);
  const [correctLabel, setCorrectLabel] = useState("");
  const resultsRef = useRef(null);

  const detect = useCallback(async () => {
    if (!text.trim()) return;
    const [premise, hypothesis] = splitText(text);
    if (!premise || !hypothesis) {
      setError("No se pudo dividir el texto en premisa e hipótesis.");
      return;
    }
    setLoading(true);
    setResult(null);
    setError(null);
    setFeedbackSent(false);
    setCorrectLabel("");
    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ premise, hypothesis }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setResult({ premise, hypothesis, ...data });
      setHistory((h) => [{ premise, hypothesis, ...data, ts: Date.now() }, ...h.slice(0, 9)]);
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [text]);

  const sendFeedback = useCallback(
    async (accepted) => {
      if (!result) return;
      setFeedbackLoading(true);
      try {
        const payload = {
          premise: result.premise,
          hypothesis: result.hypothesis,
          prediction: result.prediction,
          confidence: result.confidence,
          correct_label: accepted ? result.prediction : correctLabel,
        };

        const res = await fetch(`${API}/feedback`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        setFeedbackSent(true);
      } catch (e) {
        setError(e.message);
      } finally {
        setFeedbackLoading(false);
      }
    },
    [result, correctLabel],
  );

  const labelKey = result?.prediction?.toLowerCase();
  const meta = LABEL_META[labelKey] ?? LABEL_META.neutral;

  return (
    <div className="relative min-h-dvh overflow-x-hidden" style={{ fontFamily: "var(--font-sans)" }}>
      {/* Background orbs */}

      <div className="relative z-10 max-w-3xl mx-auto px-4 py-12 space-y-10">
        {/* Header */}
        <header className="text-center space-y-3">
          <div
            className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium mb-2"
            style={{
              background: "oklch(58% 0.26 270 / 0.15)",
              border: "1px solid oklch(58% 0.26 270 / 0.3)",
              color: "oklch(76% 0.16 270)",
            }}
          >
            <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: "oklch(76% 0.16 270)" }} />
            IA · Aprendizaje por Refuerzo
          </div>
          <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight gradient-text">
            Detector de contradicciones en texto
          </h1>
          <p className="text-white/50 text-base max-w-md mx-auto leading-relaxed">
            Detecta contradicciones entre enunciados usando un modelo entrenado en Google Colab.
          </p>
        </header>

        {/* Input Card */}
        <div className="glass rounded-2xl p-6 space-y-4">
          <div className="space-y-2">
            <label htmlFor="text-input" className="text-xs font-semibold uppercase tracking-widest text-white/40">
              Texto
            </label>
            <div className="glow-focus rounded-xl transition-all duration-300">
              <textarea
                id="text-input"
                rows={4}
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="El cielo es azul durante el día. El cielo es completamente negro a mediodía."
                onKeyDown={(e) => {
                  if (e.key === "Enter" && e.ctrlKey) detect();
                }}
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

          <button
            id="detect-btn"
            onClick={detect}
            disabled={loading || !text.trim()}
            className="w-full flex items-center justify-center gap-2 py-3 rounded-xl font-semibold text-sm transition-all duration-200 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed"
            style={{
              background: "linear-gradient(135deg, oklch(58% 0.26 270), oklch(50% 0.28 270))",
              color: "white",
              boxShadow: "0 0 20px oklch(58% 0.26 270 / 0.3)",
            }}
          >
            {loading ? (
              <>
                <Spinner /> Analizando…
              </>
            ) : (
              <>
                <span>🔍</span> Detectar contradicción
              </>
            )}
          </button>
          <p className="text-center text-white/25 text-xs">Ctrl + Enter para analizar</p>
        </div>

        {/* Error */}
        {error && (
          <div
            className="animate-fade-in-up glass rounded-xl px-5 py-4 flex items-center gap-3 text-sm"
            style={{ borderColor: "oklch(65% 0.25 25 / 0.4)", color: "oklch(75% 0.22 25)" }}
          >
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
                    <span className={`px-3 py-1 rounded-full text-sm font-bold ${meta.badgeClass}`}>{meta.label}</span>
                    {result.source === "cache" && (
                      <span className="ml-2 text-xs text-white/40 font-mono">· caché humano</span>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Scores */}
            {result.confidence !== undefined && (
              <div className="space-y-3">
                <p className="text-xs font-semibold uppercase tracking-widest text-white/40">Confianza</p>
                <div className="space-y-3">
                  <ScoreBar label={meta.label} score={result.confidence} color={meta.barColor} />
                </div>
              </div>
            )}

            {/* Feedback */}
            {!feedbackSent ? (
              <div className="space-y-3 pt-2 border-t" style={{ borderColor: "oklch(30% 0.025 270 / 0.4)" }}>
                <p className="text-xs font-semibold uppercase tracking-widest text-white/40">
                  ¿Es correcto este resultado?
                </p>
                <div className="flex gap-3 flex-wrap">
                  <button
                    onClick={() => sendFeedback(true)}
                    disabled={feedbackLoading}
                    className="flex-1 py-2 rounded-xl text-sm font-semibold transition-all duration-200 active:scale-95 disabled:opacity-40"
                    style={{
                      background: "oklch(70% 0.2 145 / 0.15)",
                      border: "1px solid oklch(70% 0.2 145 / 0.4)",
                      color: "oklch(75% 0.18 145)",
                    }}
                  >
                    {feedbackLoading ? <Spinner /> : "👍 Correcto"}
                  </button>
                  <button
                    onClick={() => sendFeedback(false)}
                    disabled={feedbackLoading || !correctLabel}
                    className="flex-1 py-2 rounded-xl text-sm font-semibold transition-all duration-200 active:scale-95 disabled:opacity-40"
                    style={{
                      background: "oklch(65% 0.25 25 / 0.15)",
                      border: "1px solid oklch(65% 0.25 25 / 0.4)",
                      color: "oklch(75% 0.22 25)",
                    }}
                  >
                    {feedbackLoading ? <Spinner /> : "👎 Incorrecto (Enviar)"}
                  </button>
                </div>
                <div className="space-y-1">
                  <label htmlFor="correct-label" className="text-xs text-white/40">
                    Etiqueta correcta (seleccionar para corregir):
                  </label>
                  <select
                    id="correct-label"
                    value={correctLabel}
                    onChange={(e) => setCorrectLabel(e.target.value)}
                    className="w-full rounded-lg px-3 py-2 text-sm"
                    style={{
                      background: "oklch(19% 0.02 270)",
                      border: "1px solid oklch(30% 0.025 270 / 0.6)",
                      color: "oklch(92% 0.01 270)",
                    }}
                  >
                    <option value="">— Seleccionar —</option>
                    <option value="contradiction">Contradicción</option>
                    <option value="not_contradiction">No Contradicción</option>
                    <option value="neutral">Neutral</option>
                  </select>
                </div>
              </div>
            ) : (
              <div
                className="animate-fade-in-up text-sm px-4 py-3 rounded-xl flex items-center gap-2"
                style={{
                  background: "oklch(58% 0.26 270 / 0.12)",
                  border: "1px solid oklch(58% 0.26 270 / 0.3)",
                  color: "oklch(76% 0.16 270)",
                }}
              >
                ✅ Feedback enviado y almacenado.
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
                const hMeta = LABEL_META[h.prediction?.toLowerCase()] ?? LABEL_META.neutral;
                return (
                  <button
                    key={h.ts}
                    onClick={() => {
                      setText(`${h.premise}. ${h.hypothesis}`);
                      setResult(h);
                      setFeedbackSent(false);
                      setCorrectLabel("");
                    }}
                    className="w-full text-left glass rounded-xl px-4 py-3 flex items-center gap-3 group transition-all duration-200 hover:border-white/20"
                  >
                    <span className="text-lg shrink-0">{hMeta.icon}</span>
                    <div className="min-w-0 flex-1">
                      <p className="text-xs text-white/50 truncate">{h.premise}</p>
                      <p className="text-xs text-white/30 truncate">{h.hypothesis}</p>
                    </div>
                    <span className={`text-xs px-2 py-0.5 rounded-full shrink-0 ${hMeta.badgeClass}`}>
                      {hMeta.label}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Footer */}
        <footer className="text-center text-white/20 text-xs space-y-1 pb-4">
          <p>ContradictAI · Aprendizaje por Refuerzo</p>
          <p>
            Backend en <code className="font-mono">localhost:8000</code>
          </p>
        </footer>
      </div>
    </div>
  );
}
