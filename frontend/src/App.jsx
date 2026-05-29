import { useCallback, useMemo, useRef, useState } from "react";

const API = "http://localhost:8000";

const LABEL_META = {
  contradiction: {
    label: "Contradiccion",
    title: "Hay contradicciones",
    badgeClass: "badge-contradiction",
    color: "oklch(64% 0.23 25)",
    barColor: "oklch(64% 0.23 25)",
  },
  entailment: {
    label: "Inferencia",
    title: "Inferencias detectadas",
    badgeClass: "badge-entailment",
    color: "oklch(66% 0.18 145)",
    barColor: "oklch(66% 0.18 145)",
  },
  neutral: {
    label: "Neutral",
    title: "Sin contradiccion fuerte",
    badgeClass: "badge-neutral",
    color: "oklch(72% 0.12 260)",
    barColor: "oklch(62% 0.16 260)",
  },
};

function Spinner() {
  return (
    <svg className="spin" width="18" height="18" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeOpacity="0.25" />
      <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

function ScoreBar({ label, score, color }) {
  const pct = Math.round((score ?? 0) * 100);

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

function RLBadge({ rl }) {
  if (!rl) return null;

  const confidenceState = rl.state?.split("|")[0] ?? "STATE";
  const stateColor = { LOW: "#f59e0b", MEDIUM: "#8b5cf6", HIGH: "#ef4444" }[confidenceState] ?? "#6b7280";
  const readableState = rl.state?.split("|").slice(0, 4).join(" / ") ?? "sin estado";

  return (
    <div className="flex items-center gap-2 flex-wrap text-xs">
      <span
        className="px-2 py-0.5 rounded-md font-mono font-medium"
        style={{ background: `${stateColor}22`, color: stateColor, border: `1px solid ${stateColor}44` }}
      >
        {readableState}
      </span>
      <span className="text-white/40">{rl.decision_type}</span>
      <span className={rl.show_alert ? "text-red-300" : "text-white/40"}>
        {rl.show_alert ? "alertar" : "observar"}
      </span>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="rounded-lg px-3 py-3" style={{ background: "oklch(18% 0.02 270)" }}>
      <p className="text-[10px] uppercase tracking-widest text-white/35">{label}</p>
      <p className="text-xl font-semibold text-white/85 mt-1">{value}</p>
    </div>
  );
}

function PairCard({ pair, active, onSelect }) {
  const meta = LABEL_META[pair.label] ?? LABEL_META.neutral;
  const confidence = Math.round((pair.confidence ?? 0) * 100);
  const sourceLabel = {
    cache: "memoria exacta",
    memory: "memoria semantica",
    model: "modelo NLI",
  }[pair.source] ?? pair.source;

  return (
    <button
      type="button"
      onClick={() => onSelect?.(pair.id)}
      className="w-full text-left glass rounded-lg p-4 space-y-3 transition-colors"
      style={{
        borderColor: active ? meta.color : "oklch(30% 0.025 270 / 0.5)",
        background: active ? "oklch(20% 0.025 270 / 0.9)" : undefined,
      }}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="text-xs font-mono text-white/45">
          P{pair.premise_paragraph + 1} F{pair.premise_index + 1}
          {" -> "}
          P{pair.hypothesis_paragraph + 1} F{pair.hypothesis_index + 1}
        </div>
        <div className="flex items-center gap-2">
          <span className={`px-2 py-0.5 rounded-md text-xs font-bold ${meta.badgeClass}`}>{meta.label}</span>
          <span className="text-xs text-white/40">{confidence}%</span>
        </div>
      </div>

      <div className="flex flex-wrap gap-2 text-[11px] text-white/40">
        <span>Fuente: {sourceLabel}</span>
        {pair.calibration && <span>Calibracion: {pair.calibration}</span>}
        {pair.shared_entities?.length > 0 && <span>Entidad: {pair.shared_entities.join(", ")}</span>}
        {pair.shared_terms?.length > 0 && <span>Terminos: {pair.shared_terms.slice(0, 4).join(", ")}</span>}
      </div>

      <div className="grid sm:grid-cols-2 gap-3 text-sm">
        <div>
          <p className="text-[10px] uppercase tracking-widest text-white/35 mb-1">Fragmento base</p>
          <p className="text-white/75 leading-relaxed">{pair.premise}</p>
        </div>
        <div>
          <p className="text-[10px] uppercase tracking-widest text-white/35 mb-1">Fragmento comparado</p>
          <p className="text-white/75 leading-relaxed">{pair.hypothesis}</p>
        </div>
      </div>

      <RLBadge rl={pair.rl_decision} />
    </button>
  );
}

function HighlightedDocument({ text, highlights, activePairId, onSelectPair }) {
  const parts = useMemo(() => {
    if (!text) return [];

    const ordered = [...(highlights ?? [])]
      .filter((mark) => Number.isInteger(mark.start) && Number.isInteger(mark.end) && mark.end > mark.start)
      .sort((a, b) => a.start - b.start);

    const output = [];
    let cursor = 0;

    ordered.forEach((mark, index) => {
      if (mark.start < cursor) return;

      if (mark.start > cursor) {
        output.push({ type: "text", text: text.slice(cursor, mark.start), key: `text-${index}-${cursor}` });
      }

      output.push({
        type: "mark",
        text: text.slice(mark.start, mark.end),
        key: `mark-${mark.segment_id}-${index}`,
        mark,
      });
      cursor = mark.end;
    });

    if (cursor < text.length) {
      output.push({ type: "text", text: text.slice(cursor), key: `text-tail-${cursor}` });
    }

    return output;
  }, [text, highlights]);

  if (!parts.length) {
    return <p className="text-sm text-white/45">El resultado resaltado aparecera despues del analisis.</p>;
  }

  return (
    <div className="doc-preview">
      {parts.map((part) => {
        if (part.type === "text") {
          return <span key={part.key}>{part.text}</span>;
        }

        const isActive =
          part.mark.pair_id === activePairId || part.mark.pair_ids?.includes(activePairId);

        return (
          <mark
            key={part.key}
            className={`doc-mark ${isActive ? "doc-mark-active" : ""}`}
            onClick={() => onSelectPair?.(part.mark.pair_id)}
            title={`Parrafo ${part.mark.paragraph_index + 1}, confianza ${Math.round(part.mark.confidence * 100)}%`}
          >
            {part.text}
          </mark>
        );
      })}
    </div>
  );
}

export default function App() {
  const [documentText, setDocumentText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [feedbackLoading, setFeedbackLoading] = useState(false);
  const [correctLabel, setCorrectLabel] = useState("");
  const [totalCorrections, setTotalCorrections] = useState(null);
  const [history, setHistory] = useState([]);
  const [activePairId, setActivePairId] = useState(null);
  const resultsRef = useRef(null);

  const stats = useMemo(() => {
    const trimmed = documentText.trim();
    return {
      words: trimmed ? trimmed.split(/\s+/).length : 0,
      chars: documentText.length,
      paragraphs: trimmed ? trimmed.split(/\n\s*\n/).filter(Boolean).length : 0,
    };
  }, [documentText]);

  const orderedComparisons = useMemo(() => {
    const comparisons = result?.comparisons ?? [];
    return [...comparisons].sort((a, b) => {
      if (a.label === "contradiction" && b.label !== "contradiction") return -1;
      if (a.label !== "contradiction" && b.label === "contradiction") return 1;
      return (b.confidence ?? 0) - (a.confidence ?? 0);
    });
  }, [result]);

  const activePair = useMemo(() => {
    if (!result) return null;
    const pairs = [...(result.conflicts ?? []), ...(result.comparisons ?? [])];
    return pairs.find((pair) => pair.id === activePairId) ?? result.strongest_pair ?? pairs[0] ?? null;
  }, [result, activePairId]);

  const detect = useCallback(async () => {
    const text = documentText.trim();
    if (!text) return;

    setLoading(true);
    setResult(null);
    setError(null);
    setFeedbackSent(false);
    setCorrectLabel("");
    setActivePairId(null);

    try {
      const res = await fetch(`${API}/detect-document`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, max_pairs: 180 }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data = await res.json();
      const firstPair = data.conflicts?.[0] ?? data.strongest_pair ?? data.comparisons?.[0] ?? null;
      setResult(data);
      setActivePairId(firstPair?.id ?? null);
      setHistory((items) => [{ text, ...data, ts: Date.now() }, ...items.slice(0, 7)]);
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [documentText]);

  const sendFeedback = useCallback(
    async (accepted) => {
      if (!result) return;

      const pair = activePair;
      const confidence = pair?.confidence ?? result.scores?.[result.label] ?? 0.5;
      const action = pair?.rl_decision?.action ?? result.rl_decision?.action ?? 1;

      setFeedbackLoading(true);

      try {
        const body = { confidence, action, accepted };
        if (pair?.agent_context) {
          body.context = pair.agent_context;
        }

        if (!accepted && correctLabel && pair) {
          body.correct_label = correctLabel;
          body.premise = pair.premise;
          body.hypothesis = pair.hypothesis;
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
      } catch (e) {
        setError(e.message);
      } finally {
        setFeedbackLoading(false);
      }
    },
    [result, activePair, correctLabel],
  );

  const labelKey = result?.label?.toLowerCase();
  const meta = LABEL_META[labelKey] ?? LABEL_META.neutral;
  const scores = result?.scores ?? {};
  const agentSummary = result?.agent_summary;
  const memory = agentSummary?.memory;
  const policy = agentSummary?.policy;

  return (
    <div className="relative min-h-dvh overflow-x-hidden" style={{ fontFamily: "var(--font-sans)" }}>
      <div className="relative z-10 max-w-7xl mx-auto px-4 py-8 space-y-8">
        <header className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-widest text-white/35">ContradictAI</p>
            <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white">
              Analisis de coherencia documental
            </h1>
            <p className="text-sm text-white/45 mt-2 max-w-2xl">
              NLP para comparar enunciados y Q-learning para ajustar cuando el agente debe alertar.
            </p>
          </div>
          <div className="grid grid-cols-3 gap-2 text-center min-w-[300px]">
            <Metric label="Palabras" value={stats.words} />
            <Metric label="Parrafos" value={stats.paragraphs} />
            <Metric label="Caracteres" value={stats.chars} />
          </div>
        </header>

        <main className="grid xl:grid-cols-[minmax(0,1fr)_380px] gap-6 items-start">
          <section className="space-y-4">
            <div className="rounded-lg p-3" style={{ background: "oklch(18% 0.02 270)", border: "1px solid oklch(30% 0.025 270 / 0.5)" }}>
              <div className="rounded-lg bg-slate-50 text-slate-950 shadow-2xl shadow-black/20 overflow-hidden">
                <div className="flex items-center justify-between gap-3 px-5 py-3 border-b border-slate-200 bg-white">
                  <input
                    value="Documento de analisis"
                    readOnly
                    className="bg-transparent text-sm font-semibold text-slate-700 outline-none w-full"
                    aria-label="Titulo del documento"
                  />
                  <span className="text-xs text-slate-400 shrink-0">editor</span>
                </div>

                <textarea
                  value={documentText}
                  onChange={(e) => setDocumentText(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && e.ctrlKey) detect();
                  }}
                  placeholder={
                    "Escribe o pega aqui un texto largo. El sistema analizara parrafos, frases y clausulas largas para ubicar contradicciones.\n\nEjemplo:\nJuan esta durmiendo profundamente. Mas tarde, el informe afirma que Juan esta despierto y trabajando en la oficina."
                  }
                  className="w-full min-h-[540px] resize-y px-8 py-8 text-base leading-8 bg-slate-50 text-slate-900 placeholder-slate-400"
                  style={{ fontFamily: "Georgia, serif" }}
                />
              </div>
            </div>

            <div className="flex flex-col sm:flex-row gap-3">
              <button
                onClick={detect}
                disabled={loading || !documentText.trim()}
                className="flex-1 flex items-center justify-center gap-2 py-3 rounded-lg font-semibold text-sm transition-all duration-200 active:scale-[0.99] disabled:opacity-40 disabled:cursor-not-allowed"
                style={{
                  background: "oklch(52% 0.17 255)",
                  color: "white",
                  boxShadow: "0 10px 30px oklch(10% 0.02 270 / 0.3)",
                }}
              >
                {loading ? (
                  <>
                    <Spinner /> Analizando documento
                  </>
                ) : (
                  "Analizar coherencia"
                )}
              </button>
              <button
                onClick={() => {
                  setDocumentText("");
                  setResult(null);
                  setError(null);
                  setFeedbackSent(false);
                  setActivePairId(null);
                }}
                disabled={loading || (!documentText && !result)}
                className="px-5 py-3 rounded-lg text-sm font-semibold transition-all duration-200 disabled:opacity-40"
                style={{
                  background: "oklch(24% 0.022 270)",
                  border: "1px solid oklch(35% 0.03 270)",
                  color: "oklch(92% 0.01 270)",
                }}
              >
                Limpiar
              </button>
            </div>

            {error && (
              <div
                className="animate-fade-in-up glass rounded-lg px-5 py-4 text-sm"
                style={{ borderColor: "oklch(65% 0.25 25 / 0.4)", color: "oklch(75% 0.22 25)" }}
              >
                {error}. Revisa que el backend este corriendo en localhost:8000.
              </div>
            )}
          </section>

          <aside className="space-y-4">
            <div className="glass rounded-lg p-5 space-y-4">
              <div>
                <p className="text-xs font-semibold uppercase tracking-widest text-white/35">Agente inteligente</p>
                <p className="text-sm text-white/70 mt-1">
                  Modelo NLI, memoria JSON y politica Q-learning contextual.
                </p>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <Metric label="Memoria" value={memory?.corrections ?? 0} />
                <Metric label="Estados" value={policy?.learned_states ?? 0} />
                <Metric label="Alertas" value={agentSummary?.alerted_conflicts ?? 0} />
                <Metric label="Conflictos" value={result?.conflicts?.length ?? 0} />
              </div>
              <div className="rounded-lg p-3 text-xs space-y-2" style={{ background: "oklch(18% 0.02 270)" }}>
                <div className="flex justify-between gap-3">
                  <span className="text-white/40">Correcciones exactas</span>
                  <span className="text-white/75">{memory?.exact_items ?? 0}</span>
                </div>
                <div className="flex justify-between gap-3">
                  <span className="text-white/40">Recompensas positivas</span>
                  <span className="text-white/75">{policy?.positive_rewards ?? 0}</span>
                </div>
                <div className="flex justify-between gap-3">
                  <span className="text-white/40">Recompensas negativas</span>
                  <span className="text-white/75">{policy?.negative_rewards ?? 0}</span>
                </div>
                <div className="flex justify-between gap-3">
                  <span className="text-white/40">Exploracion epsilon</span>
                  <span className="text-white/75">{policy?.epsilon ?? 0}</span>
                </div>
              </div>
              {result?.truncated && (
                <p className="text-xs text-yellow-200/80">
                  El documento fue limitado a los primeros {result.segments?.length ?? 0} fragmentos para mantener respuesta rapida.
                </p>
              )}
            </div>

            {history.length > 0 && (
              <div className="glass rounded-lg p-5 space-y-3">
                <p className="text-xs font-semibold uppercase tracking-widest text-white/35">Historial</p>
                {history.map((item) => {
                  const itemMeta = LABEL_META[item.label] ?? LABEL_META.neutral;
                  return (
                    <button
                      key={item.ts}
                      onClick={() => {
                        const firstPair = item.conflicts?.[0] ?? item.strongest_pair ?? item.comparisons?.[0] ?? null;
                        setDocumentText(item.text);
                        setResult(item);
                        setFeedbackSent(false);
                        setActivePairId(firstPair?.id ?? null);
                      }}
                      className="w-full text-left rounded-lg px-3 py-3 transition-colors hover:bg-white/5"
                      style={{ border: "1px solid oklch(30% 0.025 270 / 0.4)" }}
                    >
                      <div className="flex items-center justify-between gap-2 mb-1">
                        <span className={`text-[10px] px-2 py-0.5 rounded-md font-bold ${itemMeta.badgeClass}`}>
                          {itemMeta.label}
                        </span>
                        <span className="text-[10px] text-white/30">{item.analysis_count ?? 0} pares</span>
                      </div>
                      <p className="text-xs text-white/50 line-clamp-2">{item.text}</p>
                    </button>
                  );
                })}
              </div>
            )}
          </aside>
        </main>

        {result && (
          <section ref={resultsRef} className="animate-fade-in-up space-y-5">
            <div className="grid xl:grid-cols-[minmax(0,1fr)_390px] gap-6">
              <div className="glass rounded-lg p-6 space-y-5">
                <div className="flex items-start justify-between flex-wrap gap-4">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-widest text-white/35">Resultado general</p>
                    <h2 className="text-2xl font-bold mt-1" style={{ color: meta.color }}>
                      {meta.title}
                    </h2>
                    <p className="text-sm text-white/45 mt-1">
                      {result.analysis_count} pares analizados de {result.total_possible_pairs} posibles.
                    </p>
                  </div>
                  <span className={`px-3 py-1 rounded-md text-sm font-bold ${meta.badgeClass}`}>{meta.label}</span>
                </div>

                {result.message && <p className="text-sm text-white/55">{result.message}</p>}

                <div className="grid md:grid-cols-3 gap-4">
                  {Object.entries(scores).map(([label, score]) => {
                    const item = LABEL_META[label] ?? LABEL_META.neutral;
                    return <ScoreBar key={label} label={item.label} score={score} color={item.barColor} />;
                  })}
                </div>
                <p className="text-xs text-white/35">
                  Las barras resumen la evidencia agregada del documento; el detalle de cada par mantiene su confianza individual.
                </p>

                <div className="space-y-3">
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-xs font-semibold uppercase tracking-widest text-white/35">Documento resaltado</p>
                    <p className="text-xs text-white/35">{result.highlights?.length ?? 0} marcas</p>
                  </div>
                  <HighlightedDocument
                    text={documentText}
                    highlights={result.highlights}
                    activePairId={activePairId}
                    onSelectPair={setActivePairId}
                  />
                </div>
              </div>

              <div className="glass rounded-lg p-5 space-y-4">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-widest text-white/35">Par activo</p>
                  <p className="text-sm text-white/55 mt-1">
                    Selecciona una marca del documento o una comparacion para revisar el detalle.
                  </p>
                </div>

                {activePair ? (
                  <PairCard pair={activePair} active onSelect={setActivePairId} />
                ) : (
                  <p className="text-sm text-white/45">No hay pares destacados.</p>
                )}

                {!feedbackSent ? (
                  <div className="space-y-3 pt-4 border-t" style={{ borderColor: "oklch(30% 0.025 270 / 0.4)" }}>
                    <p className="text-xs font-semibold uppercase tracking-widest text-white/35">Feedback del agente</p>
                    <div className="grid grid-cols-2 gap-3">
                      <button
                        onClick={() => sendFeedback(true)}
                        disabled={feedbackLoading || !activePair}
                        className="py-2.5 rounded-lg text-sm font-semibold transition-all disabled:opacity-40"
                        style={{
                          background: "oklch(70% 0.2 145 / 0.15)",
                          border: "1px solid oklch(70% 0.2 145 / 0.4)",
                          color: "oklch(75% 0.18 145)",
                        }}
                      >
                        {feedbackLoading ? <Spinner /> : "Correcto"}
                      </button>
                      <button
                        onClick={() => sendFeedback(false)}
                        disabled={feedbackLoading || !activePair}
                        className="py-2.5 rounded-lg text-sm font-semibold transition-all disabled:opacity-40"
                        style={{
                          background: "oklch(65% 0.25 25 / 0.15)",
                          border: "1px solid oklch(65% 0.25 25 / 0.4)",
                          color: "oklch(75% 0.22 25)",
                        }}
                      >
                        Incorrecto
                      </button>
                    </div>
                    <select
                      value={correctLabel}
                      onChange={(e) => setCorrectLabel(e.target.value)}
                      className="w-full rounded-lg px-3 py-2 text-sm"
                      style={{
                        background: "oklch(19% 0.02 270)",
                        border: "1px solid oklch(30% 0.025 270 / 0.6)",
                        color: "oklch(92% 0.01 270)",
                      }}
                    >
                      <option value="">Etiqueta correcta</option>
                      <option value="contradiction">Contradiccion</option>
                      <option value="entailment">Inferencia</option>
                      <option value="neutral">Neutral</option>
                    </select>
                  </div>
                ) : (
                  <div
                    className="text-sm px-4 py-3 rounded-lg"
                    style={{
                      background: "oklch(58% 0.26 270 / 0.12)",
                      border: "1px solid oklch(58% 0.26 270 / 0.3)",
                      color: "oklch(76% 0.16 270)",
                    }}
                  >
                    Feedback registrado. {totalCorrections !== null && `${totalCorrections} correcciones en memoria.`}
                    {" "}La politica Q-learning fue actualizada para este tipo de caso.
                  </div>
                )}
              </div>
            </div>

            {orderedComparisons.length > 0 && (
              <div className="space-y-3">
                <p className="text-xs font-semibold uppercase tracking-widest text-white/35">
                  Comparaciones principales
                </p>
                <div className="grid lg:grid-cols-2 gap-3">
                  {orderedComparisons.map((pair) => (
                    <PairCard
                      key={pair.id}
                      pair={pair}
                      active={pair.id === activePairId}
                      onSelect={setActivePairId}
                    />
                  ))}
                </div>
              </div>
            )}

            {result.segments?.length > 0 && (
              <div className="glass rounded-lg p-5 space-y-3">
                <p className="text-xs font-semibold uppercase tracking-widest text-white/35">Estructura detectada</p>
                <div className="grid md:grid-cols-2 gap-2">
                  {result.segments.map((segment) => (
                    <button
                      key={segment.id}
                      type="button"
                      onClick={() => {
                        const pair = orderedComparisons.find(
                          (item) => item.premise_index === segment.id || item.hypothesis_index === segment.id,
                        );
                        if (pair) setActivePairId(pair.id);
                      }}
                      className="rounded-lg px-3 py-2 text-sm text-left text-white/65 transition-colors hover:bg-white/5"
                      style={{ background: "oklch(19% 0.02 270)" }}
                    >
                      <span className="font-mono text-white/35 mr-2">
                        P{segment.paragraph_index + 1} F{segment.id + 1}
                      </span>
                      {segment.text}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </section>
        )}
      </div>
    </div>
  );
}
