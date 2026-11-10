import { useEffect, useState } from "react";

const TYPES = ["choice", "score", "noul"];

function blankQuestion() {
  return {
    id: crypto.randomUUID(),
    qid: "",
    type: "choice",
    instructions: "",
    options: [{ key: "", description: "" }],
  };
}

// {qid, type, instructions, options:[{key,description}]} -> the API's question shape.
function toApiQuestions(rows) {
  const out = {};
  for (const row of rows) {
    if (!row.qid.trim()) continue;
    if (row.type === "choice") {
      const criteria = {};
      for (const o of row.options) if (o.key.trim()) criteria[o.key.trim()] = o.description;
      out[row.qid.trim()] = { type: "choice", instructions: row.instructions, criteria };
    } else if (row.type === "score") {
      out[row.qid.trim()] = {
        type: "score",
        instructions: row.instructions,
        criteria: row.options.map((o) => o.description || o.key),
      };
    } else {
      out[row.qid.trim()] = { type: "noul", instructions: row.instructions };
    }
  }
  return out;
}

// The API's question shape -> editable rows (the inverse of toApiQuestions).
function fromApiQuestions(questions) {
  return Object.entries(questions).map(([qid, q]) => {
    let options = [{ key: "", description: "" }];
    if (q.type === "choice") {
      options = Object.entries(q.criteria).map(([key, description]) => ({ key, description }));
    } else if (q.type === "score") {
      options = q.criteria.map((description, i) => ({ key: `level ${i}`, description }));
    }
    return { id: crypto.randomUUID(), qid, type: q.type, instructions: q.instructions, options };
  });
}

function Bar({ label, prob, isTop }) {
  return (
    <div className="bar-row">
      <span className="bar-label">{label}</span>
      <div className="bar-track">
        <div className={"bar-fill" + (isTop ? " top" : "")} style={{ width: `${prob * 100}%` }} />
      </div>
      <span className="bar-pct">{(prob * 100).toFixed(1)}%</span>
    </div>
  );
}

function Panel({ step, title, children, footer }) {
  return (
    <section className="panel">
      <div className="panel-bar">
        <span className="panel-step">{step}</span>
        <span className="panel-title">{title}</span>
      </div>
      <div className="panel-body">{children}</div>
      {footer}
    </section>
  );
}

export default function App() {
  const [state, setState] = useState("");
  const [questions, setQuestions] = useState([]);
  const [status, setStatus] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch("/api/status").then((r) => r.json()).then(setStatus).catch(() => {});
    loadExample();
  }, []);

  function loadExample() {
    fetch("/api/example")
      .then((r) => r.json())
      .then((data) => {
        setState(data.state);
        setQuestions(fromApiQuestions(data.questions));
      })
      .catch((e) => setError(String(e)));
  }

  function updateQuestion(id, patch) {
    setQuestions((qs) => qs.map((q) => (q.id === id ? { ...q, ...patch } : q)));
  }

  function updateOption(qId, idx, patch) {
    setQuestions((qs) =>
      qs.map((q) =>
        q.id !== qId
          ? q
          : { ...q, options: q.options.map((o, i) => (i === idx ? { ...o, ...patch } : o)) }
      )
    );
  }

  function addOption(qId) {
    setQuestions((qs) =>
      qs.map((q) => (q.id === qId ? { ...q, options: [...q.options, { key: "", description: "" }] } : q))
    );
  }

  function removeOption(qId, idx) {
    setQuestions((qs) =>
      qs.map((q) => (q.id === qId ? { ...q, options: q.options.filter((_, i) => i !== idx) } : q))
    );
  }

  function addQuestion() {
    setQuestions((qs) => [...qs, blankQuestion()]);
  }

  function removeQuestion(id) {
    setQuestions((qs) => qs.filter((q) => q.id !== id));
  }

  async function ask() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state, questions: toApiQuestions(questions) }),
      });
      if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
      setResult(await res.json());
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header>
        <div className="brand">
          <img src="/logo.svg" alt="" className="brand-mark" />
          <div>
            <div className="brand-name">openjev</div>
            <div className="brand-caption">decision playground</div>
          </div>
        </div>
        {status && <div className={`caveat caveat-${status.mode}`}>{status.caveat}</div>}
      </header>

      <div className="workspace">
        <Panel step="1" title="state">
          <textarea value={state} onChange={(e) => setState(e.target.value)} rows={10} />
          <button className="link-button" onClick={loadExample}>load example case</button>
        </Panel>

        <Panel
          step="2"
          title="questions"
          footer={
            <button className="ask-button" onClick={ask} disabled={loading || !state.trim()}>
              {loading ? "asking..." : "ask → see the decision"}
            </button>
          }
        >
          <button className="add-question-button" onClick={addQuestion}>+ question</button>
          {questions.map((q) => (
            <div className="question-card" key={q.id}>
              <div className="question-row">
                <input
                  className="qid-input"
                  placeholder="question id"
                  value={q.qid}
                  onChange={(e) => updateQuestion(q.id, { qid: e.target.value })}
                />
                <select value={q.type} onChange={(e) => updateQuestion(q.id, { type: e.target.value })}>
                  {TYPES.map((t) => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
                <button className="icon-button" onClick={() => removeQuestion(q.id)}>remove</button>
              </div>
              <input
                className="instructions-input"
                placeholder="instructions"
                value={q.instructions}
                onChange={(e) => updateQuestion(q.id, { instructions: e.target.value })}
              />
              {q.type !== "noul" && (
                <div className="options-list">
                  {q.options.map((o, idx) => (
                    <div className="option-row" key={idx}>
                      <input
                        placeholder={q.type === "score" ? `level ${idx}` : "option"}
                        value={o.key}
                        onChange={(e) => updateOption(q.id, idx, { key: e.target.value })}
                      />
                      <input
                        placeholder="description (optional)"
                        value={o.description}
                        onChange={(e) => updateOption(q.id, idx, { description: e.target.value })}
                      />
                      <button className="icon-button" onClick={() => removeOption(q.id, idx)}>x</button>
                    </div>
                  ))}
                  <button className="link-button" onClick={() => addOption(q.id)}>+ option</button>
                </div>
              )}
            </div>
          ))}
        </Panel>

        <Panel step="3" title="decision">
          {error && <div className="error">{error}</div>}
          {!error && !result && <div className="empty-hint">ask to see typed answers here</div>}
          {result && (
            <>
              <div className="usage">
                {result.usage.forward_passes} forward pass &middot; {result.usage.questions} questions &middot;{" "}
                {result.usage.ms.toFixed(0)}ms
                {result.usage.truncated && <span className="truncated-badge">state truncated</span>}
              </div>
              {Object.entries(result.answers).map(([qid, answer]) => {
                const probs = Object.entries(answer.probabilities).sort((a, b) => b[1] - a[1]);
                const top = probs[0][0];
                return (
                  <div className="answer-card" key={qid}>
                    <h3>{qid}</h3>
                    {probs.map(([label, prob]) => (
                      <Bar key={label} label={label} prob={prob} isTop={label === top} />
                    ))}
                  </div>
                );
              })}
            </>
          )}
        </Panel>
      </div>
    </div>
  );
}
