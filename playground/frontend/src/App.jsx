import { useEffect, useState } from "react";

const TYPES = ["choice", "score", "noul"];

// The field each answer type's value lives under, in both the API response
// and the Python call -- drives the "result.answers.x.y" breadcrumb below.
const VALUE_FIELD = { choice: "choice", score: "score", noul: "noul" };

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

// The Python one would actually write to get this exact result -- mirrors
// the request, the way Jev's playground keeps its "send code" box in sync
// with the request editor.
function pythonCall(state, apiQuestions) {
  const stateLiteral = JSON.stringify(state.length > 60 ? state.slice(0, 57) + "..." : state);
  const questionsLiteral = JSON.stringify(apiQuestions, null, 4).replace(/\n/g, "\n");
  return `import openjev

agent = openjev.load("checkpoint")
result = agent.predict(\n    ${stateLiteral},\n    ${questionsLiteral},\n)`;
}

function StepHeading({ step, title, hint }) {
  return (
    <h2>
      <span className="step">{step}</span>
      {title}
      {hint && <small>{hint}</small>}
    </h2>
  );
}

function Bar({ label, prob, isTop }) {
  return (
    <div className={"bar-row" + (isTop ? " top" : "")}>
      <span className="bar-label">{label}</span>
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${prob * 100}%` }} />
      </div>
      <span className="bar-pct">{(prob * 100).toFixed(1)}%</span>
    </div>
  );
}

function AnswerCard({ qid, answer, type }) {
  const probs = Object.entries(answer.probabilities).sort((a, b) => b[1] - a[1]);
  const top = probs[0][0];
  const field = VALUE_FIELD[type];
  const value = field === "choice" ? answer.choice : answer[field].toFixed(2);

  return (
    <div className={"answer " + type}>
      <div className="answer-head">
        <span className="badge">{type}</span>
        <span className="answer-id">{qid}</span>
        <span className="answer-value">{value}</span>
      </div>
      <div className="answer-path">
        result["answers"]["{qid}"]["{field}"]
      </div>
      {probs.map(([label, prob]) => (
        <Bar key={label} label={label} prob={prob} isTop={label === top} />
      ))}
      <div className="confidence">
        confidence <b>{answer.answer_confidence.toFixed(2)}</b>
      </div>
    </div>
  );
}

export default function App() {
  const [state, setState] = useState("");
  const [questions, setQuestions] = useState([]);
  const [status, setStatus] = useState(null);
  const [result, setResult] = useState(null);
  const [askedTypes, setAskedTypes] = useState({});
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
    const apiQuestions = toApiQuestions(questions);
    // Snapshot qid -> type now: the result card needs it to pick a color and a
    // path, and the live `questions` list can drift (edited, reordered) by the
    // time the response lands.
    const types = {};
    for (const [qid, q] of Object.entries(apiQuestions)) types[qid] = q.type;
    setAskedTypes(types);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state, questions: apiQuestions }),
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

      <main className="workspace">
        <section className="pane">
          <StepHeading step="1" title="state" hint="the text being evaluated" />
          <textarea value={state} onChange={(e) => setState(e.target.value)} rows={10} />
          <button className="link-button" onClick={loadExample}>load example case</button>

          <StepHeading step="2" title="questions" hint="typed questions about the state" />
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
          <button className="run" onClick={ask} disabled={loading || !state.trim()}>
            {loading ? "asking..." : "ask ⌘↵"}
          </button>
        </section>

        <section className="pane">
          <StepHeading step="3" title="decision" hint="one forward pass answers every question" />
          {error && <div className="error">{error}</div>}
          {!error && !result && <div className="placeholder">ask to see typed answers here</div>}
          {result && (
            <>
              <div className="usage">
                <span>{result.usage.forward_passes} forward pass</span>
                <span>{result.usage.questions} questions</span>
                <span>{result.usage.ms.toFixed(0)}ms</span>
                {result.usage.truncated && <span className="truncated-badge">state truncated</span>}
              </div>
              {Object.entries(result.answers).map(([qid, answer]) => (
                <AnswerCard key={qid} qid={qid} answer={answer} type={askedTypes[qid]} />
              ))}
            </>
          )}

          <StepHeading step="4" title="how it's called" hint="the equivalent openjev call" />
          <pre className="code">{pythonCall(state, toApiQuestions(questions))}</pre>
        </section>
      </main>
    </div>
  );
}
