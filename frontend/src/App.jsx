import { useState } from "react";
import "./App.css";

function App() {
  const [feedback, setFeedback] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const analyzeFeedback = async () => {
    if (!feedback.trim()) {
      setError("Please enter customer feedback.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(
        "https://ai-customer-feedback-analyzer-x2zn.onrender.com/analyze",        
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            feedback: feedback,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Something went wrong.");
      }

      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">

      <header className="header">
        <h1>AI Customer Feedback Analyzer</h1>
        <p>
          Analyze customer sentiment, aspects and generate AI insights
        </p>
      </header>

      <main className="container">

        <section className="input-card">

          <h2>Customer Feedback</h2>

          <textarea
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            placeholder="Enter customer feedback here..."
            rows="6"
          />

          <button
            onClick={analyzeFeedback}
            disabled={loading}
          >
            {loading ? "Analyzing..." : "Analyze Feedback"}
          </button>

          {error && (
            <p className="error">
              {error}
            </p>
          )}

        </section>


        {result && (
          <section className="results">
            <div className="result-card">
  <h2>Analyzed Feedback</h2>
  <p className="feedback-display">
    {result.feedback}
  </p>
</div>

            <div className="result-card">
              <h2>Sentiment</h2>

              <div
  className={`sentiment ${result.sentiment.toLowerCase()}`}
>
  {result.sentiment}
</div>
            </div>


            <div className="result-card">
              <h2>Prediction Confidence</h2>

              {Object.entries(result.confidence).map(
                ([label, value]) => (
                  <div className="confidence-row" key={label}>

                    <span>{label}</span>

                    <div className="progress-container">
                      <div
                        className="progress-bar"
                        style={{
                          width: `${value}%`,
                        }}
                      ></div>
                    </div>

                    <span>{value}%</span>

                  </div>
                )
              )}

            </div>


            <div className="result-card">
              <h2>Detected Aspects</h2>

              {result.aspects.length > 0 ? (
                <div className="aspect-list">
                  {result.aspects.map((aspect) => (
                    <span
                      className="aspect"
                      key={aspect}
                    >
                      {aspect}
                    </span>
                  ))}
                </div>
              ) : (
                <p>No specific aspects detected.</p>
              )}

            </div>


            <div className="result-card">
              <h2>AI Insights</h2>

              <div className="ai-insights">
  {result.ai_insights
    .replace(/\*\*/g, "")
    .split("\n")
    .map((line, index) => (
      <p key={index}>
        {line}
      </p>
    ))}
</div>
            </div>

          </section>
        )}

      </main>

    </div>
  );
}

export default App;