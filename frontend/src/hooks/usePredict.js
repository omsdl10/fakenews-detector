/**
 * usePredict — manages the full prediction lifecycle:
 *   idle → loading → success | error
 *
 * Returns { predict, result, loading, error, reset }
 */

import { useState, useCallback } from "react";
import { api } from "../api/client";

export function usePredict() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const predict = useCallback(async ({ text, url, deepExplain = false }) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const { data } = await api.predict({
        text: text || undefined,
        url: url || undefined,
        deep_explain: deepExplain,
        retrieve_evidence: true,
      });
      setResult(data);
    } catch (err) {
      setError(err.displayMessage || "Prediction failed");
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { predict, result, loading, error, reset };
}
