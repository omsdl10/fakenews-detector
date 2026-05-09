/**
 * Centralised API client.
 * All HTTP calls go through here — base URL, auth header, error normalisation.
 */

import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "/api/v1";

export const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 30_000, // 30s — model inference can be slow
  headers: { "Content-Type": "application/json" },
});

// Attach JWT from localStorage on every request
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem("access_token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// Normalise errors to always have a .message string
apiClient.interceptors.response.use(
  (res) => res,
  (err) => {
    const detail =
      err.response?.data?.detail ||
      err.response?.data?.message ||
      err.message ||
      "Unknown error";
    err.displayMessage = Array.isArray(detail)
      ? detail.map((d) => d.msg || d).join("; ")
      : String(detail);
    return Promise.reject(err);
  }
);

// ── API calls ────────────────────────────────────────────────────────────────

export const api = {
  predict: (payload) => apiClient.post("/predict/", payload),
  uploadArticle: (payload) => apiClient.post("/upload-article/", payload),
  searchEvidence: (payload) => apiClient.post("/search-evidence/", payload),
  health: () => apiClient.get("/health"),
  login: (email, password) =>
    apiClient.post("/auth/token", { email, password }),
  register: (email, password) =>
    apiClient.post("/auth/register", { email, password }),
};
