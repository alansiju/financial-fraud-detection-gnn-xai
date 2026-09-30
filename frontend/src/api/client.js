import axios from 'axios';

const API_BASE = 'http://localhost:8000';

const client = axios.create({
  baseURL: API_BASE,
  timeout: 60000,   // 60 s default; scan requests override to 300 s per-call
});


// Attach JWT token to every request
client.interceptors.request.use((config) => {
  const token = localStorage.getItem('fs_token');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// Auto-logout on 401
client.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('fs_token');
      localStorage.removeItem('fs_user');
      window.location.href = '/login';
    }
    return Promise.reject(err);
  }
);

export default client;
