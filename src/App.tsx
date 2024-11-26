import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Auth } from './pages/Auth';
import { Dashboard } from './pages/Dashboard';
import { useAuthStore } from './lib/store';
import { FacebookCallback } from './routes/auth/facebook/callback';
import { SocialAccounts } from './components/dashboard/SocialAccounts';
import { VideoRedirect } from './pages/VideoRedirect';
import { Toaster } from 'react-hot-toast';
import './styles/datepicker.css';

function App() {
  const { isAuthenticated } = useAuthStore();

  return (
    <>
      <Router>
        <Routes>
          <Route 
            path="/auth" 
            element={isAuthenticated ? <Navigate to="/dashboard" replace /> : <Auth />} 
          />
          <Route 
            path="/dashboard/*" 
            element={isAuthenticated ? <Dashboard /> : <Navigate to="/auth" replace />} 
          />
          <Route 
            path="/" 
            element={<Navigate to={isAuthenticated ? "/dashboard" : "/auth"} replace />} 
          />
          <Route path="/dashboard/social-accounts" element={<SocialAccounts />} />
          <Route path="/auth/facebook/callback" element={<FacebookCallback />} />
          <Route path="/v/:shortCode" element={<VideoRedirect />} />
        </Routes>
      </Router>
      <Toaster position="top-right" />
    </>
  );
}

export default App;