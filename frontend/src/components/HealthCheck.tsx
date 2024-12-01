import { useEffect, useState } from 'react';
import { API_BASE_URL } from '../api-service';

export const HealthCheck = () => {
  const [status, setStatus] = useState<'loading' | 'connected' | 'error'>('loading');

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
          setStatus('connected');
        } else {
          setStatus('error');
        }
      } catch (error) {
        setStatus('error');
        console.error('Health check failed:', error);
      }
    };

    checkHealth();
  }, []);

  return (
    <div className="health-status">
      API Status: {status}
    </div>
  );
}; 