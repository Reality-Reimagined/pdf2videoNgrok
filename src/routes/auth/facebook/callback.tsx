import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { socialService } from '../../../lib/social-service';

export function FacebookCallback() {
  const navigate = useNavigate();

  useEffect(() => {
    const handleCallback = async () => {
      try {
        const code = new URLSearchParams(window.location.search).get('code');
        if (!code) throw new Error('No code provided');
        
        await socialService.handleFacebookCallback(code);
        navigate('/dashboard/social-accounts', { 
          state: { success: 'Facebook account connected successfully!' }
        });
      } catch (error) {
        console.error('Facebook auth error:', error);
        navigate('/dashboard/social-accounts', { 
          state: { error: 'Failed to connect Facebook account' }
        });
      }
    };

    handleCallback();
  }, [navigate]);

  return <div>Connecting to Facebook...</div>;
} 