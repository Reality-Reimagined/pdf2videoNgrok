import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { tiktokService } from '../../../lib/social-service';

export function TikTokCallback() {
  const navigate = useNavigate();

  useEffect(() => {
    const handleCallback = async () => {
      try {
        const code = new URLSearchParams(window.location.search).get('code');
        if (!code) throw new Error('No code provided');

        await tiktokService.handleTikTokAuth(code);

        navigate('/dashboard/social-accounts', { 
          state: { success: 'TikTok account connected successfully!' }
        });
      } catch (error) {
        console.error('TikTok auth error:', error);
        navigate('/dashboard/social-accounts', { 
          state: { error: 'Failed to connect TikTok account' }
        });
      }
    };

    handleCallback();
  }, [navigate]);

  return (
    <div className="flex items-center justify-center h-screen">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-black"></div>
    </div>
  );
} 