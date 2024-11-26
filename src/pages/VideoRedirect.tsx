import { useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { supabase } from '../lib/supabase';

export const VideoRedirect = () => {
  const { shortCode } = useParams();
  const navigate = useNavigate();

  useEffect(() => {
    const redirectToVideo = async () => {
      try {
        // Update click count and get original URL
        const { data, error } = await supabase
          .from('short_urls')
          .update({ clicks: supabase.raw('clicks + 1') })
          .eq('short_code', shortCode)
          .select('original_url')
          .single();

        if (error || !data) {
          navigate('/404');
          return;
        }

        window.location.href = data.original_url;
      } catch (error) {
        navigate('/404');
      }
    };

    redirectToVideo();
  }, [shortCode, navigate]);

  return (
    <div className="flex items-center justify-center h-screen">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
    </div>
  );
}; 