import { useEffect, useState } from 'react';
import { useVideoStore } from '../../lib/store';
import { CreateVideoDialog } from '../video/CreateVideoDialog';
import { Video } from 'lucide-react';
import { ShareButton } from '../share/ShareButton';

export function VideoList() {
  const { videos, refreshVideos } = useVideoStore();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const loadVideos = async () => {
      try {
        setLoading(true);
        await refreshVideos();
      } catch (err) {
        console.error('Error loading videos:', err);
        setError('Failed to load videos');
      } finally {
        setLoading(false);
      }
    };

    loadVideos();
  }, [refreshVideos]);

  // Callback when video is generated
  const handleVideoGenerated = async () => {
    try {
      setLoading(true);
      await refreshVideos();
    } catch (err) {
      console.error('Error refreshing videos:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-end">
        <button
          onClick={() => setOpen(true)}
          className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
        >
          <Video className="w-4 h-4 mr-2" />
          Create New Video
        </button>
      </div>

      {loading && <div className="text-center py-4">Loading videos...</div>}
      
      {error && <div className="text-center py-4 text-red-600">{error}</div>}
      
      {!loading && !error && videos.length === 0 && (
        <div className="text-center py-8 text-gray-500">
          No videos yet. Click "Create New Video" to get started!
        </div>
      )}

      {!loading && !error && videos.length > 0 && (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {videos.map((video, index) => (
            <div key={index} className="p-4 border rounded-lg shadow">
              <video 
                controls 
                className="w-full rounded-lg mb-2"
                src={video.url}
              >
                Your browser does not support the video tag.
              </video>
              <div className="flex justify-between items-center">
                <div>
                  <h3 className="font-semibold">{video.title}</h3>
                  <p className="text-sm text-gray-600">
                    {new Date(video.created_at).toLocaleDateString()}
                  </p>
                </div>
                <ShareButton 
                  videoUrl={video.url} 
                  title={video.title}
                />
              </div>
              {video.has_subtitles && (
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                  Subtitles
                </span>
              )}
            </div>
          ))}
        </div>
      )}

      <CreateVideoDialog 
        open={open} 
        onOpenChange={setOpen} 
        onVideoGenerated={handleVideoGenerated}
      />
    </div>
  );
}