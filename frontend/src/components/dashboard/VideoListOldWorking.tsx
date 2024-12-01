import { useEffect, useState } from 'react';
import { apiService } from '../../lib/api-service';
import { CreateVideoDialog } from '../video/CreateVideoDialog';
import { Video } from 'lucide-react';

interface Video {
  title: string;
  description: string;
  videoPath: string;
  createdAt: string;
  hasSubtitles: boolean;
}
export function VideoList() {
  const [videos, setVideos] = useState<Video[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

  // Function to fetch videos
  const fetchVideos = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiService.getVideos();
      setVideos(data);
    } catch (error) {
      console.error('Error fetching videos:', error);
      setError('Failed to load videos. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchVideos();
  }, []);

  // Callback when video is generated
  const handleVideoGenerated = () => {
    fetchVideos(); // Refresh the video list
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
                // src={`http://localhost:5050/videos/${video.videoPath}`}
                src={video.url}
              >
                Your browser does not support the video tag.
              </video>
              <h3 className="font-semibold">{video.title}</h3>
              <p className="text-sm text-gray-600">
                {new Date(video.createdAt).toLocaleDateString()}
              </p>
              {video.hasSubtitles && (
                <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                  With Subtitles
                </span>
              )}
            </div>
          ))}
        </div>
      )}

      <CreateVideoDialog 
        open={open} 
        onOpenChange={setOpen} 
        onVideoGenerated={handleVideoGenerated} // Pass callback to refresh videos
      />
    </div>
  );
}

// export function VideoList() {
//   const [videos, setVideos] = useState<Video[]>([]);
//   const [loading, setLoading] = useState(true);
//   const [error, setError] = useState<string | null>(null);
//   const [open, setOpen] = useState(false);

//   useEffect(() => {
//     const fetchVideos = async () => {
//       try {
//         setLoading(true);
//         setError(null);
//         const data = await apiService.getVideos();
//         setVideos(data);
//       } catch (error) {
//         console.error('Error fetching videos:', error);
//         setError('Failed to load videos. Please try again later.');
//       } finally {
//         setLoading(false);
//       }
//     };

//     fetchVideos();
//   }, []);

//   return (
//     <div className="space-y-4">
//       <div className="flex justify-end">
//         <button
//           onClick={() => setOpen(true)}
//           className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
//         >
//           <Video className="w-4 h-4 mr-2" />
//           Create New Video
//         </button>
//       </div>

//       {loading && <div className="text-center py-4">Loading videos...</div>}
      
//       {error && <div className="text-center py-4 text-red-600">{error}</div>}
      
//       {!loading && !error && videos.length === 0 && (
//         <div className="text-center py-8 text-gray-500">
//           No videos yet. Click "Create New Video" to get started!
//         </div>
//       )}

//       {!loading && !error && videos.length > 0 && (
//         <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
//           {videos.map((video, index) => (
//             <div key={index} className="p-4 border rounded-lg shadow">
//               <video 
//                 controls 
//                 className="w-full rounded-lg mb-2"
//                 src={video.url}
//               >
//                 Your browser does not support the video tag.
//               </video>
//               <h3 className="font-semibold">{video.title}</h3>
//               <p className="text-sm text-gray-600">
//                 {new Date(video.createdAt).toLocaleDateString()}
//               </p>
//               {video.hasSubtitles && (
//                 <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
//                   With Subtitles
//                 </span>
//               )}
//             </div>
//           ))}
//         </div>
//       )}

//       <CreateVideoDialog open={open} onOpenChange={setOpen} />
//     </div>
//   );
// }