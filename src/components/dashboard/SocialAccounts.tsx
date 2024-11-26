import React, { useEffect, useState, useRef } from 'react';
import { Facebook, Instagram, Twitter, Youtube, Video, Calendar, CheckCircle, Plus, Trash2, Clock, Share2, ChevronUp, ChevronDown } from 'lucide-react';
import { toast } from 'react-hot-toast';
import { supabase } from '../../lib/supabase';
import { useAuthStore } from '../../lib/store';
import DatePicker from 'react-datepicker';
import "react-datepicker/dist/react-datepicker.css";

interface SocialAccount {
  platform: 'facebook' | 'instagram' | 'twitter' | 'youtube' | 'tiktok';
  connected: boolean;
  username?: string;
  access_token?: string;
}

interface ScheduledPost {
  id: string;
  platform: string;
  title: string;
  scheduled_for: string;
  status: 'pending' | 'completed' | 'failed';
  video_url: string;
}

interface VideoPreviewProps {
  url: string;
}

const VideoPreview: React.FC<VideoPreviewProps> = ({ url }) => {
  if (!url) return null;
  
  return (
    <div className="relative aspect-video rounded-lg overflow-hidden bg-gray-100">
      <video 
        src={url} 
        className="w-full h-full object-cover"
        controls
      />
    </div>
  );
};

// Add a delay hook for video hover
const useDelayedHover = (delay: number = 500) => {
  const [isHovering, setIsHovering] = useState(false);
  const timerRef = useRef<NodeJS.Timeout>();

  const handleMouseEnter = () => {
    timerRef.current = setTimeout(() => {
      setIsHovering(true);
    }, delay);
  };

  const handleMouseLeave = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }
    setIsHovering(false);
  };

  return { isHovering, handleMouseEnter, handleMouseLeave };
};

// Modify the video grid item component
const VideoGridItem = ({ video, isSelected, onSelect }: {
  video: { name: string; url: string; thumbnail?: string };
  isSelected: boolean;
  onSelect: (url: string) => void;
}) => {
  const { isHovering, handleMouseEnter, handleMouseLeave } = useDelayedHover(800);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (videoRef.current) {
      if (isHovering) {
        videoRef.current.play().catch(() => {});
      } else {
        videoRef.current.pause();
        videoRef.current.currentTime = 0;
      }
    }
  }, [isHovering]);

  return (
    <div
      onClick={() => onSelect(video.url)}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      className={`relative rounded-lg overflow-hidden cursor-pointer border-2 transition-all duration-200 ${
        isSelected ? 'border-blue-500' : 'border-transparent'
      } ${isHovering ? 'transform scale-105' : ''}`}
    >
      <video
        ref={videoRef}
        src={video.url}
        className="w-full h-24 object-cover" // Decreased size
        // preload="none" // Only load when needed
        muted
        playsInline
      />
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black to-transparent p-2">
        <p className="text-white text-xs truncate">{video.name}</p>
      </div>
    </div>
  );
};

// Update the scheduled posts section to include video previews
const ScheduledPostCard = ({ post }: { post: ScheduledPost }) => {
  const [showPreview, setShowPreview] = useState(false);

  return (
    <div className="bg-gray-50 dark:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-600 overflow-hidden">
      <div className="p-4">
        <div className="flex justify-between items-start">
          <div className="space-y-1">
            <p className="font-medium text-gray-900 dark:text-white">{post.title}</p>
            <div className="flex items-center space-x-2 text-sm text-gray-500 dark:text-gray-400">
              <Clock className="h-4 w-4" />
              <span>{new Date(post.scheduled_for).toLocaleString()}</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                post.status === 'pending' ? 'bg-yellow-100 text-yellow-800' :
                post.status === 'completed' ? 'bg-green-100 text-green-800' :
                'bg-red-100 text-red-800'
              }`}>
                {post.status}
              </span>
              <span className="text-sm text-gray-500">
                {post.platform.charAt(0).toUpperCase() + post.platform.slice(1)}
              </span>
            </div>
          </div>
          <div className="flex space-x-2">
            <button
              onClick={() => setShowPreview(!showPreview)}
              className="p-2 text-gray-400 hover:text-blue-600 rounded-full hover:bg-gray-100 dark:hover:bg-gray-600"
            >
              {showPreview ? (
                <ChevronUp className="h-5 w-5" />
              ) : (
                <ChevronDown className="h-5 w-5" />
              )}
            </button>
            <button
              onClick={() => deleteScheduledPost(post.id)}
              className="p-2 text-gray-400 hover:text-red-600 rounded-full hover:bg-gray-100 dark:hover:bg-gray-600"
            >
              <Trash2 className="h-5 w-5" />
            </button>
          </div>
        </div>
      </div>
      {showPreview && (
        <div className="border-t border-gray-200 dark:border-gray-600">
          <video
            src={post.video_url}
            className="w-full h-48 object-cover"
            controls
            preload="none"
          />
        </div>
      )}
    </div>
  );
};

export const SocialAccounts = () => {
  const { user } = useAuthStore();
  const [accounts, setAccounts] = useState<SocialAccount[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [scheduledPosts, setScheduledPosts] = useState<ScheduledPost[]>([]);
  const [selectedDate, setSelectedDate] = useState<Date | null>(null);
  const [selectedPlatform, setSelectedPlatform] = useState<string>('');
  const [postTitle, setPostTitle] = useState('');
  const [selectedVideo, setSelectedVideo] = useState<string>('');
  const [userVideos, setUserVideos] = useState<Array<{name: string, url: string}>>([]);
  const [videoSource, setVideoSource] = useState<'upload' | 'existing'>('existing');

  useEffect(() => {
    loadConnectedAccounts();
    loadScheduledPosts();
    loadUserVideos();
  }, []);

  const loadConnectedAccounts = async () => {
    try {
      if (!user) return;

      const { data, error } = await supabase
        .from('social_accounts')
        .select('*')
        .eq('user_id', user.id);

      if (error) throw error;

      setAccounts(data || []);
    } catch (err) {
      console.error('Error loading social accounts:', err);
      setError('Failed to load connected accounts');
    } finally {
      setLoading(false);
    }
  };

  const loadScheduledPosts = async () => {
    const { data } = await supabase
      .from('scheduled_posts')
      .select('*')
      .order('scheduled_for', { ascending: true });
    
    setScheduledPosts(data || []);
  };

  const loadUserVideos = async () => {
    try {
      const { data: videos, error } = await supabase
        .from('videos')  // assuming your table is called 'videos'
        .select('*')
        .eq('user_id', user?.id)
        .order('created_at', { ascending: false });

      if (error) throw error;
      setUserVideos(videos || []);
    } catch (error) {
      console.error('Error loading videos:', error);
      toast.error('Failed to load your videos');
    }
  };

  const handleSchedulePost = async () => {
    if (!selectedDate || !selectedPlatform || !postTitle || !selectedVideo) {
      toast.error('Please fill in all fields');
      return;
    }

    try {
      await supabase.from('scheduled_posts').insert({
        platform: selectedPlatform,
        title: postTitle,
        scheduled_for: selectedDate.toISOString(),
        status: 'pending',
        video_url: selectedVideo
      });

      toast.success('Post scheduled successfully!');
      loadScheduledPosts();
      setSelectedDate(null);
      setSelectedPlatform('');
      setPostTitle('');
      setSelectedVideo('');
    } catch (error) {
      toast.error('Failed to schedule post');
    }
  };

  const deleteScheduledPost = async (id: string) => {
    try {
      await supabase.from('scheduled_posts').delete().eq('id', id);
      toast.success('Scheduled post deleted');
      loadScheduledPosts();
    } catch (error) {
      toast.error('Failed to delete scheduled post');
    }
  };

  const connectFacebook = () => {
    const redirectUri = `${window.location.origin}/auth/facebook/callback`;
    const facebookAuthUrl = `https://www.facebook.com/v12.0/dialog/oauth?client_id=${
      import.meta.env.VITE_FACEBOOK_APP_ID
    }&redirect_uri=${encodeURIComponent(redirectUri)}&scope=public_profile,email`;
    
    window.location.href = facebookAuthUrl;
  };

  const connectInstagram = () => {
    const redirectUri = `${window.location.origin}/auth/instagram/callback`;
    const instagramAuthUrl = `https://api.instagram.com/oauth/authorize?client_id=${
      import.meta.env.VITE_INSTAGRAM_APP_ID
    }&redirect_uri=${encodeURIComponent(redirectUri)}&scope=user_profile,user_media&response_type=code`;
    
    window.location.href = instagramAuthUrl;
  };

  const connectTwitter = () => {
    const redirectUri = `${window.location.origin}/auth/twitter/callback`;
    const twitterAuthUrl = `https://twitter.com/i/oauth2/authorize?client_id=${
      import.meta.env.VITE_TWITTER_CLIENT_ID
    }&redirect_uri=${encodeURIComponent(redirectUri)}&scope=tweet.write,tweet.read,users.read&response_type=code`;
    
    window.location.href = twitterAuthUrl;
  };

  const connectYoutube = () => {
    const redirectUri = `${window.location.origin}/auth/youtube/callback`;
    const scope = 'https://www.googleapis.com/auth/youtube.upload';
    const youtubeAuthUrl = `https://accounts.google.com/o/oauth2/v2/auth?client_id=${
      import.meta.env.VITE_YOUTUBE_CLIENT_ID
    }&redirect_uri=${encodeURIComponent(redirectUri)}
    &response_type=code&scope=${encodeURIComponent(scope)}
    &access_type=offline&prompt=consent`;
    
    window.location.href = youtubeAuthUrl;
  };

  const connectTikTok = () => {
    const clientKey = import.meta.env.VITE_TIKTOK_CLIENT_KEY;
    const redirectUri = `${window.location.origin}/auth/tiktok/callback`;
    
    if (!clientKey) {
      console.error('TikTok client key is not defined');
      return;
    }

    const params = new URLSearchParams({
      client_key: clientKey,
      redirect_uri: redirectUri,
      scope: 'video.upload,user.info.basic',
      response_type: 'code'
    });

    const tiktokAuthUrl = `https://www.tiktok.com/auth/authorize?${params.toString()}`;
    
    console.log('TikTok Auth URL:', tiktokAuthUrl); // For debugging
    window.location.href = tiktokAuthUrl;
  };

  // Custom styles for the DatePicker
  const datePickerCustomStyles = {
    datePickerContainer: "relative",
    datePickerInput: "w-full px-4 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500",
    datePickerCalendar: "bg-white shadow-lg rounded-lg border border-gray-200 mt-2"
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded relative">
          {error}
        </div>
      )}

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {/* Facebook Card */}
        <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Facebook className="h-8 w-8 text-blue-600" />
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-white">Facebook</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {accounts.find(a => a.platform === 'facebook')?.username || 'Not connected'}
                </p>
              </div>
            </div>
            <button
              onClick={connectFacebook}
              className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-white bg-blue-500 hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              {accounts.find(a => a.platform === 'facebook') ? (
                <CheckCircle className="h-5 w-5" />
              ) : (
                <Plus className="h-5 w-5" />
              )}
            </button>
          </div>
        </div>

        {/* Instagram Card */}
        <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Instagram className="h-8 w-8 text-pink-600" />
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-white">Instagram</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {accounts.find(a => a.platform === 'instagram')?.username || 'Not connected'}
                </p>
              </div>
            </div>
            <button
              onClick={connectInstagram}
              className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-white bg-pink-500 hover:bg-pink-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-pink-500"
            >
              {accounts.find(a => a.platform === 'instagram') ? (
                <CheckCircle className="h-5 w-5" />
              ) : (
                <Plus className="h-5 w-5" />
              )}
            </button>
          </div>
        </div>

        {/* Twitter Card */}
        <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Twitter className="h-8 w-8 text-sky-500" />
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-white">Twitter</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {accounts.find(a => a.platform === 'twitter')?.username || 'Not connected'}
                </p>
              </div>
            </div>
            <button
              onClick={connectTwitter}
              className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-white bg-sky-500 hover:bg-sky-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-sky-500"
            >
              {accounts.find(a => a.platform === 'twitter') ? (
                <CheckCircle className="h-5 w-5" />
              ) : (
                <Plus className="h-5 w-5" />
              )}
            </button>
          </div>
        </div>

        {/* YouTube Card */}
        <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Youtube className="h-8 w-8 text-red-600" />
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-white">YouTube</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {accounts.find(a => a.platform === 'youtube')?.username || 'Not connected'}
                </p>
              </div>
            </div>
            <button
              onClick={connectYoutube}
              className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-white bg-red-500 hover:bg-red-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
            >
              {accounts.find(a => a.platform === 'youtube') ? (
                <CheckCircle className="h-5 w-5" />
              ) : (
                <Plus className="h-5 w-5" />
              )}
            </button>
          </div>
        </div>

        {/* TikTok Card */}
        <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Video className="h-8 w-8 text-black dark:text-white" />
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-white">TikTok</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {accounts.find(a => a.platform === 'tiktok')?.username || 'Not connected'}
                </p>
              </div>
            </div>
            <button
              onClick={connectTikTok}
              className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-white bg-black hover:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500"
            >
              {accounts.find(a => a.platform === 'tiktok') ? (
                <CheckCircle className="h-5 w-5" />
              ) : (
                <Plus className="h-5 w-5" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Scheduling Dashboard */}
      <div className="bg-white dark:bg-gray-800 shadow rounded-lg overflow-hidden">
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
              Schedule Posts
            </h2>
            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800">
              <Clock className="h-4 w-4 mr-1" />
              Upcoming: {scheduledPosts.filter(p => p.status === 'pending').length}
            </span>
          </div>
        </div>

        <div className="p-6">
          <div className="grid gap-8 lg:grid-cols-2">
            {/* Left Column - Schedule Form */}
            <div className="space-y-6">
              {/* Video Selection */}
              <div className="space-y-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Video Source
                </label>
                <div className="flex space-x-4 mb-4">
                  <button
                    onClick={() => setVideoSource('existing')}
                    className={`px-4 py-2 rounded-md text-sm font-medium ${
                      videoSource === 'existing'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    Existing Videos
                  </button>
                  <button
                    onClick={() => setVideoSource('upload')}
                    className={`px-4 py-2 rounded-md text-sm font-medium ${
                      videoSource === 'upload'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    Upload New
                  </button>
                </div>

                {videoSource === 'existing' ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4 max-h-[300px] overflow-y-auto">
                      {userVideos.map((video) => (
                        <VideoGridItem
                          key={video.name}
                          video={video}
                          isSelected={selectedVideo === video.url}
                          onSelect={(url) => setSelectedVideo(url)}
                        />
                      ))}
                    </div>
                    {userVideos.length === 0 && (
                      <p className="text-gray-500 text-center py-4">
                        No videos available. Try uploading one!
                      </p>
                    )}
                  </div>
                ) : (
                  <div>
                    <input
                      type="file"
                      accept="video/*"
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) {
                          const url = URL.createObjectURL(file);
                          setSelectedVideo(url);
                        }
                      }}
                      className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                    />
                  </div>
                )}

                {selectedVideo && (
                  <div className="mt-4">
                    <VideoPreview url={selectedVideo} />
                  </div>
                )}
              </div>

              {/* Platform Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Platform
                </label>
                <select
                  value={selectedPlatform}
                  onChange={(e) => setSelectedPlatform(e.target.value)}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600"
                >
                  <option value="">Select Platform</option>
                  {accounts.map(account => (
                    <option key={account.platform} value={account.platform}>
                      {account.platform.charAt(0).toUpperCase() + account.platform.slice(1)}
                    </option>
                  ))}
                </select>
              </div>

              {/* Date Picker */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Schedule Date & Time
                </label>
                <div className={datePickerCustomStyles.datePickerContainer}>
                  <DatePicker
                    selected={selectedDate}
                    onChange={setSelectedDate}
                    showTimeSelect
                    timeFormat="HH:mm"
                    timeIntervals={15}
                    dateFormat="MMMM d, yyyy h:mm aa"
                    minDate={new Date()}
                    className={datePickerCustomStyles.datePickerInput}
                    placeholderText="Select date and time"
                    calendarClassName={datePickerCustomStyles.datePickerCalendar}
                  />
                </div>
              </div>

              {/* Post Title */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Post Title
                </label>
                <input
                  type="text"
                  value={postTitle}
                  onChange={(e) => setPostTitle(e.target.value)}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600"
                  placeholder="Enter post title"
                />
              </div>

              <button
                onClick={handleSchedulePost}
                disabled={!selectedVideo || !selectedPlatform || !selectedDate || !postTitle}
                className="w-full inline-flex justify-center items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Calendar className="h-4 w-4 mr-2" />
                Schedule Post
              </button>
            </div>

            {/* Right Column - Scheduled Posts */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white flex items-center">
                <Share2 className="h-5 w-5 mr-2" />
                Upcoming Posts
              </h3>
              <div className="space-y-3 max-h-[600px] overflow-y-auto">
                {scheduledPosts.length === 0 ? (
                  <p className="text-gray-500 dark:text-gray-400 text-center py-8">
                    No scheduled posts yet
                  </p>
                ) : (
                  scheduledPosts.map(post => (
                    <ScheduledPostCard key={post.id} post={post} />
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Development Mode Section */}
      {import.meta.env.DEV && (
        <div className="mt-8 bg-white dark:bg-gray-800 shadow rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Development Mode: Manual Token Input
          </h3>
          <div className="space-y-4">
            {['facebook', 'instagram', 'twitter', 'youtube', 'tiktok'].map((platform) => (
              <div key={platform} className="space-y-2">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 capitalize">
                  {platform} Access Token
                </label>
                <input
                  type="password"
                  onChange={(e) => {
                    if (!user) return;
                    supabase
                      .from('social_accounts')
                      .upsert({
                        user_id: user.id,
                        platform,
                        access_token: e.target.value,
                        username: 'Manual Connection',
                        updated_at: new Date().toISOString()
                      })
                      .then(() => loadConnectedAccounts());
                  }}
                  className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  placeholder={`Enter ${platform} access token`}
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}; 