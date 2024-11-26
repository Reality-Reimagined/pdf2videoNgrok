import React, { useState, useEffect } from 'react';
import { Share2, Facebook, Instagram, Twitter, Copy, Check, Calendar, Video } from 'lucide-react';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import { createShortUrl } from '../../lib/url-service';
import { tiktokService } from '../../lib/social-service';
import { toast } from 'react-hot-toast';
import { useNavigate } from 'react-router-dom';
import { socialService } from '../../lib/social-service';
import { ScheduleUpload } from './ScheduleUpload';
import { supabase } from '../../lib/supabase';

interface ShareButtonProps {
  videoId: string;
  videoUrl: string;
  title?: string;
}

interface ConnectedAccount {
  platform: string;
  username: string;
}

export const ShareButton = ({ videoId, videoUrl, title = 'Check out my video!' }: ShareButtonProps) => {
  const [shortUrl, setShortUrl] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [connectedAccounts, setConnectedAccounts] = useState<ConnectedAccount[]>([]);
  const [scheduledDate, setScheduledDate] = useState<Date | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    createShortUrl(videoId, videoUrl).then(setShortUrl);
    loadConnectedAccounts();
  }, [videoId, videoUrl]);

  const loadConnectedAccounts = async () => {
    const { data: accounts } = await supabase
      .from('social_accounts')
      .select('platform, username')
      .order('platform');
    
    setConnectedAccounts(accounts || []);
  };

  const handleCopyLink = async () => {
    try {
      await navigator.clipboard.writeText(videoUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const handleShare = async (platform: string) => {
    try {
      setIsLoading(true);
      
      if (scheduledDate && scheduledDate > new Date()) {
        // Schedule the post
        await supabase.from('scheduled_posts').insert({
          platform,
          video_url: videoUrl,
          title,
          scheduled_for: scheduledDate.toISOString(),
          status: 'pending'
        });
        
        toast.success(`Post scheduled for ${scheduledDate.toLocaleString()}`);
      } else {
        // Post immediately
        await socialService.shareToService(platform, videoUrl, title);
        toast.success(`Shared to ${platform} successfully!`);
      }
    } catch (error) {
      console.error(`Error with ${platform}:`, error);
      toast.error(`Failed to process ${platform} request`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleTikTokShare = async () => {
    try {
      setIsLoading(true);
      await tiktokService.uploadVideo(
        videoUrl,
        title || 'Check out my video!'
      );
      toast.success('Video uploaded to TikTok successfully!');
    } catch (error) {
      console.error('Error sharing to TikTok:', error);
      
      // Check if the error is due to not being connected
      if (error.message === 'TikTok account not connected') {
        toast.error('Please connect your TikTok account first');
        // Optional: Redirect to social accounts page
        const shouldRedirect = window.confirm('Would you like to connect your TikTok account now?');
        if (shouldRedirect) {
          navigate('/dashboard/social-accounts');
        }
      } else {
        toast.error('Failed to upload video to TikTok');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const platformIcons = {
    facebook: Facebook,
    instagram: Instagram,
    twitter: Twitter,
    youtube: Video,
    youtube_shorts: Video,
    tiktok: Video
  };

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <button className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
          <Share2 className="h-4 w-4 mr-2" />
          Share
        </button>
      </DropdownMenu.Trigger>

      <DropdownMenu.Portal>
        <DropdownMenu.Content
          className="min-w-[220px] bg-white rounded-md shadow-lg p-1 dark:bg-gray-800"
          sideOffset={5}
        >
          <DropdownMenu.Item
            className="flex items-center px-3 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md cursor-pointer"
            onClick={() => handleShare('facebook')}
          >
            <Facebook className="h-4 w-4 mr-2" />
            Share to Facebook
          </DropdownMenu.Item>

          <DropdownMenu.Item
            className="flex items-center px-3 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md cursor-pointer"
            onClick={() => handleShare('twitter')}
          >
            <Twitter className="h-4 w-4 mr-2" />
            Share to X (Twitter)
          </DropdownMenu.Item>

          <DropdownMenu.Item
            className="flex items-center px-3 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md cursor-pointer"
            onClick={handleCopyLink}
          >
            {copied ? (
              <Check className="h-4 w-4 mr-2 text-green-500" />
            ) : (
              <Copy className="h-4 w-4 mr-2" />
            )}
            {copied ? 'Copied!' : 'Copy Link'}
          </DropdownMenu.Item>

          <DropdownMenu.Item
            className="flex items-center px-3 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md cursor-pointer"
            onClick={handleTikTokShare}
          >
            {isLoading ? (
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-black"></div>
            ) : (
              <Instagram className="h-4 w-4 mr-2" />
            )}
            Share to TikTok
          </DropdownMenu.Item>

          <DropdownMenu.Item
            className="flex items-center px-3 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md cursor-pointer"
            onClick={() => handleShare('youtube')}
          >
            <Video className="h-4 w-4 mr-2" />
            Share to YouTube
          </DropdownMenu.Item>

          <DropdownMenu.Item
            className="flex items-center px-3 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md cursor-pointer"
            onClick={() => handleShare('youtube_shorts')}
          >
            <Video className="h-4 w-4 mr-2" />
            Share to YouTube Shorts
          </DropdownMenu.Item>

          <ScheduleUpload 
            onSchedule={setScheduledDate} 
            scheduledDate={scheduledDate}
          />
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}; 