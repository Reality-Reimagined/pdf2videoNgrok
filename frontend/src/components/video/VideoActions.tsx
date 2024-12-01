import React, { useState } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { Pencil, Trash2, MoreVertical } from 'lucide-react';
import { supabaseService } from '../../lib/supabase-service';
import type { Video } from '../../lib/supabase-service';

interface VideoActionsProps {
  videoId: string;
  currentTitle: string;
  onDelete: () => void;
  onUpdate: () => void;
}

export const VideoActions: React.FC<VideoActionsProps> = ({
  videoId,
  currentTitle,
  onDelete,
  onUpdate,
}) => {
  const [isEditOpen, setIsEditOpen] = useState(false);
  const [newTitle, setNewTitle] = useState(currentTitle);
  const [isLoading, setIsLoading] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDelete = async () => {
    if (!window.confirm('Are you sure you want to delete this video?')) {
      return;
    }

    try {
      setIsLoading(true);
      setError(null);
      
      await supabaseService.deleteVideo(videoId);
      console.log('Video deleted successfully');
      setShowDropdown(false);
      onDelete();
    } catch (error) {
      console.error('Error deleting video:', error);
      setError('Failed to delete video. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleUpdate = async () => {
    if (!newTitle.trim()) {
      setError('Title cannot be empty');
      return;
    }

    try {
      setIsLoading(true);
      setError(null);
      
      const updatedVideo = await supabaseService.updateVideoTitle(videoId, newTitle.trim());
      console.log('Video updated:', updatedVideo);
      
      setIsEditOpen(false);
      setShowDropdown(false);
      onUpdate();
    } catch (error: any) {
      console.error('Error updating video:', error);
      setError(error.message || 'Failed to update video title');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="relative">
      <button
        onClick={() => setShowDropdown(!showDropdown)}
        className="p-2 hover:bg-gray-100 rounded-full"
      >
        <MoreVertical className="w-4 h-4" />
      </button>

      {showDropdown && (
        <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg z-10 border border-gray-200">
          <button
            onClick={() => {
              setIsEditOpen(true);
              setShowDropdown(false);
            }}
            className="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-100 flex items-center gap-2"
          >
            <Pencil className="w-4 h-4" />
            Edit Title
          </button>

          <button
            onClick={handleDelete}
            disabled={isLoading}
            className="w-full px-4 py-2 text-left text-sm text-red-600 hover:bg-red-50 flex items-center gap-2"
          >
            <Trash2 className="w-4 h-4" />
            Delete
          </button>
        </div>
      )}

      <Dialog.Root open={isEditOpen} onOpenChange={setIsEditOpen}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-black/50" />
          <Dialog.Content className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 bg-white rounded-lg shadow-lg p-6 w-full max-w-md">
            <Dialog.Title className="text-lg font-medium mb-4">
              Edit Video Title
            </Dialog.Title>

            {error && (
              <div className="mb-4 p-2 bg-red-50 text-red-600 rounded-md text-sm">
                {error}
              </div>
            )}

            <input
              type="text"
              value={newTitle}
              onChange={(e) => setNewTitle(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Enter new title"
            />

            <div className="mt-4 flex justify-end gap-2">
              <button
                onClick={() => {
                  setIsEditOpen(false);
                  setNewTitle(currentTitle);
                  setError(null);
                }}
                className="px-4 py-2 text-sm text-gray-600 hover:bg-gray-100 rounded-md"
              >
                Cancel
              </button>
              <button
                onClick={handleUpdate}
                disabled={isLoading || !newTitle.trim()}
                className="px-4 py-2 text-sm text-white bg-blue-600 hover:bg-blue-700 rounded-md disabled:opacity-50"
              >
                {isLoading ? 'Saving...' : 'Save'}
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  );
}; 