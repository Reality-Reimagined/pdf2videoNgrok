import React from 'react';
import * as Switch from '@radix-ui/react-switch';
import { useVideoStore } from '../lib/store';

export const SubtitlesToggle = () => {
  const { addSubtitles, setAddSubtitles } = useVideoStore();

  return (
    <div className="flex items-center gap-2">
      <Switch.Root
        checked={addSubtitles}
        onCheckedChange={setAddSubtitles}
        className="w-11 h-6 bg-gray-200 rounded-full relative data-[state=checked]:bg-blue-600 outline-none cursor-pointer"
      >
        <Switch.Thumb className="block w-5 h-5 bg-white rounded-full shadow-lg transition-transform duration-100 translate-x-0.5 will-change-transform data-[state=checked]:translate-x-[22px]" />
      </Switch.Root>
      <label className="text-sm font-medium text-gray-700">
        Add Subtitles
      </label>
    </div>
  );
};