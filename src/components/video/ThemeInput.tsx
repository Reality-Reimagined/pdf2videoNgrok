import React from 'react';
import { useVideoStore } from '../../lib/store';

export const ThemeInput = () => {
  const { theme, setTheme } = useVideoStore();

  return (
    <div className="flex flex-col gap-2">
      <label htmlFor="theme" className="text-sm font-medium text-gray-700">
        Background Theme
      </label>
      <input
        id="theme"
        type="text"
        value={theme}
        onChange={(e) => setTheme(e.target.value)}
        className="rounded-md border border-gray-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        placeholder="Enter theme (e.g., city, nature)"
      />
    </div>
  );
};