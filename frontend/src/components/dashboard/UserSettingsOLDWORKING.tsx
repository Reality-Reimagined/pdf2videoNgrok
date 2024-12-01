import React from 'react';
import { useAuthStore } from '../../lib/store';
import { Bell, Moon, Key } from 'lucide-react';
import * as Switch from '@radix-ui/react-switch';

export const UserSettings = () => {
  const { user, updateUser } = useAuthStore();

  if (!user) return null;

  const handleNotificationToggle = (checked: boolean) => {
    updateUser({
      settings: {
        ...user.settings,
        notifications: checked,
      },
    });
  };

  const handleThemeToggle = (checked: boolean) => {
    updateUser({
      settings: {
        ...user.settings,
        theme: checked ? 'dark' : 'light',
      },
    });
  };

  const handleApiKeyUpdate = (provider: keyof typeof user.apiKeys, value: string) => {
    updateUser({
      apiKeys: {
        ...user.apiKeys,
        [provider]: value,
      },
    });
  };

  return (
    <div className="space-y-6">
      <div className="bg-white shadow rounded-lg p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Notifications</h3>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Bell className="w-5 h-5 text-gray-500" />
            <span className="text-sm text-gray-700">Enable notifications</span>
          </div>
          <Switch.Root
            checked={user.settings.notifications}
            onCheckedChange={handleNotificationToggle}
            className="w-11 h-6 bg-gray-200 rounded-full relative data-[state=checked]:bg-blue-600"
          >
            <Switch.Thumb className="block w-5 h-5 bg-white rounded-full shadow-lg transition-transform duration-100 translate-x-0.5 will-change-transform data-[state=checked]:translate-x-[22px]" />
          </Switch.Root>
        </div>
      </div>

      <div className="bg-white shadow rounded-lg p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Appearance</h3>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Moon className="w-5 h-5 text-gray-500" />
            <span className="text-sm text-gray-700">Dark mode</span>
          </div>
          <Switch.Root
            checked={user.settings.theme === 'dark'}
            onCheckedChange={handleThemeToggle}
            className="w-11 h-6 bg-gray-200 rounded-full relative data-[state=checked]:bg-blue-600"
          >
            <Switch.Thumb className="block w-5 h-5 bg-white rounded-full shadow-lg transition-transform duration-100 translate-x-0.5 will-change-transform data-[state=checked]:translate-x-[22px]" />
          </Switch.Root>
        </div>
      </div>

      <div className="bg-white shadow rounded-lg p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">API Keys</h3>
        <div className="space-y-4">
          {(['openai', 'groq', 'gemini','together'] as const).map((provider) => (
            <div key={provider} className="space-y-2">
              <label className="block text-sm font-medium text-gray-700 capitalize">
                {provider} API Key
              </label>
              <div className="relative">
                <input
                  type="password"
                  value={user.apiKeys[provider] || ''}
                  onChange={(e) => handleApiKeyUpdate(provider, e.target.value)}
                  className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 pl-10"
                  placeholder={`Enter your ${provider} API key`}
                />
                <Key className="absolute left-3 top-2.5 h-5 w-5 text-gray-400" />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};