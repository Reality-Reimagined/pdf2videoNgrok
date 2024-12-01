import React, { useEffect } from 'react';
import { useAuthStore } from '../../lib/store';
import { Bell, Moon, Key } from 'lucide-react';
import * as Switch from '@radix-ui/react-switch';
import { supabase } from '../../lib/supabase';

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

  const handleThemeToggle = async (checked: boolean) => {
    const newTheme = checked ? 'dark' : 'light';
    
    try {
      // Get current settings first
      const { data: currentProfile } = await supabase
        .from('user_profiles')
        .select('settings')
        .eq('id', user.id)
        .single();

      const currentSettings = currentProfile?.settings || {
        notifications: true,
        defaultModel: 'groq',
        theme: 'light'
      };

      // Update user settings in Supabase
      const { error } = await supabase
        .from('user_profiles')
        .upsert({ 
          id: user.id,
          settings: {
            ...currentSettings,
            theme: newTheme
          },
          updated_at: new Date().toISOString()
        });

      if (error) throw error;

      // Update local state
      updateUser({
        settings: {
          ...user.settings,
          theme: newTheme,
        },
      });

      // Apply theme to document
      if (newTheme === 'dark') {
        document.documentElement.classList.add('dark');
      } else {
        document.documentElement.classList.remove('dark');
      }

    } catch (error) {
      console.error('Error updating theme:', error);
    }
  };

  const handleApiKeyUpdate = async (provider: keyof typeof user.apiKeys, value: string) => {
    try {
      // First ensure user profile exists
      const { error: profileError } = await supabase
        .from('user_profiles')
        .upsert({ 
          id: user.id,
          updated_at: new Date().toISOString()
        });

      if (profileError) throw profileError;

      // Then update API key
      const { error } = await supabase
        .from('user_api_keys')
        .upsert({
          user_id: user.id,
          provider: provider,
          api_key: value,
          updated_at: new Date().toISOString()
        }, {
          onConflict: 'user_id,provider'
        });

      if (error) throw error;

      // Update local state
      updateUser({
        apiKeys: {
          ...user.apiKeys,
          [provider]: value,
        },
      });

    } catch (error) {
      console.error('Error updating API key:', error);
      // Optionally add error handling UI
    }
  };

  // Add useEffect to load API keys on component mount
  useEffect(() => {
    const loadApiKeys = async () => {
      try {
        const { data, error } = await supabase
          .from('user_api_keys')
          .select('provider, api_key')
          .eq('user_id', user.id);

        if (error) throw error;

        if (data) {
          const apiKeys = data.reduce((acc, { provider, api_key }) => ({
            ...acc,
            [provider]: api_key
          }), {});

          updateUser({ apiKeys });
        }
      } catch (error) {
        console.error('Error loading API keys:', error);
      }
    };

    loadApiKeys();
  }, [user.id, updateUser]);

  // Add useEffect to apply theme on mount
  useEffect(() => {
    if (user?.settings?.theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [user?.settings?.theme]);

  return (
    <div className="space-y-6">
      <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Notifications</h3>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Bell className="w-5 h-5 text-gray-500 dark:text-gray-400" />
            <span className="text-sm text-gray-700 dark:text-gray-300">Enable notifications</span>
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

      <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Appearance</h3>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Moon className="w-5 h-5 text-gray-500 dark:text-gray-400" />
            <span className="text-sm text-gray-700 dark:text-gray-300">Dark mode</span>
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

      <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">API Keys</h3>
        <div className="space-y-4">
          {(['openai', 'groq', 'gemini','together'] as const).map((provider) => (
            <div key={provider} className="space-y-2">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 capitalize">
                {provider} API Key
              </label>
              <div className="relative">
                <input
                  type="password"
                  value={user.apiKeys[provider] || ''}
                  onChange={(e) => handleApiKeyUpdate(provider, e.target.value)}
                  className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 pl-10 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  placeholder={`Enter your ${provider} API key`}
                />
                <Key className="absolute left-3 top-2.5 h-5 w-5 text-gray-400 dark:text-gray-500" />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};