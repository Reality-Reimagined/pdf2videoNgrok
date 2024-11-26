import React from 'react';
import * as Select from '@radix-ui/react-select';
import { Check, ChevronDown } from 'lucide-react';
import { useVideoStore } from '../lib/store';

const models = [
  { value: 'groq', label: 'Groq' },
  { value: 'openai', label: 'OpenAI' },
  { value: 'gemini', label: 'Google Gemini' },
  { value: 'together', label: 'Together' },
];

export const ModelSelector = () => {
  const { selectedModel, setSelectedModel } = useVideoStore();

  return (
    <div className="flex flex-col gap-2">
      <label className="text-sm font-medium text-gray-700">
        Select AI Model
      </label>
      <Select.Root value={selectedModel} onValueChange={setSelectedModel}>
        <Select.Trigger className="inline-flex items-center justify-between rounded-md px-4 py-2 text-sm bg-white border border-gray-300 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 min-w-[180px]">
          <Select.Value placeholder="Select LLM Model" />
          <Select.Icon>
            <ChevronDown className="h-4 w-4" />
          </Select.Icon>
        </Select.Trigger>

        <Select.Portal>
          <Select.Content className="overflow-hidden bg-white rounded-md shadow-lg border border-gray-200">
            <Select.Viewport className="p-1">
              {models.map(({ value, label }) => (
                <Select.Item
                  key={value}
                  value={value}
                  className="relative flex items-center px-8 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer outline-none"
                >
                  <Select.ItemText>{label}</Select.ItemText>
                  <Select.ItemIndicator className="absolute left-2 inline-flex items-center">
                    <Check className="h-4 w-4" />
                  </Select.ItemIndicator>
                </Select.Item>
              ))}
            </Select.Viewport>
          </Select.Content>
        </Select.Portal>
      </Select.Root>
    </div>
  );
};