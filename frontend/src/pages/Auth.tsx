import React from 'react';
import * as Tabs from '@radix-ui/react-tabs';
import { LoginForm } from '../components/auth/LoginForm';
import { RegisterForm } from '../components/auth/RegisterForm';
import { Bot } from 'lucide-react';

export const Auth = () => {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
      <div className="sm:mx-auto sm:w-full sm:max-w-md">
        <div className="flex justify-center">
          <Bot className="w-12 h-12 text-blue-600" />
        </div>
        <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
          Welcome to PDF to Video
        </h2>
        <p className="mt-2 text-center text-sm text-gray-600">
          Transform your PDFs into engaging videos
        </p>
      </div>

      <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
        <div className="bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10">
          <Tabs.Root defaultValue="login" className="space-y-6">
            <Tabs.List className="flex space-x-4 border-b border-gray-200">
              <Tabs.Trigger
                value="login"
                className="px-4 py-2 text-sm font-medium text-gray-500 hover:text-gray-700 border-b-2 border-transparent data-[state=active]:border-blue-500 data-[state=active]:text-blue-600"
              >
                Sign in
              </Tabs.Trigger>
              <Tabs.Trigger
                value="register"
                className="px-4 py-2 text-sm font-medium text-gray-500 hover:text-gray-700 border-b-2 border-transparent data-[state=active]:border-blue-500 data-[state=active]:text-blue-600"
              >
                Create account
              </Tabs.Trigger>
            </Tabs.List>

            <Tabs.Content value="login" className="focus:outline-none">
              <LoginForm />
            </Tabs.Content>

            <Tabs.Content value="register" className="focus:outline-none">
              <RegisterForm />
            </Tabs.Content>
          </Tabs.Root>
        </div>
      </div>
    </div>
  );
};