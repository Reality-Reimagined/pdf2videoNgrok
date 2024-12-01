import React from 'react';
import { FileText, MessageSquare, Music, Film, Loader2 } from 'lucide-react';

interface ProcessingStepsProps {
  currentStep: number;
}

const ProcessingSteps: React.FC<ProcessingStepsProps> = ({ currentStep }) => {
  const steps = [
    { icon: FileText, label: 'Extracting text' },
    { icon: Music, label: 'Creating voiceover' },
    { icon: Film, label: 'Composing video' },
    { icon: MessageSquare, label: 'Finishing Touches' },
  ];

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-medium text-gray-900 text-center">
        Processing your PDF
      </h3>
      <div className="space-y-4">
        {steps.map((step, index) => {
          const StepIcon = step.icon;
          const isActive = currentStep === index + 1;
          const isComplete = currentStep > index + 1;

          return (
            <div
              key={index}
              className={`flex items-center gap-4 p-4 rounded-lg ${
                isActive ? 'bg-indigo-50' : ''
              }`}
            >
              <div
                className={`p-2 rounded-full ${
                  isComplete
                    ? 'bg-green-100 text-green-600'
                    : isActive
                    ? 'bg-indigo-100 text-indigo-600'
                    : 'bg-gray-100 text-gray-400'
                }`}
              >
                <StepIcon className="h-5 w-5" />
              </div>
              <span
                className={`flex-1 ${
                  isComplete
                    ? 'text-green-600'
                    : isActive
                    ? 'text-indigo-600'
                    : 'text-gray-500'
                }`}
              >
                {step.label}
              </span>
              {isActive && (
                <Loader2 className="h-5 w-5 text-indigo-600 animate-spin" />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ProcessingSteps;