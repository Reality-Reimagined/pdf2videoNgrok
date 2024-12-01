import React, { useState, useRef } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { X, Upload, FileText, Video as VideoIcon, Wand2, AlertCircle } from 'lucide-react';
import { FileUploader } from './FileUploader';
import { ThemeInput } from './ThemeInput';
import { SubtitlesToggle } from '../SubtitlesToggle';
import { ModelSelector } from '../ModelSelector';
import { useVideoStore } from '../../lib/store';
import { useAuthStore } from '../../lib/store';
import { aiService } from '../../lib/ai-service';
import { apiService } from '../../lib/api-service';
import ProcessingSteps from './ProcessingSteps';
// import React, { useRef } from 'react';


interface CreateVideoDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onVideoGenerated: () => void;
}

export const CreateVideoDialog: React.FC<CreateVideoDialogProps> = ({ open, onOpenChange, onVideoGenerated }) => {
  const [step, setStep] = useState(1);
  const [pdfText, setPdfText] = useState<string>('');
  const [generatedScript, setGeneratedScript] = useState('');
  const [error, setError] = useState<string | null>(null);
  const { isProcessing, setIsProcessing, selectedModel, theme, addSubtitles } = useVideoStore();
  const { user } = useAuthStore();
  const [currentProcessingStep, setCurrentProcessingStep] = useState(0);
  const [videoTitle, setVideoTitle] = useState('');

  const handleDialogClose = () => {
    // Reset state when dialog is closed
    setStep(1);
    setPdfText('');
    setGeneratedScript('');
    setVideoTitle('');
    setError(null);
    setCurrentProcessingStep(0);
    setIsProcessing(false);
  };

  const handlePdfUpload = async (file: File) => {
    try {
      setError(null);
      setIsProcessing(true);
      setCurrentProcessingStep(1);
      const response = await apiService.extractText(file);
      
      if (typeof response.text === 'string') {
        setPdfText(response.text);
        setStep(2);
      } else {
        throw new Error('Invalid response format from server');
      }
    } catch (error) {
      setError('Failed to extract text from PDF. Please try again.');
      console.error('Error processing PDF:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleScriptGeneration = async () => {
    const apiKey = user?.apiKeys[selectedModel];
    if (!apiKey) {
      setError(`Please add your ${selectedModel.toUpperCase()} API key in settings.`);
      return;
    }

    try {
      setError(null);
      setIsProcessing(true);
      setCurrentProcessingStep(2);
      
      if (!pdfText) {
        throw new Error('No text available to generate script');
      }

      const generatedScript = await aiService.generateScript(pdfText, selectedModel);
      setGeneratedScript(generatedScript);
      setStep(3);
    } catch (error) {
      console.error('Error generating script:', error);
      setError(error.message || 'Failed to generate script');
    } finally {
      setIsProcessing(false);
    }
  };
  // Ref for progress section
    const progressSectionRef = useRef<HTMLDivElement>(null);

    const handleVideoGeneration = async () => {
    try {
      if (!videoTitle.trim()) {
        setError('Please provide a title for your video');
        return;
      }

      if (!user) {
        setError('User not authenticated');
        return;
      }

      setError(null);
      setIsProcessing(true);
      setCurrentProcessingStep(3);

      if (!generatedScript) {
        throw new Error('No script available');
      }

      const result = await apiService.generateVideo(
        generatedScript, 
        theme, 
        addSubtitles,
        videoTitle.trim()
      );
      console.log('Video generation result:', result);

      setCurrentProcessingStep(4);

      if (typeof onVideoGenerated === 'function') {
        onVideoGenerated();
      }

      if (progressSectionRef.current) {
        progressSectionRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
      setStep(4);
    } catch (error) {
      console.error('Video generation error:', error);
      if (!error.message.includes('404')) {
        setError('Failed to generate video. Please try again.');
      } else {
        setStep(4);
      }
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <Dialog.Root open={open} onOpenChange={(isOpen) => {
      onOpenChange(isOpen);
      if (!isOpen) {
        handleDialogClose(); // Reset state when dialog is closed
      }
    }}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/50" />
        <Dialog.Content 
          className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 bg-white rounded-lg shadow-lg p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto"
          aria-describedby="dialog-description"
        >
          <Dialog.Title className="text-xl font-semibold mb-4">Create Video</Dialog.Title>
          <Dialog.Description id="dialog-description" className="sr-only">
            Create a new video from PDF document
          </Dialog.Description>
          <Dialog.Close className="absolute top-4 right-4 text-gray-400 hover:text-gray-600">
            <X className="w-5 h-5" />
          </Dialog.Close>

          {error && (
            <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-md">
              <div className="flex items-center gap-2 text-red-600">
                <AlertCircle className="w-5 h-5" />
                <p className="text-sm">{error}</p>
              </div>
            </div>
          )}

          <div className="space-y-6">
            {step === 1 && (
              <div className="space-y-4">
                <div className="flex items-center gap-2 text-blue-600">
                  <Upload className="w-5 h-5" />
                  <h3 className="font-medium">Upload PDF</h3>
                </div>
                <FileUploader onUpload={handlePdfUpload} />
              </div>
            )}

            {step === 2 && (
              <div className="space-y-4">
                <div className="flex items-center gap-2 text-blue-600">
                  <FileText className="w-5 h-5" />
                  <h3 className="font-medium">Generate Script</h3>
                </div>
                {pdfText && (
                  <div className="p-4 bg-gray-50 rounded-md">
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Extracted Text</h4>
                    <p className="text-sm text-gray-600 max-h-40 overflow-y-auto">{pdfText}</p>
                  </div>
                )}
                <div className="space-y-4">
                  <ModelSelector />
                  <button
                    onClick={handleScriptGeneration}
                    disabled={isProcessing}
                    className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
                  >
                    <Wand2 className="w-4 h-4" />
                    {isProcessing ? 'Generating...' : 'Generate Script'}
                  </button>
                </div>
              </div>
            )}

            {step === 3 && (
              <div ref={progressSectionRef} className="space-y-4">
                <div className="flex items-center gap-2 text-blue-600">
                  <VideoIcon className="w-5 h-5" />
                  <h3 className="font-medium">Generate Video</h3>
                </div>
                <div className="space-y-2">
                  <label htmlFor="video-title" className="block text-sm font-medium text-gray-700">
                    Video Title
                  </label>
                  <input
                    id="video-title"
                    type="text"
                    value={videoTitle}
                    onChange={(e) => setVideoTitle(e.target.value)}
                    placeholder="Enter video title"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    required
                  />
                </div>
                {generatedScript && (
                  <div className="p-4 bg-gray-50 rounded-md">
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Generated Script</h4>
                    <textarea
                      className="w-full h-40 p-2 border border-gray-300 rounded-md"
                      value={generatedScript}
                      onChange={(e) => setGeneratedScript(e.target.value)}
                    />
                    <div className="text-sm text-gray-600 mt-2">
                      Character Count: {generatedScript.length}
                    </div>
                  </div>
                )}
                <div className="space-y-4">
                  <ThemeInput />
                  <SubtitlesToggle />
                  <button
                    onClick={handleVideoGeneration}
                    disabled={isProcessing}
                    className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
                  >
                    <VideoIcon className="w-4 h-4" />
                    {isProcessing ? 'Generating...' : 'Generate Video'}
                  </button>
                </div>
                {isProcessing && <ProcessingSteps currentStep={currentProcessingStep} />}
              </div>
            )}

            {step === 4 && (
              <div className="text-center">
                <h3 className="text-lg font-medium text-gray-900">Video Generated Successfully!</h3>
                <p className="mt-2 text-sm text-gray-500">Your video has been created.</p>
              </div>
            )}
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
};