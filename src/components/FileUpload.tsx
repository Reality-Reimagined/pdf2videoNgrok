import React from 'react';
import { Dropzone, FileItem, FileValidated } from '@dropzone-ui/react';
import { useStore } from '../lib/store';
import { Upload } from 'lucide-react';
import { extractText } from '../lib/api-service';

export const FileUpload = () => {
  const { setIsLoading, setPdfText } = useStore();
  const [files, setFiles] = React.useState<FileValidated[]>([]);

  const handleChange = async (incomingFiles: FileValidated[]) => {
    setFiles(incomingFiles);
    if (incomingFiles.length > 0) {
      setIsLoading(true);
      try {
        const file = incomingFiles[0].file;
        const response = await extractText(file);
        
        setPdfText(response.text);
        
      } catch (error) {
        console.error('Error processing PDF:', error);
        setPdfText('Error processing PDF. Please try again.');
      } finally {
        setIsLoading(false);
      }
    }
  };
  return (
    <div className="w-full max-w-2xl mx-auto">
      <Dropzone
        onChange={handleChange}
        value={files}
        maxFiles={1}
        header={false}
        footer={false}
        accept=".pdf"
        label="Drag & Drop your PDF here"
        minHeight="200px"
        style={{
          border: '2px dashed #e2e8f0',
          borderRadius: '0.5rem',
          backgroundColor: '#f8fafc',
        }}
      >
        {files.length === 0 && (
          <div className="flex flex-col items-center justify-center p-6">
            <Upload className="w-12 h-12 text-gray-400 mb-2" />
            <p className="text-gray-600">Drag & Drop your PDF here</p>
            <p className="text-sm text-gray-400">or click to browse</p>
          </div>
        )}
      </Dropzone>
      {files.map((file) => (
        <FileItem {...file} preview key={file.id} />
      ))}
      {pdfText && typeof pdfText === 'string' && (
        <p className="mt-4 text-sm text-gray-600">{pdfText}</p>
      )}
    </div>
  );
};