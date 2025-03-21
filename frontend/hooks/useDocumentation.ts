import { useState } from 'react';
import { NebiusModel, CodeBody } from '@/types/types';
import { ApiService } from '@/services/api';

/**
 * Custom hook for handling documentation API calls
 */
export const useDocumentation = () => {
  const [loading, setLoading] = useState<boolean>(false);
  const [output, setOutput] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  /**
   * Sends code to the API for documentation
   */
  const documentCode = async (
    inputCode: string,
    model: NebiusModel,
    modifyExistingDocumentation: boolean,
    doWriteArgumentsAnnotations: boolean,
    doWriteDocstrings: boolean,
    doWriteComments: boolean,
    apiKey?: string
  ) => {
    if (!inputCode.trim()) {
      setError('Please enter some Python code first.');
      return;
    }

    setLoading(true);
    setOutput('');
    setError(null);

    try {
      // Prepare request body
      const body: CodeBody = {
        inputCode,
        model,
        modifyExistingDocumentation,
        doWriteArgumentsAnnotations,
        doWriteDocstrings,
        doWriteComments,
        apiKey: apiKey || '',
      };

      const response = await ApiService.documentCode(body);
      
      await ApiService.processStream(response, (chunk) => {
        setOutput(chunk);
      });

      setLoading(false);
    } catch (err) {
      console.error(err);
      setLoading(false);
      setError(err instanceof Error ? err.message : 'Error streaming from backend.');
    }
  };

  return {
    loading,
    output,
    error,
    documentCode,
  };
}; 