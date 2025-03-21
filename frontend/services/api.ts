import { CodeBody, DocumentResponse } from '@/types/types';

/**
 * API service for handling all API calls
 */
export const ApiService = {
  /**
   * Documents Python code using the API
   */
  documentCode: async (body: CodeBody): Promise<Response> => {
    const response = await fetch('/api/document', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      throw new Error('Error calling backend /api/document.');
    }

    return response;
  },

  /**
   * Reads and processes the streaming response from the API
   */
  processStream: async (
    response: Response,
    onChunk: (chunk: string) => void
  ): Promise<void> => {
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No readable stream returned by the server.');
    }

    const decoder = new TextDecoder();
    let done = false;

    while (!done) {
      const { value, done: doneReading } = await reader.read();
      done = doneReading;
      if (value) {
        const chunkValue = decoder.decode(value);
        onChunk(chunkValue);
      }
    }
  },
}; 