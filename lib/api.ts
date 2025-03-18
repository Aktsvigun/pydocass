import { CodeBody, DocumentResponse, NebiusModel } from '@/types';

/**
 * Sends a request to the document API to process code
 */
export async function documentCode(body: CodeBody): Promise<DocumentResponse> {
  try {
    const response = await fetch('/api/pydocass/document', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      throw new Error(`Error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error documenting code:', error);
    throw error;
  }
} 