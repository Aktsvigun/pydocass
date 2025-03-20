import { CodeBody, DocumentResponse, NebiusModel } from '@/types';

/**
 * Get the backend API URL based on environment
 */
function getBackendUrl(): string {
  // In Docker, services can communicate using service names
  // In local development, we use localhost
  const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:4000';
  return backendUrl;
}

/**
 * Sends a request to the document API to process code
 */
export async function documentCode(body: CodeBody): Promise<DocumentResponse> {
  try {
    // Frontend API route that acts as a proxy
    const apiRoute = '/api/pydocass/document';
    
    const response = await fetch(apiRoute, {
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