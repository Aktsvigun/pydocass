import { CodeBody, DocumentResponse } from '@/types/types';
import type { NextApiRequest, NextApiResponse } from 'next';

// Remove the Edge Runtime config
// export const config = {
//   runtime: 'edge',
// };

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const {
      inputCode,
      model,
      modifyExistingDocumentation,
      doWriteArgumentsAnnotations,
      doWriteDocstrings,
      doWriteComments,
      // apiKey
    } = req.body as CodeBody;

    // Use environment variable for backend URL with fallback
    const backendUrl = process.env.BACKEND_URL || 'http://172.20.0.2:4000';
    
    console.log(`Connecting to backend at: ${backendUrl}`);
    
    const response = await fetch(`${backendUrl}/document`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        code: inputCode,
        model_checkpoint: model,
        modify_existing_documentation: modifyExistingDocumentation,
        do_write_arguments_annotations: doWriteArgumentsAnnotations,
        do_write_docstrings: doWriteDocstrings,
        do_write_comments: doWriteComments,
        // api_key: apiKey
      }),
    });

    if (!response.ok) {
      throw new Error(`Backend error: ${response.statusText}`);
    }

    // Set up streaming response
    res.setHeader('Content-Type', 'text/plain');
    res.setHeader('Transfer-Encoding', 'chunked');
    
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No readable stream returned by the server.');
    }

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          res.end();
          break;
        }
        // Forward the chunk to the client
        res.write(value);
      }
    } catch (error) {
      console.error('Streaming error:', error);
      res.end();
    }

  } catch (error: any) {
    console.error('API error:', error);
    return res.status(500).json({ error: error.message || 'Unknown error' });
  }
}