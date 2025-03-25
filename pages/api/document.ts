import { CodeBody, DocumentResponse } from '@/types/types';
import type { NextApiRequest, NextApiResponse } from 'next';

export const config = {
  api: {
    bodyParser: false, // disable body parsing so we can stream
  },
};

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

    // Try both hostname and IP to ensure connectivity
    const backendUrl = process.env.NODE_ENV === 'production' 
      ? 'http://backend:4000' 
      : 'http://172.20.0.2:4000';
    
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

    // For streaming responses in traditional API routes
    const data = await response.text();
    return res.status(200).send(data);

  } catch (error: any) {
    console.error('API error:', error);
    return res.status(500).json({ error: error.message || 'Unknown error' });
  }
}