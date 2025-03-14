import Head from 'next/head';
import { useEffect, useState } from 'react';
import { CodeBlock } from '@/components/CodeBlock';
import { APIKeyInput } from '@/components/APIKeyInput';
import { ModelSelect } from '@/components/ModelSelect';
import { NebiusModel } from '@/types/types';
import { CodeBody } from '@/types/types';


export default function Home() {
  // We only have one code input and one code output
  const [inputCode, setInputCode] = useState<string>('');
  const [outputCode, setOutputCode] = useState<string>('');

  // Model selection
  const [model, setModel] = useState<NebiusModel>(
    'Qwen/Qwen2.5-Coder-32B-Instruct-fast',
  );

  // Four booleans
  const [modifyExistingDocumentation, setModifyExistingDocumentation] = useState<boolean>(true);
  const [doWriteArgumentsAnnotations, setDoWriteArgumentsAnnotations] = useState<boolean>(true);
  const [doWriteDocstrings, setDoWriteDocstrings] = useState<boolean>(true);
  const [doWriteComments, setDoWriteComments] = useState<boolean>(true);

  // General state
  const [loading, setLoading] = useState<boolean>(false);
  const [apiKey, setApiKey] = useState<string>('');

  // Load any existing API key from localStorage if you want it
  useEffect(() => {
    const storedKey = localStorage.getItem('apiKey');
    if (storedKey) {
      setApiKey(storedKey);
    }
  }, []);

  const handleApiKeyChange = (value: string) => {
    setApiKey(value);
    localStorage.setItem('apiKey', value);
  };

  const handleDocument = async () => {
    // Basic checks
    if (!inputCode.trim()) {
      alert('Please enter some Python code first.');
      return;
    }

    setLoading(true);
    setOutputCode('');

    try {
      // Prepare request body
      const body: CodeBody = {
        inputCode,
        outputCode: '', // Not used by the backend, but typed in CodeBody
        model,
        apiKey, // not explicitly needed by your local server, but included in the type
        modifyExistingDocumentation,
        doWriteArgumentsAnnotations,
        doWriteDocstrings,
        doWriteComments,
      };

      const response = await fetch('/api/document', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        setLoading(false);
        alert('Error calling backend /api/document.');
        return;
      }

      // SSE streaming response
      const reader = response.body?.getReader();
      if (!reader) {
        setLoading(false);
        alert('No readable stream returned by the server.');
        return;
      }

      const decoder = new TextDecoder();
      let done = false;
      let docString = '';

      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        if (value) {
          const chunkValue = decoder.decode(value);
          docString = chunkValue; // Instead of accumulating, store only the latest chunk
          setOutputCode(chunkValue); // Set state with the latest response
        }
      }


      setLoading(false);
    } catch (err) {
      console.error(err);
      setLoading(false);
      alert('Error streaming from backend.');
    }
  };

  return (
    <>
      <Head>
        <title>Python Code Documentation Assistant</title>
        <meta name="description" content="Use Nebius AI Studio to document your Python code." />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div className="flex h-full min-h-screen flex-col items-center bg-white px-4 pb-20 text-black sm:px-10">
        <div className="mt-10 sm:mt-14 text-center">
          <h1 className="text-3xl font-bold">Python Code Documentation Assistant</h1>
        </div>

        {/* API Key */}
        <div className="mt-6">
          <APIKeyInput apiKey={apiKey} onChange={handleApiKeyChange} />
        </div>

        {/* Model + Submit button */}
        <div className="mt-4 flex items-center space-x-2">
          <ModelSelect model={model} onChange={(val) => setModel(val)}/>
          <button
            onClick={handleDocument}
            disabled={loading}
            className={`cursor-pointer rounded-md bg-blue-600 px-4 py-2 font-bold text-white hover:bg-blue-700 transition-all whitespace-nowrap ${
              loading ? 'w-[200px]' : 'w-[120px]'
            }`}
          >
            {loading ? 'Annotating your code...' : 'Submit'}
          </button>
        </div>

        {/* The checkboxes for booleans */}
        <div className="mt-6 flex flex-col space-y-2">
          <label className="flex items-center space-x-2">
            <input
                type="checkbox"
                checked={modifyExistingDocumentation}
                onChange={(e) => setModifyExistingDocumentation(e.target.checked)}
            />
            <span>Modify existing docstrings</span>
          </label>

          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={doWriteArgumentsAnnotations}
              onChange={(e) => setDoWriteArgumentsAnnotations(e.target.checked)}
            />
            <span>Write arguments annotation</span>
          </label>

          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={doWriteDocstrings}
              onChange={(e) => setDoWriteDocstrings(e.target.checked)}
            />
            <span>Write docstrings</span>
          </label>

          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={doWriteComments}
              onChange={(e) => setDoWriteComments(e.target.checked)}
            />
            <span>Write inline comments</span>
          </label>
        </div>

        {/* Input + Output blocks */}
        <div className="mt-8 w-full max-w-[1200px] flex flex-col space-y-8 sm:flex-row sm:space-y-0 sm:space-x-8">
          {/* Input block */}
          <div className="sm:w-1/2">
            <h2 className="mb-2 text-center text-xl font-bold">Input Code</h2>
            <CodeBlock
              code={inputCode}
              editable={!loading}
              onChange={(val) => setInputCode(val)}
            />
          </div>

          {/* Output block */}
          <div className="sm:w-1/2">
            <h2 className="mb-2 text-center text-xl font-bold">Output (Documentation)</h2>
            <CodeBlock code={outputCode} editable={false} />
          </div>
        </div>
      </div>
    </>
  );
}