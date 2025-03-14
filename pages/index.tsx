import Head from 'next/head';
import { useEffect, useState } from 'react';
import { CodeBlock } from '@/components/CodeBlock';
// import { APIKeyInput } from '@/components/APIKeyInput';
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
        model,
        modifyExistingDocumentation,
        doWriteArgumentsAnnotations,
        doWriteDocstrings,
        doWriteComments,
        apiKey, // not explicitly needed by your local server, but included in the type
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

      <div className="flex min-h-screen flex-col items-center
                bg-gradient-to-br from-gray-50 to-gray-100
                px-4 pb-20 text-gray-900 sm:px-10">
        <div className="mt-10 sm:mt-14 text-center">
          <h1 className="text-4xl font-extrabold text-transparent bg-clip-text
               bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500
               drop-shadow-sm">
            Python Code Documentation Assistant
          </h1>
        </div>

        {/*/!* API Key *!/*/}
        {/*<div className="mt-6">*/}
        {/*  <APIKeyInput apiKey={apiKey} onChange={handleApiKeyChange}/>*/}
        {/*</div>*/}

        {/* Model + Submit button */}
        <div className="mt-4 flex items-center space-x-2">
          <ModelSelect model={model} onChange={(val) => setModel(val)}/>
          <button
              onClick={handleDocument}
              disabled={loading}
              className={`${
                  loading ? 'w-[200px]' : 'w-[120px]'
              } cursor-pointer rounded-lg bg-gradient-to-r 
     from-indigo-500 to-blue-500 px-4 py-2 font-semibold text-white 
     shadow-lg transition-all hover:from-indigo-600 hover:to-blue-600 
     disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            {loading ? 'Annotating your code...' : 'Submit'}
          </button>
        </div>

        {/* The checkboxes for booleans */}
        <div className="mt-6 flex flex-col space-y-2">
          <label className="relative inline-flex items-center cursor-pointer">
            <input
                type="checkbox"
                className="sr-only peer"
                checked={modifyExistingDocumentation}
                onChange={(e) => setModifyExistingDocumentation(e.target.checked)}
            />
            <div
                className="w-11 h-6 bg-gray-200 peer-focus:outline-none
               rounded-full peer peer-checked:bg-indigo-500
               peer-checked:after:translate-x-5
               peer-checked:after:border-white
               after:content-[''] after:absolute after:top-[2px]
               after:left-[2px] after:bg-white after:border-gray-300
               after:border after:rounded-full after:h-5 after:w-5
               after:transition-all"
            ></div>
            <span className="ml-3 text-sm text-gray-700">Modify existing documentation</span>
          </label>

          <label className="relative inline-flex items-center cursor-pointer">
            <input
                type="checkbox"
                className="sr-only peer"
                checked={doWriteArgumentsAnnotations}
                onChange={(e) => setDoWriteArgumentsAnnotations(e.target.checked)}
            />
            <div
                className="w-11 h-6 bg-gray-200 peer-focus:outline-none
               rounded-full peer peer-checked:bg-indigo-500
               peer-checked:after:translate-x-5
               peer-checked:after:border-white
               after:content-[''] after:absolute after:top-[2px]
               after:left-[2px] after:bg-white after:border-gray-300
               after:border after:rounded-full after:h-5 after:w-5
               after:transition-all"
            ></div>
            <span className="ml-3 text-sm text-gray-700">Write arguments annotations</span>
          </label>

          <label className="relative inline-flex items-center cursor-pointer">
            <input
                type="checkbox"
                className="sr-only peer"
                checked={doWriteDocstrings}
                onChange={(e) => setDoWriteDocstrings(e.target.checked)}
            />
            <div
                className="w-11 h-6 bg-gray-200 peer-focus:outline-none
               rounded-full peer peer-checked:bg-indigo-500
               peer-checked:after:translate-x-5
               peer-checked:after:border-white
               after:content-[''] after:absolute after:top-[2px]
               after:left-[2px] after:bg-white after:border-gray-300
               after:border after:rounded-full after:h-5 after:w-5
               after:transition-all"
            ></div>
            <span className="ml-3 text-sm text-gray-700">Write docstrings</span>
          </label>

          <label className="relative inline-flex items-center cursor-pointer">
            <input
                type="checkbox"
                className="sr-only peer"
                checked={doWriteComments}
                onChange={(e) => setDoWriteComments(e.target.checked)}
            />
            <div
                className="w-11 h-6 bg-gray-200 peer-focus:outline-none
               rounded-full peer peer-checked:bg-indigo-500
               peer-checked:after:translate-x-5
               peer-checked:after:border-white
               after:content-[''] after:absolute after:top-[2px]
               after:left-[2px] after:bg-white after:border-gray-300
               after:border after:rounded-full after:h-5 after:w-5
               after:transition-all"
            ></div>
            <span className="ml-3 text-sm text-gray-700">Write comments</span>
          </label>
        </div>

        {/* Input + Output blocks */}
        <div className="mt-8 w-full max-w-[1200px] flex flex-col space-y-8 sm:flex-row sm:space-y-0 sm:space-x-8">
          {/* Input block */}
          <div className="sm:w-1/2">
            <h2 className="text-2xl font-extrabold text-transparent
               bg-clip-text bg-gradient-to-r from-green-500 to-blue-500
               mb-4 text-center drop-shadow-sm">
              Input Code
            </h2>

            <CodeBlock
                code={inputCode}
                editable={!loading}
                onChange={(val) => setInputCode(val)}
            />
          </div>

          {/* Output block */}
          <div className="sm:w-1/2">
            <h2 className="text-2xl font-extrabold text-transparent
               bg-clip-text bg-gradient-to-r from-blue-500 to-purple-500
               mb-4 text-center drop-shadow-sm">
              Output (Documentation)
            </h2>

            <CodeBlock code={outputCode} editable={false}/>
          </div>
        </div>
      </div>
    </>
  );
}