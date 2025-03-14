import { StreamLanguage } from '@codemirror/language';
import { python } from '@codemirror/legacy-modes/mode/python';
import { githubLight } from '@uiw/codemirror-theme-github';
import CodeMirror from '@uiw/react-codemirror';
import { FC, useEffect, useState } from 'react';

interface Props {
  code: string;
  editable?: boolean;
  onChange?: (value: string) => void;
}

export const CodeBlock: FC<Props> = ({
  code,
  editable = false,
  onChange = () => {},
}) => {
  const [copyText, setCopyText] = useState<string>('Copy');

  useEffect(() => {
    const timeout = setTimeout(() => {
      setCopyText('Copy');
    }, 2000);

    return () => clearTimeout(timeout);
  }, [copyText]);

  return (
      <div className="relative bg-white border border-gray-300 rounded-md p-2">
          <button
              className="absolute right-2 top-2 z-10 rounded bg-blue-100 px-2 py-1 text-xs text-blue-800 hover:bg-blue-200 active:bg-blue-300"
              onClick={() => {
                  navigator.clipboard.writeText(code);
                  setCopyText('Copied!');
              }}
          >
              {copyText}
          </button>

          <CodeMirror
              editable={editable}
              value={code}
              minHeight="500px"
              extensions={[StreamLanguage.define(python)]}
              theme={githubLight}
              onChange={(value) => onChange(value)}
              className="bg-white"
          />

      </div>
  );
};
