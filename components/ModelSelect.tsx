// ./components/ModelSelect.tsx

import { NebiusModel } from '@/types/types';
import { FC } from 'react';

interface Props {
  model: NebiusModel;
  onChange: (model: NebiusModel) => void;
}

export const ModelSelect: FC<Props> = ({ model, onChange }) => {
  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    onChange(e.target.value as NebiusModel);
  };

  return (
      <select
          className="h-10 w-64 rounded-lg border border-gray-300 bg-white px-3 py-2
             text-gray-700 shadow-sm focus:border-indigo-500 focus:outline-none
             focus:ring-1 focus:ring-indigo-500"
          value={model}
          onChange={handleChange}
      >
          <optgroup label="Fast Models">
              <option value="Qwen/Qwen2.5-Coder-32B-Instruct-fast">
                  Qwen2.5-Coder-32B-Instruct-fast
              </option>
              <option value="Qwen/QwQ-32B-fast">QwQ-32B-fast</option>
              <option value="meta-llama/Llama-3.3-70B-Instruct-fast">
                  Llama-3.3-70B-Instruct-fast
              </option>
          </optgroup>

          <optgroup label="Normal Models">
              <option value="deepseek-ai/DeepSeek-V3">DeepSeek-V3</option>
              <option value="Qwen/Qwen2.5-Coder-32B-Instruct">
                  Qwen2.5-Coder-32B-Instruct
              </option>
              <option value="Qwen/QwQ-32B">QwQ-32B</option>
              <option value="meta-llama/Llama-3.3-70B-Instruct">
                  Llama-3.3-70B-Instruct
              </option>
          </optgroup>
      </select>
  );
};
