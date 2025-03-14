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
      className="h-[40px] w-[300px] rounded-md bg-white border border-gray-300 px-4 py-2 text-black"
      value={model}
      onChange={handleChange}
    >
      <option value="Qwen/Qwen2.5-Coder-32B-Instruct-fast">
        Qwen2.5-Coder-32B-Instruct-fast
      </option>
      <option value="deepseek-ai/DeepSeek-V3">DeepSeek-V3</option>
      <option value="Qwen/QwQ-32B-fast">QwQ-32B-fast</option>
      <option value="meta-llama/Llama-3.3-70B-Instruct-fast">
        Llama-3.3-70B-Instruct-fast
      </option>
      <option value="Qwen/Qwen2.5-Coder-32B-Instruct">
        Qwen2.5-Coder-32B-Instruct
      </option>
      <option value="Qwen/QwQ-32B">QwQ-32B</option>
      <option value="meta-llama/Llama-3.3-70B-Instruct">
        Llama-3.3-70B-Instruct
      </option>
    </select>
  );
};
