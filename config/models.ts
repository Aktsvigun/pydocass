import { NebiusModel } from '@/types';

export const MODEL_OPTIONS: { 
  label: string; 
  value: NebiusModel; 
  isFast: boolean;
  description: string; 
}[] = [
  {
    label: 'Qwen2.5-Coder-32B (Fast)',
    value: 'Qwen/Qwen2.5-Coder-32B-Instruct-fast',
    isFast: true,
    description: 'Optimized for code documentation with fast response times'
  },
  {
    label: 'DeepSeek-V3',
    value: 'deepseek-ai/DeepSeek-V3',
    isFast: true,
    description: 'General purpose model with good code understanding'
  },
  {
    label: 'QwQ-32B (Fast)',
    value: 'Qwen/QwQ-32B-fast',
    isFast: true,
    description: 'Fast variant of QwQ-32B model'
  },
  {
    label: 'Llama-3.3-70B (Fast)',
    value: 'meta-llama/Llama-3.3-70B-Instruct-fast',
    isFast: true,
    description: 'Fast variant of Llama 3.3 70B model'
  },
  {
    label: 'Qwen2.5-Coder-32B',
    value: 'Qwen/Qwen2.5-Coder-32B-Instruct',
    isFast: false,
    description: 'High quality code documentation (standard speed)'
  },
  {
    label: 'QwQ-32B',
    value: 'Qwen/QwQ-32B',
    isFast: false,
    description: 'Standard variant of QwQ-32B model'
  },
  {
    label: 'Llama-3.3-70B',
    value: 'meta-llama/Llama-3.3-70B-Instruct',
    isFast: false,
    description: 'Standard variant of Llama 3.3 70B model'
  }
]; 