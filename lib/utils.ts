/**
 * Helper function to safely parse JSON
 */
export function safeJsonParse<T>(json: string, fallback: T): T {
  try {
    return JSON.parse(json) as T;
  } catch (e) {
    console.error('Error parsing JSON:', e);
    return fallback;
  }
}

/**
 * Helper function to safely retrieve data from localStorage
 */
export function getFromLocalStorage<T>(key: string, fallback: T): T {
  if (typeof window === 'undefined') {
    return fallback;
  }
  
  try {
    const value = localStorage.getItem(key);
    if (!value) return fallback;
    return safeJsonParse(value, fallback);
  } catch (e) {
    console.error(`Error getting ${key} from localStorage:`, e);
    return fallback;
  }
}

/**
 * Helper function to safely store data to localStorage
 */
export function setToLocalStorage(key: string, value: any): void {
  if (typeof window === 'undefined') {
    return;
  }
  
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch (e) {
    console.error(`Error setting ${key} to localStorage:`, e);
  }
} 