import { useCallback, useState } from "react";

interface UseFetchState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

interface UseFetchReturn<T> extends UseFetchState<T> {
  execute: () => Promise<void>;
  reset: () => void;
}

export function useFetch<T>(fetcher: () => Promise<T>): UseFetchReturn<T> {
  const [state, setState] = useState<UseFetchState<T>>({
    data: null,
    loading: false,
    error: null,
  });

  const execute = useCallback(async () => {
    setState((prev) => ({ ...prev, loading: true, error: null }));
    try {
      const data = await fetcher();
      setState({ data, loading: false, error: null });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setState({ data: null, loading: false, error: message });
    }
  }, [fetcher]);

  const reset = useCallback(() => {
    setState({ data: null, loading: false, error: null });
  }, []);

  return { ...state, execute, reset };
}
