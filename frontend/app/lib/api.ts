export const API = process.env.NEXT_PUBLIC_AETHEL_API ?? "http://localhost:8000";
export const fetcher = (url: string) => fetch(url).then((r) => r.json());
