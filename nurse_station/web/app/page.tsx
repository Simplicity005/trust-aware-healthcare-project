import { Suspense } from "react";
import Dashboard from "./dashboard";

export default function Home() {
  return <Suspense fallback={<main className="loading">Opening Nurse Station…</main>}><Dashboard /></Suspense>;
}
