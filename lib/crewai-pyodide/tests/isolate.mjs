// Run the smoke test inside a worker_threads Worker: a separate V8 isolate with
// its own heap and memory limit. JSPI comes from the parent's V8 flags, which
// apply to the whole process.
//
//   npm run build && npm run test:isolate
import { Worker } from "node:worker_threads";
const t0 = performance.now();
const worker = new Worker(new URL("./smoke.mjs", import.meta.url), {
  resourceLimits: { maxOldGenerationSizeMb: 256 },
});
worker.on("error", (e) => { console.error("worker error:", e); process.exitCode = 1; });
worker.on("exit", (code) => {
  console.log(`isolate exited ${code} after ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  if (code !== 0) process.exitCode = code;
});
