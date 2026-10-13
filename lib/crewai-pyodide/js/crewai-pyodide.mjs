// Load Pyodide with CrewAI installed and ready to import.
//
//   import { loadCrewAI } from "./crewai-pyodide.mjs";
//   const pyodide = await loadCrewAI({ wheelBaseURL: "./wheels/" });
//   await pyodide.runPythonAsync(`
//     import crewai_pyodide
//     from crewai import Agent, Crew, Task
//     ...
//     result = await crewai_pyodide.run(crew.akickoff())
//   `);

export const PYODIDE_VERSION = "0.29.3";
export const PYODIDE_CDN_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

/** Whether this runtime supports WebAssembly JavaScript Promise Integration. */
export function hasJSPI() {
  // Older V8 builds (e.g. Node.js 22) expose the earlier `Suspender` API instead.
  return typeof WebAssembly !== "undefined" && ("Suspending" in WebAssembly || "Suspender" in WebAssembly);
}

/**
 * Load Pyodide and install CrewAI into it.
 *
 * @param {object} [options]
 * @param {object} [options.pyodide] An already loaded Pyodide instance to install into.
 * @param {string} [options.indexURL] Where to load Pyodide from when `pyodide` is not given.
 *   Defaults to the jsDelivr CDN.
 * @param {string} [options.wheelBaseURL] URL of a directory built by `build.py`, holding
 *   `manifest.json` and the crewai wheels it lists. Without it, the released versions are
 *   installed from PyPI.
 * @param {string[]} [options.extras] Optional provider SDKs to install, e.g. `["anthropic"]`.
 * @returns {Promise<object>} The Pyodide instance, with `crewai` importable.
 */
export async function loadCrewAI({ pyodide, indexURL = PYODIDE_CDN_URL, wheelBaseURL, extras = [] } = {}) {
  if (!hasJSPI()) {
    throw new Error(
      "CrewAI on Pyodide needs WebAssembly JSPI: use Chrome/Edge 137+, " +
        "or Node.js with --experimental-wasm-jspi.",
    );
  }
  if (!pyodide) {
    const { loadPyodide } = await import(`${indexURL}pyodide.mjs`);
    pyodide = await loadPyodide({ indexURL });
  }

  let bootstrap = "crewai-pyodide";
  let crewai = [];
  if (wheelBaseURL) {
    if (!globalThis.location && !URL.canParse(wheelBaseURL)) {
      // Node.js fetch cannot load file: URLs, so a path has nothing to resolve against.
      throw new Error(`wheelBaseURL must be an absolute http(s) URL outside a browser: ${wheelBaseURL}`);
    }
    const base = new URL(wheelBaseURL, globalThis.location?.href);
    const manifestURL = new URL("manifest.json", base);
    const response = await fetch(manifestURL);
    if (!response.ok) {
      throw new Error(`Failed to load ${manifestURL}: HTTP ${response.status}`);
    }
    const manifest = await response.json();
    bootstrap = new URL(manifest.bootstrap, base).href;
    crewai = manifest.crewai.map((wheel) => new URL(wheel, base).href);
  }

  await pyodide.loadPackage("micropip");
  await pyodide.pyimport("micropip").install(bootstrap);
  const namespace = pyodide.toPy({ crewai, extras });
  try {
    await pyodide.runPythonAsync(
      "import crewai_pyodide\nawait crewai_pyodide.install(crewai, extras=extras)",
      { globals: namespace },
    );
  } finally {
    namespace.destroy();
  }
  return pyodide;
}
