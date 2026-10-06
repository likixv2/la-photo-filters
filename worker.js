// Runs photo_filters_la.py unchanged inside Pyodide (Python compiled to WebAssembly).
importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

let ready = null;
const say = (text) => postMessage({ type: "status", text });

async function init() {
  say("Loading Python (about 10 MB, cached after the first visit)...");
  const py = await loadPyodide();
  say("Loading numpy, matplotlib and Pillow...");
  await py.loadPackage(["numpy", "matplotlib", "pillow"]);
  return py;
}

onmessage = async (e) => {
  try {
    ready = ready || init();
    const py = await ready;

    py.setStdout({ batched: (s) => postMessage({ type: "out", text: s + "\n" }) });
    py.setStderr({ batched: (s) => postMessage({ type: "out", text: s + "\n" }) });

    say("Fetching photo_filters_la.py...");
    let src = await (await fetch("photo_filters_la.py")).text();
    // The script saves figures to a fixed path; point it at the in-browser filesystem.
    src = src.replace(/^OUT = .*$/m, 'OUT = "/out/"');

    py.FS.mkdirTree("/out");
    let photoPath = "";
    if (e.data.photo) {
      py.FS.writeFile("/tmp/photo", new Uint8Array(e.data.photo));
      photoPath = "/tmp/photo";
    }
    py.globals.set("SRC", src);
    py.globals.set("PHOTO", photoPath);

    say("Running the program...");
    const t0 = performance.now();
    await py.runPythonAsync(`
import sys
sys.argv = ["photo_filters_la.py"] + ([PHOTO] if PHOTO else [])
exec(compile(SRC, "photo_filters_la.py", "exec"), {"__name__": "__main__"})
`);
    const secs = ((performance.now() - t0) / 1000).toFixed(1);

    const names = ["fig1_filters.png", "fig2_inversion.png", "fig3_eigen_projection.png"];
    const figs = names.map((n) => ({ name: n, bytes: py.FS.readFile("/out/" + n) }));
    postMessage({ type: "done", secs, figs });
  } catch (err) {
    postMessage({ type: "error", text: String(err && err.message ? err.message : err) });
  }
};
