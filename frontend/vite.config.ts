// @lovable.dev/vite-tanstack-config already includes the following — do NOT add them manually
// or the app will break with duplicate plugins:
//   - tanstackStart, viteReact, tailwindcss, tsConfigPaths, cloudflare (build-only),
//     componentTagger (dev-only), VITE_* env injection, @ path alias, React/TanStack dedupe,
//     error logger plugins, and sandbox detection (port/host/strictPort).
// You can pass additional config via defineConfig({ vite: { ... } }) if needed.
import { defineConfig } from "@lovable.dev/vite-tanstack-config";
import type { Plugin } from "vite";

// Workaround: @tanstack/start-server-core 1.168.x imports this virtual module in dev mode
// but no installed plugin registers a resolver for it. Provide an empty stub so dev starts.
const injectedHeadScriptsStub: Plugin = {
  name: "tanstack-start-injected-head-scripts-stub",
  resolveId(id) {
    if (id === "tanstack-start-injected-head-scripts:v") {
      return "\0tanstack-start-injected-head-scripts:v";
    }
  },
  load(id) {
    if (id === "\0tanstack-start-injected-head-scripts:v") {
      return "export const injectedHeadScripts = undefined;";
    }
  },
};

// Redirect TanStack Start's bundled server entry to src/server.ts (our SSR error wrapper).
// @cloudflare/vite-plugin builds from this — wrangler.jsonc main alone is insufficient.
export default defineConfig({
  tanstackStart: {
    server: { entry: "server" },
  },
  vite: {
    plugins: [injectedHeadScriptsStub],
  },
});
