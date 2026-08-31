import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  // Dual ESM and CJS: the package has to be requireable from a plain Node service and
  // importable from a bundler without either one getting a half-working copy.
  format: ["esm", "cjs"],
  dts: true,
  sourcemap: true,
  clean: true,
  target: "node18",
  treeshake: true,
});
