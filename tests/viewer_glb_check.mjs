// Read a GLB the Python encoder wrote, using the viewer's own reader.
//
// Run by tests/test_viewer_assets.py through Node, which is the only way to check that
// the two halves of the wire format agree without opening a browser. A mismatch here --
// a padding rule, an accessor offset, an index width -- shows up in a browser as a bone
// that silently fails to appear.
//
//     node tests/viewer_glb_check.mjs <path to .glb> <expected vertices> <expected indices>

import { readFileSync } from "node:fs";
import { parseGlb } from "../web/glb.js";

const [path, expectedVertices, expectedIndices] = process.argv.slice(2);
const file = readFileSync(path);
// A Buffer is a view into a larger pool, so its own slice has to be taken before the
// offsets inside the file mean anything.
const buffer = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);

const { positions, indices, min, max } = parseGlb(buffer);

const report = {
  vertices: positions.length / 3,
  indices: indices.length,
  min,
  max,
  extent: max.map((value, axis) => value - min[axis]),
  maxIndex: indices.reduce((best, value) => (value > best ? value : best), 0),
};

const problems = [];
if (report.vertices !== Number(expectedVertices)) {
  problems.push(`vertices ${report.vertices} != ${expectedVertices}`);
}
if (report.indices !== Number(expectedIndices)) {
  problems.push(`indices ${report.indices} != ${expectedIndices}`);
}
if (report.maxIndex >= report.vertices) {
  problems.push(`index ${report.maxIndex} points past ${report.vertices} vertices`);
}
report.problems = problems;

process.stdout.write(JSON.stringify(report));
process.exit(problems.length ? 1 : 0);
