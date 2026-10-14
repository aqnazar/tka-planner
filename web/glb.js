// Reading the binary glTF the server sends.
//
// three.js ships a GLTFLoader, and this is not it. What the server emits is one mesh
// with one primitive, positions and indices, and nothing else: no materials, no
// animations, no textures, no scene graph. Parsing exactly that is forty lines, against
// vendoring a loader that handles a format we deliberately do not use. Colour lives on
// the node in our scene model, not on the geometry -- except on a fit map, whose
// per-vertex colours and values are the server's and come with the mesh.

const MAGIC = 0x46546c67; // "glTF"
const JSON_CHUNK = 0x4e4f534a;
const BIN_CHUNK = 0x004e4942;

export function parseGlb(buffer) {
  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== MAGIC) {
    throw new Error("Not a GLB file.");
  }
  if (view.getUint32(4, true) !== 2) {
    throw new Error("Only glTF 2.0 is understood.");
  }

  let offset = 12;
  let document = null;
  let binary = null;
  while (offset < view.byteLength) {
    const length = view.getUint32(offset, true);
    const kind = view.getUint32(offset + 4, true);
    const start = offset + 8;
    if (kind === JSON_CHUNK) {
      document = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, start, length)));
    } else if (kind === BIN_CHUNK) {
      binary = { start, length };
    }
    offset = start + length;
  }
  if (!document || !binary) {
    throw new Error("The GLB is missing its JSON or its binary chunk.");
  }

  const primitive = document.meshes[0].primitives[0];
  const positions = readAccessor(document, buffer, binary, primitive.attributes.POSITION, Float32Array);
  const indices = readAccessor(document, buffer, binary, primitive.indices, Uint32Array);
  const accessor = document.accessors[primitive.attributes.POSITION];
  // A colour map also carries a colour and the value it stands for at every vertex,
  // both decided by the server. They are read here and drawn as they come.
  const { COLOR_0: colour, _VALUE_MM: value } = primitive.attributes;
  const colours = colour === undefined
    ? null : readAccessor(document, buffer, binary, colour, Float32Array);
  const values = value === undefined
    ? null : readAccessor(document, buffer, binary, value, Float32Array);
  return { positions, indices, colours, values, min: accessor.min, max: accessor.max };
}

function readAccessor(document, buffer, binary, index, ArrayType) {
  const accessor = document.accessors[index];
  const view = document.bufferViews[accessor.bufferView];
  const components = accessor.type === "VEC3" ? 3 : 1;
  // Offsets in the document are relative to the buffer; the buffer starts partway
  // into the file, which is what `binary.start` corrects for.
  const start = binary.start + (view.byteOffset || 0) + (accessor.byteOffset || 0);
  return new ArrayType(buffer, start, accessor.count * components);
}
