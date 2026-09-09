# Vendored JavaScript

Committed rather than fetched from a content delivery network, so the planner works on
a machine with no internet connection. A tool that holds patient geometry should not
need to reach the public internet to draw a bone, and a hospital machine frequently
cannot.

| File | Source | Version | Licence |
|---|---|---|---|
| `three.module.js` | https://github.com/mrdoob/three.js | r169 | MIT, `LICENSE-three.txt` |
| `OrbitControls.js` | https://github.com/mrdoob/three.js `examples/jsm/controls` | r169 | MIT, `LICENSE-three.txt` |

MIT is compatible with this project's Apache 2.0 licence. Nothing here is GPL, which
keeps the shipped set clean in the same way the geometry kernel choice does.

The glTF loader that ships with three.js is deliberately **not** vendored. The server
emits one mesh with one primitive, positions and indices; `web/glb.js` reads exactly
that in forty lines rather than pulling in a loader for a format we do not use.

## Updating

Fetch the two files at the same release tag and replace the licence text with that
tag's. Anything that imports from `three` resolves through the import map in
`web/index.html`, so a version bump touches no other file.

    curl -o three.module.js https://cdn.jsdelivr.net/npm/three@<version>/build/three.module.js
    curl -o OrbitControls.js https://cdn.jsdelivr.net/npm/three@<version>/examples/jsm/controls/OrbitControls.js
