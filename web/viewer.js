// The 3D view. It draws what the server sends and computes nothing clinical.
//
// Two things here are worth knowing before reading the rest.
//
// The scene is two groups, femoral and tibial, each holding its bone, its implant parts
// and its landmarks. A group carries one pose and everything inside it carries a rest
// transform, exactly as the engine's scene model does, so a trial pose is one matrix
// assignment on a group rather than a walk over its contents.
//
// Plan mode does not cut anything. It clips. A planar resection is a half-space, and a
// clipping plane evaluates the same half-space per pixel instead of per triangle, so the
// previewed cut follows a slider at display refresh with no geometry work at all. The
// only difference from committed geometry is the triangulation of the cut face, which is
// why the preview is honest rather than approximate.

import * as THREE from "./vendor/three.module.js";
import { OrbitControls } from "./vendor/OrbitControls.js";
import { parseGlb } from "./glb.js";

// The engine works in millimetres and glTF's convention is metres. Vertices are
// converted by the encoder; a node transform's translation is converted here, which is
// the matching half of the same boundary.
const MM_PER_M = 1000;

const RESECTED_COLOUR = 0x8c8580;
const BACKGROUND = 0x1a1d21;
const CLEAN_BACKGROUND = 0xffffff;

export class Viewer {
  constructor(container) {
    this.container = container;
    this.meshCache = new Map();
    this.nodes = new Map();
    this.groups = new Map();
    this.clipPlanes = new Map();
    this.clipping = true;
    this.meshUrl = null;

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.localClippingEnabled = true;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(BACKGROUND);

    this.camera = new THREE.PerspectiveCamera(35, 1, 0.005, 50);
    this.camera.position.set(0.35, -0.45, 0.25);
    this.camera.up.set(0, 0, 1); // the anatomy's proximal axis is +Z, not +Y

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;

    this.root = new THREE.Group();
    this.scene.add(this.root);
    for (const setId of ["femoral", "tibial", "world"]) {
      const group = new THREE.Group();
      group.matrixAutoUpdate = false;
      this.groups.set(setId, group);
      this.root.add(group);
    }

    this.scene.add(new THREE.HemisphereLight(0xdfe7f2, 0x2b2f36, 2.2));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(0.6, -0.9, 0.8);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.5);
    fill.position.set(-0.7, 0.6, -0.3);
    this.scene.add(fill);

    this.grid = new THREE.GridHelper(1.0, 20, 0x3a4048, 0x2a2f36);
    this.grid.rotation.x = Math.PI / 2; // the grid lies in XY, like the anatomy's floor
    this.scene.add(this.grid);

    this.resize();
    window.addEventListener("resize", () => this.resize());
    this.renderer.setAnimationLoop(() => this.tick());
  }

  resize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  tick() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  // --------------------------------------------------------------
  // Building
  // --------------------------------------------------------------

  async setScene(snapshot, meshUrl) {
    this.meshUrl = meshUrl;
    this.clear();

    for (const [setId, matrix] of Object.entries(snapshot.set_poses || {})) {
      this.setPose(setId, matrix);
    }
    await Promise.all(snapshot.nodes.map((node) => this.addNode(node)));
    this.frame();
  }

  clear() {
    for (const { mesh, cap } of this.nodes.values()) {
      mesh.parent?.remove(mesh);
      mesh.material.dispose();
      if (cap) {
        cap.parent?.remove(cap);
        cap.material.dispose();
      }
    }
    this.nodes.clear();
  }

  async addNode(node) {
    if (!node.mesh) return;
    const geometry = await this.geometry(node.mesh);
    const material = this.material(node);
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = node.name;
    mesh.matrixAutoUpdate = false;
    mesh.matrix.copy(toMatrix(node.rest));
    mesh.visible = node.visible;
    mesh.renderOrder = node.tags?.landmark ? 2 : 0;

    const group = this.groups.get(node.set) || this.groups.get("world");
    group.add(mesh);

    // A bone gets a second copy drawn inside out. Clipped open, the back faces of a
    // closed surface are what you see through the cut, so drawing them in resected-bone
    // colour makes the preview read as solid rather than as a hollow shell. It is not a
    // true cut face, and it is a fraction of the cost of a stencil pass that would be.
    let cap = null;
    if (node.tags?.bone) {
      cap = new THREE.Mesh(geometry, new THREE.MeshLambertMaterial({
        color: RESECTED_COLOUR,
        side: THREE.BackSide,
      }));
      cap.matrixAutoUpdate = false;
      cap.matrix.copy(mesh.matrix);
      cap.visible = node.visible && this.clipping;
      group.add(cap);
    }

    const entry = { mesh, cap, node };
    this.nodes.set(node.name, entry);
    this.applyClip(entry);
    return entry;
  }

  material(node) {
    const colour = new THREE.Color(...(node.colour || [0.6, 0.6, 0.6]));
    const transparent = (node.alpha ?? 1) < 1;
    return new THREE.MeshLambertMaterial({
      color: colour,
      transparent,
      opacity: node.alpha ?? 1,
      side: node.tags?.plane || node.tags?.axis ? THREE.DoubleSide : THREE.FrontSide,
      depthWrite: !transparent,
    });
  }

  async geometry(meshId) {
    if (this.meshCache.has(meshId)) return this.meshCache.get(meshId);

    const pending = fetch(this.meshUrl(meshId))
      .then((response) => {
        if (!response.ok) throw new Error(`Mesh ${meshId} could not be fetched.`);
        return response.arrayBuffer();
      })
      .then((buffer) => {
        const { positions, indices } = parseGlb(buffer);
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
        // The server sends no normals: for flat-shaded bone they are cheaper to derive
        // here than to transfer, and it halves what crosses the wire.
        geometry.computeVertexNormals();
        geometry.computeBoundingSphere();
        return geometry;
      });

    this.meshCache.set(meshId, pending);
    return pending;
  }

  // --------------------------------------------------------------
  // Updating
  // --------------------------------------------------------------

  async applyDelta(delta) {
    for (const [name, matrix] of Object.entries(delta.poses || {})) {
      if (name === "TibialSet") {
        this.setPose("tibial", matrix);
      } else if (name === "FemoralSet") {
        this.setPose("femoral", matrix);
      } else {
        const entry = this.nodes.get(name);
        if (entry) {
          entry.mesh.matrix.copy(toMatrix(matrix));
          entry.cap?.matrix.copy(entry.mesh.matrix);
        }
      }
    }

    for (const [name, visible] of Object.entries(delta.visibility || {})) {
      const entry = this.nodes.get(name);
      if (!entry) continue;
      entry.mesh.visible = visible;
      if (entry.cap) entry.cap.visible = visible && this.clipping;
    }

    // Meshes are addressed by content hash, so a replacement is a different id and a
    // node whose geometry did not change is never refetched.
    await Promise.all(
      Object.entries(delta.meshes || {}).map(([name, meshId]) =>
        this.replaceMesh(name, meshId, delta.nodes?.[name])
      )
    );
  }

  async replaceMesh(name, meshId, description) {
    const entry = this.nodes.get(name);
    // A commit in cutting-block mode adds the bone shells, which were not in the scene
    // the viewer drew from. The server sends the node beside its mesh, so a name the
    // viewer does not know is added rather than dropped.
    if (!entry) {
      return description ? this.addNode({ ...description, mesh: meshId }) : null;
    }
    const geometry = await this.geometry(meshId);
    entry.mesh.geometry = geometry;
    if (entry.cap) entry.cap.geometry = geometry;
    entry.node.mesh = meshId;
    return entry;
  }

  setPose(setId, matrix) {
    const group = this.groups.get(setId);
    if (group) group.matrix.copy(toMatrix(matrix));
  }

  // --------------------------------------------------------------
  // The Plan-mode preview
  // --------------------------------------------------------------

  setClip(clip) {
    this.clipPlanes.clear();
    for (const [bone, plane] of Object.entries(clip || {})) {
      this.clipPlanes.set(
        bone,
        new THREE.Plane(
          new THREE.Vector3(...plane.normal),
          plane.constant_mm / MM_PER_M
        )
      );
    }
    for (const entry of this.nodes.values()) this.applyClip(entry);
  }

  setClipping(enabled) {
    this.clipping = enabled;
    for (const entry of this.nodes.values()) this.applyClip(entry);
  }

  applyClip(entry) {
    if (!entry.node.tags?.bone) return;
    const plane = this.clipPlanes.get(entry.node.name);
    const planes = this.clipping && plane ? [plane] : null;
    entry.mesh.material.clippingPlanes = planes;
    entry.mesh.material.needsUpdate = true;
    if (entry.cap) {
      entry.cap.material.clippingPlanes = planes;
      entry.cap.material.needsUpdate = true;
      entry.cap.visible = entry.mesh.visible && Boolean(planes);
    }
  }

  // --------------------------------------------------------------
  // Presentation
  // --------------------------------------------------------------

  setCleanViewport(clean) {
    this.scene.background = new THREE.Color(clean ? CLEAN_BACKGROUND : BACKGROUND);
    this.grid.visible = !clean;
  }

  frame() {
    const box = new THREE.Box3();
    for (const { mesh } of this.nodes.values()) {
      if (mesh.visible && mesh.geometry.boundingSphere) {
        box.expandByObject(mesh);
      }
    }
    if (box.isEmpty()) return;

    const centre = box.getCenter(new THREE.Vector3());
    const radius = box.getSize(new THREE.Vector3()).length() / 2 || 0.2;
    const distance = radius / Math.sin((this.camera.fov * Math.PI) / 360);

    this.controls.target.copy(centre);
    this.camera.position.copy(centre).add(
      new THREE.Vector3(0.55, -0.78, 0.3).normalize().multiplyScalar(distance * 1.15)
    );
    this.camera.near = Math.max(distance / 500, 0.001);
    this.camera.far = distance * 20;
    this.camera.updateProjectionMatrix();
    this.controls.update();

    this.grid.scale.setScalar(Math.max(radius * 4, 0.2));
    this.grid.position.set(centre.x, centre.y, box.min.z);
  }
}

function toMatrix(rows) {
  // Row major from the server, and the translation column converted from millimetres.
  // Only the translation is scaled: a transform acting on millimetres becomes the same
  // transform on metres with its rotation and scale untouched.
  const m = new THREE.Matrix4();
  m.set(
    rows[0][0], rows[0][1], rows[0][2], rows[0][3] / MM_PER_M,
    rows[1][0], rows[1][1], rows[1][2], rows[1][3] / MM_PER_M,
    rows[2][0], rows[2][1], rows[2][2], rows[2][3] / MM_PER_M,
    rows[3][0], rows[3][1], rows[3][2], rows[3][3]
  );
  return m;
}
