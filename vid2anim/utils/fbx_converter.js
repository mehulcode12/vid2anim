#!/usr/bin/env node
/**
 * Standalone high-performance FBX to GLB Converter using Three.js & fflate.
 * Converts binary/ASCII FBX character models (Mixamo, etc.) into standardized binary GLTF (.glb).
 */

const fs = require('fs');
const path = require('path');
const THREE = require('three');
global.THREE = THREE;
global.window = global;

const blobStore = new Map();
let blobCounter = 0;
global.window.URL = {
  createObjectURL: (blob) => {
    const id = 'blob:mem-' + (++blobCounter);
    blobStore.set(id, blob);
    return id;
  },
  revokeObjectURL: (id) => {
    blobStore.delete(id);
  }
};

class MockBlob {
  constructor(parts, opts) {
    this.parts = parts || [];
    this.type = (opts && opts.type) || 'application/octet-stream';
    let len = 0;
    for (const p of this.parts) {
      if (typeof p === 'string') len += Buffer.byteLength(p, 'utf8');
      else if (p && p.byteLength !== undefined) len += p.byteLength;
      else if (p && p.size !== undefined) len += p.size;
    }
    this.size = len;
  }
  arrayBuffer() {
    const bufs = this.parts.map(p => {
      if (typeof p === 'string') return Buffer.from(p, 'utf8');
      if (p instanceof ArrayBuffer) return Buffer.from(p);
      if (ArrayBuffer.isView(p)) return Buffer.from(p.buffer, p.byteOffset, p.byteLength);
      if (p instanceof MockBlob) return Buffer.concat(p.parts.map(sub => {
        if (typeof sub === 'string') return Buffer.from(sub, 'utf8');
        if (sub instanceof ArrayBuffer) return Buffer.from(sub);
        if (ArrayBuffer.isView(sub)) return Buffer.from(sub.buffer, sub.byteOffset, sub.byteLength);
        return Buffer.from(sub);
      }));
      return Buffer.from(p);
    });
    const combined = Buffer.concat(bufs);
    return Promise.resolve(combined.buffer.slice(combined.byteOffset, combined.byteOffset + combined.byteLength));
  }
}
global.Blob = MockBlob;

class MockFileReader {
  readAsArrayBuffer(blob) {
    blob.arrayBuffer().then(ab => {
      this.result = ab;
      if (this.onloadend) this.onloadend();
    });
  }
}
global.window.FileReader = MockFileReader;

global.HTMLImageElement = class HTMLImageElement {
  constructor() {
    this._src = '';
    this.complete = false;
    this.width = 0;
    this.height = 0;
    this.listeners = {};
  }
  set src(val) {
    this._src = val;
    const b = blobStore.get(val);
    if (b) {
      this._rawBlob = b;
    }
    this.complete = true;
    this.width = 1024;
    this.height = 1024;
    const cbs = this.listeners['load'] || [];
    cbs.forEach(cb => cb.call(this));
    if (this.onload) this.onload.call(this);
  }
  get src() { return this._src; }
  setAttribute() {}
  addEventListener(evt, cb) {
    this.listeners[evt] = this.listeners[evt] || [];
    this.listeners[evt].push(cb);
  }
  removeEventListener(evt, cb) {
    if (this.listeners[evt]) {
      this.listeners[evt] = this.listeners[evt].filter(f => f !== cb);
    }
  }
};

let lastDrawnImage = null;
global.HTMLCanvasElement = class HTMLCanvasElement {};
class MockCanvas extends global.HTMLCanvasElement {
  constructor() {
    super();
    this.width = 1024;
    this.height = 1024;
  }
  getContext() {
    return {
      translate: () => {},
      scale: () => {},
      drawImage: (img) => { lastDrawnImage = img; },
      putImageData: () => {}
    };
  }
  toBlob(cb) {
    if (lastDrawnImage && lastDrawnImage._rawBlob) {
      cb(lastDrawnImage._rawBlob);
    } else {
      cb(new MockBlob([Buffer.alloc(100)], { type: 'image/png' }));
    }
  }
}

global.document = {
  createElementNS: (ns, tag) => (tag === 'canvas' ? new MockCanvas() : new global.HTMLImageElement()),
  createElement: (tag) => (tag === 'canvas' ? new MockCanvas() : new global.HTMLImageElement())
};

const { FBXLoader } = require('three/examples/jsm/loaders/FBXLoader.js');
const { GLTFExporter } = require('three/examples/jsm/exporters/GLTFExporter.js');
const fflate = require('fflate');
global.fflate = fflate;

const inputPath = process.argv[2];
const outputPath = process.argv[3];

if (!inputPath || !outputPath) {
  console.error('Usage: node fbx_converter.js <input.fbx> <output.glb>');
  process.exit(1);
}

try {
  const buf = fs.readFileSync(inputPath);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);

  const loader = new FBXLoader();
  const fbx = loader.parse(ab, '');

  // Convert materials to modern PBR MeshStandardMaterial preserving diffuse, normal & alpha textures
  fbx.traverse(child => {
    if (child.isMesh && child.material) {
      const mats = Array.isArray(child.material) ? child.material : [child.material];
      const newMats = mats.map(m => {
        // Build PBR material with preserved embedded textures
        const std = new THREE.MeshStandardMaterial({
          name: m.name,
          color: m.color || new THREE.Color(0xffffff),
          map: m.map || null,
          normalMap: m.normalMap || null,
          roughness: 0.7,
          metalness: 0.1,
          skinning: true,
          transparent: !!m.transparent,
          opacity: m.opacity !== undefined ? m.opacity : 1.0,
          side: THREE.FrontSide
        });

        // Ensure texture maps have sRGB encoding
        if (std.map) {
          std.map.encoding = THREE.sRGBEncoding;
        }

        // Apply natural studio fallback color only if mesh has no texture map and is flat white
        if (!std.map && std.color && (std.color.getHex() === 0xffffff || std.color.getHex() === 0x000000)) {
          const n = (child.name || '').toLowerCase();
          if (n.includes('hair')) std.color.setHex(0x2d1d16);
          else if (n.includes('top') || n.includes('shirt') || n.includes('jacket')) std.color.setHex(0x3b82f6);
          else if (n.includes('bottom') || n.includes('pant') || n.includes('jean')) std.color.setHex(0x1e293b);
          else if (n.includes('shoe') || n.includes('foot')) std.color.setHex(0x111827);
          else if (n.includes('body') || n.includes('skin') || n.includes('face')) std.color.setHex(0xd4a373);
          else if (n.includes('eye')) std.color.setHex(0x1e293b);
          else std.color.setHex(0x64748b);
        }

        return std;
      });
      child.material = Array.isArray(child.material) ? newMats : newMats[0];
    }
  });

  const exporter = new GLTFExporter();
  exporter.parse(
    fbx,
    result => {
      if (result instanceof ArrayBuffer) {
        fs.writeFileSync(outputPath, Buffer.from(result));
        console.log(`[FBX_CONVERTER] Successfully converted ${inputPath} -> ${outputPath} (${result.byteLength} bytes)`);
        process.exit(0);
      } else {
        console.error('[FBX_CONVERTER] Expected binary ArrayBuffer, got JSON object');
        process.exit(1);
      }
    },
    { binary: true, trs: true, embedImages: true }
  );
} catch (err) {
  console.error('[FBX_CONVERTER] Error:', err);
  process.exit(1);
}
