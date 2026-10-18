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

const makeImg = () => ({
  setAttribute: () => {},
  style: {},
  addEventListener: () => {}
});

global.document = {
  createElementNS: makeImg,
  createElement: makeImg
};

class MockFileReader {
  readAsArrayBuffer(blob) {
    blob.arrayBuffer().then(buf => {
      this.result = buf;
      if (this.onloadend) this.onloadend();
    });
  }
}
global.window.FileReader = MockFileReader;

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

  // Sanitize materials: clear unresolvable texture references so GLTFExporter doesn't crash on undefined image.width
  fbx.traverse(child => {
    if (child.isMesh && child.material) {
      const mats = Array.isArray(child.material) ? child.material : [child.material];
      mats.forEach(m => {
        ['map', 'normalMap', 'bumpMap', 'roughnessMap', 'metalnessMap', 'specularMap', 'emissiveMap'].forEach(texKey => {
          if (m[texKey] && (!m[texKey].image || !m[texKey].image.width)) {
            m[texKey] = null;
          }
        });
        m.needsUpdate = true;
      });
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
    { binary: true, trs: true }
  );
} catch (err) {
  console.error('[FBX_CONVERTER] Error:', err);
  process.exit(1);
}
