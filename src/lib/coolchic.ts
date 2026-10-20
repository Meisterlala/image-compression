// Lazy, separate WASM module: no tensor runtime, Python, GPU, models or network
// service. Training and actual bitstream decoding run only inside codec.worker.
export async function loadCoolChic(quality: number, effort: number, progress: (text: string) => void) {
  const { default: init, CoolChicEncoder, decodeCoolChic } = await import('../wasm-coolchic/coolchic');
  await init();
  return {
    async encode(image: ImageData): Promise<Uint8Array<ArrayBuffer>> {
      let encoder: InstanceType<typeof CoolChicEncoder> | undefined;
      try {
        encoder = new CoolChicEncoder(new Uint8Array(image.data.buffer, image.data.byteOffset, image.data.byteLength), image.width, image.height, quality, effort);
        let done = false;
        while (!done) {
          done = encoder.step();
          progress(`Training Cool-chic on your CPU · ${encoder.completed} / ${encoder.passes} passes…`);
          // Let the worker dispatch progress between bounded native calls. Cancel
          // terminates the whole worker, including an in-flight WASM call.
          await new Promise<void>(resolve => setTimeout(resolve, 0));
        }
        progress('Quantizing the per-image model and entropy-coding the .cool bitstream…');
        const trained = encoder;
        encoder = undefined; // finish consumes/frees the Rust encoder, including on error.
        return new Uint8Array(trained.finish());
      } finally { encoder?.free(); }
    },
    decode(buffer: ArrayBuffer): ImageData {
      const image = decodeCoolChic(new Uint8Array(buffer));
      try { return new ImageData(new Uint8ClampedArray(image.rgba()), image.width, image.height); }
      finally { image.free(); }
    },
  };
}
