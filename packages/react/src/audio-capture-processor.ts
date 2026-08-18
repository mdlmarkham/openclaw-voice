// AudioWorkletProcessor source for the React widget.
//
// The React package is distributed as an npm package with no static file
// server, so the worklet module is embedded as a string and loaded from a
// Blob URL (audioContext.audioWorklet.addModule(URL.createObjectURL(...))).
//
// Keep this in sync with src/client/audio-capture-processor.js — the demo
// client serves that file directly. Both post ~4096-sample Float32Array
// chunks to the main thread, which base64-encodes them into
// { type: 'audio', data: <base64 float32 PCM> }.

export const AUDIO_CAPTURE_PROCESSOR_SOURCE = `
class AudioCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._chunkSize = 4096;
    this._buffer = new Float32Array(this._chunkSize);
    this._bufferLength = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input.length > 0) {
      const channel = input[0];
      for (let i = 0; i < channel.length; i++) {
        this._buffer[this._bufferLength++] = channel[i];
        if (this._bufferLength === this._chunkSize) {
          this.port.postMessage(this._buffer, [this._buffer.buffer]);
          this._buffer = new Float32Array(this._chunkSize);
          this._bufferLength = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor('audio-capture-processor', AudioCaptureProcessor);
`;
