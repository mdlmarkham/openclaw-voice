// AudioWorkletProcessor that captures microphone input and posts
// ~4096-sample chunks to the main thread for encoding/transmission.
//
// Loaded via audioContext.audioWorklet.addModule(...) in the demo client
// (src/client/index.html) and via a Blob URL in the React widget
// (packages/react/src/audio-capture-processor.ts). Keep the two copies in
// sync — the React package embeds this source as a string for portability.
//
// Wire format is unchanged: the main thread receives Float32Array chunks and
// base64-encodes them into { type: 'audio', data: <base64 float32 PCM> }.

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
