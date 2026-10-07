// AudioWorklet: downsample the microphone to 16 kHz mono and post 20 ms PCM16 frames.
class MicCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetRate = (options.processorOptions && options.processorOptions.targetRate) || 16000;
    this.ratio = sampleRate / this.targetRate; // input samples per output sample
    this.frameSize = Math.round(this.targetRate * 0.02);
    this.frame = new Int16Array(this.frameSize);
    this.filled = 0;
    this.position = 0; // fractional read position into the current input block
    this.acc = 0;
    this.accCount = 0;
  }

  process(inputs) {
    const input = inputs[0] && inputs[0][0];
    if (!input) return true;
    // Box-filter decimation: average all input samples that fall into one output sample.
    for (let i = 0; i < input.length; i++) {
      this.acc += input[i];
      this.accCount += 1;
      this.position += 1;
      if (this.position >= this.ratio) {
        this.position -= this.ratio;
        const value = Math.max(-1, Math.min(1, this.acc / this.accCount));
        this.frame[this.filled++] = value * 32767;
        this.acc = 0;
        this.accCount = 0;
        if (this.filled === this.frameSize) {
          this.port.postMessage(this.frame.buffer, [this.frame.buffer]);
          this.frame = new Int16Array(this.frameSize);
          this.filled = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor("mic-capture", MicCapture);
