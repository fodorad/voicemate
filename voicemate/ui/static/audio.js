// Browser audio bridge: microphone → server, synthesized speech → speakers.
//
// Protocol on /ws/audio?sid=<session>:
//   browser → server  binary: PCM16 mono 16 kHz frames (20 ms)
//                     text:   {"type": "playback", "state": "started" | "ended"}
//                             {"type": "mic", "state": "off"}  (drop a half-heard utterance)
//   server → browser  binary: [uint32 LE sample rate][PCM16 mono]
//                     text:   {"type": "stop"}   (barge-in: drop queued speech)
window.voicemate = (() => {
  const TARGET_RATE = 16000;
  let ws = null;
  let ctx = null;
  let micStream = null;
  let micSource = null;
  let micNode = null;
  let workletReady = null;
  let micOn = false;
  let playHead = 0;
  const sources = new Set();
  let sessionId = null;

  function send(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  }

  function connect(sid) {
    sessionId = sid;
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${scheme}://${location.host}/ws/audio?sid=${encodeURIComponent(sid)}`);
    ws.binaryType = "arraybuffer";
    ws.onmessage = (event) => {
      if (typeof event.data === "string") {
        const message = JSON.parse(event.data);
        if (message.type === "stop") stopPlayback();
      } else {
        play(event.data);
      }
    };
    ws.onclose = () => setTimeout(() => sessionId && connect(sessionId), 1000);
  }

  async function ensureAudio() {
    if (!ctx) ctx = new AudioContext({ latencyHint: "interactive" });
    // Never await resume(): without a user gesture the promise stays pending forever.
    if (ctx.state === "suspended") ctx.resume().catch(() => {});
    return ctx;
  }

  function play(buffer) {
    if (!ctx) return; // no user gesture yet: audio cannot start
    const rate = new DataView(buffer).getUint32(0, true);
    const pcm = new Int16Array(buffer, 4);
    if (!pcm.length) return;
    const audio = ctx.createBuffer(1, pcm.length, rate);
    const channel = audio.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 32768;
    const source = ctx.createBufferSource();
    source.buffer = audio;
    source.connect(ctx.destination);
    const startAt = Math.max(ctx.currentTime + 0.02, playHead);
    source.start(startAt);
    playHead = startAt + audio.duration;
    if (sources.size === 0) send({ type: "playback", state: "started" });
    sources.add(source);
    source.onended = () => {
      sources.delete(source);
      if (sources.size === 0) send({ type: "playback", state: "ended" });
    };
  }

  function stopPlayback() {
    for (const source of sources) {
      source.onended = null;
      try { source.stop(); } catch (e) { /* already stopped */ }
    }
    const wasPlaying = sources.size > 0;
    sources.clear();
    playHead = 0;
    if (wasPlaying) send({ type: "playback", state: "ended" });
  }

  async function startMic() {
    const context = await ensureAudio();
    workletReady ??= context.audioWorklet.addModule("/static/mic-worklet.js");
    await workletReady;
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
    micSource = context.createMediaStreamSource(micStream);
    micNode = new AudioWorkletNode(context, "mic-capture", {
      processorOptions: { targetRate: TARGET_RATE },
    });
    micNode.port.onmessage = (event) => {
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(event.data);
    };
    micSource.connect(micNode);
    micOn = true;
  }

  // Tear the whole capture graph down. A worklet node that is only disconnected from its
  // outputs keeps posting (silent) frames, and after a restart those interleave with the
  // live microphone, so the server never hears a clean utterance.
  function stopMic() {
    if (micStream) micStream.getTracks().forEach((track) => track.stop());
    if (micSource) micSource.disconnect();
    if (micNode) {
      micNode.port.onmessage = null;
      micNode.port.close();
      micNode.disconnect();
    }
    micStream = null;
    micSource = null;
    micNode = null;
    micOn = false;
    send({ type: "mic", state: "off" });
  }

  // Send a WAV file through the same path as the microphone (used by E2E tests).
  // Frames are sent without real-time pacing: the VAD counts samples, not wall-clock time,
  // and timers are throttled to 1 Hz in background tabs.
  async function playFakeMic(url) {
    while (!ws || ws.readyState !== WebSocket.OPEN) {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    // Decode offline: works before any user gesture (a live AudioContext would stay suspended).
    const data = await (await fetch(url)).arrayBuffer();
    const decoded = await new OfflineAudioContext(1, 1, TARGET_RATE).decodeAudioData(data);
    const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * TARGET_RATE), TARGET_RATE);
    const src = offline.createBufferSource();
    src.buffer = decoded;
    src.connect(offline.destination);
    src.start();
    const samples = (await offline.startRendering()).getChannelData(0);
    const frame = TARGET_RATE * 0.02;
    for (let i = 0; i < samples.length; i += frame) {
      const chunk = new Int16Array(frame);
      for (let j = 0; j < frame && i + j < samples.length; j++) {
        chunk[j] = Math.max(-1, Math.min(1, samples[i + j])) * 32767;
      }
      ws.send(chunk.buffer);
    }
    const silence = new Int16Array(frame);
    for (let k = 0; k < 60; k++) ws.send(silence.buffer.slice(0)); // 1.2 s ends the utterance
    return samples.length / TARGET_RATE;
  }

  async function toggleMic() {
    if (micOn) {
      stopMic();
    } else {
      try {
        await startMic();
      } catch (error) {
        console.error("[voicemate] microphone unavailable", error);
      }
    }
    return micOn;
  }

  // Ctrl+M toggles the microphone, Ctrl+H the hold. Both press the visible buttons, so the
  // page and the server stay in sync. (Cmd+M minimizes Chrome on macOS before the page can
  // intercept it.)
  const SHORTCUTS = { KeyM: ".vm-mic", KeyH: ".vm-hold" };
  document.addEventListener("keydown", (event) => {
    if (!event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return;
    const button = SHORTCUTS[event.code] && document.querySelector(SHORTCUTS[event.code]);
    if (!button) return;
    event.preventDefault();
    button.click();
  });

  return { connect, ensureAudio, toggleMic, stopPlayback, playFakeMic, isMicOn: () => micOn };
})();
