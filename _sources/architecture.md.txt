# How it works

voicemate is a pipeline of small local models around one LangGraph agent. Everything runs
in one Python process except the language model, which runs in Ollama.

## The whole system

```mermaid
flowchart LR
    subgraph Browser["Browser tab (localhost:8080)"]
        MIC[Microphone<br/>echo cancellation] --> WORKLET[AudioWorklet<br/>16 kHz PCM]
        PLAYER[Speaker<br/>gapless scheduler]
        UI[Status rail, conversation,<br/>activity trace, notes]
    end

    subgraph Server["voicemate process (CPU)"]
        VAD[Silero VAD<br/>endpointing] --> ASR[Parakeet v3<br/>speech to text]
        ASR --> LID[Language<br/>detection]
        LID --> AGENT[LangGraph agent]
        AGENT --> CHUNK[Sentence chunker]
        CHUNK --> TTS[Kokoro EN / Piper HU<br/>text to speech]
        AGENT <--> MEM[(Memory<br/>LanceDB + bge-m3)]
        AGENT <--> NOTES[(Notes<br/>SQLite)]
        AGENT <--> TOOLS[Tools]
        BUS{{Event bus}}
    end

    subgraph GPU["Ollama (GPU)"]
        LLM[Chat model<br/>one profile: gemma4 or qwen]
    end

    WORKLET -- WebSocket --> VAD
    TTS -- WebSocket --> PLAYER
    AGENT <--> LLM
    TOOLS -- only these go online --> NET((Web search, pages,<br/>arXiv, weather))
    VAD & ASR & AGENT & TTS -.-> BUS -.-> UI
```

The speech models run on the CPU on purpose: the GPU and most of the memory belong to the
language model (see the spec, §10).

## The LangGraph agent

This diagram is generated from the compiled graph (`graph.get_graph().draw_mermaid()`):

```mermaid
graph TD;
    __start__([start]) --> recall;
    recall --> agent;
    agent -.->|tool calls| tools;
    tools --> agent;
    agent -.->|answer| __end__([end]);
    tools -.->|"interrupt(): needs permission"| pause([paused, waiting for you]);
    pause -.->|"Command(resume=your answer)"| tools;
```

- **recall** embeds what you said and loads relevant facts, earlier conversations and
  research into a per-turn context block (about 50 ms).
- **agent** calls the model with every tool bound. The system prompt never changes, so
  Ollama reuses its cached prompt; only the newest message carries the context.
- **tools** runs the requested tools. Errors go back to the model as text, so it can retry
  or explain. After 6 tool rounds the tools are removed and the model must answer.
- A **checkpointer** (SQLite) keeps the conversation, including interrupted answers and
  paused runs. The current thread id lives in `data/conversation.json`, so a reloaded page
  or a restarted server continues where you left off; **New conversation** starts a new one.
- If the model goes silent after a tool round, the agent asks it once more to answer.

## Asking before risky actions (human-in-the-loop)

```mermaid
sequenceDiagram
    actor You
    participant S as VoiceSession
    participant G as LangGraph
    participant T as write_file tool

    You->>S: "Save my notes to plan.md"
    S->>G: run
    G->>T: write_file(plan.md)
    T->>T: file exists
    T-->>G: interrupt("plan.md already exists. Replace it?")
    G-->>S: run paused (checkpointed)
    S-->>You: dialog + spoken question
    You->>S: "no, call it plan2" (voice, typing or a button)
    S->>G: Command(resume="no, call it plan2")
    G->>T: answer is not a yes → nothing replaced
    T-->>G: "The user did not approve; they said: no, call it plan2"
    G->>G: model writes plan2.md instead
```

The tool receives your exact words, so a "no, but…" can redirect the action. The pause
survives page reloads because it lives in the checkpointer.

## Choosing the model

```mermaid
flowchart TD
    P[Preferred profile<br/>fast / gemma / qwen] --> I{Installed in Ollama?}
    I -- no --> F[Fallback profile]
    I -- yes --> M{"1.5 × size on disk + 1 GB<br/>+ 5 GB reserve fits in<br/>free RAM?"}
    M -- yes --> U[Use it]
    M -- no --> F
    F --> W[Use fallback, warn in the UI]
```

A model that does not fit gets paged out by macOS between turns; replies then take 15+ s
instead of about 1 s. Each profile is one model used for everything.

## One spoken turn

```mermaid
sequenceDiagram
    actor You
    participant B as Browser
    participant S as VoiceSession
    participant A as ASR
    participant G as LangGraph
    participant L as Ollama
    participant T as TTS

    You->>B: speak
    B->>S: 20 ms audio frames
    S-->>B: Listening
    Note over S: about 1 s of speech:<br/>live partial transcript
    You->>B: stop talking
    Note over S: 700 ms of silence ends the turn
    S->>A: utterance
    A-->>S: text (~0.3 s)
    S->>G: astream(text, language)
    G->>G: recall memory (~50 ms)
    G->>L: prompt (cached prefix)
    L-->>G: first tokens (~1 s)
    loop each sentence while the model keeps writing
        G-->>S: tokens
        S->>T: complete sentence
        T-->>B: audio
        B-->>You: speech
    end
```

The first sentence is spoken while the model is still writing the rest. A network tool
(search, weather) adds a short spoken filler ("One moment, let me check.").

## Talking while it works

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Listening: you speak
    Listening --> Thinking: silence ends the turn
    Thinking --> Speaking: first sentence ready
    Speaking --> Idle: playback finished

    Thinking --> Listening: you speak again
    note right of Thinking
        Nothing was said yet: the unfinished turn is dropped and,
        within 20 s, your two utterances become one question.
    end note

    Speaking --> Listening: you speak again
    note right of Speaking
        Playback stops at once. The part you heard is kept in the
        history, marked as interrupted; cut-off tool calls are closed.
        Your next words are treated as a correction or a new question.
    end note
```

**Hold until I'm finished.** The Hold button (Ctrl+H) stops a running answer and suspends
endpointing: each pause still produces a transcript, but it is collected rather than
answered. **I'm finished**, or turning the microphone off, sends everything as one
question. Turning the microphone off without a hold also answers what was heard so far,
including a sentence cut off mid-word.

While it speaks, barge-in needs louder and longer speech (`vad.barge_in_*`), so its own
voice from the speakers does not interrupt it.

## Research memory

```mermaid
flowchart TD
    Q[web_search query] --> C{Similar search stored<br/>and still fresh?}
    C -- yes --> M[Answer from memory<br/>badge: from memory]
    C -- no --> W[Search the web<br/>badge: sent to web]
    W --> S[(Store the search and<br/>its snippets as documents)]
    S --> R[Answer]
    P[fetch_page / download_paper] --> D[(Chunk, embed and store<br/>the full text)]
    D -.-> REC[recall node finds it<br/>in later conversations]
    S -.-> REC
```

Freshness depends on the topic: news and jobs expire after a day, general pages after 30
days, papers after a year.
