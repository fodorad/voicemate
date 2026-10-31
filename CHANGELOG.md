# Changelog

## [1.2.0](https://github.com/fodorad/voicemate/compare/voicemate-v1.1.0...voicemate-v1.2.0) (2026-10-31)


### Features

* **hots:** ask A or B and learn hero name spellings ([#43](https://github.com/fodorad/voicemate/issues/43)) ([b600671](https://github.com/fodorad/voicemate/commit/b6006711067b37250ee326d9951bbd293936c341))
* **hots:** brief the two teams on the loading screen ([#57](https://github.com/fodorad/voicemate/issues/57)) ([0788887](https://github.com/fodorad/voicemate/commit/078888760ffa1cae7031df9f39e0978d99162fb9))
* **hots:** detect locked vs pending heroes by label colour ([#47](https://github.com/fodorad/voicemate/issues/47)) ([1c9cddf](https://github.com/fodorad/voicemate/commit/1c9cddf788f707a4ed21dcba34288742353641dc))
* **hots:** draft mode commands and short answers ([#51](https://github.com/fodorad/voicemate/issues/51)) ([322f6a2](https://github.com/fodorad/voicemate/commit/322f6a2668b3138b212529f79d5966bfd72dfe9d))
* **hots:** identify misread hero labels by learned portraits ([#56](https://github.com/fodorad/voicemate/issues/56)) ([7c74ade](https://github.com/fodorad/voicemate/commit/7c74ade99e50bcca308f0df128d5c54c6736af87))
* **hots:** interpret draft screens from text ([#46](https://github.com/fodorad/voicemate/issues/46)) ([f4626f3](https://github.com/fodorad/voicemate/commit/f4626f379d17136ed75b61dbb0c83857916a5b48))
* **hots:** log captured draft screens ([#45](https://github.com/fodorad/voicemate/issues/45)) ([ffd72f7](https://github.com/fodorad/voicemate/commit/ffd72f7ef71e68a931a6b7fc24d4f6a188a49b3a))
* **hots:** measure the screen layout so any resolution or window reads the draft ([#55](https://github.com/fodorad/voicemate/issues/55)) ([c9820fd](https://github.com/fodorad/voicemate/commit/c9820fd7920dfc5593735cfc2d11d9c7a0eb4b3e))
* **hots:** numbered name questions, screen watched only in draft mode ([#54](https://github.com/fodorad/voicemate/issues/54)) ([205daf4](https://github.com/fodorad/voicemate/commit/205daf465924199ce4c115e8ef659ed7ca8ec8b9))
* **hots:** read ban slots by portrait matching ([#48](https://github.com/fodorad/voicemate/issues/48)) ([3bfad42](https://github.com/fodorad/voicemate/commit/3bfad4227a00ff3fbc7430456307d2d619acc09e))
* **hots:** screen reader seam and macOS adapter ([#44](https://github.com/fodorad/voicemate/issues/44)) ([a79b4b6](https://github.com/fodorad/voicemate/commit/a79b4b6ec7176441853d5d82e3d7dd7a4b129ccf))
* **hots:** sound-alike hero name matching ([#42](https://github.com/fodorad/voicemate/issues/42)) ([6410011](https://github.com/fodorad/voicemate/commit/6410011a3751f414aae6722474c63ea25f372a4a))
* **hots:** watch the draft and keep it current ([#50](https://github.com/fodorad/voicemate/issues/50)) ([44fc856](https://github.com/fodorad/voicemate/commit/44fc856f077bc2a877506c6b270b2de3c61e8c1f))
* **pipeline:** skills answer asynchronously and can run in the background ([#49](https://github.com/fodorad/voicemate/issues/49)) ([37b6fea](https://github.com/fodorad/voicemate/commit/37b6fea2a18f62c1c354b24e3aa181bd7e77f8c2))
* **tts:** female English default voice ([#40](https://github.com/fodorad/voicemate/issues/40)) ([dd8e1a5](https://github.com/fodorad/voicemate/commit/dd8e1a5de81c8b03e3226f6a8a4fa062b3cd17a2))
* **ui:** global hold-to-talk hotkey ([#58](https://github.com/fodorad/voicemate/issues/58)) ([3c5903d](https://github.com/fodorad/voicemate/commit/3c5903d18ee2b97e391162817137db25589af185))


### Bug Fixes

* **hots:** review fixes for draft vision ([#53](https://github.com/fodorad/voicemate/issues/53)) ([3f66bc0](https://github.com/fodorad/voicemate/commit/3f66bc08e189493cf269c8d34aba3b58adf9ba05))


### Documentation

* describe reading the HotS draft from the screen ([#52](https://github.com/fodorad/voicemate/issues/52)) ([474e4de](https://github.com/fodorad/voicemate/commit/474e4deb7fd318c60a1f543a2526e53625b2c64f))

## [1.1.0](https://github.com/fodorad/voicemate/compare/voicemate-v1.0.1...voicemate-v1.1.0) (2026-10-26)


### Features

* **hots:** draft advisor for counters, picks and bans ([#35](https://github.com/fodorad/voicemate/issues/35)) ([83a72ff](https://github.com/fodorad/voicemate/commit/83a72ff0ac3ee510338c2196b759af05e8457755))
* **hots:** hero roster and Icy Veins page parsers ([#33](https://github.com/fodorad/voicemate/issues/33)) ([34904f9](https://github.com/fodorad/voicemate/commit/34904f9f631a7d6dfc5c1a9e8049681819b1cf58))
* **hots:** knowledge base with startup freshness check ([#34](https://github.com/fodorad/voicemate/issues/34)) ([063d6ed](https://github.com/fodorad/voicemate/commit/063d6ed8f7ab3434266e6ad42e585ff6057d9bc9))
* **hots:** voice intents and draft state ([#36](https://github.com/fodorad/voicemate/issues/36)) ([50aa65a](https://github.com/fodorad/voicemate/commit/50aa65ab5c5c5ea46ef6512656f591b86546c13c))
* **pipeline:** skill seam answers HotS questions without the LLM ([#37](https://github.com/fodorad/voicemate/issues/37)) ([c27517b](https://github.com/fodorad/voicemate/commit/c27517b72b9e93b3d5b43c54b6d71b3761ffdbc6))
* **pipeline:** wake-word activation mode ([#38](https://github.com/fodorad/voicemate/issues/38)) ([a474630](https://github.com/fodorad/voicemate/commit/a474630c52e57af8925a3603fe9839a59649ccab))
* **tts:** phonon English voice, Kokoro stays selectable ([#31](https://github.com/fodorad/voicemate/issues/31)) ([c56fd93](https://github.com/fodorad/voicemate/commit/c56fd93ab789b153dbfb8a7dc6d1b07f3b241285))


### Documentation

* describe the HotS draft advisor ([#39](https://github.com/fodorad/voicemate/issues/39)) ([b4ff2c0](https://github.com/fodorad/voicemate/commit/b4ff2c0619f25103c786b9eec22fc66405261230))

## [1.0.1](https://github.com/fodorad/voicemate/compare/voicemate-v1.0.0...voicemate-v1.0.1) (2026-10-18)


### Bug Fixes

* **agent:** read addresses the user gives with fetch_page ([#18](https://github.com/fodorad/voicemate/issues/18)) ([d538213](https://github.com/fodorad/voicemate/commit/d5382136d6f758d920b1b549c5dc9275fbacf91b))
* repair the arXiv integration test, raise the coverage gate and quiet the test output ([#25](https://github.com/fodorad/voicemate/issues/25)) ([b75613d](https://github.com/fodorad/voicemate/commit/b75613d9b6a40b77391f2506ce0ad54e7d61327a))


### Documentation

* describe the cleanup ([#30](https://github.com/fodorad/voicemate/issues/30)) ([f053f42](https://github.com/fodorad/voicemate/commit/f053f42927ad1e15ee24b595521a9ad8f87d9cf2))
* describe the deepened modules ([#16](https://github.com/fodorad/voicemate/issues/16)) ([0e8e3f4](https://github.com/fodorad/voicemate/commit/0e8e3f41f4956cd85ef187fa2a8b92e879813cad))
* describe the round-2 modules ([#24](https://github.com/fodorad/voicemate/issues/24)) ([cae7c42](https://github.com/fodorad/voicemate/commit/cae7c42aed41a6e0774594a5ae1a63515864855f))

## [1.0.0](https://github.com/fodorad/voicemate/compare/voicemate-v0.1.0...voicemate-v1.0.0) (2026-10-07)


### Features

* baseline hungarian offline model ([823774d](https://github.com/fodorad/voicemate/commit/823774df1bcb740a4249207f8344645f3dd3636d))
* local bilingual voice assistant with LangGraph agent, memory and browser UI ([#1](https://github.com/fodorad/voicemate/issues/1)) ([568f6da](https://github.com/fodorad/voicemate/commit/568f6da78b4ec5b912e2c87b76528fbefe6c382d))


### Bug Fixes

* keep the tool-call message when closing an interrupted turn ([#8](https://github.com/fodorad/voicemate/issues/8)) ([23a66de](https://github.com/fodorad/voicemate/commit/23a66de51c361626c04588ac862536833c171d85))


### Documentation

* add an early-look screenshot of the UI ([#7](https://github.com/fodorad/voicemate/issues/7)) ([35f20f2](https://github.com/fodorad/voicemate/commit/35f20f26cb5758bec5f17e682f858d393fb7344a))

## Changelog

All notable changes are recorded here by release-please from Conventional Commits.
