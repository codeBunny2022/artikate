## Section 1 — Diagnose a Failing LLM Pipeline


### Problem 1: The Bot Confidently Gives Wrong Pricing Answers

**What I would investigate first:** I would start by checking whether pricing information lives in the system prompt or is retrieved dynamically. My first assumption in a system with "no changes after launch" is that pricing was hardcoded into the system prompt at deployment time and has since gone stale. I would compare the prices the bot quotes against current pricing to see if they match the original launch-time values and if they do, the diagnosis is confirmed without needing logs.

**What I would rule out:**

* Temperature issue: high temperature causes random variance and different answers on the same question. If the wrong answers are consistently the same wrong value, temperature is not the cause. I would verify by sending the same pricing question 5 times and checking for consistency.
* Retrieval issue: the scenario states no retrieval layer exists, so I would eliminate this immediately.
* Model knowledge cutoff: GPT 4o has no access to proprietary company pricing from training data, so this cannot be the source of confident pricing answers. The model must be reading it from somewhere in the prompt.

**Root cause I would identify: Stale hardcoded system prompt.** My conclusion would be that pricing was written directly into the system prompt at launch and never updated. The model confidently reproduces whatever is in its context and it has no mechanism to know the data is outdated.

**Fix I would propose:** Replacing hardcoded pricing with a live injection at request time.

I would also add a monitoring alert that fires if the pricing data injected into the prompt is older than 24 hours, so stale data cannot silently persist again.


### Problem 2: Bot Responds in English When User Writes in Hindi or Arabic

**What I would investigate first:** I would look at the system prompt. My first assumption is that the system prompt is written entirely in English with no language instruction. In a system prompt + user message architecture, GPT 4o determines output language by weighting the dominant language in its full context. A long English system prompt will override a short Hindi or Arabic user message and the model defaults to the language it has seen the most of. **This is documented emergent behaviour in large language models.**

**What I would rule out:**

* The user message not being passed correctly: I would verify the message reaches the model in the original language by logging the raw API payload.
* A preprocessing step stripping non-ASCII characters: I would check whether any middleware normalises or translates input before it hits the LLM.

**Root cause I would identify: System prompt language dominance with no language instruction.** When the system prompt is 400 to 600 tokens of English and the user writes a short sentence in Hindi, the English context dominates. The model has no explicit instruction to mirror the user's language, so it defaults to English.

**Specific prompt fix:**

I will add this as the **final instruction** in the system prompt (end position gives it recency weight):

LANGUAGE RULE → HIGHEST PRIORITY: Always respond in the exact language the user writes in.

* User writes in Hindi → respond entirely in Hindi
* User writes in Arabic → respond entirely in Arabic
* User writes in English → respond in English Never mix languages. Never default to English if the user did not write in English. This rule overrides all other formatting preferences.


Explicit per-language enumeration is more reliable than a generic "match user language" instruction, which models sometimes interpret loosely. I would test this by sending 50 messages across 5 languages and asserting that detected language response is same as detected language user message on each.


### Problem 3: Response Time Degraded from 1.2s to 8–12s Over Two Weeks

**How I would approach this:**
I would first instrument latency at three distinct layers before touching any code: (1) total request time at the load balancer, (2) time spent inside the app server before and after the OpenAI call, (3) OpenAI's own reported latency via response headers. This tells me immediately whether the bottleneck is inside our system or external.

**Three distinct causes I would investigate:**

**Cause A → Unbounded conversation history growth (will investigate it first):** My strongest assumption for this pattern and latency creeping up gradually with no code changes is that the app stores conversation history and appends it to every request. As the user base grows and sessions get longer, average context length grows. GPT-4o inference time scales with total token count. A context that grew from 1,000 to 6,000 tokens over two weeks would produce exactly this latency curve. I would check by logging average prompt per request over time. If the trend is rising, I would implement history truncation or summarisation immediately.

**Cause B → OpenAI rate limit queue back-pressure:** As request volume grew, the system may be hitting per-minute token limits and silently absorbing the retry after delays inside the SDK's retry loop. This manifests as degraded latency without surfaced errors. I would check OpenAI API response headers for x_ratelimit_remaining_tokens and review the SDK retry configuration to see if retries are being swallowed silently.

**Cause C → No response streaming enabled:** If stream is false the application waits for the complete response before returning anything to the user. As response length or context grows, wall-clock wait time grows linearly. This is the easiest fix by enabling stream, immediately improves perceived latency with no infrastructure changes required.

I would investigate Cause A first because it produces a gradual, correlated degradation pattern that matches the two-week timeline exactly.


### Post-Mortem Summary (for non-technical stakeholders)

Well over the past two weeks, three separate issues were identified and resolved in our customer support chatbot.

The first issue caused the bot to quote incorrect product prices with full confidence. The likely cause is that pricing information was written into the bot's instructions at launch and was never updated when prices changed. The bot had no way to detect that the information was outdated and it simply repeated what it was given. The fix is to connect the bot to a live pricing source so it always uses current data, rather than relying on instructions written at a point in time.

The second issue caused the bot to respond in English when customers wrote in Hindi or Arabic. This is a known behaviour in large language models: when the bot's operating instructions are written entirely in English, the model tends to reply in English regardless of what the customer wrote. The fix is to add an explicit language-matching rule to the bot's instructions, which has been tested across multiple languages.

The third issue caused response times to rise from roughly one second to eight to twelve seconds. The most probable cause is that the bot was accumulating full conversation history and sending it with every message, making each request progressively larger and slower to process. Capping and summarising conversation history would keep request sizes consistent and restore original response times.