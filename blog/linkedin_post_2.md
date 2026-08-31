# LinkedIn — blog post 2 (release post)

Post after the Substack URL is live. Paste the URL, then run it through LinkedIn's Post
Inspector once to force a card refresh — the og:image changed since these URLs were last
scraped.

---

We built the benchmark that judges our own fashion attribute models, then published it — along
with every run we lost.

Three models are out today. Here is what they do on frozen test sets, against FashionCLIP 2.0
with matched supervised heads:

→ Catalogue product images, 10 attributes: 0.8292 vs 0.6657. We win all 10 fields, by 31% on
average.
→ Full-body photos, 18 attributes: 0.6917 vs 0.5943. We win 17 of 18. The one we lose is
lower_fabric, where honestly both systems are close to useless.

The numbers matter less than how they were produced.

Four tracks, never averaged — a model can be strong on clean product shots and weak on street
photography, and one headline number lets the good result hide the bad one. Predictions are
generated without labels and SHA-256 committed before the scorer opens anything. The scorer
fails closed: an unknown value raises instead of being quietly dropped. Every interval is a
10,000-sample paired bootstrap clustered at the unit that would actually leak.

We also score whether a model knows an attribute is *not visible* rather than inventing one.
That is the difference between a catalogue you can populate unattended and one you cannot.

And the prediction files ship — including the losses. Three zero-shot VLM runs that did not
win. A concatenation result that looked like a real gain until a control showed it was just
extra width, so we retracted it.

Weights: one MIT, two CC BY-NC because their evaluation corpora are research-only. That
restriction binds us too — those exact weights are not in our paid product, and we checked
before publishing that no serving path loads them.

If you work on product data and want to argue with our numbers, the repository has the scorers,
the split builders and our predictions. We would rather be corrected in public than in a
customer's pipeline.

[BLOG URL]

Code: github.com/hopit-ai/Moda_ner
Benchmarks: hopit-ai.github.io/Moda_ner

#MachineLearning #ComputerVision #FashionTech #RetailAI #OpenSource
