# LinkedIn — post 2 (the numbers and the release)

Written to continue the voice of post 1: no hashtags, no bullet arrows, numbers inline in
prose, link in the comments rather than the body. Shorter than post 1.

Run the URL through LinkedIn's Post Inspector once before posting — the og:image changed since
these links were last scraped.

---

As promised in Post 1, here are the numbers.

Three models are out today, scored on frozen test sets that existed before the models did.

On catalogue product images, across 10 attributes, we score 0.8292 against 0.6657 for
FashionCLIP 2.0. On full-body photos, across 18 attributes, 0.6917 against 0.5943. Field by
field, that is all 10 catalogue attributes and 17 of 18 on full-body. The one we lose is
lower_fabric, where both systems are close to useless and neither deserves your trust.

We are also publishing what did not work.

One of our own ideas scored +0.0095 over the baseline. Small, but real. Then we ran the
control: instead of adding a second encoder, we made the original features wider by the same
amount, at random. That scored +0.0096. The gain was width, not the idea. So we retracted it.

Same instinct as the training run in Post 1. If you only check the number you were hoping for,
you will ship it.

We also ran Gemini 3.5 Flash against a stop rule fixed before the run. It did not stay
competitive over the first 100 rows, so we stopped and did not buy the rest.

Every prediction file ships with the models, hashes included, so you can recompute all of it
rather than take our word.

Post 3 is about what none of this tells you about your own catalogue.

Post link in comments.
