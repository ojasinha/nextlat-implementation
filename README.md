# NextLat Implementation

My attempt at implementing [Teoh et al.](https://arxiv.org/pdf/2511.05963) NextLat model. Wasn't really able to make it perform better than GPT at d=1 at the Countdown task :(.

I used Karpathy's nanoGPT as the base for the GPT model and implemented the NextLat model by referencing the paper and the shared code. The loss was too complex for my liking so I tried to simplify. Might be why it's not producing the results that I wanted.

Code for generating the training set was vibe-coded using the original repo. Model and training pipeline is human-written slop. Utility functions and making it easier to run via CLI is vibe-coded.