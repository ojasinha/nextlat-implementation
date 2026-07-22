# NextLat Implementation

My attempt at implementing [Teoh et al.](https://arxiv.org/pdf/2511.05963) NextLat model. Wasn't really able to make it perform better than GPT at d=1 at the Countdown task :(.

Trained using the same hyperparameters as the paper and used the model the model with the lowest validation losses for both GPT (93M parameters) and NextLat (95M parameters). I do not have the compute unfortunately to train three different models for comparison across three different seeds.

<table align="center">
  <tr>
    <th>Model</th>
    <th align="right">Score</th>
  </tr>
  <tr>
    <td>GPT</td>
    <td align="right">46.15%</td>
  </tr>
  <tr>
    <td>NextLat</td>
    <td align="right">45.68%</td>
  </tr>
</table>

<p align="center"><em>Comparison of GPT and NextLat performance on 10,000 test samples.</em></p>


I used Karpathy's nanoGPT as the base for the GPT model and implemented the NextLat model by referencing the paper and the shared code. The loss was too complex for my liking so I tried to simplify. Might be why it's not producing the results that I wanted.

Code for generating the training set was vibe-coded using the original repo. Model and training pipeline is human-written slop. Utility functions and making it easier to run via CLI is vibe-coded.
