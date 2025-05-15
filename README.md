# Markov-Mamba
Code to experiment on Mamba (Selective State-Space Models) with Markov data.

### A few pointers

-   [Markov-Mamba](Markov-Mamba) contains the full Mamba model for binary k-order Markov chains.
-   [Markov-Mamba-States](Markov-Mamba-States) contains the full Mamba model finite-state k-order Markov chains.
-   [Markov-Mamba-L1](Markov-Mamba-L1) contains the Mamba model with L1 normalization in place of softmax.
-   [Markov-Mamba-Switch](Markov-Mamba-Switch) contains the full Mamba model for switching Markov chains.
-   [Attention-Conv](Attention-Conv) contains a transformer model with convolution for the Q, K and V matrices.

The script to run the experiments is in src/main.py.

The Mamba code is based on the original [Mamba-2 implementation](https://github.com/state-spaces/mamba), Copyright (c) 2023 Tri Dao, Albert Gu.
The Transformer code is based on the [NanoGPT implementation](https://github.com/karpathy/nanoGPT), Copyright (c) 2022 Andrej Karpathy.