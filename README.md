Building encoder-decoder transformer based on [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper, for educatinal purposes. [BPE](https://en.wikipedia.org/wiki/Byte-pair_encoding) algorithm is used for tokenization. [OPUS](https://opus.nlpl.eu/) english to russian translation dataset is used to train the model. Heavily inspired by [Andrej Karpathy](https://github.com/karpathy) [Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) video course.


<img width="776" height="711" alt="image" src="https://github.com/user-attachments/assets/37baff2b-edf3-4041-8988-bae90971fd1e" />

After 100_000 iterations with 10 sentences batch size on single Nvidia RTX 4000 GPU it can translate some basic phrases that were not originally in the training dataset.

<img width="1108" height="623" alt="image" src="https://github.com/user-attachments/assets/c2a57ef2-e2fc-4bce-84b1-5c771e76d4f8" />
