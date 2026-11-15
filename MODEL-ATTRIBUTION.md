# Model attribution

Vedaris Advisor was developed and integrated by Ved Prajapati. The pretrained base used for this case study is [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), published by Qwen under Apache-2.0. The CUDA training configuration pins revision `cdbee75f17c01a7cc42f958dc650907174af0554`.

The macOS integration uses the [MLX community 4-bit conversion](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit) at revision `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`. The domain adaptation run trains separate low-rank adapter tensors over the frozen pretrained base. The case study documents that adaptation, the checkpoint transfer, and the native inference integration.
