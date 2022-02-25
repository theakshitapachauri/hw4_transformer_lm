# Homework 4. Transformer language model

This homework has two parts.

1. In the first part of the homework, you will implement the key concept of Transformer: multi-head attention.

1. In the second part, you need to
   1. Use your multi-head attention implementation and `torch.nn` layers to implement Transformer Encoder,
   2. Implement training script `train.py` and train a language model.

## Setting up the environment

Feel free to use Jupyter Lab for the first part of the homework, but we stongly recommend to use a code editor like VSCode or PyCharm for the second part, as it involves more interaction with `.py` files. Here is a good tutorial on how to [setup VSCode for Python](https://www.youtube.com/watch?v=Z3i04RoI9Fk). Both of them also support jupyter notebooks, you just need to specify which jupyter kernel you want to use (most probably its `nlp_class`). For VSCode you may want to additionally install a [markdown extention](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one) to render files like this README.md.

You will developing the package that is located inside `transformer_lm`. In order to be able to import this package from the notebooks, training script, and anywhere else, you need to 

1. If you are working in a new environment (for example, a server), you need to create a new conda environment (`conda create -n nlp_class python=3.7`).
2. Activate your python environment (e.g., `conda activate nlp_class`).
3. Go to the homework directory that contains `setup.py` file (the same directory this `README.md` is in).
4. Install the package using the command `pip install -e .`. It should download all of the dependencies and install your module.
5. If you are on a GPU machine, you need to install a GPU version of pytorch. To do that, first check what CUDA version your server has with `nvidia-smi`.
   1. If your CUDA version is below 10.2, don't use this server
   2. If your CUDA version is below 11, run `pip install torch`
   3. Else, `pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
   4. Check that pytorch-GPU works via `python -c "import torch; print(torch.cuda.is_available())"`. If it returns False, reinstall pytorch via one of the above command (usually this helps), if it doesn't help, describe your problem in `#discussion`.
   5. If you are using 30XX, A100 or A6000 GPU, you have to use CUDA 11.3 and above. 

**Explaining pip install -e . command**:
`pip` is the python package manager. `install` is the pip command to install the module. `-e` is the argument that says to install the module in *editable* mode which allows you to edit the package and import from it right away without the need to reinstall it. The final argument is `.` which says "the package is located in the current directory".

## Connecting to DAN 417 machines

You should have a UML CS account to do that (it's not the same as your UML account). If you don't have it for any reason, please [get one ASAP](https://www.uml.edu/Sciences/computer-science/CS-Infrastructure/Request-an-account.aspx).

If you are using the eduroam wifi, you can directly connect to the machine using command `ssh YOURCSUSERNAME@cs.uml.edu` which connects you to CS network, and then `ssh dan417-XX.uml.edu` where XX could be any number from 01 to 16. For example `ssh dan417-02.uml.edu`.
If you are not in the eduroam network (say, at home), you should first activate [Umass Lowell VPN](https://www.uml.edu/it/services/get-connected/remote-access/) and then you should be able to connect to cs and dan417 as usual.

> If you are unfamiliar with SSH, look up this [tutorial](https://www.digitalocean.com/community/tutorials/how-to-use-ssh-to-connect-to-a-remote-server) and if you are using Windows, [this one (Windows 10 native SSH)](https://www.howtogeek.com/336775/how-to-enable-and-use-windows-10s-built-in-ssh-commands/) or [this one (PUTTY)](https://mediatemple.net/community/products/dv/204404604/using-ssh-in-putty-(windows)).

**Check that nobody is using the GPU on that machine.** Use `nvidia-smi` command to check if GPU has free RAM. By free we mean >90% of it is unused. If this one is occupied, its better to use a differnet machine. The number of servers is limited, so please do not postpone training until the last day, when every one of them will be occupied.

## Connecting to Google Cloud

If you are unfamiliar with cloud, we recommend you to stick to DAN 417 for now. If you want to use GCP for this homework, ask Vlad for instructions and a $50 coupon that you can spend for this class. It is enough for about 6 hours of Nvidia V100 time. Please don't spend all of the money for a single homework. We have additional coupons, but the supply is limited.

## How To Keep Running Commands After SSH Session Disconnection

It takes hours to run language model training on a GPU. However, if you would just run `python cli/train.py <blahblah>` in your SSH session and then disconnect, your script will automatically be shutdown by the system. To avoid this, you can use [tmux sessions](https://leimao.github.io/blog/Tmux-Tutorial/), you can also use `screen`, if you are familiar with it, but **do not** use nohup as it is not flexible enough for our purposes.


### How to sync your laptop code and server code

We strongly recommend to use Github and Git for that. These are essential professional instuments and you will need to learn them at some point anyway. Create a **private** repository for your homework (your grade will be lowered, if the repository is public). And use git commands to synchronize your code. You will mostly only need these ones: `git commit`, `git push`, `git clone`, and `git pull` .

> If you are unfamiliar with Git and Github: [read this tutorial](https://docs.github.com/en/get-started/using-git/about-git).

> Git and GitHub are extremely popular. If you see an error — google it! You will find the answer way quicker than contacting a TA.

If you have troubles with understanding git, you can use `scp` command to sync your code, but **it can cause loosing your changes** if you are not careful. We advise **not** to use `scp` to sync your code, but only to use it for large file upload/download.

## Part 1: Implementing Multi-Head Attention

Start with `notebooks/part1_multi_head_atention.ipynb`. It contains guides and instructions to implementing multi-head attention. Your implementation should pass all of the tests.

There are 5 coding tasks and 5 iniline questions in total. Task 1.2 is probably the hardest.
If you have trouble answering inline questions, feel free to search for them, but don't copy your peer answers.

## Part 2: Training a Transformer Language Model

This part is way bigger than Part 1, and we recommend to start working on it **as early as you can**. Preferrably, before the Part 1 deadline.

There are 8 coding tasks in `modeling_transformer.py`, 1 task in `cli/create_tokenizer.py`, and 5 tasks in `cli/train.py`. The hardest one is probably 4.4. There are also 2 inline questions in `cli/train.py` which require you to read and understand a piece of code.

In this part, you will need to implement the rest of the Transformer architecture, including `TransformerEncoderLayer` and `TransformerEncoder`. Detailed instructions are provided in `notebooks/part2_transformer.ipynb`.

After implementing Transformer, its time to train a language model. We will be using a small wikipedia corpus provided to us via 🤗 Datasets. You can read more about it [here](https://huggingface.co/datasets/wikitext). It is a medium-sized 0.5Gb dataset containing the texts of Good and Featured articles on Wikipedia in English. If you want to use a different dataset, maybe in a differnet langauge, ask TA for recommendation and approval.

The next thing you need is a tokenizer. We will be using a BPE tokenizer which you need to train on our dataset to learn the subword vocabulary. 🤗 Tokenizers provides you with tools to do this in a couple of lines of code. The script `cli/create_tokenizer.py` contains instructions how to train the tokenizer.

After finishing the tasks in the script, you can use it to train the tokenizer and save it to a file. Use `--vocab_size` of 8192 and save your tokenizer to `output_dir`.

Then, you need to open `cli/train.py` and complete tasks there. It contans several coding tasks and 2 inline questions. Please remember to answer them.

> If you are unsure how to use any of the scripts, you can call `python path/to/script.py --help` to see all of the arguments.

Finally, you need to train your language model. We would recommend to go with a small or base-sized model. Here are the configs:

```
# Small model
n_layers = 6
hidden = 512
ffn_hidden = 2048
n_heads = 8
max_length = 128

# Base model
n_layers = 12
hidden = 768
ffn_hidden = 3072
n_heads = 12
max_length = 256
```

Feel free to change these hyperparameters, but remember that:
* It takes hours to train a single model using a GPU.
  * Meaning you need to start training at least a day or two before the deadline
  * Meaning you won't be able to play with hyperparameters a lot
  * Meaning you can't use your laptop (Core i9 laptop needs more than 100hours to train the model)
  * Meaning you need to learn how to work with a GPU server.
* Smaller models are faster to train until convergence, but larger models sometimes can reach the same loss in fewer steps and less time.
* Try to maximize your batch size and fill all of the GPU memory.
  * Batch size should be at least 8 and preferrably around 64 or even more.
  * If you see an out-of-memory error, probably your batch size or max_length or your network parameters are too big. Do not reduce max_length beyond 128.
  * To reduce memory consumption, you can use Adafactor optimizer instead of Adam. Here is the [documentation on how to use it](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.Adafactor)
* Keep all of your shapes and sizes divisible by 8, this makes the GPUs a bit more efficient.
* You can use an empirical [scaling law](https://arxiv.org/abs/2001.08361) to estimate your learning rate `lr = 0.003239 - 0.0001395 log(N)`. Here, N is the number of parameters of your model, excluding embeddings (should be around 1-100M if you are using somethins like small or base transformer).
* Your final model will be evaulated based on it's validation loss. Undertrained or weak models with validatoin perplexity significnatly above 100 might loose points.

Finally, run `train.py` and provide your selected hyperparameters to it. Save your model to `output_dir`.

**Monitor your model performance while it is training** in WadnB. This is literally the main purpose of this tool. If the model is not improving at all or is diverging, look up our "my model does not converge" checklist from Lecture 3. At the same time, if you get a very high test accuracy, your model might be cheating and your causal masking or data preprocessing is not implemented correctly. To help you understand what correct training loss plot and eval perplixity/accuracy should look like, we will post a couple of images in Slack. Your runs will most probaly look different, because of different hyperparemeters, but they should not be extremely different.

> To interrupt the script, press CTRL+C

> You can download your model and tokenizer from the server to your laptop using `scp` command line tool ([how to use it](https://linuxize.com/post/how-to-use-scp-command-to-securely-transfer-files/)).

## Try out your model for text generation

After training, use `notebooks/interact.ipynb` to generate text using your language model.

## Submitting this homework

Submission instructions are the same for Part 1 and Part 2.

> NOTE: Do not add `model.pt` and other large files to your git repository.

1. Restart your `.ipynb` notebooks (part 1, part 2, and interact) and reexecute them top-to-bottom via "Restart and run all" button.
Not executed notebooks or the notebooks with the cells not executed in order will receive 0 points.
1. **If you are using GitHub**, add `github.com/guitaricet` and `github.com/NamrataRShivagunde` to your [repository collaborators](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-github-user-account/managing-access-to-your-personal-repositories/inviting-collaborators-to-a-personal-repository), so we could access your code. Push your last changes (inclusing the reexecuted notebooks) and subimt the link to your GitHub repository to the Blackboard.
2. **If you are not using GitHub**, delete `output_dir` (or move its contents somewhere else if you want to reuse them later) and `wandb` diretories. Zip this directory and submit the archive it to the Blackaord.
3. **Part 2 only:** Submit a link to your best wandb run to the Blackboard too. You will be evaluated based on the perplexity of your model. Make sure your wandb project is public.
