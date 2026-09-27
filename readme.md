# Module 0: Programming Best Practices

## What I learned

**Git basics**
- `git clone` to get the repo, `git commit` to record changes, `git push` to upload them.
- I created `readme.md` with "hello world", committed it, and later changed it to "aloha" on `main`.

**Branches**
- I created a branch `for_fun` and made a change on it, so `main` and `for_fun` diverged.

**Merging and conflicts**
- Merging `for_fun` into `main` produced a conflict in `readme.md`, because both branches had
  edited the same line. I resolved it by keeping `aloha` and deleting the conflict markers
  (`<<<<<<<`, `=======`, `>>>>>>>`), then committed the merge.
- It's a little bit difficult to understand this, because I  could not merge for_fun into main at first due to the commit of inference.py in the main branch. But I successfully solved it by adding extra modifications to the branch "for_fun".

**HEAD and reflog**
- `git checkout <commit>` moves HEAD to an older commit (detached HEAD) without creating a branch.
- The reflog records every movement of HEAD, so such a checkout can be verified afterwards.

**HuggingFace / ResNet inference**
- I set up a virtual environment and installed torch, transformers and datasets.
- I loaded the pretrained ResNet-50 from HuggingFace and ran inference on MNIST.
- MNIST images are 28x28 grayscale, so I converted them to RGB and resized them to 224x224
  to match the input size of the model.
- The accuracy was 0.0000: the model was pretrained on ImageNet (1000 object classes) and was
  never finetuned on MNIST.
- Datasets and model weights must stay out of the repo.