# Diffusion Language Models from Scratch

## Motivation
This is a WIP with the goal of exploring Diffusion Language Models. Inspired by the work in [LLaDA](https://github.com/ML-GSAI/LLaDA/).

## Features
This contains a lighter (in terms of LoC) LLaDA model and an adapted AR training loop using the instructions for pre-training in [LLaDA](https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md#pre-training).

## A first run

### Model
The model that was trained is a bidirectionnal Transformer, with vanilla multi head attention (no bias), uses RoPE for positionnal embeddings and RMS Layer norm.
- Model config
    - number of heads: 16
    - hidden dim: 512
    - number of layers: 4
    - number of heads: 16 
    - max sequence length: 256

### Data
The dataset used is [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

### A few examples
These generations were done using semi-autoregressive sampling with random remasking using an early checkpoint.

--- 

**Setup:** 256 steps, max gen length of 256 and a block length of 256

**Prompt:**   

```
Oh wow, it's Judy!
``` 

**Generated:** 

```
Oh wow, it's Judy! 
The girl smiled and said goodbye to her. She was happy and she decided to share the ball and play with it. 
The girl and the girl took the little girl and ran back to the park. She was so happy that she had played in the park.
``` 


**Prompt** 

```
I love Judy, 
```

**Generated:** 

```
I love Judy, but I want to play?" Ben asks.
"Yes, let's play too," Ben says. "Yes, Ben, we can slide the park," Ben says.
They play together in the park. They are very happy. They like to slide the slide. They run to the slide and slide the slide. They are very happy and have fun. They love the park every day.
.
Once upon a time, in a big small house, there was a little cat. The cat had many friends in the park. She was happy and a little cat. She loved to play with her friends. One day, she saw a little cat named Lily went to the park. In the park, she saw a big cat under a big tree. The cat was big and soft. Lily liked to play with the cat.
One day, the cat saw a cat. The cat wanted to catch it. The cat wanted to play with it.
But then, she saw a little cat on the cat. She was very sad and wanted to play with it. The cat saw the cat. It tried the cat, but the cat was too small. The cat was sad. the cat and the cat became friends. They and the cat played together.
```

---

**Setup:** 256 steps, max gen length of 256 and a block length of 128

**Prompt:** 

```
Oh wow, it's Judy!
```

**Generated:** 

```
Oh wow, it's Judy! That is not a ball!" Lily said.
Lily. Lily wanted to play. They played with the ball and the ball together. They laughed and laughed and played with the ball.
They played with the ball. They had fun day at the park. They were happy and proud anymore. Lily and Lily became friends. They had lots of fun together, and the ball. They had a great day.
```

**Prompt:** 

```
I love Judy, 
```

**Generated:** 

```
I love Judy, Tim!" Tim said. Tim said, "Yes, I can. Tim and the dog played together.
One day, Tim saw a big ball. The dog saw the ball and wanted to catch it. Then, he saw a big dog. It was big and shiny. In the end, Tim and the dog ran away, and the dog was very happy. Tim was very happy and played with his ball.
```
