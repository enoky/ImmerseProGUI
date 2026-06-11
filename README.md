<div align="center">

<div class="logo">
   <a>
      <img src="misc/imp-logo.png" style="width:180px;">
   </a>
</div>

<h1>ImmersePro: End-to-End Stereo Video Synthesis Via Implicit Disparity Learning</h1>

<div>
    <a href='#' target='_blank'>Jian Shi</a>&emsp;
    <a href='#' target='_blank'>ZhenYu Li</a>&emsp;
    <a href='#' target='_blank'>Peter Wonka</a>&emsp;
</div>
<div>
    KAUST&emsp; 
</div>


<div>
    <h4 align="center">
        <a href="https://shijianjian.github.io/ImmersePro" target='_blank'>
        <img src="https://img.shields.io/badge/🐳-Project%20Page-blue">
        </a>
        <a href="https://arxiv.org/abs/2410.00262" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2410.00262-b31b1b.svg">
        </a>
        <a href="https://youtu.be/Lhu0hHsDvao" target='_blank'>
        <img src="https://img.shields.io/badge/Demo%20Video-%23FF0000.svg?logo=YouTube&logoColor=white">
        </a>
    </h4>
</div>

⭐ If you like ImmersePro or if it is helpful to your projects, please help star this repo. Thanks! 🤗


</div>



<p align="center">
  <img src="./misc/jumbotron.png" />
</p>

## Updates

- [11th Jun] We added DepthAnything-based models and weights.

## Running

1. Download the checkpoints from [here](https://huggingface.co/shijianjian/ImmersePro).
    ```
    $ git clone https://huggingface.co/shijianjian/ImmersePro
    $ mv ImmersePro experiments_model
    ```

2. Then construct the project folder as:
    ```
    ├── experiments_model
    │   └── immersepro_model_da_inference_da
    │       ├── dis_035000.pth
    │       ├── gen_035000.pth
    │       └── opt_035000.pth
    │   └── immersepro_model_inference
    │       ├── dis_030000.pth
    │       ├── gen_030000.pth
    │       └── opt_030000.pth
    ├── inference_video.py
    ├── model
    ...
    ```
3. Run with provided videos with DepthAnything-based model -> `python inference_video.py -c configs/inference_da.json` and MiDAS-based model -> `python inference_video.py -c configs/inference.json`.


## Demo

If you have VR devices already, you may try out the converted videos on the following link:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/Lhu0hHsDvao/0.jpg)](https://www.youtube.com/watch?v=Lhu0hHsDvao)