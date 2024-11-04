# Extract (Transform), Train, Eval, and Predict (Inference)

Completed package of Transform, Train, Eval, Deploy loop

## Installation

```sh
## pip
pip install -r requirements.txt

## conda
conda create --name entropy --file entropy-env.txt
conda activate entropy
```


## Usage

Overview: videos -> covers -> design matrix -> train -> eval -> predict

1. place videos to extract in folder `1-input_videos`
2. run `python extract.py`
3. the extracted covers are placed in `2-covers`
    - The resulting CSVs will have three columns, namely Video Path, Frame Number, Frame size (in bytes)
4. run `python collate.py` to collate covers into a single well formatted design matrix, saved in `3_model_input_data`
5. run `python train.py`, checkpoints saved in `4_checkpoints`
6. run `python eval.py` to evaluate, be sure to change `CHECKPOINT_PATH` before running
7. run `python predict.py` to predict on new dataset, which needs to be in a design matrix. Change `MODEL_PATH` and `data_path` accordingly. 




## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated. 

Please to fork and create a pull request!


## Contact

Felix Gan -  reachfelixgan@gmail.com


## License

Distributed under the [MIT License](https://opensource.org/licenses/MIT).


