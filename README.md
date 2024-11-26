# LCF-ATEPC 
## Requirement

* Python >= 3.7
* PyTorch >= 1.0
* transformers >= 4.5.1
* Set `use_bert_spc = True` to improve the APC performance while only APC is considered.

```sh
pip install -r requirements.txt
```

## Training
We use the configuration file to manage experiments setting.

Training in batches by experiments configuration file, refer to the [experiments.json](experiments.json) to manage experiments.

Then, 
```sh
python train.py --config_path experiments.json
```


