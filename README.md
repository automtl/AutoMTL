# AutoMTL

Open-source **anonymous** code for paper *Automatic Multi-Task Learning Framework with Neural
Architecture Search in Recommendations*

## Configurations

Download corresponding datasets from

- IJCAI-2015: https://tianchi.aliyun.com/dataset/dataDetail?dataId=472
- UserBehavior-2017: https://tianchi.aliyun.com/dataset/dataDetail?dataId=6493
- KuaiRand-Pure: https://kuairand.com/
- QB-Video: https://static.qblv.qq.com/qblv/h5/algo-frontend/tenrec_dataset.html
- AliCCP: https://tianchi.aliyun.com/dataset/408

Preprocess the datasets by code or jupyter notbooks in `src/datasets/preprocesses/`.

Edit configuration files according to these in `configs/`.

## Architecture Search

Search with AutoMTL: 
```bash
python src/run_nas.py --dataset_name UserBehavior --device_ids=0
```

It will return the training and test performance of searched architecture.

## Valid Searched Architectures

- Set the architecture file path and checkpoint path in `src/test_nas.py`.
- Run script: `src/test_nas.py --dataset_name UserBehavior --device_ids=0`


## License

The codes and models in this repo are released under the GNU GPLv3 license.