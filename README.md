# AutoMTL

The codes for paper *AutoMTL: Automatic Multi-Task Learning Framework with Neural Architecture Search for Recommender Systems*

## Configurations

Download corresponding datasets from

- IJCAI-2015: https://tianchi.aliyun.com/dataset/dataDetail?dataId=472
- UserBehavior-2017: https://tianchi.aliyun.com/dataset/dataDetail?dataId=6493
- AliExpress: https://tianchi.aliyun.com/dataset/dataDetail?dataId=74690

Preprocess the datasets by jupyter notbooks in `preprocesses/`.

Edit configuration files according to these in `tests/`.

## Architecture Search

- Set the search space in `src/supernets/search_space_config.py`.
- Use `sh scripts/search.sh` to perform differentiable architecture search.

## Retrain

Use `sh scripts/retrain.sh` to retrain the searched architectures, remember to config the architecture file path.

## License

The codes and models in this repo are released under the GNU GPLv3 license.