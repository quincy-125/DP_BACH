# DP_BACH
This is the codebase of Tensorflow-Version Deep Neural Networks Implementations for the paper cited in below:

[Gu, Q., Prodduturi, N., Hart, S. Deep Learning in Automating Breast Cancer Diagnosis from Microscopy Images. MedRxiv. 2023. (https://www.medrxiv.org/content/10.1101/2023.06.15.23291437v1)

The structure of the codebase
```
|- README.md
|- clam_tf_module
    |- requirements.txt
    |- configs
        |- data_config.yaml
        |- train_config.yaml
        |- eval_config.yaml
    |- data_module
        |- components
            |- cross_val_split.py
            |- patch_extract.py
            |- patch_resnet_feature.py
            |- utils.py
        |- main.py
    |- training_module
        |- components
            |- model_attention.py
            |- model_ins_classifier.py
            |- model_bag_classifier.py
            |- model_clam.py
            |- model_train.py
            |- model_val.py
            |- model_test.py
            |- model_main.py
            |- util.py
    |- main.py
    |- run.sh
    |- notebooks
        |- data_prep.ipynb
        |- bach_plot.ipynb
    |- logs
        |- data.out
        |- train.out
        |- eval.out
|- cnn_module
|- oneshort_module
```
