# DP_BACH
This is the codebase of Tensorflow-Version Deep Neural Networks Implementations for the paper cited in below:

[Gu, Q., Prodduturi, N., Hart, S., Meroueh, C., Flotte, T. Deep Learning in Automated Breast Cancer Diagnosis by Learning the Breast Histology from Microscopy Images. J Path Info](https://www.jpathinformatics.org/article.asp?issn=2153-3539;year=2021;volume=12;issue=1;spage=44;epage=44;aulast=;t=6)

The structure of the codebase
```
|- README.md
|-clam_tf_module
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
```
