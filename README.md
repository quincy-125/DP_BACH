# DigiPath_CLAM_TF
This is the codebase of Tensorflow-Version Attention-based Multi Instance Learning for WSI Classification Framework. This work is original proposed by [Mahmood Lab from Harvard University.](https://faisal.ai/?fireglass_rsn=true#fireglass_params&tabid=a00145eb29ce3894&start_with_session_counter=2&application_server_address=mc1.prod.fire.glass) Their original work was under the following citation section

Our group re-implemented their original Pytorch-Version codebase for breast cancer normal versus tumor binary classification project, our work is also under the following citation section

Our re-implementation comes with the following contributions:

1. Unabled a [Tenforflow-Version](https://www.tensorflow.org/) of the codebase for end-users who is devising, and deploying AI models on Tensorflow Framework
2. Accpeting .tif microscopy image format as input to train the model, which is not accepted by the original [Pytoch-Version](https://pytorch.org/) of the codebase

To use this Code, you have to cite the following documentation


1. [Gu, Q., Prodduturi, N., Hart, S., Meroueh, C., Flotte, T. Deep Learning in Automated Breast Cancer Diagnosis by Learning the Breast Histology from Microscopy Images. J Path Info](https://www.jpathinformatics.org/article.asp?issn=2153-3539;year=2021;volume=12;issue=1;spage=44;epage=44;aulast=;t=6)
2. [Lu, MY., Williamson, DFK., Chen, TY., Chen, RJ., Barbieri, M., Mahmood, F. Data-efficient and weakly supervised computational pathology on whole-slide images. Nature Biomedical Engineering](https://www.nature.com/articles/s41551-020-00682-w) 


The structure of the codebase
```
|- README.md
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
    |- extract_patches.ipynb
    |- extract_patch_features.ipynb
    |- model.ipynb
    |- model_train_eval.ipynb
    |- main.ipynb
    |- util.ipynb
    |- bach_plot.ipynb
|- logs
    |- data.out
    |- train.out
    |- eval.out
```
