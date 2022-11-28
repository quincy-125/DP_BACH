import pandas as pd
import sklearn
from sklearn import metrics
import os
import random
import time

from UTILITY.util import get_data_from_tf, s_clam_call, most_frequent, m_clam_call


def m_test_per_sample(
    n_class,
    top_k_percent,
    att_gate,
    att_only,
    m_clam_op,
    mil_ins,
    mut_ex,
    c_model,
    dim_compress_features,
    img_features,
    slide_label,
    n_test_steps,
):

    slide_pred_per_sample = list()

    for i in range(n_test_steps):
        if m_clam_op:
            (
                att_score,
                A,
                h,
                ins_labels,
                ins_logits_unnorm,
                ins_logits,
                slide_score_unnorm,
                Y_prob,
                Y_hat,
                Y_true,
                predict_label,
            ) = m_clam_call(
                att_net=c_model[0],
                ins_net=c_model[1],
                bag_net=c_model[2],
                img_features=img_features,
                slide_label=slide_label,
                n_class=n_class,
                dim_compress_features=dim_compress_features,
                top_k_percent=top_k_percent,
                att_gate=att_gate,
                att_only=att_only,
                mil_ins=mil_ins,
                mut_ex=mut_ex,
            )
        else:
            (
                att_score,
                A,
                h,
                ins_labels,
                ins_logits_unnorm,
                ins_logits,
                slide_score_unnorm,
                Y_prob,
                Y_hat,
                Y_true,
                predict_label,
            ) = s_clam_call(
                att_net=c_model[0],
                ins_net=c_model[1],
                bag_net=c_model[2],
                img_features=img_features,
                slide_label=slide_label,
                n_class=n_class,
                top_k_percent=top_k_percent,
                att_gate=att_gate,
                att_only=att_only,
                mil_ins=mil_ins,
                mut_ex=mut_ex,
            )

        slide_pred_per_sample.append(predict_label)
        predict_slide_label = most_frequent(slide_pred_per_sample)

        return predict_slide_label


def test_step(
    n_class,
    top_k_percent,
    att_gate,
    att_only,
    mil_ins,
    mut_ex,
    m_clam_op,
    imf_norm_op,
    c_model,
    dim_compress_features,
    test_path,
    result_path,
    result_file_name,
    n_test_steps,
):

    start_time = time.time()

    slide_true_label = list()
    slide_predict_label = list()
    sample_names = list()

    test_sample_list = os.listdir(test_path)
    test_sample_list = random.sample(test_sample_list, len(test_sample_list))

    for i in test_sample_list:
        print(">", end="")
        single_test_data = test_path + i
        img_features, slide_label = get_data_from_tf(
            single_test_data, imf_norm_op=imf_norm_op
        )

        predict_slide_label = m_test_per_sample(
            n_class=n_class,
            top_k_percent=top_k_percent,
            att_gate=att_gate,
            att_only=att_only,
            m_clam_op=m_clam_op,
            mil_ins=mil_ins,
            mut_ex=mut_ex,
            c_model=c_model,
            dim_compress_features=dim_compress_features,
            img_features=img_features,
            slide_label=slide_label,
            n_test_steps=n_test_steps,
        )

        slide_true_label.append(slide_label)
        slide_predict_label.append(predict_slide_label)
        sample_names.append(i)

        test_results = pd.DataFrame(
            list(zip(sample_names, slide_true_label, slide_predict_label)),
            columns=["Sample Names", "Slide True Label", "Slide Predict Label"],
        )
        test_results.to_csv(
            os.path.join(result_path, result_file_name), sep="\t", index=False
        )

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(
        slide_true_label, slide_predict_label
    ).ravel()
    test_tn = int(tn)
    test_fp = int(fp)
    test_fn = int(fn)
    test_tp = int(tp)

    test_sensitivity = round(test_tp / (test_tp + test_fn), 2)
    test_specificity = round(test_tn / (test_tn + test_fp), 2)
    test_acc = round((test_tp + test_tn) / (test_tn + test_fp + test_fn + test_tp), 2)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        slide_true_label, slide_predict_label, pos_label=1
    )
    test_auc = round(sklearn.metrics.auc(fpr, tpr), 2)

    test_run_time = time.time() - start_time

    template = "\n Test Accuracy: {}, Test Sensitivity: {}, Test Specificity: {}, Test Running Time: {}"
    print(
        template.format(
            f"{float(test_acc):.4%}",
            f"{float(test_sensitivity):.4%}",
            f"{float(test_specificity):.4%}",
            "--- %s mins ---" % int(test_run_time / 60),
        )
    )
