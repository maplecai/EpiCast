import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from gosai_io import load_pred_df
from gosai_masks import get_mask


def safe_metric(metric_fn, x, y):
    valid = ~(pd.isna(x) | pd.isna(y))
    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return np.nan

    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan

    return metric_fn(x, y)


def compute_heldout_cell_correlation(
    true_df,
    pred_df,
    masks,
    target_cell_type,
    split,
    metric_fn,
    metric_name,
):
    eval_mask = get_mask(split, masks, cell_type=target_cell_type)

    x = true_df.loc[eval_mask, f"{target_cell_type}_true"]
    y = pred_df.loc[eval_mask, f"{target_cell_type}_pred"]
    value = safe_metric(metric_fn, x, y)

    return pd.Series(
        {
            "split": split,
            "cell_type": target_cell_type,
            "metric": metric_name,
            "n_eval": int(eval_mask.sum()),
            "value": value,
        }
    )


def summarize_metric_rows(model_name, split, metric_name, split_df):
    return pd.Series(
        {
            "model": model_name,
            "split": split,
            "cell_type": "loo_mean",
            "metric": metric_name,
            "n_eval": split_df["n_eval"].mean(),
            "value": split_df["value"].mean(skipna=True),
        }
    )


def evaluate_full_model(
    model_name,
    pred_df,
    true_df,
    cell_types,
    masks,
    split,
    metric_fn_map,
):
    all_rows = []

    for metric_name, metric_fn in metric_fn_map.items():
        split_rows = []

        for cell_type in cell_types:
            row = compute_heldout_cell_correlation(
                true_df=true_df,
                pred_df=pred_df,
                masks=masks,
                target_cell_type=cell_type,
                split=split,
                metric_fn=metric_fn,
                metric_name=metric_name,
            )
            row["model"] = model_name
            split_rows.append(row)
            all_rows.append(row)

        all_rows.append(summarize_metric_rows(model_name, split, metric_name, pd.DataFrame(split_rows)))

    cols = ["model", "split", "cell_type", "metric", "n_eval", "value"]
    return pd.DataFrame(all_rows)[cols]


def evaluate_one_model(
    model_name,
    pred_paths,
    true_df,
    cell_types,
    masks,
    split,
    metric_fn_map,
):
    all_rows = []

    for metric_name, metric_fn in metric_fn_map.items():
        split_rows = []

        for heldout_cell_type in cell_types:
            pred_path = pred_paths[heldout_cell_type]
            print("loading", model_name, heldout_cell_type, pred_path)
            pred_df = load_pred_df(pred_path, cell_types)

            row = compute_heldout_cell_correlation(
                true_df=true_df,
                pred_df=pred_df,
                masks=masks,
                target_cell_type=heldout_cell_type,
                split=split,
                metric_fn=metric_fn,
                metric_name=metric_name,
            )
            row["model"] = model_name
            split_rows.append(row)
            all_rows.append(row)

        all_rows.append(summarize_metric_rows(model_name, split, metric_name, pd.DataFrame(split_rows)))

    cols = ["model", "split", "cell_type", "metric", "n_eval", "value"]
    result_df = pd.DataFrame(all_rows)[cols]
    print(result_df[result_df["cell_type"] == "loo_mean"])

    return result_df


def precision_recall_at_k(scores, labels, k):
    n_pos = int(labels.sum())

    if n_pos == 0:
        return np.nan, np.nan

    k = min(k, len(scores))
    order = np.argsort(-scores)
    topk = order[:k]
    pos_in_topk = int(labels[topk].sum())

    return pos_in_topk / k, pos_in_topk / n_pos


def retrieval_metrics(scores, labels, k_list):
    out = {}

    if int(labels.sum()) == 0:
        out["ap"] = np.nan
        for k in k_list:
            out[f"p@{k}"] = np.nan
            out[f"r@{k}"] = np.nan
        return out

    out["ap"] = average_precision_score(labels, scores)

    for k in k_list:
        p_k, r_k = precision_recall_at_k(scores, labels, k)
        out[f"p@{k}"] = p_k
        out[f"r@{k}"] = r_k

    return out


def get_specific_scores(pred_df, target_cell_type, cell_types, mode="max"):
    other_cell_types = [ct for ct in cell_types if ct != target_cell_type]
    target_scores = pred_df[f"{target_cell_type}_pred"].to_numpy(dtype=float)
    other_scores = pred_df[[f"{ct}_pred" for ct in other_cell_types]]

    if mode == "max":
        background_scores = other_scores.max(axis=1).to_numpy(dtype=float)
    else:
        background_scores = other_scores.mean(axis=1).to_numpy(dtype=float)

    return target_scores - background_scores


def compute_heldout_cell_retrieval(
    pred_df,
    masks,
    cell_types,
    target_cell_type,
    split,
    k_list,
    specific_score_mode="max",
):
    eval_mask = get_mask(split, masks)
    idx = np.where(eval_mask)[0]
    scores_all = get_specific_scores(pred_df, target_cell_type, cell_types, mode=specific_score_mode)
    labels_all = get_mask("specific", masks, cell_type=target_cell_type)
    scores = scores_all[idx]
    labels = labels_all[idx]

    metrics = retrieval_metrics(scores, labels, k_list)
    metrics["cell_type"] = target_cell_type
    metrics["n_eval"] = len(idx)
    metrics["n_pos"] = int(labels.sum())

    return pd.Series(metrics)


def evaluate_loo_model_retrieval(
    model_name,
    pred_paths,
    cell_types,
    masks,
    split,
    k_list,
    specific_score_mode="max",
):
    all_rows = []
    split_rows = []

    for heldout_cell_type in cell_types:
        pred_path = pred_paths[heldout_cell_type]
        print("loading", model_name, heldout_cell_type, pred_path)
        pred_df = load_pred_df(pred_path, cell_types)

        row = compute_heldout_cell_retrieval(
            pred_df=pred_df,
            masks=masks,
            cell_types=cell_types,
            target_cell_type=heldout_cell_type,
            split=split,
            k_list=k_list,
            specific_score_mode=specific_score_mode,
        )
        row["model"] = model_name
        row["split"] = split
        split_rows.append(row)
        all_rows.append(row)

    split_df = pd.DataFrame(split_rows)
    metric_cols = ["ap"] + [f"p@{k}" for k in k_list] + [f"r@{k}" for k in k_list]
    summary = split_df[metric_cols].mean(axis=0, skipna=True)
    summary["model"] = model_name
    summary["split"] = split
    summary["cell_type"] = "loo_mean"
    summary["n_eval"] = split_df["n_eval"].mean()
    summary["n_pos"] = split_df["n_pos"].mean()
    all_rows.append(summary)

    cols = ["model", "split", "cell_type", "n_eval", "n_pos"] + metric_cols
    result_df = pd.DataFrame(all_rows)[cols]
    print(result_df[result_df["cell_type"] == "loo_mean"])

    return result_df


def safe_binary_metric(y_true_continuous, y_score, positive_threshold=0.0):
    valid = ~(pd.isna(y_true_continuous) | pd.isna(y_score))
    y_true_continuous = np.asarray(y_true_continuous[valid])
    y_score = np.asarray(y_score[valid])

    if len(y_true_continuous) == 0:
        return None

    y_true = (y_true_continuous > positive_threshold).astype(int)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)

    if n_pos == 0 or n_neg == 0:
        return None

    return {
        "n": len(y_true),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "positive_rate": n_pos / len(y_true),
        "auroc": roc_auc_score(y_true, y_score),
        "auprc": average_precision_score(y_true, y_score),
    }


def evaluate_binary_one_cell(
    model_name,
    true_df,
    pred_df,
    masks,
    target_cell_type,
    split,
    positive_threshold=0.0,
):
    eval_mask = get_mask(split, masks, cell_type=target_cell_type)
    y_true_continuous = true_df.loc[eval_mask, f"{target_cell_type}_true"]
    y_score = pred_df.loc[eval_mask, f"{target_cell_type}_pred"]

    result = safe_binary_metric(
        y_true_continuous=y_true_continuous,
        y_score=y_score,
        positive_threshold=positive_threshold,
    )

    if result is None:
        return pd.Series(
            {
                "model": model_name,
                "split": split,
                "cell_type": target_cell_type,
                "positive_threshold": positive_threshold,
                "n_eval": int(eval_mask.sum()),
                "n_pos": np.nan,
                "n_neg": np.nan,
                "positive_rate": np.nan,
                "auroc": np.nan,
                "auprc": np.nan,
            }
        )

    return pd.Series(
        {
            "model": model_name,
            "split": split,
            "cell_type": target_cell_type,
            "positive_threshold": positive_threshold,
            "n_eval": result["n"],
            "n_pos": result["n_pos"],
            "n_neg": result["n_neg"],
            "positive_rate": result["positive_rate"],
            "auroc": result["auroc"],
            "auprc": result["auprc"],
        }
    )


def summarize_binary(model_name, split, positive_threshold, result_df):
    return {
        "model": model_name,
        "split": split,
        "cell_type": "loo_mean",
        "positive_threshold": positive_threshold,
        "n_eval": result_df["n_eval"].mean(),
        "n_pos": result_df["n_pos"].mean(),
        "n_neg": result_df["n_neg"].mean(),
        "positive_rate": result_df["positive_rate"].mean(skipna=True),
        "auroc": result_df["auroc"].mean(skipna=True),
        "auprc": result_df["auprc"].mean(skipna=True),
    }


def evaluate_full_model_binary(
    model_name,
    pred_df,
    true_df,
    cell_types,
    masks,
    split,
    positive_threshold=0.0,
):
    rows = [
        evaluate_binary_one_cell(
            model_name=model_name,
            true_df=true_df,
            pred_df=pred_df,
            masks=masks,
            target_cell_type=cell_type,
            split=split,
            positive_threshold=positive_threshold,
        )
        for cell_type in cell_types
    ]
    result_df = pd.DataFrame(rows)
    summary = summarize_binary(model_name, split, positive_threshold, result_df)
    return pd.concat([result_df, pd.DataFrame([summary])], axis=0, ignore_index=True)


def evaluate_loo_model_binary(
    model_name,
    pred_paths,
    true_df,
    cell_types,
    masks,
    split,
    positive_threshold=0.0,
):
    rows = []

    for heldout_cell_type in cell_types:
        pred_path = pred_paths[heldout_cell_type]
        print("loading", model_name, heldout_cell_type, pred_path)
        pred_df = load_pred_df(pred_path, cell_types)
        rows.append(
            evaluate_binary_one_cell(
                model_name=model_name,
                true_df=true_df,
                pred_df=pred_df,
                masks=masks,
                target_cell_type=heldout_cell_type,
                split=split,
                positive_threshold=positive_threshold,
            )
        )

    result_df = pd.DataFrame(rows)
    summary = summarize_binary(model_name, split, positive_threshold, result_df)
    result_df = pd.concat([result_df, pd.DataFrame([summary])], axis=0, ignore_index=True)
    print(result_df[result_df["cell_type"] == "loo_mean"])

    return result_df


def summarize_test_rows(model_name, split, metric_name, split_df):
    return pd.Series(
        {
            "model": model_name,
            "split": split,
            "cell_type": "test_mean",
            "metric": metric_name,
            "n_eval": split_df["n_eval"].mean(),
            "value": split_df["value"].mean(skipna=True),
        }
    )


def evaluate_full_model_on_cells(
    model_name,
    pred_df,
    true_df,
    eval_cell_types,
    masks,
    split,
    metric_fn_map,
):
    all_rows = []

    for metric_name, metric_fn in metric_fn_map.items():
        split_rows = []

        for cell_type in eval_cell_types:
            row = compute_heldout_cell_correlation(
                true_df=true_df,
                pred_df=pred_df,
                masks=masks,
                target_cell_type=cell_type,
                split=split,
                metric_fn=metric_fn,
                metric_name=metric_name,
            )
            row["model"] = model_name
            split_rows.append(row)
            all_rows.append(row)

        all_rows.append(summarize_test_rows(model_name, split, metric_name, pd.DataFrame(split_rows)))

    cols = ["model", "split", "cell_type", "metric", "n_eval", "value"]
    return pd.DataFrame(all_rows)[cols]


def evaluate_single_pred_model(
    model_name,
    pred_path,
    true_df,
    all_cell_types,
    test_cell_types,
    masks,
    split,
    metric_fn_map,
):
    print("loading", model_name, pred_path)
    pred_df = load_pred_df(pred_path, all_cell_types)
    return evaluate_full_model_on_cells(
        model_name=model_name,
        pred_df=pred_df,
        true_df=true_df,
        eval_cell_types=test_cell_types,
        masks=masks,
        split=split,
        metric_fn_map=metric_fn_map,
    )


def evaluate_retrieval_single_pred(
    model_name,
    pred_df,
    all_cell_types,
    test_cell_types,
    masks,
    split,
    k_list,
    specific_score_mode="max",
):
    all_rows = []
    split_rows = []

    for target_cell_type in test_cell_types:
        row = compute_heldout_cell_retrieval(
            pred_df=pred_df,
            masks=masks,
            cell_types=all_cell_types,
            target_cell_type=target_cell_type,
            split=split,
            k_list=k_list,
            specific_score_mode=specific_score_mode,
        )
        row["model"] = model_name
        row["split"] = split
        split_rows.append(row)
        all_rows.append(row)

    split_df = pd.DataFrame(split_rows)
    metric_cols = ["ap"] + [f"p@{k}" for k in k_list] + [f"r@{k}" for k in k_list]
    summary = split_df[metric_cols].mean(axis=0, skipna=True)
    summary["model"] = model_name
    summary["split"] = split
    summary["cell_type"] = "test_mean"
    summary["n_eval"] = split_df["n_eval"].mean()
    summary["n_pos"] = split_df["n_pos"].mean()
    all_rows.append(summary)

    cols = ["model", "split", "cell_type", "n_eval", "n_pos"] + metric_cols
    result_df = pd.DataFrame(all_rows)[cols]
    print(result_df[result_df["cell_type"] == "test_mean"])

    return result_df


def summarize_binary_test(model_name, split, positive_threshold, result_df):
    return {
        "model": model_name,
        "split": split,
        "cell_type": "test_mean",
        "positive_threshold": positive_threshold,
        "n_eval": result_df["n_eval"].mean(),
        "n_pos": result_df["n_pos"].mean(),
        "n_neg": result_df["n_neg"].mean(),
        "positive_rate": result_df["positive_rate"].mean(skipna=True),
        "auroc": result_df["auroc"].mean(skipna=True),
        "auprc": result_df["auprc"].mean(skipna=True),
    }


def evaluate_full_model_binary_on_cells(
    model_name,
    pred_df,
    true_df,
    eval_cell_types,
    masks,
    split,
    positive_threshold=0.0,
):
    rows = [
        evaluate_binary_one_cell(
            model_name=model_name,
            true_df=true_df,
            pred_df=pred_df,
            masks=masks,
            target_cell_type=cell_type,
            split=split,
            positive_threshold=positive_threshold,
        )
        for cell_type in eval_cell_types
    ]
    result_df = pd.DataFrame(rows)
    summary = summarize_binary_test(model_name, split, positive_threshold, result_df)
    return pd.concat([result_df, pd.DataFrame([summary])], axis=0, ignore_index=True)


def evaluate_single_pred_model_binary(
    model_name,
    pred_path,
    true_df,
    all_cell_types,
    test_cell_types,
    masks,
    split,
    positive_threshold=0.0,
):
    print("loading", model_name, pred_path)
    pred_df = load_pred_df(pred_path, all_cell_types)
    result_df = evaluate_full_model_binary_on_cells(
        model_name=model_name,
        pred_df=pred_df,
        true_df=true_df,
        eval_cell_types=test_cell_types,
        masks=masks,
        split=split,
        positive_threshold=positive_threshold,
    )
    print(result_df[result_df["cell_type"] == "test_mean"])
    return result_df


def summarize_test_rows(model_name, split, metric_name, split_df):
    return pd.Series(
        {
            "model": model_name,
            "split": split,
            "cell_type": "test_mean",
            "metric": metric_name,
            "n_eval": split_df["n_eval"].mean(),
            "value": split_df["value"].mean(skipna=True),
        }
    )


def evaluate_full_model_on_cells(
    model_name,
    pred_df,
    true_df,
    eval_cell_types,
    masks,
    split,
    metric_fn_map,
):
    all_rows = []

    for metric_name, metric_fn in metric_fn_map.items():
        split_rows = []

        for cell_type in eval_cell_types:
            row = compute_heldout_cell_correlation(
                true_df=true_df,
                pred_df=pred_df,
                masks=masks,
                target_cell_type=cell_type,
                split=split,
                metric_fn=metric_fn,
                metric_name=metric_name,
            )
            row["model"] = model_name
            split_rows.append(row)
            all_rows.append(row)

        all_rows.append(summarize_test_rows(model_name, split, metric_name, pd.DataFrame(split_rows)))

    cols = ["model", "split", "cell_type", "metric", "n_eval", "value"]
    return pd.DataFrame(all_rows)[cols]


def evaluate_single_pred_model(
    model_name,
    pred_path,
    true_df,
    all_cell_types,
    test_cell_types,
    masks,
    split,
    metric_fn_map,
):
    print("loading", model_name, pred_path)
    pred_df = load_pred_df(pred_path, all_cell_types)
    return evaluate_full_model_on_cells(
        model_name=model_name,
        pred_df=pred_df,
        true_df=true_df,
        eval_cell_types=test_cell_types,
        masks=masks,
        split=split,
        metric_fn_map=metric_fn_map,
    )


def evaluate_retrieval_single_pred(
    model_name,
    pred_df,
    all_cell_types,
    test_cell_types,
    masks,
    split,
    k_list,
    specific_score_mode="max",
):
    all_rows = []
    split_rows = []

    for target_cell_type in test_cell_types:
        row = compute_heldout_cell_retrieval(
            pred_df=pred_df,
            masks=masks,
            cell_types=all_cell_types,
            target_cell_type=target_cell_type,
            split=split,
            k_list=k_list,
            specific_score_mode=specific_score_mode,
        )
        row["model"] = model_name
        row["split"] = split
        split_rows.append(row)
        all_rows.append(row)

    split_df = pd.DataFrame(split_rows)
    metric_cols = ["ap"] + [f"p@{k}" for k in k_list] + [f"r@{k}" for k in k_list]
    summary = split_df[metric_cols].mean(axis=0, skipna=True)
    summary["model"] = model_name
    summary["split"] = split
    summary["cell_type"] = "test_mean"
    summary["n_eval"] = split_df["n_eval"].mean()
    summary["n_pos"] = split_df["n_pos"].mean()
    all_rows.append(summary)

    cols = ["model", "split", "cell_type", "n_eval", "n_pos"] + metric_cols
    result_df = pd.DataFrame(all_rows)[cols]
    print(result_df[result_df["cell_type"] == "test_mean"])

    return result_df


def summarize_binary_test(model_name, split, positive_threshold, result_df):
    return {
        "model": model_name,
        "split": split,
        "cell_type": "test_mean",
        "positive_threshold": positive_threshold,
        "n_eval": result_df["n_eval"].mean(),
        "n_pos": result_df["n_pos"].mean(),
        "n_neg": result_df["n_neg"].mean(),
        "positive_rate": result_df["positive_rate"].mean(skipna=True),
        "auroc": result_df["auroc"].mean(skipna=True),
        "auprc": result_df["auprc"].mean(skipna=True),
    }


def evaluate_full_model_binary_on_cells(
    model_name,
    pred_df,
    true_df,
    eval_cell_types,
    masks,
    split,
    positive_threshold=0.0,
):
    rows = [
        evaluate_binary_one_cell(
            model_name=model_name,
            true_df=true_df,
            pred_df=pred_df,
            masks=masks,
            target_cell_type=cell_type,
            split=split,
            positive_threshold=positive_threshold,
        )
        for cell_type in eval_cell_types
    ]
    result_df = pd.DataFrame(rows)
    summary = summarize_binary_test(model_name, split, positive_threshold, result_df)
    return pd.concat([result_df, pd.DataFrame([summary])], axis=0, ignore_index=True)


def evaluate_single_pred_model_binary(
    model_name,
    pred_path,
    true_df,
    all_cell_types,
    test_cell_types,
    masks,
    split,
    positive_threshold=0.0,
):
    print("loading", model_name, pred_path)
    pred_df = load_pred_df(pred_path, all_cell_types)
    result_df = evaluate_full_model_binary_on_cells(
        model_name=model_name,
        pred_df=pred_df,
        true_df=true_df,
        eval_cell_types=test_cell_types,
        masks=masks,
        split=split,
        positive_threshold=positive_threshold,
    )
    print(result_df[result_df["cell_type"] == "test_mean"])
    return result_df
