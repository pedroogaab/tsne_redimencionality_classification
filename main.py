import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
)
from collections import Counter

import warnings

warnings.filterwarnings("ignore")


class KNN:
    def __init__(self, k, metric):
        self.k = k
        self.metric = metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def cosine_distance(self, x1, x2):
        return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def predict_proba(self, X):
        probas = []
        classes = np.unique(self.y_train)

        for x in X:
            if self.metric == "euclidean":
                distances = [
                    self.euclidean_distance(x, x_train) for x_train in self.X_train
                ]
            else:  # cosine
                distances = [
                    self.cosine_distance(x, x_train) for x_train in self.X_train
                ]

            k_indices = np.argsort(distances)[: self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]

            proba = np.zeros(len(classes))
            count = Counter(k_nearest_labels)
            for i, cls in enumerate(classes):
                proba[i] = count.get(cls, 0) / self.k

            probas.append(proba)

        return np.array(probas)

    def _predict(self, x):
        if self.metric == "euclidean":
            distances = [
                self.euclidean_distance(x, x_train) for x_train in self.X_train
            ]
        else:  # cosine
            distances = [self.cosine_distance(x, x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]


def load_data(file_path="data/mini_gm_public_v0.1.p"):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def structure_data(data):
    embeddings, syndromes, subject_ids, image_ids = [], [], [], []

    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():
                embeddings.append(embedding)
                syndromes.append(syndrome_id)
                subject_ids.append(subject_id)
                image_ids.append(image_id)

    embeddings = np.array(embeddings)
    syndrome_df = pd.DataFrame({"syndrome": syndromes})
    embedding_df = pd.DataFrame(
        embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])]
    )

    return pd.concat([syndrome_df, embedding_df], axis=1)


def calculate_f1_score(y_true, y_pred):
    classes = np.unique(y_true)
    f1_scores = []

    for cls in classes:
        true_binary = (y_true == cls).astype(int)
        pred_binary = (y_pred == cls).astype(int)

        true_positives = np.sum(true_binary & pred_binary)
        false_positives = np.sum(pred_binary) - true_positives
        false_negatives = np.sum(true_binary) - true_positives

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0

        f1_scores.append(f1)

    return np.mean(f1_scores)


def calculate_top_k_accuracy(y_true, y_proba, k=3):
    n_samples = len(y_true)
    top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]

    correct = 0
    for i in range(n_samples):
        if y_true[i] in top_k_preds[i]:
            correct += 1

    return correct / n_samples


def calculate_auc_roc(y_true, y_score):
    classes = np.unique(y_true)
    n_classes = len(classes)

    y_true_one_hot = np.zeros((len(y_true), n_classes))
    for i, cls in enumerate(classes):
        y_true_one_hot[:, i] = (y_true == cls).astype(int)

    roc_auc = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_one_hot[:, i], y_score[:, i])
        roc_auc.append(auc(fpr, tpr))

    return np.mean(roc_auc)


def plot_roc_curves(y_true, y_proba_euc, y_proba_cos, class_names, img_dir):
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, len(class_names)))

    for i, color, cls in zip(range(len(class_names)), colors, class_names):
        y_true_bin = (y_true == cls).astype(int)

        fpr_euc, tpr_euc, _ = roc_curve(y_true_bin, y_proba_euc[:, i])
        roc_auc_euc = auc(fpr_euc, tpr_euc)
        plt.plot(
            fpr_euc,
            tpr_euc,
            color=color,
            linestyle="-",
            label=f"ROC curve of class {cls} (Euclidean, AUC = {roc_auc_euc:.2f})",
        )

        fpr_cos, tpr_cos, _ = roc_curve(y_true_bin, y_proba_cos[:, i])
        roc_auc_cos = auc(fpr_cos, tpr_cos)
        plt.plot(
            fpr_cos,
            tpr_cos,
            color=color,
            linestyle="--",
            label=f"ROC curve of class {cls} (Cosine, AUC = {roc_auc_cos:.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.savefig(f"{img_dir}/roc_curves.png")
    plt.close()


def plot_tsne_visualization(X_tsne, y, class_names, img_dir):
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="tab10", s=50, alpha=0.7
    )
    plt.colorbar(scatter, label="Syndrome Class")
    plt.title("t-SNE Visualization of Embeddings")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")

    handles, labels = scatter.legend_elements()

    legend_labels = [class_names[int(label)] for label in labels if label.isdigit()]
    plt.legend(handles, legend_labels, title="Syndromes")

    plt.savefig(f"{img_dir}/tsne_visualization.png")
    plt.close()


def cross_validate_knn(
    X, y, n_folds=10, k_range=range(1, 16), img_dir="images", tsne_params=None
):
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    class_names = np.unique(y)

    results = {
        "euclidean": {
            k: {"accuracy": [], "f1": [], "auc": [], "top3_acc": []} for k in k_range
        },
        "cosine": {
            k: {"accuracy": [], "f1": [], "auc": [], "top3_acc": []} for k in k_range
        },
    }

    best_proba_euc = None
    best_proba_cos = None
    test_y_for_roc = None
    best_k_euc = None
    best_auc_euc = -1
    best_k_cos = None
    best_auc_cos = -1

    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for metric in ["euclidean", "cosine"]:
            for k in k_range:
                clf = KNN(k=k, metric=metric)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                y_proba = clf.predict_proba(X_test)

                accuracy = np.mean(y_pred == y_test)
                f1 = calculate_f1_score(y_test, y_pred)
                auc_score = calculate_auc_roc(y_test, y_proba)
                top3_acc = calculate_top_k_accuracy(y_test, y_proba, k=3)

                results[metric][k]["accuracy"].append(accuracy)
                results[metric][k]["f1"].append(f1)
                results[metric][k]["auc"].append(auc_score)
                results[metric][k]["top3_acc"].append(top3_acc)

                if metric == "euclidean" and auc_score > best_auc_euc:
                    best_auc_euc = auc_score
                    best_k_euc = k
                    best_proba_euc = y_proba
                    test_y_for_roc = y_test

                if metric == "cosine" and auc_score > best_auc_cos:
                    best_auc_cos = auc_score
                    best_k_cos = k
                    best_proba_cos = y_proba

    avg_results = {}
    for metric in ["euclidean", "cosine"]:
        avg_results[metric] = {}
        for k in k_range:
            avg_results[metric][k] = {
                "accuracy": np.mean(results[metric][k]["accuracy"]),
                "f1": np.mean(results[metric][k]["f1"]),
                "auc": np.mean(results[metric][k]["auc"]),
                "top3_acc": np.mean(results[metric][k]["top3_acc"]),
            }

    best_k = {}
    for metric in ["euclidean", "cosine"]:
        aucs = [avg_results[metric][k]["auc"] for k in k_range]
        best_k[metric] = k_range[np.argmax(aucs)]

    if (
        best_proba_euc is not None
        and best_proba_cos is not None
        and test_y_for_roc is not None
    ):
        plot_roc_curves(
            test_y_for_roc, best_proba_euc, best_proba_cos, class_names, img_dir
        )

    summary = pd.DataFrame(
        columns=[
            "Metric",
            "Best K",
            "Accuracy",
            "F1 Score",
            "AUC",
            "Top-3 Accuracy",
            "t-SNE Components",
            "t-SNE Perplexity",
            "t-SNE Max Iterations",
        ]
    )

    for i, metric in enumerate(["euclidean", "cosine"]):
        k = best_k[metric]
        summary.loc[i] = [
            metric,
            k,
            avg_results[metric][k]["accuracy"],
            avg_results[metric][k]["f1"],
            avg_results[metric][k]["auc"],
            avg_results[metric][k]["top3_acc"],
            tsne_params["n_components"],
            tsne_params["perplexity"],
            tsne_params["max_iter"],
        ]

    plt.figure(figsize=(15, 10))
    metrics = ["accuracy", "f1", "auc", "top3_acc"]
    titles = ["Accuracy", "F1 Score", "AUC", "Top-3 Accuracy"]

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.subplot(2, 2, i + 1)
        for dist in ["euclidean", "cosine"]:
            values = [avg_results[dist][k][metric] for k in k_range]
            plt.plot(k_range, values, marker="o", label=f"{dist} distance")
        plt.title(title)
        plt.xlabel("k")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{img_dir}/knn_comparison.png")
    plt.close()

    return summary, avg_results, best_k


def exploratory_data_analysis(df, img_dir):
    print("Dataset Statistics:")
    print(f"Total samples: {len(df)}")

    syndromes = df["syndrome"].unique()
    print(f"Number of unique syndromes: {len(syndromes)}")

    print("\nSyndrome Distribution:")
    total = len(df)
    syndrome_counts = df["syndrome"].value_counts()

    for syndrome, count in syndrome_counts.items():
        percentage = (count / total) * 100
        print(f"  - {syndrome}: {count} samples ({percentage:.2f}%)")

    plt.figure(figsize=(12, 6))
    syndrome_counts.plot(kind="bar")
    plt.title("Number of Samples per Syndrome")
    plt.xlabel("Syndrome")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{img_dir}/class_distribution.png")
    plt.close()


def main():
    # Load and transform data to DataFrame
    data = load_data()
    df = structure_data(data)

    img_dir = "images"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # Perform EDA
    exploratory_data_analysis(df, img_dir)

    # Extract features and labels
    X = df.drop("syndrome", axis=1).values
    y_original = df["syndrome"].values

    # t-SNE parameters
    tsne_params = {
        "n_components": 2,
        "perplexity": 89,
        "max_iter": 790,
        "random_state": 42,
    }

    # Reduce dimensionality with t-SNE
    print("Applying t-SNE dimensionality reduction...")
    tsne = TSNE(**tsne_params)
    X_tsne = tsne.fit_transform(X)

    # Normalize syndrome labels to integers
    label_map = {syndrome: idx for idx, syndrome in enumerate(np.unique(y_original))}
    y = np.array([label_map[val] for val in y_original], dtype=int)

    inverse_label_map = {idx: syndrome for syndrome, idx in label_map.items()}

    # Plot t-SNE visualization
    plot_tsne_visualization(
        X_tsne, y, [inverse_label_map[i] for i in range(len(label_map))], img_dir
    )

    # Perform cross-validation
    print("Performing 10-fold cross-validation...")
    summary, avg_results, best_k = cross_validate_knn(
        X_tsne, y, n_folds=10, img_dir=img_dir, tsne_params=tsne_params
    )

    # Print results
    print("\nResults Summary:")
    print(summary)

    # Save detailed results to CSV
    detailed_results = summary.copy()

    # Add k values tested
    detailed_results["K Values Tested"] = str(list(range(1, 16)))

    # Add all metrics for both distance methods
    for metric in ["euclidean", "cosine"]:
        detailed_results[f"All {metric.capitalize()} Accuracy"] = str(
            [round(avg_results[metric][k]["accuracy"], 4) for k in range(1, 16)]
        )
        detailed_results[f"All {metric.capitalize()} F1"] = str(
            [round(avg_results[metric][k]["f1"], 4) for k in range(1, 16)]
        )
        detailed_results[f"All {metric.capitalize()} AUC"] = str(
            [round(avg_results[metric][k]["auc"], 4) for k in range(1, 16)]
        )
        detailed_results[f"All {metric.capitalize()} Top-3"] = str(
            [round(avg_results[metric][k]["top3_acc"], 4) for k in range(1, 16)]
        )

    # Add dataset information
    detailed_results["Number of Samples"] = len(df)
    detailed_results["Number of Classes"] = len(df["syndrome"].unique())
    detailed_results["Class Distribution"] = str(
        df["syndrome"].value_counts().to_dict()
    )

    # Save detailed summary to CSV
    detailed_results.to_csv("knn_results_summary.csv", index=False)

    # Compare best models
    print("\nBest k for Euclidean distance:", best_k["euclidean"])
    print("Best k for Cosine distance:", best_k["cosine"])

    # Get the scores for the best models
    euc_best = avg_results["euclidean"][best_k["euclidean"]]
    cos_best = avg_results["cosine"][best_k["cosine"]]

    print(f"\nEuclidean distance (k={best_k['euclidean']}):")
    print(f"  - Accuracy: {euc_best['accuracy']:.4f}")
    print(f"  - F1 Score: {euc_best['f1']:.4f}")
    print(f"  - AUC: {euc_best['auc']:.4f}")
    print(f"  - Top-3 Accuracy: {euc_best['top3_acc']:.4f}")

    print("\nCosine distance (k={}):".format(best_k["cosine"]))
    print(f"  - Accuracy: {cos_best['accuracy']:.4f}")
    print(f"  - F1 Score: {cos_best['f1']:.4f}")
    print(f"  - AUC: {cos_best['auc']:.4f}")
    print(f"  - Top-3 Accuracy: {cos_best['top3_acc']:.4f}")

    # Determine best overall metric
    best_metric = "euclidean" if euc_best["auc"] > cos_best["auc"] else "cosine"
    print(f"\nBest overall distance metric based on AUC: {best_metric}")

    # Final model testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_tsne, y, test_size=0.2, random_state=42
    )

    # Test the best models
    for metric in ["euclidean", "cosine"]:
        k = best_k[metric]
        clf = KNN(k=k, metric=metric)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Reds)
        plt.title(f"Confusion Matrix - {metric.capitalize()} (k={k})")
        plt.colorbar()

        syndrome_names = [inverse_label_map[i] for i in range(len(label_map))]
        tick_marks = np.arange(len(syndrome_names))
        plt.xticks(tick_marks, syndrome_names, rotation=45)
        plt.yticks(tick_marks, syndrome_names)

        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        plt.savefig(f"{img_dir}/confusion_matrix_{metric}.png")
        plt.close()


if __name__ == "__main__":
    main()
