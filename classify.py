#!/usr/bin/env python3
"""
CS 440 PS7 - Text Classification

Usage:
    python3 classify.py TRAIN_FILE TEST_FILE FUNCTION

Functions:
    tf      Build term-frequency counts and write tf.csv
    tfgrep  Classify using the most discriminating term
    priors  Classify using the majority-class baseline
    mnb     Multinomial Naive Bayes
    df      Build document-frequency probabilities and write df.csv
    nb      Multivariate Bernoulli Naive Bayes
    mine    Reduced-vocabulary Multinomial Naive Bayes
"""

import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

VALID_FUNCTIONS = {"tf", "tfgrep", "priors", "mnb", "df", "nb", "mine"}


def read_documents(filename):
    """Read labeled text documents. First token is the class label."""
    documents = []
    with open(filename, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            label = parts[0]
            words = parts[1:]
            documents.append((label, words))
    if not documents:
        raise ValueError(f"No valid documents found in {filename}")
    return documents


def get_labels(documents):
    """Return binary labels in the order they appear in the training data."""
    labels = []
    for label, _ in documents:
        if label not in labels:
            labels.append(label)
        if len(labels) == 2:
            break
    if len(labels) != 2:
        raise ValueError("This assignment expects exactly two class labels.")
    return labels[0], labels[1]


def class_counts(documents, labels):
    counts = {label: 0 for label in labels}
    for label, _ in documents:
        counts[label] += 1
    return counts


def build_tf(documents, labels):
    """Build term-frequency counts per class."""
    tf_counts = {label: Counter() for label in labels}
    vocab = set()

    for label, words in documents:
        tf_counts[label].update(words)
        vocab.update(words)

    return tf_counts, vocab


def write_tf_csv(tf_counts, vocab, labels, output_file="tf.csv"):
    """Write word, tf(label0), tf(label1) to tf.csv."""
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        for word in sorted(vocab):
            writer.writerow([word, tf_counts[labels[0]][word], tf_counts[labels[1]][word]])


def print_top_tf(train_file, tf_counts, labels):
    print("===== Results =====")
    print(f"Data: {train_file}")

    for index, label in enumerate(labels):
        print(f"Class Label: {index}")
        print(dict(tf_counts[label].most_common(5)))
        if index == 0:
            print()


def build_df(documents, labels):
    """Build document-frequency counts per class."""
    df_counts = {label: Counter() for label in labels}
    vocab = set()

    for label, words in documents:
        unique_words = set(words)
        df_counts[label].update(unique_words)
        vocab.update(unique_words)

    return df_counts, vocab


def write_df_csv(df_counts, vocab, labels, counts_by_class, output_file="df.csv"):
    """Write word, P(word appears | class0), P(word appears | class1) to df.csv."""
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        for word in sorted(vocab):
            p0 = df_counts[labels[0]][word] / counts_by_class[labels[0]]
            p1 = df_counts[labels[1]][word] / counts_by_class[labels[1]]
            writer.writerow([word, p0, p1])


def print_top_df(train_file, df_counts, labels, counts_by_class):
    print("===== Results =====")
    print(f"Data: {train_file}")

    for index, label in enumerate(labels):
        probabilities = {
            word: df_counts[label][word] / counts_by_class[label]
            for word in df_counts[label]
        }
        top_five = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:5])
        print(f"Class Label: {index}")
        print(top_five)
        if index == 0:
            print()


def confusion_matrix(documents, predict, positive_label):
    """Return TP, FN, FP, TN using the first training label as the positive class."""
    tp = fn = fp = tn = 0

    for actual_label, words in documents:
        predicted_label = predict(words)

        if predicted_label == positive_label and actual_label == positive_label:
            tp += 1
        elif predicted_label == positive_label and actual_label != positive_label:
            fn += 1
        elif predicted_label != positive_label and actual_label == positive_label:
            fp += 1
        else:
            tn += 1

    return tp, fn, fp, tn


def print_confusion(title, data_file, matrix):
    tp, fn, fp, tn = matrix
    print(f"===== {title} ======")
    print(f"Data:  {data_file}")
    print("*** Confusion Matrices ***")
    print(f"TP   FN  | {tp} {fn}")
    print(f"FP   TN  | {fp} {tn}")
    print()


def run_tf(train_file):
    train_docs = read_documents(train_file)
    labels = get_labels(train_docs)
    tf_counts, vocab = build_tf(train_docs, labels)
    write_tf_csv(tf_counts, vocab, labels)
    print_top_tf(train_file, tf_counts, labels)


def run_df(train_file):
    train_docs = read_documents(train_file)
    labels = get_labels(train_docs)
    counts_by_class = class_counts(train_docs, labels)
    df_counts, vocab = build_df(train_docs, labels)
    write_df_csv(df_counts, vocab, labels, counts_by_class)
    print_top_df(train_file, df_counts, labels, counts_by_class)


def run_tfgrep(train_file, test_file):
    train_docs = read_documents(train_file)
    test_docs = read_documents(test_file)
    labels = get_labels(train_docs)
    tf_counts, vocab = build_tf(train_docs, labels)
    write_tf_csv(tf_counts, vocab, labels)

    best_word = max(
        vocab,
        key=lambda word: abs(tf_counts[labels[0]][word] - tf_counts[labels[1]][word]),
    )

    label_if_present = (
        labels[0]
        if tf_counts[labels[0]][best_word] >= tf_counts[labels[1]][best_word]
        else labels[1]
    )
    label_if_absent = labels[1] if label_if_present == labels[0] else labels[0]

    def predict(words):
        return label_if_present if best_word in set(words) else label_if_absent

    for data_file, docs in [(train_file, train_docs), (test_file, test_docs)]:
        print("===== tfgrep result ======")
        print(f"Data:  {data_file}")
        print(f"Most discriminating term: {best_word}\n")
        print("*** Confusion Matrices ***")
        tp, fn, fp, tn = confusion_matrix(docs, predict, labels[0])
        print(f"TP   FN  | {tp} {fn}")
        print(f"FP   TN  | {fp} {tn}")
        print()


def run_priors(train_file, test_file):
    train_docs = read_documents(train_file)
    test_docs = read_documents(test_file)
    labels = get_labels(train_docs)
    counts_by_class = class_counts(train_docs, labels)
    majority_label = max(labels, key=lambda label: counts_by_class[label])

    def predict(_words):
        return majority_label

    for data_file, docs in [(train_file, train_docs), (test_file, test_docs)]:
        matrix = confusion_matrix(docs, predict, labels[0])
        print_confusion("PRIORS Result", data_file, matrix)


def mnb_predictor(train_docs, labels, reduced_vocab=None):
    tf_counts, vocab = build_tf(train_docs, labels)
    if reduced_vocab is not None:
        vocab = set(reduced_vocab)

    counts_by_class = class_counts(train_docs, labels)
    total_docs = len(train_docs)
    token_totals = {
        label: sum(tf_counts[label][word] for word in vocab)
        for label in labels
    }
    vocab_size = max(len(vocab), 1)

    log_prior = {
        label: math.log(counts_by_class[label] / total_docs)
        for label in labels
    }

    # Laplacian smoothing: (1 + tf_wc) / (vocab_size + total_tokens_c)
    log_prob = {
        label: {
            word: math.log((1 + tf_counts[label][word]) / (vocab_size + token_totals[label]))
            for word in vocab
        }
        for label in labels
    }

    unknown_log_prob = {
        label: math.log(1 / (vocab_size + token_totals[label]))
        for label in labels
    }

    def predict(words):
        word_counts = Counter(word for word in words if word in vocab)
        scores = {}
        for label in labels:
            score = log_prior[label]
            for word, count in word_counts.items():
                score += count * log_prob[label].get(word, unknown_log_prob[label])
            scores[label] = score
        return max(labels, key=lambda label: scores[label])

    return predict, tf_counts, vocab


def run_mnb(train_file, test_file):
    train_docs = read_documents(train_file)
    test_docs = read_documents(test_file)
    labels = get_labels(train_docs)
    tf_counts, vocab = build_tf(train_docs, labels)
    write_tf_csv(tf_counts, vocab, labels)
    predict, _tf_counts, _vocab = mnb_predictor(train_docs, labels)

    for data_file, docs in [(train_file, train_docs), (test_file, test_docs)]:
        matrix = confusion_matrix(docs, predict, labels[0])
        print_confusion("MNB Result", data_file, matrix)


def nb_predictor(train_docs, labels):
    df_counts, vocab = build_df(train_docs, labels)
    counts_by_class = class_counts(train_docs, labels)
    total_docs = len(train_docs)

    log_prior = {
        label: math.log(counts_by_class[label] / total_docs)
        for label in labels
    }

    # Bernoulli smoothing: (1 + df_wc) / (2 + number_of_docs_in_class).
    # To keep runtime reasonable, start each class score with all words absent,
    # then adjust only for words that are present in the document.
    base_absent_score = {}
    present_adjustment = {label: {} for label in labels}

    for label in labels:
        base = log_prior[label]
        denom = 2 + counts_by_class[label]
        for word in vocab:
            p_present = (1 + df_counts[label][word]) / denom
            p_absent = 1 - p_present
            base += math.log(p_absent)
            present_adjustment[label][word] = math.log(p_present) - math.log(p_absent)
        base_absent_score[label] = base

    def predict(words):
        present_words = set(word for word in words if word in vocab)
        scores = {}
        for label in labels:
            score = base_absent_score[label]
            for word in present_words:
                score += present_adjustment[label][word]
            scores[label] = score
        return max(labels, key=lambda label: scores[label])

    return predict, df_counts, vocab


def run_nb(train_file, test_file):
    train_docs = read_documents(train_file)
    test_docs = read_documents(test_file)
    labels = get_labels(train_docs)
    counts_by_class = class_counts(train_docs, labels)
    df_counts, vocab = build_df(train_docs, labels)
    write_df_csv(df_counts, vocab, labels, counts_by_class)
    predict, _df_counts, _vocab = nb_predictor(train_docs, labels)

    for data_file, docs in [(train_file, train_docs), (test_file, test_docs)]:
        matrix = confusion_matrix(docs, predict, labels[0])
        print_confusion("NB Result", data_file, matrix)


def run_mine(train_file, test_file):
    train_docs = read_documents(train_file)
    test_docs = read_documents(test_file)
    labels = get_labels(train_docs)
    tf_counts, vocab = build_tf(train_docs, labels)

    reduced_vocab = {
        word for word in vocab
        if tf_counts[labels[0]][word] != tf_counts[labels[1]][word]
    }

    # Keep a fallback in case a tiny dataset removes everything.
    if not reduced_vocab:
        reduced_vocab = vocab

    write_tf_csv(tf_counts, reduced_vocab, labels, output_file="tfmine.csv")
    predict, _tf_counts, _vocab = mnb_predictor(train_docs, labels, reduced_vocab=reduced_vocab)

    print()
    print("The changes in my implementation focus on reducing false negatives and false positives.")
    print("I modified the multinomial Naive Bayes model by using a reduced vocabulary.")
    print("The reduced vocabulary removes terms that appear equally often in both class labels because those words add complexity without helping the classifier separate the two classes.")
    print("This keeps the model simpler while still using discriminating terms for prediction.")
    print()

    for data_file, docs in [(train_file, train_docs), (test_file, test_docs)]:
        matrix = confusion_matrix(docs, predict, labels[0])
        print_confusion("MINE / Reduced MNB Result", data_file, matrix)


def print_help():
    print("*** ERROR ***")
    print("\nPlease enter data in the correct format:")
    print("python3 classify.py (Training Data) (Testing Data) (Execution function)\n")
    print("Allowable functions:")
    print("  tf      - build term frequencies and write tf.csv")
    print("  tfgrep  - classify using the most discriminating term")
    print("  priors  - classify using the majority class baseline")
    print("  mnb     - run multinomial Naive Bayes")
    print("  df      - build document frequencies and write df.csv")
    print("  nb      - run multivariate Bernoulli Naive Bayes")
    print("  mine    - run a reduced-vocabulary MNB model")


def main():
    if len(sys.argv) != 4:
        print_help()
        return

    train_file, test_file, function_name = sys.argv[1], sys.argv[2], sys.argv[3]

    if function_name not in VALID_FUNCTIONS:
        print_help()
        return

    if not Path(train_file).exists() or not Path(test_file).exists():
        print("*** ERROR ***")
        print("Training or testing file was not found.")
        return

    if function_name == "tf":
        run_tf(train_file)
    elif function_name == "df":
        run_df(train_file)
    elif function_name == "tfgrep":
        run_tfgrep(train_file, test_file)
    elif function_name == "priors":
        run_priors(train_file, test_file)
    elif function_name == "mnb":
        run_mnb(train_file, test_file)
    elif function_name == "nb":
        run_nb(train_file, test_file)
    elif function_name == "mine":
        run_mine(train_file, test_file)


if __name__ == "__main__":
    main()
