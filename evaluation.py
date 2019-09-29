"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from collections import OrderedDict
from gen_ner import read_dataset


def chunk_finder(current_token, previous_token, tag):
    current_tag = current_token.split('-', 1)[-1]
    previous_tag = previous_token.split('-', 1)[-1]
    if previous_tag != tag:
        previous_tag = 'O'
    if current_tag != tag:
        current_tag = 'O'
    if (previous_tag == 'O' and current_token == 'B-' + tag) or \
            (previous_token == 'I-' + tag and current_token == 'B-' + tag) or \
            (previous_token == 'B-' + tag and current_token == 'B-' + tag) or \
            (previous_tag == 'O' and current_token == 'I-' + tag):
        create_chunk = True
    else:
        create_chunk = False

    if (previous_token == 'I-' + tag and current_token == 'B-' + tag) or \
            (previous_token == 'B-' + tag and current_token == 'B-' + tag) or \
            (current_tag == 'O' and previous_token == 'I-' + tag) or \
            (current_tag == 'O' and previous_token == 'B-' + tag):
        pop_out = True
    else:
        pop_out = False
    return create_chunk, pop_out


def precision_recall_f1(y_true, y_pred, print_results=True, short_report=False, entity_of_interest=None):
    # Find all tags
    tags = set()
    for tag in y_true + y_pred:
        if tag != 'O':
            current_tag = tag[2:]
            tags.add(current_tag)
    tags = sorted(list(tags))

    results = OrderedDict()
    for tag in tags:
        results[tag] = OrderedDict()
    results['__total__'] = OrderedDict()
    n_tokens = len(y_true)
    total_correct = 0
    # Firstly we find all chunks in the ground truth and prediction
    # For each chunk we write starting and ending indices

    for tag in tags:
        count = 0
        true_chunk = list()
        pred_chunk = list()
        y_true = [str(y) for y in y_true]
        y_pred = [str(y) for y in y_pred]
        prev_tag_true = 'O'
        prev_tag_pred = 'O'
        while count < n_tokens:
            yt = y_true[count]
            yp = y_pred[count]

            create_chunk_true, pop_out_true = chunk_finder(yt, prev_tag_true, tag)
            if pop_out_true:
                true_chunk[-1].append(count - 1)
            if create_chunk_true:
                true_chunk.append([count])

            create_chunk_pred, pop_out_pred = chunk_finder(yp, prev_tag_pred, tag)
            if pop_out_pred:
                pred_chunk[-1].append(count - 1)
            if create_chunk_pred:
                pred_chunk.append([count])
            prev_tag_true = yt
            prev_tag_pred = yp
            count += 1

        if len(true_chunk) > 0 and len(true_chunk[-1]) == 1:
            true_chunk[-1].append(count - 1)
        if len(pred_chunk) > 0 and len(pred_chunk[-1]) == 1:
            pred_chunk[-1].append(count - 1)

        # Then we find all correctly classified intervals
        # True positive results
        tp = 0
        for start, stop in true_chunk:
            for start_p, stop_p in pred_chunk:
                if start == start_p and stop == stop_p:
                    tp += 1
                if start_p > stop:
                    break
        total_correct += tp
        # And then just calculate errors of the first and second kind
        # False negative
        fn = len(true_chunk) - tp
        # False positive
        fp = len(pred_chunk) - tp
        if tp + fp > 0:
            precision = tp / (tp + fp) * 100
        else:
            precision = 0
        if tp + fn > 0:
            recall = tp / (tp + fn) * 100
        else:
            recall = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        results[tag]['precision'] = precision
        results[tag]['recall'] = recall
        results[tag]['f1'] = f1
        results[tag]['n_predicted_entities'] = len(pred_chunk)
        results[tag]['n_true_entities'] = len(true_chunk)
    total_true_entities = 0
    total_predicted_entities = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    for tag in results:
        if tag == '__total__':
            continue
        n_pred = results[tag]['n_predicted_entities']
        n_true = results[tag]['n_true_entities']
        total_true_entities += n_true
        total_predicted_entities += n_pred
        total_precision += results[tag]['precision'] * n_pred
        total_recall += results[tag]['recall'] * n_true
        total_f1 += results[tag]['f1'] * n_true
    if total_true_entities > 0:
        accuracy = total_correct / total_true_entities * 100
        total_recall = total_recall / total_true_entities
    else:
        accuracy = 0
        total_recall = 0
    if total_predicted_entities > 0:
        total_precision = total_precision / total_predicted_entities
    else:
        total_precision = 0

    if total_precision + total_recall > 0:
        total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall)
    else:
        total_f1 = 0

    results['__total__']['n_predicted_entities'] = total_predicted_entities
    results['__total__']['n_true_entities'] = total_true_entities
    results['__total__']['precision'] = total_precision
    results['__total__']['recall'] = total_recall
    results['__total__']['f1'] = total_f1

    if print_results:
        s = 'processed {len} tokens ' \
            'with {tot_true} phrases; ' \
            'found: {tot_pred} phrases;' \
            ' correct: {tot_cor}.\n\n'.format(len=n_tokens,
                                              tot_true=total_true_entities,
                                              tot_pred=total_predicted_entities,
                                              tot_cor=total_correct)

        s += 'precision:  {tot_prec:.2f}%; ' \
             'recall:  {tot_recall:.2f}%; ' \
             'FB1:  {tot_f1:.2f}\n\n'.format(acc=accuracy,
                                             tot_prec=total_precision,
                                             tot_recall=total_recall,
                                             tot_f1=total_f1)

        if not short_report:
            for tag in tags:
                if entity_of_interest is not None:
                    if entity_of_interest in tag:
                        s += '\t' + tag + ': precision:  {tot_prec:.2f}%; ' \
                                          'recall:  {tot_recall:.2f}%; ' \
                                          'F1:  {tot_f1:.2f} ' \
                                          '{tot_predicted}\n\n'.format(tot_prec=results[tag]['precision'],
                                                                       tot_recall=results[tag]['recall'],
                                                                       tot_f1=results[tag]['f1'],
                                                                       tot_predicted=results[tag]['n_predicted_entities'])
                elif tag != '__total__':
                    s += '\t' + tag + ': precision:  {tot_prec:.2f}%; ' \
                                      'recall:  {tot_recall:.2f}%; ' \
                                      'F1:  {tot_f1:.2f} ' \
                                      '{tot_predicted}\n\n'.format(tot_prec=results[tag]['precision'],
                                                                   tot_recall=results[tag]['recall'],
                                                                   tot_f1=results[tag]['f1'],
                                                                   tot_predicted=results[tag]['n_predicted_entities'])
        elif entity_of_interest is not None:
            s += '\t' + entity_of_interest + ': precision:  {tot_prec:.2f}%; ' \
                              'recall:  {tot_recall:.2f}%; ' \
                              'F1:  {tot_f1:.2f} ' \
                              '{tot_predicted}\n\n'.format(tot_prec=results[entity_of_interest]['precision'],
                                                           tot_recall=results[entity_of_interest]['recall'],
                                                           tot_f1=results[entity_of_interest]['f1'],
                                                           tot_predicted=results[entity_of_interest]['n_predicted_entities'])
        print(s)
    return results

def chain(file):
    res_pred = []
    res_gold = []
    id2label={"0": "O",
              "1": "B-ment",
              "2": "I-ment",}
    datasets = read_dataset(file)
    for d in datasets:
        pred = eval(d.split("###")[0])
        pred = [id2label[i] for i in pred]
        res_pred.extend(pred)
        gold = eval(d.split("###")[1])
        gold = [id2label[i] for i in gold]
        res_gold.extend(gold)
    return res_pred, res_gold




if __name__ == "__main__":
    res_pred, res_gold = chain("./models/test_results_epoch_0.tsv")
    precision_recall_f1(y_pred=res_pred, y_true=res_gold)
