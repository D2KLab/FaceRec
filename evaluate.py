"""Compare the output of bulk run with a ground truth"""
import argparse
import json
import sys

import pandas as pd
import numpy as np
from src import clusterize
import matplotlib.pyplot as plt


def get_match(person, res):
    # print('*****', person, sep=' ')
    person = person.lower()
    if person is '0':
        return False
    for r in res:
        # print(r['name'], '$$' if r['name'].person == person else '', sep=' ')
        if r['name'].lower() == person:
            return True
    return False


def parse_person(x):
    p = x['person'].split(',')
    p.reverse()
    p = ' '.join(p).strip()
    return p


def main(results, ground_truth):
    gt = pd.read_csv(ground_truth)
    with open(results, 'r', encoding='utf-8') as f:
        res = json.load(f)

    persons = [parse_person(x) for i, x in gt.iterrows()]
    persons = ['Elisabeth II' if p == "Elizabeth d'Angleterre" else p for p in persons]

    precision = []
    recall = []
    thresholds = np.arange(0.4, 1, step=0.05)
    dominant = np.arange(0.4, 1, step=0.05)
    weighted_dominant = np.arange(0.4, 1, step=0.05)
    wc = 0.4
    dom = 0.6
    for val in thresholds:
        # for dom in dominant:
        # for wc in weighted_dominant:
        clusters = []
        for r in res:
            c = clusterize.main(clusterize.from_dict(r), dominant_ratio=dom, weighted_dominant_ratio=wc,
                                confidence_threshold=val, merge_cluster=True, min_length=1)
            clusters.append(c)

        predictions = [len(x) > 0 for x in clusters]
        matches = [get_match(person, r) for person, r in zip(persons, clusters)]

        # see https://github.com/rafaelpadilla/Object-Detection-Metrics
        true_positive = np.sum(matches)  # hit
        false_positive = np.sum([not m and pred for pred, m in zip(predictions, matches)])  # wrong
        false_negative = np.sum([1 if not m and p != '0' else 0 for p, m in zip(persons, matches)])  # miss

        p = true_positive / sum(predictions)  # (true_positive + false_positive)
        r = true_positive / sum([1 if p != '0' else 0 for p in persons])  # (true_positive + false_negative)
        print('%.2f' % val, true_positive, false_positive, false_negative, p, r, sep='\t|\t')

        precision.append(p)
        recall.append(r)

    plt.figure(1)
    plt.plot(recall, precision)
    plt.xlabel('recall')
    plt.ylabel('precision')
    # plt.legend(loc='best')
    plt.show()

    clusters = []
    for r in res:
        c = clusterize.main(clusterize.from_dict(r), dominant_ratio=dom, weighted_dominant_ratio=wc,
                            confidence_threshold=0.6, merge_cluster=True, min_length=1)
        clusters.append(c)

    persons = [parse_person(x) for i, x in gt.iterrows()]
    persons = ['Elisabeth II' if p == "Elizabeth d'Angleterre" else p for p in persons]
    matches = [get_match(person, res) for person, res in zip(persons, clusters)]

    df = pd.DataFrame(persons, columns=['person'])
    df['match'] = matches
    df['n_prediction'] = [len(x) for x in clusters]

    ppl = df[df['person'] != '0']
    print('Scores: ')
    print(f'* Total: {len(persons)}')
    print('* Matches: %d (%.2f%%)' % (sum(matches), sum(matches) * 100 / len(persons)))
    unpredicted = len(df[df['n_prediction'] == 0])
    print('* Unpredicted: %d (%.2f%%)' % (unpredicted, unpredicted / len(persons)))
    print(f'* Total people: {len(ppl)}')
    print('* Matches people: %d (%.2f%%)' % (sum(ppl['match']), sum(ppl['match']) * 100 / len(ppl)))
    unpredicted = len(ppl[ppl['n_prediction'] == 0])
    print('* Unpredicted people: %d (%.2f%%)' % (unpredicted, unpredicted * 100 / len(ppl)))
    predicted = ppl[ppl['n_prediction'] != 0]
    wrong = len(predicted[predicted['match'] == False])
    print('* Wrong people: %d (%.2f%%)' % (wrong, wrong * 100 / len(ppl)))

    ppl_aggr = df.groupby('person').agg({'match': ['count', 'sum']})
    ppl_aggr['perc'] = ppl_aggr['match']['sum'] / ppl_aggr['match']['count']
    print(ppl_aggr)

    gt['match'] = matches
    gt['locator'] = gt.apply(
        lambda
            x: f"https://okapi.ina.fr/antract/Media/AF/{x['media'].split('/')[-1]}.mp4#t={int(x['start'])},{int(x['end'])}",
        axis=1)
    gt.to_csv(ground_truth.rsplit('.', 1)[0] + '_res.csv')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='The JSON output of bulk run.')
    parser.add_argument('--gt', type=str, required=True,
                        help='The ground truth csv')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args.input, args.gt)

# python evaluate.py -i results_antract.json --gt evaluation/dataset_antract.csv
