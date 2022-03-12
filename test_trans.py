
import csv
import sys
from pytorch_t import load_nmt
from da_multiese import compose_testing
from multiese2_test import test_code


def test(tuples, gen_fn, w=None):
    testing_fn = compose_testing(gen_fn)
    count = 0
    for src, tgt, params in tuples:
        print(src, tgt, params)
        src, gen, tgt = testing_fn(src, tgt)
        tgt_tested = test_code(tgt, params)
        gen_tested = test_code(gen, params)
        check = 1 if tgt_tested == gen_tested else 0
        count += check
        results = (check, src, gen, tgt, gen_tested, tgt_tested)
        if w is not None:
            w.writerow(results)
    print(count, len(tuples), count/len(tuples))


def load_tsv(file, tuples):
    sep = ',' if file.endswith('.csv') else '\t'
    with open(file, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=sep)
        for row in reader:
            if len(row) > 3:
                tuples.append((row[2], row[1], row[3]))
    return tuples


def _main():
    tuples = []
    gen_fn = None
    for f in sys.argv[1:]:
        if f.endswith('pt'):
            gen_fn = load_nmt(f)
        elif f.endswith('.tsv') or f.endswith('.tsv'):
            load_tsv(f, tuples)
    with open('test_code.tsv', 'w') as w:
        w = csv.writer(w, delimiter='\t')
        test(tuples, gen_fn, w)


if __name__ == '__main__':
    _main()
