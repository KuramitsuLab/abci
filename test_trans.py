
from pytorch_t import load_nmt
from da_multiese import compose_testing


def test(tuples, gen_fn, w=None):
    testing_fn = compose_testing(gen_fn)
    for src, tgt, params in tuples:
        src, gen, tgt = testing_fn(src, tgt)
        tgt_tested = test_code(tgt, params)
        gen_tested = test_code(gen, params)
        check = 1 if tgt_tested == gen_tested else 0
        results = (check, src, gen, tgt, gen_tested, tgt_tested)
        if w is not None:
            w.writerow(results)


def load_t(file):
    generate = load_nmt(file)


def load_tsv(file, tuples):
    sep = ',' if file.endswith('.csv') else '\t'
    with open(file, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=sep)
        for row in reader:
            if len(row) > 3:
                tuples.append((row[2], row[1], row[3]))
    return tuples
