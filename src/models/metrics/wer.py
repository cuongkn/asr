import Levenshtein as Lev


def cal_wer(s1, s2):
    # build mapping of words to integers
    b = set(s1.split() + s2.split())
    word2char = dict(zip(b, range(len(b))))

    # map the words to a char array
    w1 = [chr(word2char[w]) for w in s1.split()]
    w2 = [chr(word2char[w]) for w in s2.split()]

    score = Lev.distance(''.join(w1), ''.join(w2)) / len(s2.split())

    return score