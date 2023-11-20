import Levenshtein as Lev


def cer(s1, s2):
    word_s1, word_s2, = s1.replace(' ', ''), s2.replace(' ', '')

    score = Lev.distance(word_s1, word_s2) / len(word_s2)

    return score
