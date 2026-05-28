from util import parse_vertical
from dataclasses import dataclass, field
import csv
import defopt

@dataclass
class Lexeme:
    """Class for keeping track of an item in inventory."""
    lemma: str
    pos: str
    frequency: int = 0
    forms: set[str] = field(default_factory=set)

def count_syllables(word: str) -> int:
    return len(word)

def main(input: str, output: str):
    lexemes: dict[str, Lexeme] = dict()

    sentences = parse_vertical(input, drop_punct=True)

    for s in sentences:
        for t in s:
            form = t[0].lower()
            lemma = t[1].upper()
            pos = t[2].upper()

            new_lexeme = Lexeme(
                    lemma=lemma,
                    pos=pos,
                )

            if not lemma + "/" + pos in lexemes:
                lexemes[lemma + "/" + pos] = new_lexeme

            lexemes[lemma + "/" + pos].frequency += 1
            lexemes[lemma + "/" + pos].forms.add(form)

    with open(output, "w", newline="") as csvfile:
        tabwriter = csv.writer(csvfile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_ALL)
        tabwriter.writerow(["LEMMA/POS", "LEMMA", "POS", "FREQUENCY", "PARADIGM_SIZE", "AVERAGE_LENGHT_OF_FORM"])

        for l in lexemes.values():
            paradigm_size = len(l.forms)
            average_lenght_of_forms = sum([count_syllables(f) for f in l.forms]) / paradigm_size

            tabwriter.writerow([l.lemma.upper() + "/" + l.pos.upper(), l.lemma, l.pos, l.frequency, paradigm_size, average_lenght_of_forms])

if __name__ == "__main__":
    defopt.run(main)
