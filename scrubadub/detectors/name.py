import re

import spacy

from .base import RegexDetector
from ..filth import NameFilth
from ..utils import CanonicalStringSet


class NameDetector(RegexDetector):
    """Use part of speech tagging to clean proper nouns out of the dirty dirty
    ``text``. Disallow particular nouns by adding them to the
    ``NameDetector.disallowed_nouns`` set.
    """

    nlp = spacy.load('en')

    filth_cls = NameFilth

    disallowed_nouns = CanonicalStringSet(["skype"])

    def iter_filth(self, text):

        if not isinstance(self.disallowed_nouns, CanonicalStringSet):
            raise TypeError(
                'NameDetector.disallowed_nouns must be CanonicalStringSet'
            )

        if type(text) is not unicode:
            text = text.decode('utf-8')

        parsed_text = self.nlp(text)
        entities = list(parsed_text.ents)
        proper_nouns = set()
        accepted_entities = ["PERSON"]
        for entity in entities:
            is_proper_noun = entity.label_ in accepted_entities
            for t in entity:
                word = t.orth_
                not_disallowed = word.lower() not in self.disallowed_nouns
                if is_proper_noun and not_disallowed:
                    proper_nouns.add(word)

        # use a regex to replace the proper nouns by first escaping any
        # lingering punctuation in the regex
        # http://stackoverflow.com/a/4202559/564709
        if proper_nouns:
            re_list = []
            for proper_noun in proper_nouns:
                re_list.append(r'\b' + re.escape(proper_noun) + r'\b')
            self.filth_cls.regex = re.compile('|'.join(re_list))
        else:
            self.filth_cls.regex = None
        return super(NameDetector, self).iter_filth(text)
