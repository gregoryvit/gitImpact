import hashlib
import time


class Container(object):
    def __init__(self, name, children):
        self.children = children
        self.name = name

    def __repr__(self):
        return "<%s: %s>" % (self.name, self.children)

    def __str__(self):
        return self.__repr__()

    def __getattr__(self, arg):
        if arg == 'atoms':
            return self.get_atoms()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name.__eq__(other.name) and self.children.__eq__(other.children)

    def get_atoms(self):
        results = []
        for child in self.children:
            results.extend(child.atoms)
        return results


class Atom(Container):
    def __init__(self, value, name, children):
        super(Atom, self).__init__(name, children)
        self.value = value

    def get_atoms(self):
        return [self]

    def __repr__(self):
        return self.value

    def __eq__(self, other):
        return self.value.__eq__(other.value)

    def __hash__(self):
        return hash(self.value)


class HashContainer(Container):
    def __init__(self, children):
        hash_str = hashlib.sha224(str(hash(time.time()))).hexdigest()[:5]
        Container.__init__(self, hash_str, children)
