import numpy as np
import copy

class Labels:
    def __init__ (self, labels = []):
        self.labels = labels

    def __getattr__ (self, item):
        if isinstance(item, str) and item[:2] == item[-2:] == '__':
            # skip non-existing dunder method lookups
            raise AttributeError(item)
        return getattr (self.labels, item)

    def __len__ (self):
        return len (self.labels)

    def __getitem__(self, sliced):
        return self.labels [sliced]

    def __repr__ (self):
        return repr (self.labels)

    def add (self, label):
        self.labels.append (label)

    def encode (self, *args):
        assert len (args) == len (self.labels)
        y = []
        for idx, label in enumerate (self.labels):
            val = args [idx]
            if isinstance (val, dict):
                y = label.setval (val, prefix = y)
            elif isinstance (val, (list, tuple)):
                y = label.ones (val, prefix = y)
            else:
                y = label.onehot (val, prefix = y)
        return y


class Label:
    def __init__ (self, items, name = "label"):
        self._origin = items
        self.name= name
        self._indexes = {}
        if isinstance (self._origin, (list, tuple)):
            self._set = list (set (self._origin))
            self._set.sort ()
        else:
            assert isinstance (items, dict)
            try:
                assert len (self._origin) > 1
                items_ = sorted (self._origin.items (), key = lambda x: x [1])
                pos = [v for k, v in items_]
                self._set = [k for k, v in items_]
                assert len (items_) == len (set (pos))
                assert pos [0] == 0
                assert pos [-1] == len (pos) - 1
            except (TypeError, AssertionError): # val is None
                self._set = sorted (self._origin.keys ())

        for idx, item in enumerate (self._set):
            self._indexes [item] = idx
        self._items = dict ([(v, k) for k, v in self._indexes.items ()])

    def __repr__ (self):
        return "<Label ({}): {}>".format (self.name, "[" + ", ".join ([str (each) for each in self._set]) + "]")

    def __getitem__ (self, index):
        return self.item (index)

    def info (self, item):
        return self._origin [item]

    def __len__ (self):
        return len (self._set)

    def index (self, item):
        return self._indexes [item]

    def item (self, index):
        return self._items [index]

    def items (self):
        return self._set

    def top_k (self, arr, k = 1):
        items = []
        for idx in np.argsort (arr)[::-1][:k]:
            items.append (self._items [idx])
        return items

    def setval (self, items, type = np.float, prefix = None):
        arr = np.zeros (len (self._set)).astype (type)
        if not isinstance (items, (dict, list, tuple)):
            items = [items]

        for item, value in items.items ():
            tid = self._indexes.get (item, -1)
            if tid == -1:
                raise KeyError ("{} Not Found".format (item))
            arr [self._indexes [item]] = value

        if prefix is not None:
            return np.concatenate ([prefix, arr])
        else:
            return arr

    def onehot (self, item, type = np.float, prefix = None):
        return self.setval ({item: 1.0}, type, prefix)
    one = onehot #lower version compatible

    def ones (self, items, type = np.float, prefix = None):
        return self.setval (dict ([(item, 1.0) for item in items]), type, prefix)

    def onehots (self, ys, type = np.float, prefix = None):
        # batch onehot
        return np.array ([self.onehot (item, type, prefix) for item in ys])


def onehots (labels, vals):
    if not isinstance (labels, list):
        labels = [labels]
    if not isinstance (vals, list):
        vals = [vals]
    y = []
    for idx, label in enumerate (labels):
        y = label.onehot (vals [idx], prefix = y)
    return y

if __name__ == "__main__":
    v1 = Label (["a", "b", "c", "d", "e"])
    v2 = Label (["a", "b", "c", "d", "e"])
    v3 = Label (["a", "b", "c", "d", "e"])
    f = Labels ([v1, v2, v3])
    print (f.encode ("c", {"a": 0.5}, ['d', 'e']))




