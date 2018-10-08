from unittest import TestCase
from gitimpact.core.calc.base import Atom, Container


class TestAtom(TestCase):
    def test_atom_name(self):
        # given
        atom = Atom("test", "Super atom", [])
        expected_result = "Super atom"

        # when
        actual_result = atom.name

        # then
        self.assertEqual(actual_result, expected_result)

    def test_get_no_atoms(self):
        # given
        atom = Atom("test", "Super atom", [])
        expected_result = ["test"]

        # when
        actual_result = [atom.value for atom in atom.atoms]

        # then
        self.assertListEqual(actual_result, expected_result)


class TestContainer(TestCase):
    def test_empty_container(self):
        # given
        container = Container("parent", [])
        expected_result = []

        # when
        actual_result = [atom.value for atom in container.atoms]

        # then
        self.assertListEqual(actual_result, expected_result)

    def test_simple_container(self):
        # given
        atoms = [Atom("atom%d" % idx, "", []) for idx in range(1, 6)]
        container = Container("parent", atoms)
        expected_result = ["atom1", "atom2", "atom3", "atom4", "atom5"]

        # when
        actual_result = [atom.value for atom in container.atoms]

        # then
        self.assertListEqual(actual_result, expected_result)

    def test_recurrent_container(self):
        # given
        atoms = [Atom("atom%d" % idx, "", []) for idx in range(1, 6)]
        containers = [
            Container("parent%d" % container_idx, child_atoms)
                for child_atoms in
                [Atom("atom%d" % idx, "", []) for idx in range(1, 6)]
            for container_idx in range(1, 6)
        ]
        container = Container("parent", containers)
        expected_result = ["atom1", "atom2", "atom3", "atom4", "atom5"]

        # when
        actual_result = [atom.value for atom in container.atoms]

        # then
        self.assertListEqual(actual_result, expected_result)
