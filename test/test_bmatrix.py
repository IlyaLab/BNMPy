import unittest
from BNMPy import BMatrix

class BNTest(unittest.TestCase):

    def setUp(self):
        # paths to initial networks
        self.network1 = 'input_files/simple_network_test.txt'

    def test_load_equation(self):
        "Tests loading equations"
        boolean_network = BMatrix.make_boolean_network(self.network1)

if __name__ == '__main__':
    unittest.main()
