import unittest
from BNMPy import BMatrix

class BNTest(unittest.TestCase):

    def setUp(self):
        # paths to initial networks
        self.network1 = 'input_files/simple_network_test.txt'
        self.network2 = 'input_files/simple_network_test_2.txt'
        self.network3 = 'input_files/simple_network_test_3.txt'

    def test_load_equation_1(self):
        "Tests loading equations - simple network"
        boolean_network = BMatrix.load_network_from_file(self.network1)
        # test boolean_network
        self.assertEqual(boolean_network.N, 4)
        # test runs
        boolean_network.setInitialValues([1, 1, 0, 0])
        results = boolean_network.update(10)
        self.assertEqual(results[1, :].tolist(), [1, 1, 1, 1])
        self.assertEqual(results[9, :].tolist(), [1, 1, 1, 1])
        boolean_network.setInitialValues([1, 1, 1, 1])
        results = boolean_network.update(2)
        self.assertEqual(results[1, :].tolist(), [1, 1, 1, 1])
        boolean_network.setInitialValues([0, 0, 1, 1])
        results = boolean_network.update(10)
        self.assertEqual(results[1, :].tolist(), [0, 0, 0, 0])
        self.assertEqual(results[9, :].tolist(), [0, 0, 0, 0])
        boolean_network.setInitialValues([1, 0, 1, 1])
        results = boolean_network.update(10)
        self.assertEqual(results[1, :].tolist(), [1, 0, 0, 1])
        self.assertEqual(results[9, :].tolist(), [1, 0, 0, 1])
        boolean_network.setInitialValues([1, 0, 0, 0])
        results = boolean_network.update(1)
        self.assertEqual(results[1, :].tolist(), [1, 0, 0, 1])

    def test_load_equation_2(self):
        "Tests loading equations - more complex network"
        boolean_network = BMatrix.load_network_from_file(self.network2)
        # test loading boolean_network
        self.assertEqual(boolean_network.N, 7)
        # test runs
        boolean_network.setInitialValues([1, 0, 0, 0, 0, 0, 0])
        results = boolean_network.update(10)
        self.assertEqual(results[1, :].tolist(), [1, 0, 0, 1, 1, 0, 1])
        boolean_network.setInitialValues([0, 0, 1, 0, 0, 0, 0])
        results = boolean_network.update(10)
        self.assertEqual(results[1, :].tolist(), [0, 1, 1, 0, 1, 1, 0])
        boolean_network.setInitialValues([0, 0, 1, 0, 1, 1, 1])
        results = boolean_network.update(10)
        self.assertEqual(results[1, :].tolist(), [0, 1, 1, 0, 1, 1, 0])

    def test_load_equation_3(self):
        "Tests loading equations - network with feedback"
        boolean_network = BMatrix.load_network_from_file(self.network3)
        # test loading boolean_network
        self.assertEqual(boolean_network.N, 7)
        # test runs
        boolean_network.setInitialValues([1, 0, 0, 0, 0, 0, 1])
        results = boolean_network.update(10)
        self.assertEqual(results[1, :].tolist(), [1, 0, 0, 0, 0, 1, 1])
        self.assertEqual(results[2, :].tolist(), [1, 0, 0, 0, 0, 1, 1])
        boolean_network.setInitialValues([1, 1, 1, 0, 0, 0, 1])
        results = boolean_network.update(10)
        self.assertEqual(results[1, :].tolist(), [1, 1, 1, 0, 1, 1, 0])



if __name__ == '__main__':
    unittest.main()
