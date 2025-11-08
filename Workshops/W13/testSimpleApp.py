import unittest
import json
from simpleApp import app

class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_home_get(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Welcome to a Simple Flask API!", response.data)

    def test_sqa(self):
        response = self.client.get('/sqa')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Welcome to the SQA course!", response.data)

    def test_ssp(self):
        response = self.client.get('/ssp')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Secure Software Process", response.data)

    def test_vanity(self):
        response = self.client.get('/vanity')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Kate Ella Moreland", response.data)

    def test_mypython(self):
        response = self.client.get('/mypython')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"3.", response.data)

    def test_csse(self):
        response = self.client.get('/csse')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Department of Computer Science and Software Engineering", response.data)

if __name__ == '__main__':
    unittest.main()
