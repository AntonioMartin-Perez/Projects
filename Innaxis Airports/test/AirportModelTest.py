import unittest
from src.AirportModel import AirportModel

class AirportModelTest(unittest.TestCase):

    pdfExampleFlights = './test/data/pdfExampleFlights.csv'
    pdfExampleConnections = './test/data/pdfExampleConnections.csv'

    sampleTwoDelaysFlights = './test/data/sampleTwoDelaysFlights.csv'
    sampleTwoDelaysConnections = './test/data/sampleTwoDelaysConnections.csv'

    sampleLoopFlights = './test/data/sampleLoopFlights.csv'
    sampleLoopConnections = './test/data/sampleLoopConnections.csv'

    @classmethod
    def setUpClass(cls):
	# Examples given in the practice pdf
        f = open(AirportModelTest.pdfExampleFlights, "w+")
        f.write('''FT6371,11:00,12:15,50
FT2124,12:15,14:00,100
FT4125,12:30,13:45,200
''')
        f.close()
        f = open(AirportModelTest.pdfExampleConnections, "w+")
        f.write('''p432,1000,FT6371,FT2124
p967,6000,FT6371,FT4125
''')
        f.close()

	# Example were one flight is delayed due to two other flights with
	# different delay times
        f = open(AirportModelTest.sampleTwoDelaysFlights, "w+")
        f.write('''FT6371,11:00,12:15,50
FT2124,12:15,14:00,100
FT4125,12:30,13:45,200
FT8479,14:00,17:30,80
''')
        f.close()
        f = open(AirportModelTest.sampleTwoDelaysConnections, "w+")
        f.write('''p432,1000,FT6371,FT2124
p967,6000,FT6371,FT4125
p968,6000,FT4125,FT8479
p969,6000,FT2124,FT8479
''')
        f.close()

	# Example where the connections f contains an error that causes a
	# loop
        f = open(AirportModelTest.sampleLoopFlights, "w+")
        f.write('''FT6371,11:00,12:15,50
FT2124,12:15,14:00,100
FT4125,12:30,13:45,200
FT8479,14:00,17:30,80
''')
        f.close()
        f = open(AirportModelTest.sampleLoopConnections, "w+")
        f.write('''p432,1000,FT6371,FT2124
p967,6000,FT6371,FT4125
p968,6000,FT4125,FT8479
p969,6000,FT2124,FT8479
p970,6000,FT8479,FT6371
''')
        f.close()


    def setUp(self):
        self.model = AirportModel()

    def testErrorOnInexistingFile(self):
        self.assertFalse(self.model.loadFlights("nofilepresent.csv"))
        self.assertFalse(self.model.loadConnections("nofilepresent.csv"))

    def testFindExistingFile(self):
        self.assertTrue(self.model.loadFlights(AirportModelTest.pdfExampleFlights))
        self.assertTrue(self.model.loadConnections(AirportModelTest.pdfExampleConnections))

    def testLoadFlightData(self):
        self.assertTrue(self.model.loadFlights(AirportModelTest.pdfExampleFlights))

        self.assertTrue("FT6371" in self.model.flights)
        flight = self.model.flights["FT6371"]
        self.assertEqual(flight["code"], "FT6371")
        self.assertEqual(flight["departure"], (60*11))
        self.assertEqual(flight["arrival"], (60*12)+15)
        self.assertEqual(flight["cost"], 50)

        self.assertTrue("FT2124" in self.model.flights)
        flight = self.model.flights["FT2124"]
        self.assertEqual(flight["code"], "FT2124")
        self.assertEqual(flight["departure"], (60*12)+15)
        self.assertEqual(flight["arrival"], (60*14))
        self.assertEqual(flight["cost"], 100)

        self.assertTrue("FT4125" in self.model.flights)
        flight = self.model.flights["FT4125"]
        self.assertEqual(flight["code"], "FT4125")
        self.assertEqual(flight["departure"], (60*12)+30)
        self.assertEqual(flight["arrival"], (60*13)+45)
        self.assertEqual(flight["cost"], 200)

    def testLoadConnectionData(self):
        self.assertTrue(self.model.loadFlights(AirportModelTest.pdfExampleFlights))
        self.assertTrue(self.model.loadConnections(AirportModelTest.pdfExampleConnections))

        flight = self.model.flights["FT6371"]
        connectionBuffers = {"FT2124":0, "FT4125":15}
        self.assertEqual(flight["connections"], connectionBuffers)

        flight = self.model.flights["FT2124"]
        connectionBuffers = {}
        self.assertEqual(flight["connections"], connectionBuffers)

        flight = self.model.flights["FT4125"]
        connectionBuffers = {}
        self.assertEqual(flight["connections"], connectionBuffers)

    def testZeroMinDelay(self):
        self.assertTrue(self.model.loadFlights(AirportModelTest.pdfExampleFlights))
        self.assertTrue(self.model.loadConnections(AirportModelTest.pdfExampleConnections))
        self.assertEqual(self.model.delayCost("FT6371", 0), 0)

    def testFifteenMinDelay(self):
        self.assertTrue(self.model.loadFlights(AirportModelTest.pdfExampleFlights))
        self.assertTrue(self.model.loadConnections(AirportModelTest.pdfExampleConnections))
        # delay cost = FT6371 cost * delay  +  FT2124 cost * propagated delay  +  FT4125 * propagated delay
        #            = 50 * 15  +  100 * 15  +  200 * 0  =  2250
        self.assertEqual(self.model.delayCost("FT6371", 15), 2250)
        self.assertEqual(self.model.delayCost("FT2124", 15), 1500)
        self.assertEqual(self.model.delayCost("FT4125", 15), 3000)

    def testThirtyMinDelay(self):
        self.assertTrue(self.model.loadFlights(AirportModelTest.pdfExampleFlights))
        self.assertTrue(self.model.loadConnections(AirportModelTest.pdfExampleConnections))
        self.assertEqual(self.model.delayCost("FT6371", 30), 7500)
        self.assertEqual(self.model.delayCost("FT2124", 30), 3000)
        self.assertEqual(self.model.delayCost("FT4125", 30), 6000)

    def testSixtyMinDelay(self):
        self.assertTrue(self.model.loadFlights(AirportModelTest.pdfExampleFlights))
        self.assertTrue(self.model.loadConnections(AirportModelTest.pdfExampleConnections))
        self.assertEqual(self.model.delayCost("FT6371", 60), 18000)
        self.assertEqual(self.model.delayCost("FT2124", 60), 6000)
        self.assertEqual(self.model.delayCost("FT4125", 60), 12000)

    def testTwoDelaysForSameFlight(self):
        self.assertTrue(self.model.loadFlights(AirportModelTest.sampleTwoDelaysFlights))
        self.assertTrue(self.model.loadConnections(AirportModelTest.sampleTwoDelaysConnections))
        # Flight FT8479 depends on passengers from both FT4125 and FT2124.
        # A delay of 40 minutes in FT6371 triggers two different delays in FT8479.
        # FT6371 delay of 40 min
        #   FT2124 delay of 40 min     (40 - 0 buffer)
        #       FT8479 delay of 40 min (40 - 0 buffer)
        #   FT4125 delay of 25 min     (40 - 15 buffer)
        #       FT8479 delay of 10 min (25 - 15 buffer)
	# Only the longest delay should be considered for any flight.
        # delay cost = FT6371 cost * delay  +  FT2124 cost * propagated delay  +  FT4125 * propagated delay  +  FT8479 cost * highest propagated delay
        #            = 50 * 40  +  100 * 40  +  200 * 25  +  80 * 40  =  14200
        self.assertEqual(self.model.delayCost("FT6371", 40), 14200)

    def testDetectLoop(self):
        # According to sampleLoopConnections.csv, flight FT8479 (arrival time
        # 14:00) has passengers connecting to FT6371 (departure time 11:00).
        # This is clearly an error in the file, but it could cause an infinite
        # loop while calculating costs as FT6371 causes a delay in FT8479 which
        # in turn causes a delay in FT6371. The code should detect this and
        # abort the loading of connections.
        self.assertTrue(self.model.loadFlights(AirportModelTest.sampleLoopFlights))
        self.assertFalse(self.model.loadConnections(AirportModelTest.sampleLoopConnections))





def main():
    unittest.main()

if __name__ == '__main__':
    main()
