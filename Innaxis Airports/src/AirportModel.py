import sys

class AirportModel:

    def __init__(self):
        self.notice = ""
        # convenience variable to store messages since the program should only output the cost.

        self.indentationLevel = 0

        self.flights = {}
        # Dictionary to store flight data parsed from csv files
        # flights = {
        #   "FT6371":{
        #             "code":FT6371, "departure":660, "arrival":735, "cost":50,
        #             "connections":{"FT2124":0, "FT4125":15}
        #            },
        #   "FT2124":{...},
        #   ...
        #   }

        self.costs = {}
        # Dictionary to store calculated costs for each delayed flight.
        # costs = {
        #   "FT6371":{"delay":15, "cost":750},
        #   "FT2124":{"delay":15, "cost":1500},
        #   "FT4125":{"delay":0, "cost":0}
        #   }



    def loadFlights(self, flightsFile):
        self.flights = {}
        try:
            f = open(flightsFile, "r")
        except FileNotFoundError:
            self.notice = "Could not find file " + flightsFile
            return False
        else:
            try:
                flightEntries = f.readlines()
                for flightString in flightEntries:
                    data = flightString.split(",")
                    flightCode = data[0].strip()
                    departure = self.transformToMinutes(data[1].strip())
                    arrival = self.transformToMinutes(data[2].strip())
                    minuteCost = int(data[3].strip())
                    flight = {"code":flightCode, "departure":departure, "arrival":arrival, "cost":minuteCost, "connections":{}}
                    self.flights[flightCode] = flight
            except Exception as err:
                self.notice = "Error while parsing this line of flight data: " + fligthString + "with the following message:\n  " + str(err)
                f.close()
                self.flights = {}
                return False

            f.close()
            return True

    def loadConnections(self, connectionsFile):
        if (self.flights == {}):
            notice = "No flights loaded yet; cannot add connections without flights."
            return False
        try:
            f = open(connectionsFile, "r")
        except FileNotFoundError:
            notice = "Could not find file " + connectionsFile
            return False
        else:
            try:
                connectionEntries = f.readlines()
                for connectionString in connectionEntries:
                    data = connectionString.split(",")
                    flight = self.flights[data[2].strip()]
                    connFlight = self.flights[data[3].strip()]
                    connBuffer = connFlight["departure"] - flight["arrival"]
                    # If the buffer is negative, consider this an error in the
                    # connections file, as the connecting flight should not be
                    # scheduled to leave before the plane it depends on arrives.
                    if connBuffer < 0:
                        raise Exception("Second flight in flight seccuence departs before first flight arrives.")
                    
                    # Populate a nested dictionary with flights that depend on
                    # passengers from this flight as key and the time buffer
                    # between arrival and departure (connection buffer) as value
                    flight["connections"][connFlight["code"]] = connBuffer

            except Exception as err:
                self.notice = "Error while parsing this line of connection data: " + connectionString + "with the following message:\n  " + str(err)
                f.close()
                return False
            f.close()
            return True


    def transformToMinutes(self, time):
        data = time.split(":")
        hours = int(data[0])
        minutes = int(data[1])
        return (60*hours) + minutes

    def delayCost(self, flightCode, delay):
        self.costs = {}  # Clear the costs hash before computing costs for this run
        self.calculateCosts(flightCode, delay)
        aggregatedCost = 0
        for flight, delayData in self.costs.items():
            aggregatedCost += delayData["cost"] 
        return aggregatedCost

    def calculateCosts(self, flightCode, delay):
        # This function will be called recursively.
        # A delay in one flight causes a tree-like propragation of delays to
        # other flights. Assuming a minimum flight duration of 30 minutes, the
        # longest branch in that tree for a 24h period will have 48 flights. Thus
        # the function may nest up to 48 calls in the worst case.
        
        if flightCode in self.costs and self.costs[flightCode]["delay"] >= delay :
            # Cost has already been calculated for this flight.
            # Unless the delay is greater in this iteration, there is no need to recalculate.
            return

        if delay == 0 :
            self.costs[flightCode] = {"delay":0, "cost":0}
            return

        flight = self.flights[flightCode]
        flightDelayCost = flight["cost"] * delay
        self.costs[flightCode] = {"delay":delay, "cost":flightDelayCost}
        for nextFlight, delayBuffer in flight["connections"].items():
            newDelay = max(delay - delayBuffer, 0)
            self.calculateCosts(nextFlight, newDelay)

    def printConnectionTree(self, flightCode, delay):
        flight = self.flights[flightCode]
        cost = flight["cost"] * delay
        print(self.indentationLevel * "  ", flightCode, "| delay:", delay, "| cost:", cost)
        if delay > 0:
            self.indentationLevel += 1
            for nextFlight, delayBuffer in flight["connections"].items():
                newDelay = max(delay - delayBuffer, 0)
                self.printConnectionTree(nextFlight, newDelay)
            self.indentationLevel -= 1


def main():
    model = AirportModel()
    if (model.loadFlights("./data/traffic.csv") and model.loadConnections('./data/passengers.csv')):
        cost = model.delayCost("AR57965", 20)
        print(cost)
        model.printConnectionTree("AR57965", 20)
    else:
        print("Could not load data.")


if __name__ == '__main__':
    main()

