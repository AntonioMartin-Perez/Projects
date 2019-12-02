from src.AirportModel import AirportModel

tableFile = "./data/flightDelayTable.csv"

model = AirportModel()
if (model.loadFlights("./data/traffic.csv") and model.loadConnections('./data/passengers.csv')):
#if (model.loadFlights("./test/data/pdfExampleFlights.csv") and model.loadConnections("./test/data/pdfExampleConnections.csv")):
    try:
        f = open(tableFile, "w+")
    except FileNotFoundError:
        print("Could not open", tableFile, "for writing the data")
        quit()
    else:
        f.write("flight, total cost 15 min. delay (€), total cost 30 min. delay (€), total cost 60 min. delay (€)\n")

        for flight in model.flights:
            print("\nNext flight:", flight)
            print("  Calculating 15 min delay for flight", flight)
            cost15 = model.delayCost(flight, 15)
            print(cost15)
            print("  Calculating 30 min delay for flight", flight)
            cost30 = model.delayCost(flight, 30)
            print(cost30)
            print("  Calculating 60 min delay for flight", flight)
            cost60 = model.delayCost(flight, 60)
            print(cost60)
            f.write(flight + "," + str(cost15) + "," + str(cost30) + "," + str(cost60) + "\n")


        f.close()
else:
    print("Could not load data.")

