
from src.AirportModel import AirportModel


def main():
    model = AirportModel()
    model.loadFlights("./test/data/pdfExampleFlights.csv")
    model.loadConnections('./test/data/pdfExampleConnections.csv')
    model.calculateDelayCost("FT6371", 15)

if __name__ == '__main__':
    main()
