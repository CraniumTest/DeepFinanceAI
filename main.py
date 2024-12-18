from models.lstm_model import LSTMModel
from models.sentiment_model import SentimentModel
from models.reinforcement_model import ReinforcementModel

def main():
    lstm_model = LSTMModel('data/historical_data.csv')
    data = lstm_model.load_and_prepare_data()
    lstm_model.build_model(input_shape=(data.shape[1], 1))
    
    # Assume data prepared for training
    # lstm_model.train(X_train, y_train)

    sentiment_model = SentimentModel('data/sentiment_data.json')
    sentiment_data = sentiment_model.analyze_and_collect()
    
    reinforcement_model = ReinforcementModel()
    reinforcement_model.train()

if __name__ == "__main__":
    main()
