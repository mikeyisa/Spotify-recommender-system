# Spotify Song Preference Prediction

## Project by: Michael Yisa

## Introduction
This project uses a dataset of 2017 songs with attributes from Spotify's API to predict whether the user likes a given song. The dataset was acquired from [Kaggle](https://www.kaggle.com/datasets/geomack/spotifyclassification). The user created two playlists on Spotify—one for songs they liked and another for songs they didn't like. The data from both playlists were combined, with a separate target column indicating whether the user liked a song or not. The goal is to predict song preference and identify the most informative features.

## Metadata
**FUN FACT:** Spotify comes from two words, "Spot" and "identify."

- **Acousticness**: The acoustic measure of the track, with values between 0.0 (low confidence) and 1.0 (high confidence).
- **Danceability**: The measure of how suitable a track is for dancing, with values between 0.0 and 1.0.
- **Duration_ms**: The duration of the track in milliseconds.
- **Energy**: The measure of intensity and activity, with values from 0.0 to 1.0.
- **Instrumentalness**: The measure of the instrumental quality of the track, with values from 0.0 to 1.0.
- **Key**: The key of the track.
- **Liveness**: The measure of how likely the track is to be a live recording, with values between 0.0 and 1.0.
- **Loudness**: The measure of the track's loudness in decibels (dB), with values between -60 and 0 dB.
- **Mode**: The modality (major or minor) of a track, with values from 0 (Minor) to 1 (Major).
- **Speechiness**: The measure of the presence of spoken words in a track.
- **Tempo**: The tempo of a track in beats per minute (BPM).
- **Time_signature**: The time signature of a track.
- **Valence**: The measure of musical positiveness, with values between 0.0 and 1.0.
- **Target**: Target values for the track - 0 (Others) and 1 (Favourites).
- **Song_Title**: The title of the song.
- **Artist**: The artist of the song.

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/spotify-song-preference-prediction.git
    ```

2. Navigate to the project directory:
    ```sh
    cd spotify-song-preference-prediction
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. Load the dataset:
    ```python
    import pandas as pd
    df = pd.read_csv('Spot_data.csv')
    ```

2. Preprocess the data:
    ```python
    # Drop unnecessary columns
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df['duration'] = df['duration_ms'].apply(lambda x: round(x/1000))
    df.drop('duration_ms', axis=1, inplace=True)
    ```

3. Standardize the features:
    ```python
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_scalPrep = df.drop(['song_title', 'artist'], axis=1)
    scaler.fit(df_scalPrep.drop("target", axis=1))
    scaled_feat = scaler.transform(df_scalPrep.drop('target', axis=1))
    ```

4. Split the data into training and testing sets:
    ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(scaled_feat, df['target'], test_size=0.35)
    ```

5. Train and evaluate models (K-Nearest Neighbors, Random Forest, etc.):
    ```python
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    ```

6. Use the recommender system to suggest similar songs:
    ```python
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_sim = cosine_similarity(df_scalPrep.drop(columns=['target']))
    
    def get_recommendations(song_index, cosine_sim=cosine_sim, df=df):
        sim_scores = list(enumerate(cosine_sim[song_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        song_indices = [i[0] for i in sim_scores]
        return df.iloc[song_indices]
    
    song_index = 2
    similar_songs = get_recommendations(song_index)
    print(similar_songs)
    ```

## Results
In this project, we successfully built a predictive model and a recommender system based on song attributes. The recommender system suggests similar songs based on their acoustic features using cosine similarity. This project demonstrates the importance of data preparation and feature engineering in creating effective models and recommendations.

## Contributing
Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) first. You can submit issues or fork the project and create a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
- **Name**: Michael Yisa
- **LinkedIn**: [Michael Yisa](https://www.linkedin.com/in/michael-yisa-382a9b249)

Feel free to explore the project, raise issues, and contribute!
