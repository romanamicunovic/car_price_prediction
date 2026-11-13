Car Price Prediction

Opis projekta

Ovaj projekat koristi mašinsko učenje za procjenu cijene polovnih automobila na osnovu njihovih karakteristika.

Ciljna varijabla je priceUSD, a model koristi informacije kao što su marka vozila, model, godina proizvodnje, kilometraža, tip goriva, zapremina motora, mjenjač, pogon i segment vozila.
Skup podataka

Korišten je skup podataka cars.csv koji sadrži informacije o polovnim automobilima i njihovim prodajnim cijenama.

Analiza podataka

Tokom analize izvršeno je:

- istraživanje strukture podataka;
- provjera nedostajućih vrijednosti;
- analiza numeričkih i kategorijskih kolona;
- analiza distribucije cijena automobila;
- kreiranje dodatnih karakteristika.

Feature Engineering
Kreirane su sljedeće dodatne karakteristike:

- car_age
- engine_volume_liters
- mileage_per_year
Testirani modeli

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

Rezultati

Model	MAE	RMSE	R²
Linear Regression	2088.43	4417.58	0.702
Decision Tree	1529.62	3444.24	0.819
Random Forest	1389.70	2970.69	0.865
Gradient Boosting	1690.58	3350.94	0.829

Finalni model

Kao najbolji model izabran je Random Forest Regressor jer je ostvario:

- najmanji MAE;
- najmanji RMSE;
- najveću vrijednost R².

Model u prosjeku griješi oko 1390 USD pri procjeni cijene automobila.

Pokretanje projekta

Pokretanje treninga:
python src/model_training.py

Evaluacija modela:
python src/model_evaluation.py

Poređenje modela:
python src/model_comparison.py

Potrebne biblioteke
Biblioteke se instaliraju komandom:
pip install -r requirements.txt
