# Пояснення коду Lab 3 (`main.ipynb`)

Цей файл розбирає **код** роботи «Дослідження ефективності TF-IDF зважених ембеддингів».
Передумова: теорію з `THEORY_GUIDE.md` вже опановано, тому тут акцент на *чому код написаний саме
так* і *що робить кожен блок*. Нумерація комірок відповідає порядку в ноутбуці (0 — заголовок).

> Позначення: **к.** = номер комірки в `main.ipynb`.

---

## 0. Загальна карта ноутбука

```
0–2   заголовок · імпорти · SEED
3–17  Етап 1  EDA: структура, баланс, довжини, артефакти, очищення
18–20 Етап 2  препроцесинг spaCy (лематизація, стоп-слова, is_alpha, lowercase)
21–23 Етап 2.5 розбиття train/test (стратифіковане, ДО навчання представлень)
24–26 Етап 3  навчання FastText (skip-gram, субсловні n-грами)
27–32 Етап 4  векторизація: Mean Pooling vs TF-IDF Weighted Mean + геометрія
33–38 Етап 5  класифікатори SVM/LogReg/KNN + метрики + матриці плутанини
39–41 Етап 5+ криві навчання (подолання «стелі»)
42–45 Етап 6  t-SNE для обох методів агрегації
46–49 Лінгвістична інтерпретація (TF-IDF маркери)
50–53 Дослідницькі питання · підсумок · висновок · література
```

Ключовий принцип: **усі перетворення, що «вчаться» (FastText, idf, scaler), навчаються лише на
train** — це запобігає data leakage (див. `THEORY_GUIDE.md`, A2).

---

## 1. Імпорти та відтворюваність (к. 0–2)

```python
SEED = 42
```

* `warnings.filterwarnings("ignore")` — глушимо технічні попередження (напр., про збіжність), щоб
  вивід був чистим; на логіку не впливає.
* `logging.getLogger("gensim").setLevel(logging.WARNING)` — gensim під час навчання репортує
  прогрес у stderr; переводимо його в тихий режим.
* `gensim.models.FastText`, `sklearn.*`, `spacy`, `TSNE`, `silhouette_score` — усе, що потрібно.
* `sns.set_theme(...)`, `plt.rcParams` — єдиний стиль графіків.

Чому фіксоване зерно важливе: `train_test_split`, `FastText`, `SVC`, `LogisticRegression`, t-SNE і
підвибірки для кривих — усі випадкові; `SEED` робить результати відтворюваними.

---

## 2. Етап 1 — EDA

### Структура (к. 4)
```python
df_raw = pd.read_csv("ai_vs_human_text_2026.csv")
pd.DataFrame({"dtype": ..., "non_null": ..., "n_unique": ...})
```
Зводимо типи/пропуски/унікальність в одну таблицю. `n_unique` для `text_content` одразу натякає на
дублікати (966 унікальних на 2000 рядків).

### Баланс класів (к. 7)
* `value_counts()` — кількість на клас; `value_counts(normalize=True)` — частки.
* Лівий бар-плот — баланс; правий — `pd.crosstab(label, domain, normalize="index")` показує, що
  домени розподілені між класами рівномірно (жоден домен не «підказує» клас).

### Розподіл довжин (к. 10)
Гістограми `word_count` та `avg_sentence_length` для кожного класу (накладені, `alpha=0.6`).
Показує, що розподіли перекриваються → довжина не є роздільною ознакою.

### Артефакти (к. 13)
```python
n_empty      = (text.strip() == "").sum()
n_dup_text   = df_raw["text_content"].duplicated().sum()
n_short      = (df_raw["word_count"] < 5).sum()
n_cross      = (df_raw.groupby("text_content")["label"].nunique() > 1).sum()
```
Перевіряємо чотири типи дефектів. Знаходимо **1034 повних дублікати** (переважно `human` — 755) і
нуль «суперечливих» міток.

### Очищення (к. 16)
```python
df = df_raw.drop_duplicates(subset=["text_content"], keep="first").reset_index(drop=True)
```
Дублікати шкідливі тим, що за випадкового розбиття той самий текст потрапить і в train, і в test,
штучно завищуючи метрики. Після очищення — 966 унікальних текстів (human 579 / ai 387).
Бічні графи — `countplot` домену та джерела за класом.

---

## 3. Етап 2 — Препроцесинг (к. 19)

```python
nlp = spacy.load("en_core_web_sm")
texts = df["text_content"].tolist()
docs = nlp.pipe(texts, batch_size=64, disable=["ner", "parser", "attribute_ruler"])
tokens = [[t.lemma_.lower() for t in d if t.is_alpha and not t.is_stop] for d in docs]
```

* `nlp.pipe(..., batch_size=64)` — обробляє документи батчами (швидше, ніж цикл `nlp(text)`).
* `disable=["ner", "parser", "attribute_ruler"]` — вимикаємо компоненти, **не потрібні** для
  лематизації/стоп-слів; це прискорює обробку в рази.
* Умова `t.is_alpha and not t.is_stop` реалізує дві вимоги методички (тільки літери, без
  стоп-слів), а `t.lemma_.lower()` — лематизацію й нижній регістр.
* У ноутбуці також задана функція `preprocess(text)` — точно як у методичці, для наочності; але для
  966 документів використовується продуктивніша форма через `nlp.pipe`.

---

## 4. Етап 2.5 — Розбиття (к. 22)

```python
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
```

Розбиття стоїть **перед** FastText і TF-IDF: саме так ми гарантуємо, що idf та word-vectors не
«бачать» test. Потім друкуємо таблицю перевірки стратифікації (`train`/`test` частки класів).

---

## 5. Етап 3 — FastText (к. 25)

```python
ft = FastText(sentences=train_tokens, vector_size=100, window=5, min_count=2,
              sg=1, workers=1, seed=SEED, epochs=30)
```

* `sentences=train_tokens` — корпус **із токенів** (не рядків); кожен документ — список слів.
* `min_count=2` — ігноруємо слова, що трапляються один раз (шум).
* `sg=1` — skip-gram; `vector_size=100` — розмірність вектора.
* `workers=1` — **один потік для повної відтворюваності**: багатопотокове навчання gensim
  оновлює ваги асинхронно й дає трохи різні вектори навіть із фіксованим `seed`. Корпус малий,
  тож на швидкість це майже не впливає, зате результати стабільні між запусками.
* OOV-демонстрація:
  ```python
  oov = "zorbulatrix"
  oov in ft.wv.key_to_index          # False — слова немає у словнику
  np.linalg.norm(ft.wv[oov])         # працює! вектор із субсловних n-грам
  ```
  Це і є ключова властивість FastText: представлення існує навіть для невідомих слів.

---

## 6. Етап 4 — Векторизація (к. 28)

Дві функції агрегації — майже дослівно з методички:

```python
def mean_pooling(tokens, model):
    vectors = [model.wv[t] for t in tokens if t in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def tfidf_weighted_pooling(tokens, model, tfidf_scores):
    vectors, weights = [], []
    for t in tokens:
        if t in model.wv and t in tfidf_scores:
            vectors.append(model.wv[t]); weights.append(tfidf_scores[t])
    weights = np.array(weights); weights /= weights.sum()
    return np.average(vectors, axis=0, weights=weights) if vectors else np.zeros(model.vector_size)
```

* `idf_scores` — словник `термін → idf`, побудований `TfidfVectorizer(smooth_idf=False, norm=None)`
  на **train**-документах (`train_docs`). `smooth_idf=False` дає `idf = log(n/df)` — точно як у
  формулі (4) методички.
* `weights /= weights.sum()` — нормування ваг (формула (6): ділення на `Σ idf`).
* `np.average(..., weights=...)` — зважене середнє векторів.
* `build_matrix(idx_list, pooling)` повертає матрицю `(n_docs × 100)`; будуємо окремо для train і
  test обома методами.

### Геометрія (к. 31)
```python
def geometry(X, y, name):
    c0, c1 = X[y==0].mean(0), X[y==1].mean(0)
    centroid = np.linalg.norm(c0 - c1)
    within   = малий розкид у кожному класі
    return {..., "Силует": silhouette_score(X, y)}
```
Кількісно порівнюємо простори Mean vs TF-IDF метриками, незалежними від класифікатора:
відстань центрів, внутрішньокласовий розкид, їх відношення та силует.

---

## 7. Етап 5 — Класифікатори (к. 34)

```python
classifiers = {
  "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=SEED),
  "LogReg":    LogisticRegression(max_iter=1000, random_state=SEED),
  "KNN":       KNeighborsClassifier(n_neighbors=5),
}
for agg, (Xtr, Xte) in {...}.items():
    scaler = StandardScaler().fit(Xtr)          # масштабування ЛИШЕ на train
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    for clf_name, clf in classifiers.items():
        clf.fit(Xtr_s, y_train)
        y_pred = clf.predict(Xte_s)
        y_prob = clf.predict_proba(Xte_s)[:, 1]  # потрібно для ROC-AUC
        ... f1_score(..., average="macro"), accuracy_score, balanced_accuracy_score, roc_auc_score
```

* `StandardScaler` навчається на train — інакше був би витік (середні/дисперсії тесту).
* `predict_proba[:, 1]` — ймовірність класу `ai` (тому для SVM потрібно `probability=True`).
* `average="macro"` — усереднення F1 по класах однаково (чутливо до меншого класу).
* Результати збираються в `results_df`; `predictions` зберігає `confusion_matrix` для кожної пари.
* `results_df` візуалізується груповим бар-плотом (`к. 35`) — 4 метрики × 3 класифікатори ×
  2 агрегації.
* Матриці плутанини (`к. 37`) — `sns.heatmap` у сітці 2×3 (рядки — агрегація, стовпці — класифікатор).

---

## 8. Етап 5+ — Криві навчання (к. 40)

Повний train дає стелю (F1=1.0) для всіх методів. Щоб **зробити різницю видимою**, навчаємо на
підвибірці з `N` прикладів на клас:

```python
sizes = [10, 15, 20, 30, 45, 70]
for N in sizes:
    sub = concat(rng.choice(ai_pool, N), rng.choice(hu_pool, N))
    ... для кожного методу агрегації та класифікатора: fit на підвибірці, predict на test
```

* `rng = np.random.default_rng(SEED)` — ізольований генератор; підвибірки **з train-пулу**, не з
  усього масиву (без витоку).
* `pos = {v: k for k, v in enumerate(train_idx)}` — переводить «глобальний» індекс у позицію в
  train-матриці, щоб правильно вибрати рядки `X_mean_train[...]`.
* Для кожної `N` — свій `StandardScaler`, навчений лише на підвибірці.
* Результат — `curve_df` і два line-плоти (по одному на метод агрегації).

Саме тут видно «прокляття розмірності»: KNN різко відстає за малих `N`, а SVM/LogReg виходять на
стелю швидко.

---

## 9. Етап 6 — t-SNE (к. 43)

```python
def plot_tsne(X, y, title, ax):
    Z = TSNE(n_components=2, perplexity=30, init="pca",
             learning_rate="auto", random_state=SEED).fit_transform(X)
    ... scatter для human/ai ...
    sil = silhouette_score(Z, y)
```

* `init="pca"` — стабільніша ініціалізація, ніж випадкова.
* Матриці `X_mean_all`/`X_tfidf_all` будуються над `idx` (усі документи) у **вихідному порядку**,
  щоб узгодити з масивом міток `y` (важливо: порядок рядків train+test не збігається з порядком `y`).
* Дві окремі фігури (Mean і TF-IDF), як вимагає методичка; у заголовку — 2D-силует.

---

## 10. Лінгвістична інтерпретація (к. 47)

```python
vec_uni = TfidfVectorizer(smooth_idf=False, norm=None)
D_uni = vec_uni.fit_transform(train_docs).toarray()
ai_m, hu_m = D_uni[y_train==1].mean(0), D_uni[y_train==0].mean(0)
diff = ai_m - hu_m
order = np.argsort(diff)[::-1]           # найбільш "AI-специфічні" терміни
```

* Те саме для біграм (`ngram_range=(2, 2)`).
* `top_terms(...)` повертає топ-`k` термінів для AI (найбільший `diff`) і для human (найменший).
* Бар-чарти (к. 48) — найважливіші терміни кожного класу. Багато AI-маркерів мають
  `TF-IDF(Human)=0` → чисті дискримінативні ознаки.

---

## 11. Підсумок і висновок (к. 51–53)

Комірка 51 динамічно друкує ключові числа з `results_df`, `geom`, `curve_df` (найкраща
конфігурація, розкид F1, силуети, KNN за N=15), щоб числа в тексті гарантовано збігалися з
виводом. Далі — markdown «Дослідницькі питання» (5 відповідей з конкретними числами) та
«Загальний висновок».

---

## 12. Головні пастки, які враховано

| Пастка | Як уникнуто |
|---|---|
| Витік через дублікати тексту | `drop_duplicates` перед розбиттям (1034 → 0) |
| Витік idf/FastText/scaler із test | усі fit — лише на train |
| Незіставність порядку рядків у t-SNE | `X_*_all` будуються над `idx`, як `y` |
| Незбалансовані класи | macro-F1, balanced accuracy, ROC-AUC, стратифікація |
| Стеля метрик на легких даних | додатковий експеримент кривих навчання |
| Недетермінованість FastText | `workers=1`: багатопотокове навчання gensim дає різні вектори навіть із `seed` |
| Відтворюваність | `SEED` + ізольовані `default_rng` |
