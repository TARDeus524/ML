# Пояснення коду Lab 2 (`main.ipynb`)

Цей файл розбирає **код** роботи. Передумова: теорію з `THEORY_GUIDE.md` вже опановано, тому тут
акцент на *чому код написаний саме так* і *що робить кожен блок*. Нумерація осередків відповідає
порядку в ноутбуці.

---

## 0. Загальна карта ноутбука

```
Етап 1  (комірки 1–20)   EDA → розбиття → відбір ознак → масштабування
Етап 2  (комірки 21–23)  клас NumPy-логрегресії + перевірка градієнтів
Етап 3  (комірки 24–30)  підбір гіперпараметрів → централізована модель → метрики
Етап 4  (комірки 31–36)  IID та Dirichlet розподіл між клієнтами
Етап 5  (комірки 37–38)  run_fed_avg
Етап 6  (комірки 39–54)  експерименти α, E, порівняння моделей, зважування
Звіт    (комірки 55–57)  теорія, підсумок, загальний висновок
```

Принцип: **усі функції — «чисті»**, тобто приймають дані/параметри й повертають результат; жодних
глобальних змінних, крім `class_names`, `NUM_CLASSES`, `NUM_FEATURES` і констант. Це дозволяє
перевикористовувати `run_fed_avg` для різних `α`, `E` та режиму зважування.

---

## 1. Імпорти та відтворюваність (комірка 1)

```python
SEED = 42
np.random.seed(SEED)
```

Усі випадкові операції прив'язані до `SEED`. Для розбиттів і локального навчання додатково
створюються **окремі** генератори `np.random.default_rng(seed)`, щоб результат не залежав від того,
скільки випадкових чисел «з'їли» попередні виклики. Саме тому IID-розподіл і кожен `α` дають
повторювані результати.

---

## 2. EDA (комірки 3–6)

- `df.info()`, `df.isnull().sum()`, `df.describe().T`, `value_counts()` — структура, пропуски,
  статистики, розподіл класів.
- Класи перетворюються в мітки 0..6:

```python
class_names = sorted(df['Class'].unique())
class_to_idx = {name: i for i, name in enumerate(class_names)}
idx_to_class = {i: name for i, name in enumerate(class_names)}
df['target'] = df['Class'].map(class_to_idx)
```

`sorted(...)` важливий: порядок міток фіксований, тому `class_names[i]` завжди відповідає
стовпцю/рядку `i` у матрицях і графіках.

---

## 3. Розбиття (комірки 12–13)

```python
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=SEED)
X_pool, X_val, y_pool, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15/0.85, stratify=y_temp, random_state=SEED)
```

Тонкість: після першого розбиття залишилось 85% даних. Щоб отримати **15% від початкового** обсягу,
у другому виклику беруть `0.15/0.85 ≈ 0.1765`, а не `0.15`. Підсумок: 70/15/15.
`stratify` зберігає частки класів; комірка 13 виводить таблицю для перевірки цього.

---

## 4. Відбір ознак (комірки 15–17)

```python
def select_features_greedy(X_tr, y_tr, names, corr_threshold=0.95):
    F, p = f_classif(X_tr, y_tr)
    corr_m = np.corrcoef(X_tr, rowvar=False)
    order = np.argsort(F)[::-1]
    kept = []
    for i in order:
        if all(abs(corr_m[i, j]) < corr_threshold for j in kept):
            kept.append(int(i))
    kept = sorted(kept, key=lambda i: -F[i])
    return kept, F, p
```

Логіка: спершу ознаки сортуються за **ANOVA F** (наскільки добре кожна окремо розділяє класи), далі
жадібно лишаємо ознаку, лише якщо вона не корелює (>0.95) з уже обраними. Результат: 16 → 10 ознак.
Зверни увагу: цикл `for` тут — **по ознаках**, не по прикладах, тому заборона методички (цикли по
прикладах для loss/градієнтів) не порушена. Далі `X_pool = X_pool[:, kept_idx]` тощо.

---

## 5. Масштабування та one-hot (комірка 20)

```python
mean = np.mean(X_pool, axis=0)
std  = np.std(X_pool, axis=0) + 1e-8
X_pool_s = (X_pool - mean) / std
X_val_s  = (X_val  - mean) / std
X_test_s = (X_test - mean) / std
```

μ і σ рахуються **лише на pool** і застосовуються до val/test (без витоку). `+1e-8` захищає від
ділення на 0 для константних ознак.

```python
def to_one_hot(y_arr, num_classes=NUM_CLASSES):
    Y = np.zeros((len(y_arr), num_classes))
    Y[np.arange(len(y_arr)), y_arr] = 1.0
    return Y
```

One-hot через **fancy indexing** — без циклів. Три форми: `Y_pool_oh`, `Y_val_oh`, `Y_test_oh`.

---

## 6. Клас моделі (комірка 22)

```python
class MultinomialLogisticRegression:
    def __init__(self, num_features, num_classes, reg_lambda=0.01, seed=0):
        self.W = np.zeros((num_features, num_classes))   # d x C
        self.b = np.zeros((1, num_classes))              # 1 x C
        self.rng = np.random.default_rng(seed)
```

Ключові методи:

```python
def softmax(self, Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))   # стабільність
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)    # axis=1 -> по класах рядка

def predict_proba(self, X):
    return self.softmax(X @ self.W + self.b)               # Z = XW + b

def predict(self, X):
    return np.argmax(self.predict_proba(X), axis=1)        # клас з макс. ймовірністю
```

Втрата та градієнти (усе векторизовано):

```python
def compute_loss(self, X, Y_oh):
    P = np.clip(self.predict_proba(X), 1e-15, 1 - 1e-15)   # щоб log(0) != -inf
    ce = -(1.0 / m) * np.sum(Y_oh * np.log(P))
    ridge = (self.reg_lambda / 2.0) * np.sum(self.W ** 2)  # b НЕ штрафується
    return ce + ridge

def compute_gradients(self, X, Y_oh):
    diff = self.predict_proba(X) - Y_oh
    dW = (1.0 / m) * (X.T @ diff) + self.reg_lambda * self.W
    db = (1.0 / m) * np.sum(diff, axis=0, keepdims=True)
    return dW, db
```

`diff = P − Y` — це та сама «магія» softmax+CE з теорії (B6). `keepdims=True` зберігає форму
`(1, C)`, інакше `db` став би `(C,)` і не транслювався б із `b`.

```python
def fit_epoch(self, X, Y_oh, lr=0.1, batch_size=64):
    idx = self.rng.permutation(m)          # перемішування щоепохи
    Xs, Ys = X[idx], Y_oh[idx]
    for i in range(0, m, batch_size):      # цикл ПО БАТЧАХ (дозволено)
        dW, db = self.compute_gradients(Xs[i:i+batch_size], Ys[i:i+batch_size])
        self.W -= lr * dW
        self.b -= lr * db
```

Цикл — по батчах; всередині — лише матричні операції. Один виклик `fit_epoch` = одна епоха.

---

## 7. Перевірка градієнтів (комірка 23)

```python
def directional_grad_check(model, X, Y_oh, eps=1e-6, seed=0):
    ...
    numerical = (lp - lm) / (2 * eps)   # центральна різниця
```

Перевірка через **похідну за напрямком**: обираємо випадковий вектор `V`, порівнюємо
`<∇J, V>` (аналітично) з `(J(w+εV) − J(w−εV)) / 2ε` (чисельно). Відносна похибка ~1e-10 ⇒
формули градієнтів правильні. Це і є «допоміжна перевірка», дозволена методичкою.

---

## 8. Метрики та тренування (комірка 25)

```python
def evaluate_model(model, X_eval, y_eval):
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)
    return {'Accuracy': accuracy_score(...),
            'Macro-F1': f1_score(..., average='macro'),
            'Balanced Acc': balanced_accuracy_score(...),
            'Log-Loss': log_loss(y_eval, y_proba, labels=list(range(NUM_CLASSES)))}
```

`labels=` в `log_loss` потрібен, щоб порядок стовпців відповідав 0..6.

```python
def train_model(X_tr, Y_tr, X_val, Y_val_oh, y_val, lr, batch_size, reg_lambda,
                epochs, seed, record_history=False):
    for _ in range(epochs):
        model.fit_epoch(...)
        if record_history: ...
    return model, hist
```

Єдина функція навчання для всіх випадків (підбір, фінал, локальні моделі) — уникає дублювання коду.

---

## 9. Підбір гіперпараметрів (комірки 26–27)

```python
for lr in [0.05, 0.1, 0.2]:
    for bs in [64, 128]:
        for lam in [0.0, 0.001, 0.01, 0.1]:
            m, _ = train_model(..., epochs=100)
            tuning_rows.append({... 'val_macro_f1': ...})
tuning_df = pd.DataFrame(tuning_rows).sort_values('val_macro_f1', ascending=False)
```

**Grid search 3×2×4 = 24 конфігурації**, сортування за `val_macro_f1`. Переможець:
`lr=0.2, batch=128, λ=0.0`. Комірка 27 будує heatmap `lr × batch` і графік залежності від `λ`. Усе —
**на Global Validation**, test не торкаємо.

---

## 10. Централізована модель (комірки 29–30)

```python
central_model, central_hist = train_model(..., lr=BEST_LR, batch_size=BEST_BATCH,
                                          reg_lambda=BEST_LAMBDA, epochs=200,
                                          record_history=True)
```

Перенавчання з найкращими гіперпараметрами на 200 епох. Метрики на **Global Test**:
Accuracy 0.9197, Macro-F1 0.9323, Balanced Acc 0.9312, Log-Loss 0.2142. Confusion matrix показує
плутанину між схожими сортами.

---

## 11. Розподіл між клієнтами (комірки 33–35)

```python
def create_iid_split(X_pool, y_pool, K=5, seed=SEED):
    idx = rng.permutation(len(y_pool))
    chunks = np.array_split(idx, K)              # приблизно рівні частини
    return {k: (X_pool[chunks[k]], y_pool[chunks[k]]) for k in range(K)}
```

```python
def create_dirichlet_non_iid_split(X_pool, y_pool, K=5, alpha=0.3, ...):
    while True:                                  # повторюємо, поки є крихітні клієнти
        for c in range(num_classes):
            proportions = rng.dirichlet(np.repeat(alpha, K))
            cut = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            for k, part in enumerate(np.split(idx_c, cut)):
                client_idx[k].extend(part.tolist())
        if min(sizes) >= min_size or attempt >= 100: break
```

Для кожного класу генеруються частки `Dir(α)` і клас ділиться між клієнтами. `min_size`-цикл
гарантує, що жоден клієнт не отримає <10 прикладів. `client_distribution_table` і
`plot_client_distribution` (heatmap абсолютних значень і часток) документують розподіл; перевірка
`n_samples < 10` виводить число проблемних клієнтів.

---

## 12. FedAvg (комірка 38)

```python
def run_fed_avg(client_data, X_val, Y_val_oh, y_val, rounds=30, local_epochs=5,
                lr=0.1, batch_size=64, reg_lambda=0.01, weighted=True, seed=SEED):
    sizes = np.array([len(client_data[k][1]) for k in range(K)], dtype=float)
    weights = sizes / sizes.sum() if weighted else np.ones(K) / K
    for r in range(rounds):
        for k in range(K):
            local = MultinomialLogisticRegression(...)      # нова копія
            local.W, local.b = global_model.W.copy(), global_model.b.copy()
            for _ in range(local_epochs): local.fit_epoch(...)
            agg_W += weights[k] * local.W                    # зважене середнє
        global_model.W, global_model.b = agg_W, agg_b
        history['val_f1'].append(...)                        # моніторинг на валідації
    return global_model, history
```

Дві важливі деталі:
1. `local.W = global_model.W.copy()` — **копія**, інакше `fit_epoch` зіпсує глобальні ваги
   (у NumPy присвоєння посиланням).
2. Прапорець `weighted` дозволяє одним і тим самим кодом порівняти зважене й незважене середнє.

`NUM_FEATURES = X_pool_s.shape[1]` визначено після функції, але використовується під час виклику,
тому проблеми немає.

---

## 13. Експерименти (комірки 41–50)

**Експеримент 1 (α).** Для `α ∈ {5, 1, 0.3}`: створити Non-IID розподіл → `run_fed_avg` → метрики на
test. Будується крива `val_f1` по раундах і bar-plot фінальної якості.

**Експеримент 2 (E).** Фіксуємо `α=0.3`, перебираємо `E ∈ {1,5,10}`. Результат: якість **монотонно
падає** зі зростанням `E` — демонстрація client drift.

**Локальні моделі (комірки 48–50).**

```python
def train_all_local_models(client_data, epochs=200, ...):
    for k, (Xk, yk) in client_data.items():
        m = MultinomialLogisticRegression(...)
        for _ in range(epochs): m.fit_epoch(...)
        models[k] = m
```

Тренуємо **усіх** клієнтів, оцінюємо кожного на Global Test, будуємо зведену таблицю з
`mean/min/max` по локальних моделях. Висновок: локальні моделі сильно варіюють (0.53–0.87 accuracy),
глобальна стабільніша, централізована — найкраща.

---

## 14. Демонстрація зважування (комірка 53)

```python
def create_noisy_client_split(X_pool, y_pool, K=5, small_size=300, seed=SEED):
    ...
    for k in range(K - 1):
        data[k] = (X_pool[s], rng.integers(0, NUM_CLASSES, len(s)))  # випадкові мітки
    data[K - 1] = (X_pool[big], y_pool[big])                          # великий чистий клієнт
```

Контрольований експеримент: 4 малі клієнти по 300 з **шумними мітками** + 1 великий чистий (~8327).
Зважене усереднення: Macro-F1 0.925, Log-Loss 0.400. Незважене: Macro-F1 0.868, Log-Loss 1.154.
Це прямий доказ питання 10–11: вага має бути пропорційною обсягу/надійності даних.

---

## 15. Підсумок і звіт (комірки 55–57)

Комірка 56 динамічно друкує підсумкові числа (не хардкодяться), тому звіт завжди узгоджений із
результатами запуску. Комірка 57 — загальний висновок.

---

## 16. Типові проблеми та як їх ловити

| Симптом | Причина | Рішення |
|---|---|---|
| `log_loss` дає `nan`/`inf` | ймовірність 0 | кліпінг `P` у `compute_loss` |
| Ваги «розходяться» (`nan`) | завеликий `lr` | зменшити `lr`, додати Ridge |
| Локальні моделі псують глобальну | присвоєння за посиланням | `global_model.W.copy()` |
| Різні результати між запусками | не зафіксоване зерно | `np.random.default_rng(SEED)` |
| Метрики «завищені» | витік (масштабування до розбиття) | розбиття ПЕРШИМ |
| Клієнт з 0 прикладів | малий `α` | `min_size`-цикл у Dirichlet |

---

## 17. Що сказати на захисті (зв'язок код ↔ теорія)

1. Порядок комірок = порядок пайплайну без витоку (A1–A4).
2. `select_features_greedy` = F-тест + боротьба з мультиколінеарністю (A6).
3. `softmax` і `compute_gradients` = формули B2/B6; перевірені чисельно (комірка 23).
4. `train_model` + grid search = підбір гіперпараметрів на validation (E1).
5. `run_fed_avg` = FedAvg із зваженим середнім (D3–D4).
6. Експерименти α та E = Non-IID і client drift (D5–D6).
7. Порівняльна таблиця = centralized vs federated vs local (D7).
