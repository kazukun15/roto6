import os
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score

# oneDNN無効化（必要に応じて）
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

# データ読み込み関数
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].iloc[:, :-1].values  # 特徴量
    y = df[numeric_cols].iloc[:, -1].values   # ラベル
    return np.array(X), np.array(y)  # NumPy 配列に変換

# データ前処理関数
def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))

    # One-Hotエンコード後の列数をクラス数とする
    n_classes = y_encoded.shape[1]
    return X_scaled, y_encoded, n_classes

# モデル構築関数
def build_nn_model(input_dim, units, dropout, learning_rate, n_classes):
    model = Sequential([
        Dense(units, activation='relu', input_dim=input_dim),
        Dropout(dropout),
        Dense(units, activation='relu'),
        Dropout(dropout),
        Dense(n_classes, activation='softmax')  # クラス数を動的に設定
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_rf_model():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model

# ハイパーパラメータ最適化関数
def optimize_hyperparameters(X_train, y_train, n_classes):
    def objective(trial):
        units = trial.suggest_int('units', 32, 256, step=32)
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        epochs = trial.suggest_int('epochs', 10, 50, step=10)
        batch_size = trial.suggest_int('batch_size', 16, 128, step=16)

        model = build_nn_model(
            input_dim=X_train.shape[1],
            units=units,
            dropout=dropout,
            learning_rate=learning_rate,
            n_classes=n_classes
        )

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        _, accuracy = model.evaluate(X_train, y_train, verbose=0)
        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    return study.best_params

# 予想番号を 5 組作成する関数
def generate_predictions(model, X_data, n_classes, num_predictions=5):
    """
    Generates prediction numbers based on model probabilities.
    """
    predictions = model.predict(X_data)  # shape = (n_samples, n_classes)
    
    # Select 5 random samples
    indices = np.random.choice(range(len(X_data)), size=num_predictions, replace=False)
    result_list = []
    
    for idx in indices:
        prob = predictions[idx]
        top6 = np.argsort(prob)[-6:]
        top6_sorted = sorted(top6)
        # Convert numpy integers to Python integers
        top6_sorted = [int(num) for num in top6_sorted]
        result_list.append(top6_sorted)
    
    return result_list

# Streamlitアプリケーション
def main():
    st.title("ロト6データ分析アプリ")

    # ファイルアップロード
    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")
    if uploaded_file is not None:
        try:
            # データ読み込み
            X, y = load_data(uploaded_file)
            st.success("データを正常に読み込みました！")

            # 分析方法の選択
            analysis_method = st.radio(
                "分析方法を選択してください",
                ("ニューラルネットワーク (単純)", "ランダムフォレスト", "Optuna + ニューラルネットワーク")
            )

            if st.button("分析を開始する"):
                st.write("データを分析しています...")
                # 前処理
                X_scaled, y_encoded, n_classes = preprocess_data(X, y)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded.argmax(axis=1)
                )

                # プログレスバー
                progress_bar = st.progress(0)
                best_params = None
                if analysis_method == "Optuna + ニューラルネットワーク":
                    best_params = optimize_hyperparameters(X_train, y_train, n_classes)
                    st.write("Optuna 最適化結果:", best_params)
                    progress_bar.progress(50)

                # モデル構築・学習
                if analysis_method == "ニューラルネットワーク (単純)":
                    model = build_nn_model(
                        input_dim=X_train.shape[1],
                        units=64,
                        dropout=0.2,
                        learning_rate=1e-3,
                        n_classes=n_classes
                    )
                    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
                elif analysis_method == "ランダムフォレスト":
                    model = build_rf_model()
                    model.fit(X_train, y_train.argmax(axis=1))
                else:  # Optuna + NN
                    model = build_nn_model(
                        input_dim=X_train.shape[1],
                        units=best_params['units'],
                        dropout=best_params['dropout'],
                        learning_rate=best_params['learning_rate'],
                        n_classes=n_classes
                    )
                    model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=1)
                progress_bar.progress(100)

                # 評価
                if analysis_method == "ランダムフォレスト":
                    # ランダムフォレストは .score() を使う
                    score = model.score(X_test, y_test.argmax(axis=1))
                    st.write(f"テストスコア（accuracy）: {score:.4f}")
                else:
                    # NNの場合
                    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
                    st.write(f"テストデータの精度: {accuracy:.4f}")

                # 予想5組作成
                st.subheader("予想番号 5 組")
                predictions_5sets = generate_predictions(model, X_test, n_classes, num_predictions=5)

                # Display predictions using columns for better readability
                for i, pred in enumerate(predictions_5sets, start=1):
                    st.markdown(f"### 予想第 {i} 組")
                    cols = st.columns(len(pred))
                    for col, num in zip(cols, pred):
                        col.markdown(f"**{num}**")

                st.success("分析 + 予想作成が完了しました！")

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
    else:
        st.info("CSVファイルを選択してください。")

if __name__ == "__main__":
    main()
