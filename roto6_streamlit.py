import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import optuna

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
def build_model(input_dim, units, dropout, learning_rate, n_classes):
    model = Sequential([
        Dense(units, activation='relu', input_dim=input_dim),
        Dropout(dropout),
        Dense(units, activation='relu'),
        Dropout(dropout),
        Dense(n_classes, activation='softmax')  # クラス数を動的に設定
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ハイパーパラメータ最適化関数
def optimize_hyperparameters(X_train, y_train, n_classes):
    def objective(trial):
        units = trial.suggest_int('units', 32, 256, step=32)
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        epochs = trial.suggest_int('epochs', 10, 50, step=10)
        batch_size = trial.suggest_int('batch_size', 16, 128, step=16)

        model = build_model(
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

            # 分析開始ボタン
            if st.button("分析を開始する"):
                st.write("データを分析しています...")
                X_scaled, y_encoded, n_classes = preprocess_data(X, y)
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

                # プログレスバー
                progress_bar = st.progress(0)
                best_params = optimize_hyperparameters(X_train, y_train, n_classes)
                progress_bar.progress(50)

                # 最適化されたハイパーパラメータでモデルを再構築
                model = build_model(
                    input_dim=X_train.shape[1],
                    units=best_params['units'],
                    dropout=best_params['dropout'],
                    learning_rate=best_params['learning_rate'],
                    n_classes=n_classes
                )
                model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=1)
                progress_bar.progress(100)

                _, accuracy = model.evaluate(X_test, y_test)
                st.write("テストデータの精度:", accuracy)
                st.success("分析が完了しました！")

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
