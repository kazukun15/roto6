import os
import streamlit as st
import pandas as pd
import numpy as np

# 機械学習関連
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score

# oneDNN無効（必要に応じて）
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

#############################
# データ読み込み＆前処理関数
#############################
def load_data(uploaded_file):
    """
    CSVファイルを読み込み、数値列のみ抜き出し、
    最後の列をラベルとして返す。
    """
    df = pd.read_csv(uploaded_file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].iloc[:, :-1].values  # 特徴量
    y = df[numeric_cols].iloc[:, -1].values   # ラベル
    return X, y

def preprocess_data(X, y):
    """
    データをスケーリングし、ラベルをOne-Hotエンコードした結果を返す。
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))

    # クラス数
    n_classes = y_encoded.shape[1]
    return X_scaled, y_encoded, n_classes

#############################
# モデル構築関数
#############################
def build_nn_model(input_dim, units, dropout, learning_rate, n_classes):
    """
    ニューラルネットワーク（全結合）モデルを構築
    """
    model = Sequential([
        Dense(units, activation='relu', input_dim=input_dim),
        Dropout(dropout),
        Dense(units, activation='relu'),
        Dropout(dropout),
        Dense(n_classes, activation='softmax')
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_rf_model():
    """
    ランダムフォレストモデル（パラメータは必要に応じて調整）
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model

#############################
# ハイパーパラメータ最適化 (Optuna + NN)
#############################
def optimize_hyperparameters(X_train, y_train, n_classes):
    """
    Optuna で NN のハイパーパラメータを最適化
    """
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

        # 学習
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # 訓練データでの精度(暫定指標)
        _, accuracy = model.evaluate(X_train, y_train, verbose=0)
        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    return study.best_params

#############################
# 予想番号を 5 組作成する関数
#############################
def generate_predictions(model, X_data, n_classes, num_predictions=5):
    """
    入力データからモデルの予測確率を算出し、
    上位6クラスを抽出して 5セット分を表示する例。
    （ロト6など「6つ選ぶ」形を仮定）
    """
    # もしクラス数が 6 未満・以上 ならロジック変更が必要
    # 仮に "クラス数 >= 6" と想定し、上位 6 を抽出
    predictions = model.predict(X_data)  # shape = (サンプル数, n_classes)

    # ここではランダムに 5 サンプル選んで、上位6クラスを取得する
    indices = np.random.choice(range(len(X_data)), size=num_predictions, replace=False)
    result_list = []

    for idx in indices:
        prob = predictions[idx]  # 単一サンプルの予測確率ベクトル
        top6 = np.argsort(prob)[-6:]  # 確率が高い上位6クラスのindex
        top6_sorted = sorted(top6)    # 昇順に並べる (お好みで)
        result_list.append(top6_sorted)

    return result_list

#############################
# Streamlitアプリ本体
#############################
def main():
    st.title("複数分析方法 + 予想番号作成デモ")

    # ファイルアップロード
    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")
    if uploaded_file is not None:
        st.success("CSVファイルのアップロードに成功しました。")
        X, y = load_data(uploaded_file)

        # 分析方法の選択
        analysis_method = st.radio(
            "分析方法を選択してください",
            ("ニューラルネットワーク (単純)", "ランダムフォレスト", "Optuna + ニューラルネットワーク")
        )

        if st.button("分析を開始"):
            st.write("分析を開始します。")
            # 前処理
            X_scaled, y_encoded, n_classes = preprocess_data(X, y)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded.argmax(axis=1)
            )

            # モデル構築・学習フロー
            if analysis_method == "ニューラルネットワーク (単純)":
                # 固定パラメータで NN 構築
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
                model.fit(X_train, y_train.argmax(axis=1))  # ランダムフォレストは one-hot ではなくクラスindexで学習

            else:  # "Optuna + ニューラルネットワーク"
                best_params = optimize_hyperparameters(X_train, y_train, n_classes)
                st.write("Optuna最適化結果:", best_params)

                model = build_nn_model(
                    input_dim=X_train.shape[1],
                    units=best_params['units'],
                    dropout=best_params['dropout'],
                    learning_rate=best_params['learning_rate'],
                    n_classes=n_classes
                )
                model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=1)

            # 評価
            if analysis_method == "ランダムフォレスト":
                # RFの場合は predict_proba で確率を取得できる
                score = model.score(X_test, y_test.argmax(axis=1))
                st.write(f"テストスコア（accuracy）: {score:.4f}")
            else:
                # NNの場合
                loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
                st.write(f"テストデータの精度: {accuracy:.4f}")

            # 予想5組作成
            st.subheader("予想番号 5 組")
            if analysis_method == "ランダムフォレスト":
                # ランダムフォレストの場合: predict_proba を使って確率を得る
                def predict_fn(X):
                    return model.predict_proba(X)  # shape=(n_samples, n_classes)
                original_predict_method = model.predict_proba
            else:
                # NNの場合
                predict_fn = model.predict

            # 予想リスト生成
            predictions_5sets = []
            predictions = predict_fn(X_test)
            # ランダムに5行サンプルを選んでトップ6を抽出
            indices = np.random.choice(len(X_test), size=5, replace=False)
            for idx in indices:
                prob = predictions[idx]
                top6 = np.argsort(prob)[-6:]
                top6_sorted = sorted(top6)
                predictions_5sets.append(top6_sorted)

            for i, pred_6 in enumerate(predictions_5sets, start=1):
                st.write(f"予想第 {i} 組: {pred_6}")

            st.success("分析 + 予想作成が完了しました！")

    else:
        st.info("CSVファイルを選択してください。")

if __name__ == "__main__":
    main()
