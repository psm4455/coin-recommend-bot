from flask import Flask, request, jsonify
from flask_cors import CORS  # ← 추가
import requests
import pandas as pd
import time
import datetime
import pytz
import json
import schedule
import os
import threading

app = Flask(__name__)
CORS(app)  # 모든 도메인에서의 접근 허용
# ===== 설정 =====
COINMARKETCAP_API_KEY = "efd185db-42f9-4c43-b08e-9d99fa6a1013"
KAKAO_REST_API_KEY = "f37a2090d8a668183699437f586bf241"
KAKAO_REDIRECT_URI = "https://my-kakao-webhook.onrender.com"
KAKAO_TOKEN_URL = "https://kauth.kakao.com/oauth/token"
TOKENS_FILE = "tokens.json"

# ===== 토큰 관리 =====
def load_tokens():
    if not os.path.exists(TOKENS_FILE):
        raise Exception("❌ tokens.json 파일이 없습니다.")
    with open(TOKENS_FILE, "r") as f:
        return json.load(f)

def save_tokens(tokens):
    with open(TOKENS_FILE, "w") as f:
        json.dump(tokens, f)

def get_valid_access_token():
    tokens = load_tokens()
    now = int(time.time())

    if now >= tokens.get("expires_at", 0):
        print("🔄 엑세스 토큰이 만료되어 갱신합니다...")
        refresh_data = {
            "grant_type": "refresh_token",
            "client_id": KAKAO_REST_API_KEY,
            "refresh_token": tokens["refresh_token"]
        }
        response = requests.post(KAKAO_TOKEN_URL, data=refresh_data)
        if response.status_code == 200:
            new_tokens = response.json()
            tokens["access_token"] = new_tokens["access_token"]
            if "refresh_token" in new_tokens:
                tokens["refresh_token"] = new_tokens["refresh_token"]
            tokens["expires_at"] = now + new_tokens.get("expires_in", 0)
            save_tokens(tokens)
            print("✅ 토큰 갱신 완료")
        else:
            raise Exception("❌ 토큰 갱신 실패: " + response.text)
    else:
        print("✅ 기존 토큰 사용 가능")

    return tokens["access_token"]

# ===== 백테스트 관련 함수 =====
def get_top_marketcap_symbols(api_key, top_n=20):
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    headers = {"X-CMC_PRO_API_KEY": api_key}
    params = {"start": "1", "limit": str(top_n), "convert": "USD"}
    response = requests.get(url, headers=headers, params=params)
    data = response.json()["data"]

    result = []
    for item in data:
        symbol = item["symbol"]
        quotes = item.get("quote", {}).get("USD", {})
        price = quotes.get("price", 0)
        percent_change = quotes.get("percent_change_24h", 0)
        volatility = abs(percent_change)
        result.append({
            "symbol": symbol,
            "volatility": volatility,
            "price": price,
        })
    return sorted(result, key=lambda x: x["volatility"], reverse=True)

def get_bybit_symbols():
    url = "https://api.bybit.com/v5/market/instruments-info?category=linear"
    response = requests.get(url)
    data = response.json()["result"]["list"]
    return {item["symbol"] for item in data}

def filter_final_coins(cmc_list, bybit_set, top_n=10):
    final = []
    for coin in cmc_list:
        symbol = coin["symbol"]
        bybit_symbol = f"{symbol}USDT"
        if bybit_symbol in bybit_set:
            coin["symbol"] = bybit_symbol
            final.append(coin)
            if len(final) >= top_n:
                break
    return final

def fetch_ohlcv(symbol):
    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval=15&limit=200"
    response = requests.get(url)
    result = response.json().get("result", {}).get("list", [])
    if not result:
        return pd.DataFrame()
    df = pd.DataFrame(result, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit='ms')
    df = df.astype(float)
    df.set_index("timestamp", inplace=True)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_indicators(df):
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["ma20"] = df["close"].rolling(window=20).mean()
    df["stddev"] = df["close"].rolling(window=20).std()
    df["upper"] = df["ma20"] + 2 * df["stddev"]
    df["lower"] = df["ma20"] - 2 * df["stddev"]
    df["rsi"] = compute_rsi(df["close"])
    df["vol_ma20"] = df["volume"].rolling(window=20).mean()
    df["volume_spike"] = df["volume"] > (df["vol_ma20"] * 2)
    return df

def backtest(df):
    success, fail, total = 0, 0, 0
    position = False
    entry_price = 0

    for i in range(20, len(df)):
        rsi = df["rsi"].iloc[i]
        close = df["close"].iloc[i]
        upper = df["upper"].iloc[i]
        lower = df["lower"].iloc[i]
        ma20 = df["ma20"].iloc[i]
        vol_spike = df["volume_spike"].iloc[i]

        if not position and rsi < 30 and close <= lower and vol_spike:
            entry_price = close
            position = True
            continue

        if position:
            if close >= ma20:
                success += 1
                position = False
            elif close <= lower * 0.98:
                fail += 1
                position = False

    total = success + fail
    return success, fail, total

# ===== 카카오 메시지 전송 =====
def send_kakao_message(token, message):
    url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
    headers = {"Authorization": f"Bearer {token}"}
    data = {
        "template_object": json.dumps({
            "object_type": "text",
            "text": message[:1000],
            "link": {
                "web_url": "https://bybit.com",
                "mobile_web_url": "https://bybit.com"
            },
            "button_title": "확인하기"
        })
    }
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        print("✅ 메시지 전송 성공")
    else:
        print(f"❌ 메시지 전송 실패: {response.status_code} {response.text}")

# ===== 백테스트 실행 함수 =====
def run_daily_backtest():
    try:
        token = get_valid_access_token()
        cmc_filtered = get_top_marketcap_symbols(COINMARKETCAP_API_KEY)
        bybit_symbols = get_bybit_symbols()
        final_coins = filter_final_coins(cmc_filtered, bybit_symbols, top_n=10)

        message = "\n📊 [자동리포트] 백테스트 결과 (최근 3일 기준)\n"
        for coin in final_coins:
            symbol = coin["symbol"]
            volatility = coin["volatility"]
            df = fetch_ohlcv(symbol)
            if df.empty or len(df) < 50:
                message += f"⚠️ 데이터 부족: {symbol}\n"
                continue

            df = compute_indicators(df)
            success, fail, total = backtest(df)
            if total == 0:
                message += f"⏳ 조건 만족 없음: {symbol}\n"
            else:
                win_rate = (success / total) * 100
                message += f"✅ {symbol} | 변동성: {volatility:.2f}% | 성공률: {win_rate:.1f}% ({success}/{total})\n"

        send_kakao_message(token, message)

    except Exception as e:
        print("❌ 오류 발생:", str(e))
        try:
            token = get_valid_access_token()
            send_kakao_message(token, f"❌ 백테스트 실패: {str(e)}")
        except:
            print("❌ 카카오톡 전송 실패")

# ===== 스케줄 설정 =====
def is_korean_time_now(hour):
    now_kst = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
    return now_kst.hour == hour and now_kst.minute == 0

def schedule_runner():
    schedule.every().minute.do(lambda: run_daily_backtest() if is_korean_time_now(7) else None)
    while True:
        schedule.run_pending()
        time.sleep(10)

threading.Thread(target=schedule_runner, daemon=True).start()

# ===== Flask 엔드포인트 =====
@app.route("/")
def index():
    return "✅ 카카오 알림 서버가 실행 중입니다."

@app.route("/send", methods=['GET', 'POST'])
def send_message():
    try:
        data = request.json
        msg = data.get("message", "")
        token = get_valid_access_token()
        send_kakao_message(token, msg)
        return jsonify({"status": "success", "message": "전송 완료"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/")
def home():
    return "카카오톡 웹훅 서버 작동 중!"

# ✅ 수동 실행용 엔드포인트 추가
@app.route("/manual-run", methods=["GET"])
def manual_run():
    try:
        run_daily_backtest()  # 당신이 정의한 백테스트 함수 실행
        return "✅ 수동 실행 완료! 카카오톡 메시지를 전송했습니다."
    except Exception as e:
        return f"❌ 수동 실행 중 오류 발생: {e}"

# ===== 앱 실행 =====
if __name__ == "__main__":
    print("🚀 Flask 서버 실행 중...")
    app.run(host="0.0.0.0", port=10000)

@app.route('/', methods=['GET'])
def home():
    return '''
    <h2>카카오 알림 시스템</h2>
    <form action="/manual-run" method="post">
        <button type="submit">🔄 수동 실행 (즉시 카카오톡 발송)</button>
    </form>
    '''

@app.route('/manual-run', methods=['POST'])
def manual_run():
    run_daily_backtest()
    return '<p>✅ 수동 실행 완료! 카카오톡으로 메시지가 전송되었습니다.</p><a href="/">돌아가기</a>'

