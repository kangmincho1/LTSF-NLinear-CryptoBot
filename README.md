# LTSF-NLinear-CryptoBot
LTSF(Long-term Time Series Forecasting) 테스크에서 transformer 적용에 대한 의문점을 제기한 "Are Transformers Effective for Time Series Forecasting?" 논문에서 간단한 단일 선형 레이어로 구성된 모델이 오히려 더 좋은 성능을 나타낸다고 주장하면서 LTSF-Linear 모델을 제시. 
그 중에서도 데이터셋에서 분포 변화가 있는 경우 LTSF-Linear의 성능을 향상시키기 위해 사용되는 NLinear 모델을 사용하여 16개 암호화폐의 향후 5step(5minutes)를 예측, 이를 롱/숏 포트폴리오 구성하는데 활용.
구성된 롱/숏 포트폴리오에 따른 5분단위 자동 매매 프로그램 구현
![image](https://github.com/kangmincho1/LTSF-NLinear-CryptoBot/assets/72463778/f8f9c9f9-675d-4da0-958d-322e66c9161a)

![image](https://github.com/kangmincho1/LTSF-NLinear-CryptoBot/assets/72463778/6493842b-38a4-4155-a09d-e68a52df0c47)


**Result:** 가상 돈으로 시뮬레이션 해본 결과, trade 10번(50분)만에 누적 수익률 5.628 % 달성



